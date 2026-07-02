#!/usr/bin/env python3
"""
Script to identify images with the largest prediction errors and download them from IDR.

Since IDR migrated to OME-Zarr (mid-2025), pixel data is no longer served through the
old webclient render_image endpoint.  Images are now accessed by reading the OME-Zarr
stores hosted on the EBI S3 bucket:

    https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/...

The Zarr path for each image is discovered via the IDR file-annotations API and then
downloaded with the ``zarr`` + ``fsspec[http]`` libraries.  No login is required.
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import requests
import zarr
import fsspec


def get_idr_metadata(image_id):
    """
    Fetch metadata for a given image ID from IDR API.

    Args:
        image_id: The image ID to query

    Returns:
        Dictionary containing channel names, project ID, etc.
    """
    base_url = f"https://idr.openmicroscopy.org/api/v0/m/images/{image_id}/"
    metadata = {
        "Channels": "N/A",
        "Project": "N/A",
        "Dataset": "N/A",
        "Name": "N/A"
    }

    try:
        # 1. Get basic image info and channels
        r = requests.get(base_url, timeout=10)
        if r.status_code == 200:
            data = r.json().get('data', {})
            metadata["Name"] = data.get('Name', 'N/A')
            pixels = data.get('Pixels', {})
            channels = pixels.get('Channels', [])
            if channels:
                metadata["Channels"] = ", ".join([c.get('Name', 'Unknown') for c in channels])

        # 2. Get Dataset/Project info (for non-HCS images)
        r_ds = requests.get(f"{base_url}datasets/", timeout=10)
        if r_ds.status_code == 200:
            datasets = r_ds.json().get('data', [])
            if datasets:
                ds = datasets[0]
                metadata["Dataset"] = ds.get('Name', 'N/A')
                ds_id = ds.get('@id')

                # 3. Get Project info
                r_prj = requests.get(f"https://idr.openmicroscopy.org/api/v0/m/datasets/{ds_id}/projects/", timeout=10)
                if r_prj.status_code == 200:
                    projects = r_prj.json().get('data', [])
                    if projects:
                        metadata["Project"] = projects[0].get('Name', 'N/A')

        # 4. Fallback for HCS images (plates/screens) — datasets/ returns empty for these
        if metadata["Dataset"] == "N/A":
            r_img_data = requests.get(
                f"https://idr.openmicroscopy.org/webclient/imgData/{image_id}/",
                timeout=10
            )
            if r_img_data.status_code == 200:
                img_data = r_img_data.json()
                well_id = img_data.get('meta', {}).get('wellId')
                if well_id:
                    r_plates = requests.get(
                        f"https://idr.openmicroscopy.org/api/v0/m/wells/{well_id}/plates/",
                        timeout=10
                    )
                    if r_plates.status_code == 200:
                        plates = r_plates.json().get('data', [])
                        if plates:
                            plate = plates[0]
                            metadata["Dataset"] = f"Plate: {plate.get('Name', 'N/A')}"
                            plate_id = plate.get('@id')
                            r_screens = requests.get(
                                f"https://idr.openmicroscopy.org/api/v0/m/plates/{plate_id}/screens/",
                                timeout=10
                            )
                            if r_screens.status_code == 200:
                                screens = r_screens.json().get('data', [])
                                if screens:
                                    metadata["Project"] = f"Screen: {screens[0].get('Name', 'N/A')}"

    except Exception:
        # Silently fail for metadata, just return N/A
        pass

    return metadata


def get_zarr_url(image_id):
    """
    Discover the OME-Zarr S3 URL for an IDR image.

    IDR stores Zarr data at ``https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/``.
    The path within that bucket is stored as a file annotation on the image (or its
    parent well/plate) in IDR.  This function queries the IDR file-annotations API
    to find that path.

    Strategy (tried in order):
    1. File annotations directly on the image.
    2. File annotations on the parent well (HCS data).
    3. File annotations on the parent plate (HCS data).

    Args:
        image_id: The IDR image ID.

    Returns:
        A full HTTPS URL to the OME-Zarr store, or ``None`` if not found.
    """
    base = "https://idr.openmicroscopy.org"
    s3_base = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4"

    def _find_zarr_in_annotations(url):
        """Return the Zarr path from file annotations at *url*, or None."""
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                return None
            for ann in r.json().get('annotations', []):
                path = ann.get('file', {}).get('path', '') or ann.get('fileName', '')
                if '.zarr' in path:
                    # Strip any leading slash and return the full S3 URL
                    return f"{s3_base}/{path.lstrip('/')}"
        except Exception:
            pass
        return None

    # 1. Annotations directly on the image
    zarr_url = _find_zarr_in_annotations(
        f"{base}/webclient/api/annotations/?type=file&image={image_id}"
    )
    if zarr_url:
        return zarr_url

    # 2 & 3. For HCS images, look up the well and plate
    try:
        r_img = requests.get(f"{base}/webclient/imgData/{image_id}/", timeout=10)
        if r_img.status_code == 200:
            meta = r_img.json().get('meta', {})
            well_id = meta.get('wellId')
            if well_id:
                zarr_url = _find_zarr_in_annotations(
                    f"{base}/webclient/api/annotations/?type=file&well={well_id}"
                )
                if zarr_url:
                    return zarr_url
                # Try the plate
                r_plates = requests.get(
                    f"{base}/api/v0/m/wells/{well_id}/plates/", timeout=10
                )
                if r_plates.status_code == 200:
                    plates = r_plates.json().get('data', [])
                    if plates:
                        plate_id = plates[0].get('@id')
                        zarr_url = _find_zarr_in_annotations(
                            f"{base}/webclient/api/annotations/?type=file&plate={plate_id}"
                        )
                        if zarr_url:
                            return zarr_url
    except Exception:
        pass

    return None


def download_idr_image(image_id, output_dir, timeout=120):
    """
    Download all channel planes for an IDR image and save as a multi-channel TIFF.

    Uses the OME-Zarr store hosted on the EBI S3 bucket, which is the access method
    recommended by IDR following their migration to OME-Zarr infrastructure (mid-2025).
    No authentication is required; the bucket is publicly readable over HTTPS.

    The Zarr URL is discovered automatically via the IDR file-annotations API.
    The OME-Zarr array layout is ``(T, C, Z, Y, X)``; this function extracts the
    first T and Z plane for every channel and saves a ``(C, Y, X)`` multi-channel TIFF.

    Args:
        image_id: The IDR image ID to download.
        output_dir: Directory to save the downloaded file.
        timeout: Not used directly (retained for API compatibility); zarr uses fsspec
                 default timeouts for HTTP.

    Returns:
        Path to the saved TIFF file, or None if the download failed.
    """
    output_path = Path(output_dir) / f"image_{image_id}.tif"

    try:
        # 1. Find the Zarr store URL for this image
        zarr_url = get_zarr_url(image_id)
        if not zarr_url:
            print(f"  [SKIP] No OME-Zarr store found for image {image_id} "
                  f"(study may not yet be converted to Zarr)")
            return None

        print(f"  Zarr: {zarr_url}")

        # 2. Open the Zarr store over HTTP using fsspec
        mapper = fsspec.get_mapper(zarr_url)
        root = zarr.open(mapper, mode='r')

        # OME-Zarr multi-scale layout: resolution levels are '0', '1', ...
        # Use the highest resolution (level '0').
        if '0' not in root:
            print(f"  [ERROR] Unexpected Zarr structure for image {image_id}: "
                  f"no '0' array found.  Keys: {list(root.keys())}")
            return None

        arr = root['0']  # shape: (T, C, Z, Y, X)
        if arr.ndim < 5:
            print(f"  [ERROR] Unexpected array ndim={arr.ndim} for image {image_id}")
            return None

        size_c = arr.shape[1]

        # 3. Extract first T=0, Z=0 plane for every channel → (C, Y, X)
        planes = [arr[0, c, 0, :, :] for c in range(size_c)]
        stack = np.stack(planes, axis=0)  # (C, Y, X)

        # 4. Save as multi-channel TIFF
        iio.imwrite(str(output_path), stack, extension='.tif')

        if output_path.stat().st_size > 0:
            return output_path
        output_path.unlink(missing_ok=True)

    except Exception as e:
        print(f"  [ERROR] Download failed for image {image_id}: {e}")
        if output_path.exists():
            output_path.unlink(missing_ok=True)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Find images with largest prediction errors and download them from IDR"
    )
    parser.add_argument(
        "--csv_file",
        default="Z:/working/barryd/hpc/python/Torch-Unet/eval_run_2025-12-16_09-45-57/test_predictions_2025-12-16_09-45-58.csv",
        help="Path to the CSV file with predictions"
    )
    parser.add_argument(
        "--output-dir",
        default="largest_errors_images",
        help="Output directory to store the downloaded images (default: largest_errors_images)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of images with largest errors to extract (default: 20)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds for image downloads (default: 120)"
    )

    args = parser.parse_args()

    # Load the CSV file
    print(f"Loading CSV file: {args.csv_file}")
    df = pd.read_csv(args.csv_file)

    # Calculate absolute error between Actual_Label and Predicted_Label
    df['Error'] = abs(df['Actual_Label'] - df['Predicted_Label'])

    # Sort by error in descending order and get top N
    df_sorted = df.nlargest(args.top_n, 'Error')

    print(f"\nTop {args.top_n} images with largest prediction errors and IDR metadata:")
    print("-" * 150)
    print(f"{'Image_ID':<10} {'Actual':<10} {'Predicted':<12} {'Error':<10} {'Channels':<30} {'Project':<30} {'Dataset':<30}")
    print("-" * 150)

    for idx, row in df_sorted.iterrows():
        image_id_val = int(row['Image_ID'])
        metadata = get_idr_metadata(image_id_val)
        
        print(
            f"{row['Image_ID']:<10} {row['Actual_Label']:<10.4f} {row['Predicted_Label']:<12.4f} {row['Error']:<10.4f} "
            f"{metadata['Channels'][:28]:<30} {metadata['Project'][:28]:<30} {metadata['Dataset'][:28]:<30}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n\nCreating output directory: {output_path}")

    # Download image files from IDR
    downloaded_count = 0
    failed_count = 0

    print(f"\nDownloading images from IDR...")
    print("-" * 70)

    for idx, row in df_sorted.iterrows():
        image_id = int(row['Image_ID'])
        error = row['Error']

        print(f"Image_ID {image_id} (error: {error:.4f}): downloading from IDR...", end=" ", flush=True)
        downloaded_file = download_idr_image(image_id, output_path, timeout=args.timeout)

        if downloaded_file:
            size_mb = downloaded_file.stat().st_size / (1024 * 1024)
            print(f"OK ({size_mb:.1f} MB) -> {downloaded_file.name}")
            downloaded_count += 1
        else:
            print("FAILED")
            failed_count += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY:")
    print(f"  Total images downloaded: {downloaded_count}")
    print(f"  Failed downloads: {failed_count}")
    print(f"  Output directory: {output_path.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
