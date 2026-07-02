#!/usr/bin/env python3
"""
Script to identify images with the largest prediction errors and download them from IDR.
"""

import argparse
from pathlib import Path
from io import BytesIO

import imageio.v3 as iio
import numpy as np
import pandas as pd
import requests


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


def _login_idr_session(session):
    """
    Log in to IDR as the public guest user to obtain a session cookie.

    The IDR webclient render endpoints require an authenticated session even
    for public data.  Credentials are the well-known public guest account.
    """
    base = "https://idr.openmicroscopy.org"
    # Fetch login page to get CSRF token
    r = session.get(f"{base}/webclient/login/", timeout=15)
    r.raise_for_status()
    csrf = session.cookies.get('csrftoken', '')
    login_data = {
        'username': 'public',
        'password': 'public',
        'csrfmiddlewaretoken': csrf,
        'server': 1,
        'noredirect': 1,
    }
    r_login = session.post(
        f"{base}/webclient/login/",
        data=login_data,
        headers={'Referer': f"{base}/webclient/login/"},
        timeout=15
    )
    r_login.raise_for_status()


def download_idr_image(image_id, output_dir, session, timeout=120):
    """
    Download all channel planes for an IDR image and save as a multi-channel TIFF.

    Uses the IDR webclient render_image endpoint (one request per channel plane),
    the same approach used by the original IDR training-data download scripts.

    The render_image endpoint requires:
      - An authenticated session (public/public login)
      - Channel specs in OMERO format: ``N|min:max$RRGGBB`` for each channel,
        with a leading ``-`` to render a channel as inactive (grayscale isolate).
      - The ``m=g`` (greyscale model) and ``format=tif`` query parameters.
      - Special characters (``|``, ``$``) must NOT be URL-encoded.

    Note: Some older IDR datasets have their pixel files archived on a storage
    volume (``demo_2``) that is no longer mounted on the rendering servers.
    These images will fail with a pixel-buffer error; this is an IDR-side
    infrastructure limitation and cannot be resolved client-side.

    Args:
        image_id: The IDR image ID to download
        output_dir: Directory to save the downloaded file
        session: requests.Session to use (shared across calls for efficiency)
        timeout: HTTP request timeout in seconds per channel request

    Returns:
        Path to the saved TIFF file, or None if the download failed
    """
    base_url = "https://idr.openmicroscopy.org"
    output_path = Path(output_dir) / f"image_{image_id}.tif"

    try:
        # 1. Fetch image dimensions and verify pixel accessibility via imgData
        r_check = session.get(f"{base_url}/webgateway/imgData/{image_id}/", timeout=30)
        if r_check.status_code == 200:
            idata = r_check.json()
            exc = idata.get('Exception')
            if exc:
                print(f"  [SKIP] Pixel buffer unavailable (archived dataset): {exc[:80]}")
                return None
            channels = idata.get('channels', [])
            size_c = len(channels)
        else:
            # Fall back to the JSON API for dimensions
            r_meta = session.get(f"{base_url}/api/v0/m/images/{image_id}/", timeout=30)
            r_meta.raise_for_status()
            size_c = r_meta.json().get('data', {}).get('Pixels', {}).get('SizeC', 1)

        if size_c == 0:
            print(f"  [WARN] Could not determine channel count for image {image_id}")
            return None

        # 2. Download each channel plane individually (Z=0, T=0)
        #    Use the simple c=N query parameter — the same approach used by the original
        #    generate-training-data-web-api.py script.  render_image renders one channel
        #    at a time as an 8-bit greyscale/RGB TIFF.  The complex "N|min:max$COLOR"
        #    channel-spec syntax is NOT used here; it causes HTTP 400 errors on IDR.
        planes = []
        for c in range(size_c):
            url = f"{base_url}/webclient/render_image/{image_id}/"
            params = {"c": c, "z": 0, "t": 0, "format": "tif"}
            r_plane = session.get(url, params=params, timeout=timeout)
            if r_plane.status_code != 200:
                msg = r_plane.content[:100].decode('utf-8', errors='replace')
                if r_plane.status_code == 404 and 'Cannot find Image' in msg:
                    print(f"  [SKIP] Image {image_id} not available on IDR render servers "
                          f"(pixel data may be on offline storage)")
                else:
                    print(f"  [ERROR] Channel {c} returned HTTP {r_plane.status_code}: {msg}")
                return None
            plane = iio.imread(BytesIO(r_plane.content))
            if plane.ndim > 2:
                plane = np.squeeze(plane)
            # render_image returns an 8-bit greyscale or RGB TIFF; flatten to 2-D
            if plane.ndim == 3:
                plane = plane[:, :, 0]
            planes.append(plane)

        # 3. Stack channels and save as multi-channel TIFF
        # Shape: (SizeC, SizeY, SizeX) — standard channel-first layout
        stack = np.stack(planes, axis=0)
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

    # Shared session for IDR requests — log in once as the public guest
    idr_session = requests.Session()
    print("Logging in to IDR as public guest...")
    try:
        _login_idr_session(idr_session)
        print("  Login OK")
    except Exception as e:
        print(f"  Login failed: {e} — downloads may fail")

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
        downloaded_file = download_idr_image(image_id, output_path, idr_session, timeout=args.timeout)

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
    idr_session.close()


if __name__ == "__main__":
    main()
