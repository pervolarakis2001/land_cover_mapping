import rasterio
from rasterio.transform import from_origin
import numpy as np
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import os, gc
from rasterio.windows import Window
from shapely.geometry import box
import matplotlib.pyplot as plt
from s2cloudless import S2PixelCloudDetector
from tqdm import tqdm


def read_multiband_tiff(filepath):
    """
    Reads a multi-band GeoTIFF file and returns the data, transform, and CRS.

    Parameters:
        filepath (str): Path to the input TIFF file.

    Returns:
        tuple: (data, transform, crs)
            - data (numpy.ndarray): 3D array (bands, height, width)
            - transform (Affine): Georeferencing transform
            - crs (CRS): Coordinate Reference System
    """
    with rasterio.open(filepath) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs
    return data, transform, crs


def write_multiband_tiff(output_path, data, transform, crs):
    """
    Writes a 3D NumPy array as a multi-band GeoTIFF file.

    Parameters:
        output_path (str): Path to save the output TIFF file.
        data (numpy.ndarray): 3D array of shape (bands, height, width) containing raster data.
        transform (Affine): Affine transform for georeferencing.
        crs (dict or rasterio.crs.CRS): Coordinate Reference System for the output raster.

    Returns:
        None
    """
    count = data.shape[0]  # Number of bands
    height = data.shape[1]  # Rows
    width = data.shape[2]  # Columns
    print(data.shape)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,  # Number of bands
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        # Write all bands
        dst.write(data)


def get_tiff_info(filepath):
    """
    Extracts metadata and spatial information from a GeoTIFF file.

    Parameters:
        filepath (str): Path to the TIFF file.

    Returns:
        dict: Dictionary containing the following keys:
            - 'size' (tuple): (width, height) of the raster.
            - 'crs' (CRS): Coordinate Reference System.
            - 'gsd' (tuple): Ground Sample Distance (pixel size) in (x, y).
            - 'bounds' (BoundingBox): Bounding box of the raster.
            - 'transform' (Affine): Affine transform of the raster.
            - 'band_count' (int): Number of bands in the file.
    """
    with rasterio.open(filepath) as src:
        width = src.width
        height = src.height
        # CRS (Coordinate Reference System)
        crs = src.crs
        # GSD (Ground Sample Distance)
        # Get pixel size from transform
        pixel_size_x = src.transform[0]
        pixel_size_y = -src.transform[4]
        # Get bounds
        bounds = src.bounds
        # Get transform
        transform = src.transform
        # Get number of bands
        band_count = src.count
        tag = src.tags()

        info = {
            "size": (width, height),
            "crs": crs,
            "gsd": (pixel_size_x, pixel_size_y),
            "bounds": bounds,
            "transform": transform,
            "band_count": band_count,
            "tags": tag,
            "description": src.descriptions,
            "dtype": src.dtypes,
            "nodata": src.nodata,
        }
        return info


def pansharpening(src_path, classi_path, output_path):
    """
    Performs pansharpening and cloud masking on raw Sentinel-2 imagery:
    - Reads spectral bands and applies cloud masks using MSK_CLASSI_B00.jp2
    - Masks out cloud-affected pixels at each band's native resolution
    - Resamples only the valid (cloud-free) pixels to a uniform 10m resolution
    - Stacks the cleaned bands into a multiband GeoTIFF
    - Saves the final cloud-free, pansharpened output to disk
    """

    # Find B02 (10m) as reference
    ref_path = next(
        os.path.join(src_path, f)
        for f in os.listdir(src_path)
        if f.endswith("_B02.jp2")
    )
    with rasterio.open(ref_path) as ref:
        ref_height = ref.height
        ref_width = ref.width
        ref_transform = ref.transform
        ref_crs = ref.crs

    # Load and resample cloud masks to 10m
    with rasterio.open(classi_path) as mask_src:
        opaque_mask = np.empty((ref_height, ref_width), dtype=np.uint8)
        cirrus_mask = np.empty((ref_height, ref_width), dtype=np.uint8)

        reproject(
            source=mask_src.read(1),
            destination=opaque_mask,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
        )
        reproject(
            source=mask_src.read(2),
            destination=cirrus_mask,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
        )

    cloud_mask = (opaque_mask == 1) | (cirrus_mask == 1)
    del opaque_mask, cirrus_mask
    gc.collect()

    band_arrays = []
    for f in sorted(os.listdir(src_path)):
        if "TCI.jp2" in f or not f.endswith(".jp2"):
            continue

        file_path = os.path.join(src_path, f)
        with rasterio.open(file_path) as src:
            band_native = src.read(1)
            native_shape = band_native.shape

            # Build cloud mask at native resolution
            if src.res[0] != 10:
                band_mask = np.empty(native_shape, dtype=np.uint8)
                reproject(
                    source=cloud_mask.astype(np.uint8),
                    destination=band_mask,
                    src_transform=ref_transform,
                    src_crs=ref_crs,
                    dst_transform=src.transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest,
                )
                clean_band = np.where(band_mask == 0, band_native, np.nan)
            else:
                clean_band = np.where(~cloud_mask, band_native, np.nan)

            # Resample to 10m if needed
            if src.res[0] != 10:
                clean_band = rasterio.warp.reproject(
                    source=clean_band,
                    destination=np.empty((ref_height, ref_width), dtype=np.float32),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                )[0]

            band_arrays.append(clean_band.astype(np.float32))
            del band_native, clean_band
            gc.collect()

    stack = np.stack(band_arrays)
    write_multiband_tiff(output_path, stack, ref_transform, ref_crs)
    del stack, band_arrays, cloud_mask
    gc.collect()


def reproject_raster(
    src_path, dst_path, dst_crs, target_resolution=None, target_bounds=None
):
    """
    Reprojects a raster file to a new coordinate reference system (CRS), with optional
    resolution and bounding box definition.

    Parameters:
        src_path (str): Path to the input raster file.
        dst_path (str): Path to save the reprojected output raster.
        dst_crs (str or dict): Target CRS
        target_resolution (tuple, optional): Target pixel size as (x_res, y_res) in CRS units.
        target_bounds (tuple, optional): Bounding box (left, bottom, right, top) in target CRS.

    Returns:
        None. The output is written to 'dst_path'.
    """
    with rasterio.open(src_path) as src:
        if target_bounds is None:
            # Use default bounds by transforming from source CRS
            transform, width, height = calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=target_resolution,
            )
        else:
            # Use explicitly specified bounds
            left, bottom, right, top = target_bounds
            if target_resolution:
                xres, yres = target_resolution
                width = int((right - left) / xres)
                height = int((top - bottom) / yres)
                transform = from_origin(left, top, xres, yres)
            else:
                # Approximate resolution from source
                src_res = src.transform[0]
                width = int((right - left) / src_res)
                height = int((top - bottom) / src_res)
                transform = from_origin(
                    left, top, (right - left) / width, (top - bottom) / height
                )

        # Prepare output metadata
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "nodata": -9999,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "YES",
            }
        )
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            # Reproject each band
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )


def create_patches_from_tile(tile_path, output_dir, patch_size=256, max_nan_ratio=0.25):
    """
    Splits a raster tile into patches of patch_size x patch_size and saves only those
    with acceptable amount of NaNs (masked/cloudy pixels).

    Parameters:
        tile_path (str): Path to the input GeoTIFF image (already cloud-masked).
        output_dir (str): Directory to save the extracted patches.
        patch_size (int): Size of the square patches (default: 256).
        max_nan_ratio (float): Max allowed NaN ratio (0–1) in a patch (default: 0.25).

    Returns:
        int: Number of patches saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(tile_path) as src:
        height, width = src.height, src.width
        profile = src.profile
        bands = src.count
        nodata_val = src.nodata

        patch_id = 0
        for top in tqdm(range(0, height, patch_size)):
            for left in range(0, width, patch_size):
                window = Window(left, top, patch_size, patch_size)

                if left + patch_size > width or top + patch_size > height:
                    continue  # skip patches at the edge

                patch = src.read(
                    window=window
                )  # shape: (bands, patch_size, patch_size)

                # Check for NaN/clouds
                nan_ratio = np.sum(patch == nodata_val) / patch.size
                if nan_ratio > max_nan_ratio:
                    continue  # skip cloudy patch

                # Save patch
                patch_transform = src.window_transform(window)
                patch_profile = profile.copy()
                patch_profile.update(
                    {
                        "height": patch_size,
                        "width": patch_size,
                        "transform": patch_transform,
                    }
                )

                patch_path = os.path.join(output_dir, f"patch_{patch_id:04d}.npy")
                np.save(patch_path, patch)

                patch_id += 1

    return patch_id


def visualize_tif(tif_path, output_plot_path="plot.png", bands=(4, 3, 2)):
    """
    Visualize and save a .tif file as an image plot.

    Parameters:
        tif_path (str): Path to the input .tif file.
        output_plot_path (str): Path to save the output .png or .jpg image.
        bands (tuple): Which bands to use for visualization (default is (1, 2, 3) for RGB).
    """
    with rasterio.open(tif_path) as src:
        if src.count >= 3 and len(bands) == 3:
            r = src.read(bands[0])
            g = src.read(bands[1])
            b = src.read(bands[2])
            rgb = np.stack([r, g, b], axis=-1)

            # Normalize to 0-1 for display
            rgb_min, rgb_max = np.percentile(rgb, (2, 98))
            rgb = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)

            plt.figure(figsize=(10, 10))
            plt.imshow(rgb)
            plt.axis("off")
            plt.title("RGB Composite")
        else:
            band = src.read(1)
            plt.figure(figsize=(10, 10))
            plt.imshow(band, cmap="gray")
            plt.axis("off")
            plt.title("Single Band")

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_plot_path}")
