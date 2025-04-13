import rasterio
from rasterio.transform import from_origin
import numpy as np
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import os
from rasterio.windows import Window
from shapely.geometry import box
import matplotlib.pyplot as plt
from s2cloudless import S2PixelCloudDetector


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


def pansharpen(filepath, output_path):
    """
    Resamples all Sentinel-2 bands in a folder to 10m resolution using bilinear interpolation
    and stacks them into a single multiband GeoTIFF file.

    Parameters:
        filepath (str): Path to the folder containing Sentinel-2 .jp2 files (IMG_DATA directory).
        output_path (str): Path to save the output multiband GeoTIFF (.tif) file.

    Returns:
        None. A GeoTIFF file is written to 'output_path' containing all bands at 10m resolution.
    """

    for f in os.listdir(filepath):
        if "TCI.jp2" in f:
            continue
        if "_B02.jp2" in f:
            ref_path = os.path.join(filepath, f)
            break
    with rasterio.open(ref_path) as ref:
        ref_height = ref.height
        ref_width = ref.width
        ref_transform = ref.transform
        ref_crs = ref.crs

    pansharpen_array = []
    for f in os.listdir(filepath):
        if "TCI.jp2" in f:
            continue
        with rasterio.open(os.path.join(filepath, f)) as src:
            original_res = src.res[0]
            if original_res == 10:
                data = src.read(1)
            else:
                data = src.read(
                    out_shape=(1, ref_height, ref_width), resampling=Resampling.bilinear
                )[0]
        pansharpen_array.append(data.astype(np.float32))

    stacked = np.stack(pansharpen_array)

    write_multiband_tiff(output_path, stacked, ref_transform, ref_crs)


def cloud_cleaning(path, output_path):
    """
    Applies cloud masking using s2cloudless to a pansharpened Sentinel-2 image read from a GeoTIFF.

    Parameters:
        tif_path (str): Path to the input pansharpened multiband GeoTIFF image.

    Returns:
        masked_array (np.ndarray): Array with clouds masked (shape: bands, height, width), np.nan in cloudy pixels.
        cloud_mask (np.ndarray): 2D binary cloud mask where True = cloud.
    """

    # Load the pansharpened TIFF image
    with rasterio.open(path) as src:
        pansharpened_array = src.read().astype(np.float32)
        crs = src.crs
        transform = src.transform

    # Ensure enough bands are present
    if pansharpened_array.shape[0] < 13:
        raise ValueError(
            f"Expected at least 13 bands, found {pansharpened_array.shape[0]}."
        )

    # Reorder to (H, W, B)
    image = np.moveaxis(pansharpened_array, 0, -1)

    # Normalize reflectance if in 0–10000 range
    if image.max() > 1.0:
        image /= 10000.0

    # Init s2cloudless detector
    cloud_detector = S2PixelCloudDetector(
        threshold=0.4, average_over=4, dilation_size=2, all_bands=True
    )

    # Get cloud probability and mask
    cloud_mask = cloud_detector.get_cloud_masks(image[np.newaxis, ...])[0]

    # Apply the mask
    masked_image = np.copy(image)
    masked_image[cloud_mask] = np.nan

    # Return to original shape (bands, height, width)
    masked_array = np.moveaxis(masked_image, -1, 0)

    write_multiband_tiff(output_path, masked_array, transform, crs)
    return masked_array, cloud_mask


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


def visualize_tif(tif_path, output_plot_path="plot.png", bands=(3, 2, 1)):
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
