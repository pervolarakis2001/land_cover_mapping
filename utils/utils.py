import rasterio
from rasterio.transform import from_origin
import numpy as np
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import os, re
import gc
from rasterio.windows import Window, from_bounds
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
        nodata=-9999,
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


def pansharpening(src_path, output_path):
    """
    Performs pansharpening and cloud masking on raw Sentinel-2 imagery:
    Parameters:
        src_path (str): Path to the TIFF file.
        output_path (str): output path to save results
    """
    band_order = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]

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

    bands_array = []

    jp2_files = [
        f for f in os.listdir(src_path) if f.endswith(".jp2") and "TCI" not in f
    ]
    identifier = jp2_files[0].split("_B")[0] + "_"
    for band_suffix in band_order:
        filename = f"{identifier}{band_suffix}.jp2"
        if filename in jp2_files:
            band_path = os.path.join(src_path, filename)
            with rasterio.open(band_path) as src:
                original_res = src.res[0]
                if original_res == 10.0:
                    data = src.read(1)

                else:
                    data = src.read(
                        out_shape=(1, ref_height, ref_width),
                        resampling=Resampling.bilinear,
                    )[0]
                bands_array.append((data).astype(np.float32))

    band_stack = np.stack(bands_array)

    write_multiband_tiff(output_path, band_stack, ref_transform, ref_crs)


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


def create_patches(
    tile_path,
    mask_path,
    output_mask_dir,
    output_dir,
    patch_size=256,
    max_nan_ratio=0.15,
):
    """
    Creates patches from a satellite tile and extracts corresponding patches from a
    full-scene ground truth mask based on geospatial alignment.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    with rasterio.open(tile_path) as src_img, rasterio.open(mask_path) as src_mask:
        height, width = src_img.height, src_img.width
        profile = src_img.profile
        nodata_val = src_img.nodata

        existing_ids = []
        for f in os.listdir(output_dir):
            match = re.match(r"patch_(\d+)\.npy", f)
            if match:
                existing_ids.append(int(match.group(1)))
        patch_id = max(existing_ids) + 1 if existing_ids else 0
        for top in tqdm(range(0, height, patch_size)):
            for left in range(0, width, patch_size):
                if left + patch_size > width or top + patch_size > height:
                    continue

                # Define image patch window
                window = Window(left, top, patch_size, patch_size)

                # Read image patch
                patch = src_img.read(window=window)
                nan_ratio = np.sum(patch == nodata_val) / patch.size
                if nan_ratio > max_nan_ratio:
                    continue

                # Convert image patch window to geospatial bounds
                patch_bounds = rasterio.windows.bounds(window, src_img.transform)

                # Get matching window in GT mask
                gt_window = from_bounds(*patch_bounds, transform=src_mask.transform)

                # Extract corresponding patch from GT mask
                mask_patch = src_mask.read(1, window=gt_window)

                # Save
                patch_path = os.path.join(output_dir, f"patch_{patch_id:04d}.npy")
                np.save(patch_path, patch)
                mask_path_out = os.path.join(
                    output_mask_dir, f"patch_{patch_id:04d}_mask.npy"
                )
                np.save(mask_path_out, mask_patch)

                patch_id += 1

    return patch_id


def generate_cloud_mask(scl_path, output_path, cloud_values=[3, 8, 9, 10]):
    """
    Generates a binary cloud mask from a reprojected SCL raster from L2A product.

    Parameters:
        scl_path (str): Path to the aligned SCL raster (GeoTIFF).
        output_path (str): Path to save the output cloud mask.
        cloud_values (list): SCL values considered as clouds.

    Returns:
        mask (np.ndarray): Binary mask array (1 for cloud/cirrus, 0 for clear).
    """
    with rasterio.open(scl_path) as src:
        scl = src.read(1)
        profile = src.profile

    # Generate binary mask
    mask = np.isin(scl, cloud_values).astype(np.uint8)

    # Save the mask
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)

    return mask


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


def visualize_sample(x, y=None, bands=(3, 2, 1), title="Augmented Sample"):
    """
    Visualize a multi-band image using selected RGB bands.

    Args:
        x (Tensor): Image tensor of shape (C, H, W)
        y (Tensor, optional): Mask tensor of shape (H, W)
        bands (tuple): Indices for RGB bands (default is Sentinel-2 B4, B3, B2)
        title (str): Title for the plot
    """
    # Select RGB bands and bring to (H, W, C)
    rgb = x[bands, :, :].clone().detach().cpu()
    rgb = rgb.permute(1, 2, 0)  # (H, W, C)

    # Normalize for display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")

    if y is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(y.cpu(), cmap="tab20")  # or 'nipy_spectral'
        plt.title("Segmentation Mask")
        plt.axis("off")

    plt.show()
