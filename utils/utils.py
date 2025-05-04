import rasterio
from rasterio.transform import from_origin
import numpy as np
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import os, re
from rasterio.windows import Window, from_bounds
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import matplotlib.patches as mpatches


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
    max_nan_ratio=0.25,
):
    """
    Extracts spatially aligned patches from a multi-band image and its corresponding ground truth mask.

    Parameters:
        tile_path (str): Path to the input image tile (e.g., GeoTIFF).
        mask_path (str): Path to the corresponding ground truth mask.
        output_mask_dir (str): Directory to save the extracted mask patches.
        output_dir (str): Directory to save the extracted image patches.
        patch_size (int): Size (in pixels) of each square patch (default is 256).
        max_nan_ratio (float): Maximum allowed ratio of nodata pixels per patch (default is 0.25).

    Returns:
        patch_id (int): Total number of patches successfully created and saved
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    with rasterio.open(tile_path) as src_img, rasterio.open(mask_path) as src_mask:
        height, width = src_img.height, src_img.width
        nodata_val = src_img.nodata

        existing_ids = []
        for f in os.listdir(output_dir):
            match = re.match(r"patch_(\d+)\.npy", f)
            if match:
                existing_ids.append(int(match.group(1)))

        for f in os.listdir(output_mask_dir):
            match = re.match(r"patch_(\d+)_mask\.npy", f)
            if match:
                existing_ids.append(int(match.group(1)))

        patch_id = max(existing_ids) + 1 if existing_ids else 0

        for top in tqdm(range(0, height, patch_size)):
            for left in range(0, width, patch_size):
                if left + patch_size > width or top + patch_size > height:
                    continue
                window = Window(left, top, patch_size, patch_size)

                # Read image patch
                patch = src_img.read(
                    window=window,
                    boundless=True,
                    fill_value=nodata_val if nodata_val is not None else 0,
                )
                nan_ratio = np.sum(patch == nodata_val) / patch.size
                if nan_ratio > max_nan_ratio:
                    continue
                patch_bounds = rasterio.windows.bounds(window, src_img.transform)

                # Get matching window in mask
                gt_window = from_bounds(*patch_bounds, transform=src_mask.transform)
                mask_patch = src_mask.read(
                    1, window=gt_window, boundless=True, fill_value=0
                )

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


def reconstruct_mosaic(patch_dir, grid_shape, patch_size=(256, 256)):
    """
    Reconstructs a mosaic from saved .npy patch predictions.

    Args:
        patch_dir: Directory containing .npy prediction patches.
        grid_shape: Tuple (rows, cols) indicating the number of patches in each dimension.
        patch_size: Size of each patch (height, width).

    Returns:
        A NumPy array representing the full stitched mosaic.
    """
    rows, cols = grid_shape
    ph, pw = patch_size
    mosaic = np.zeros((rows * ph, cols * pw), dtype=np.uint8)

    # Sort patch files to ensure correct placement
    patch_files = sorted([f for f in os.listdir(patch_dir) if f.endswith(".npy")])
    assert (
        len(patch_files) == rows * cols
    ), "Number of patches does not match grid size."

    idx = 0
    for r in range(rows):
        for c in range(cols):
            patch = np.load(os.path.join(patch_dir, patch_files[idx]))
            mosaic[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw] = patch
            idx += 1

    return mosaic


def plot_prediction_vs_groundtruth(
    pred, groundtruth_tif_path, class_colors, title="Model vs Ground Truth"
):
    """
    Visualizes the predicted mosaic and the ground truth TIFF side by side,
    using the same colormap and legend for both.

    Args:
        pred_npy_path: Path to .npy file with predicted mosaic.
        groundtruth_tif_path: Path to ground truth .tif file.
        class_colors: Dictionary {class_id: (label, RGB tuple)}, e.g., {10: ("Tree", (0, 100, 0))}
        title: Title of the figure.
    """

    # Load ground truth
    with rasterio.open(groundtruth_tif_path) as src:
        gt = src.read(1)

        # Remap ground truth from ESA codes to 0–7
        remap_dict = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6, 90: 7}
        remapped_gt = np.full_like(gt, fill_value=255)  # fill ignored
        for k, v in remap_dict.items():
            remapped_gt[gt == k] = v

    # Create custom colormap from RGB values
    unique_classes = sorted(class_colors.keys())
    color_list = [np.array(class_colors[c][1]) / 255.0 for c in unique_classes]
    cmap = plt.matplotlib.colors.ListedColormap(color_list)
    bounds = unique_classes + [max(unique_classes) + 1]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(remapped_gt, cmap=cmap, norm=norm)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(pred, cmap=cmap, norm=norm)
    axes[1].set_title("Model Prediction")
    axes[1].axis("off")

    # Add legend
    legend_patches = [
        mpatches.Patch(color=cmap(i), label=class_colors[c][0])
        for i, c in enumerate(unique_classes)
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5, fontsize=10)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


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


def visualize_samples(
    x_list, y_list=None, bands=(4, 3, 2), max_samples=5, title="Samples"
):
    """
    Visualize multiple multi-band samples side-by-side.

    Args:
        x_list (list of Tensor): List of image tensors of shape (C, H, W)
        y_list (list of Tensor, optional): List of mask tensors of shape (H, W)
        bands (tuple): Band indices for RGB visualization
        max_samples (int): Number of samples to visualize
        title (str): Plot title
    """
    num_samples = min(len(x_list), max_samples)
    show_masks = y_list is not None

    fig, axes = plt.subplots(
        2 if show_masks else 1,
        num_samples,
        figsize=(4 * num_samples, 4 * (2 if show_masks else 1)),
    )
    if num_samples == 1:
        axes = [axes]  # make iterable

    fig.suptitle(title, fontsize=16)

    for i in range(num_samples):
        # Prepare RGB image
        rgb = x_list[i][bands, :, :].clone().detach().cpu()
        rgb = rgb.permute(1, 2, 0)  # (H, W, C)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        if num_samples == 1:
            ax_img = axes[0] if show_masks else axes
        else:
            ax_img = axes[0][i] if show_masks else axes[i]

        ax_img.imshow(rgb)
        ax_img.set_title(f"Sample {i}")
        ax_img.axis("off")

        if show_masks:
            mask_np = y_list[i].cpu().numpy()
            display_mask = np.copy(mask_np)
            display_mask[display_mask == 255] = 8  # Optional: separate color for ignore

            ax_mask = axes[1][i] if num_samples > 1 else axes[1]
            ax_mask.imshow(display_mask, cmap="tab20", vmin=0, vmax=8)
            ax_mask.set_title(f"Mask {i}")
            ax_mask.axis("off")

    plt.tight_layout()
    plt.show()
