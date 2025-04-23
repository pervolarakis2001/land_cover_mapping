import rasterio
import utils.utils as utils
from rasterio.warp import transform_bounds
from shapely.geometry import box
import numpy as np
import os
from shapely.ops import unary_union
import time
from concurrent.futures import ProcessPoolExecutor
from rasterio.windows import Window, from_bounds
import matplotlib as plt


# directories
SATELITE_IMG_1 = "data/raw/S2A_MSIL1C_20210727T092031_N0500_R093_T34SEJ_20230131T010144.SAFE/S2A_MSIL1C_20210727T092031_N0500_R093_T34SEJ_20230131T010144.SAFE/GRANULE/L1C_T34SEJ_A031836_20210727T092343/IMG_DATA"
SATELITE_IMG_2 = "data/raw/S2A_MSIL1C_20210727T092031_N0500_R093_T34SFJ_20230131T010144.SAFE/S2A_MSIL1C_20210727T092031_N0500_R093_T34SFJ_20230131T010144.SAFE/GRANULE/L1C_T34SFJ_A031836_20210727T092343/IMG_DATA"
SATELITE_IMG_3 = "data/raw/S2A_MSIL1C_20210727T092031_N0500_R093_T34TEK_20230131T010144.SAFE/S2A_MSIL1C_20210727T092031_N0500_R093_T34TEK_20230131T010144.SAFE/GRANULE/L1C_T34TEK_A031836_20210727T092343/IMG_DATA"
SATELITE_IMG_4 = "data/raw/S2A_MSIL1C_20210727T092031_N0500_R093_T34TFK_20230131T010144.SAFE/S2A_MSIL1C_20210727T092031_N0500_R093_T34TFK_20230131T010144.SAFE/GRANULE/L1C_T34TFK_A031836_20210727T092343/IMG_DATA"

paths = [SATELITE_IMG_1, SATELITE_IMG_2, SATELITE_IMG_3, SATELITE_IMG_4]

"""Perform pansharpening on raw Sentinel-2 tiles using cloud masks"""

for i, path in enumerate(paths):
    utils.pansharpening(
        path,
        f"data/processed/pansharpened_tiles/newpansharpened_tile_{i}.tif",
    )


"""Reproject pansharpened tiles to match the CRS and extent of the groundtruth image"""
gt_info = utils.get_tiff_info(
    "/home/ubuntu/bdrs_ex_1/data/ground_truth/GBDA24_ex2_ref_data.tif"
)
gt_box = box(*gt_info["bounds"])
gt_crs = gt_info["crs"]
gsd = gt_info["gsd"]

input_tiles = [
    "data/processed/pansharpened_tiles/newpansharpened_tile_0.tif",
    "data/processed/pansharpened_tiles/newpansharpened_tile_1.tif",
    "data/processed/pansharpened_tiles/newpansharpened_tile_2.tif",
    "data/processed/pansharpened_tiles/newpansharpened_tile_3.tif",
]
final_dir = "data/processed/final_tiles"
covered_area = []

for i, tile_path in enumerate(input_tiles):
    # Get original tile bounds and reproject
    tile_info = utils.get_tiff_info(tile_path)
    sat_bounds = transform_bounds(tile_info["crs"], gt_crs, *tile_info["bounds"])
    sat_box = box(*sat_bounds)

    intersection = gt_box.intersection(sat_box)

    final_path = f"{final_dir}/tile_{i}.tif"
    if i == 0:
        utils.reproject_raster(
            src_path=tile_path,
            dst_path=final_path,
            dst_crs=gt_crs,
            target_bounds=intersection.bounds,
            target_resolution=gsd,
        )
        covered_area.append(intersection)
        continue

    tmp_path = "data/tmp_reprojected_tile.tif"

    # Reproject and crop to GT
    utils.reproject_raster(
        src_path=tile_path,
        dst_path=tmp_path,
        dst_crs=gt_crs,
        target_bounds=intersection.bounds,
        target_resolution=gsd,
    )

    # Remove overlap with previous
    non_overlap = intersection.difference(unary_union(covered_area))

    if non_overlap.is_empty:
        os.remove(tmp_path)
        continue

    # Save only the non-overlapping part
    utils.reproject_raster(
        src_path=tmp_path,
        dst_path=final_path,
        dst_crs=gt_crs,
        target_bounds=non_overlap.bounds,
        target_resolution=gsd,
    )

    os.remove(tmp_path)
    covered_area.append(non_overlap)

"""Cleaning tile_2 from clouds"""

# Reprojected SLC mask
gt_info = utils.get_tiff_info(
    "/home/ubuntu/bdrs_ex_1/data/processed/final_tiles/tile_2.tif"
)
gt_crs = gt_info["crs"]
gt_bounds = gt_info["bounds"]
gt_box = box(*gt_bounds)
gsd = gt_info["gsd"]

scl_path = "/home/ubuntu/bdrs_ex_1/data/raw/T34TEK_20210727T092031_SCL_20m.jp2"
utils.reproject_raster(
    src_path=scl_path,
    dst_path="/home/ubuntu/bdrs_ex_1/data/processed/final_tiles/rep_SLC.tif",
    dst_crs=gt_crs,
    target_bounds=gt_bounds,
    target_resolution=gsd,
)
# generate scl cloud mask as a tif file
utils.generate_cloud_mask(
    scl_path="/home/ubuntu/bdrs_ex_1/data/processed/final_tiles/rep_SLC.tif",
    output_path="/home/ubuntu/bdrs_ex_1/data/cloud_mask.tif",
)

# apply scl mask to tile with clouds
with rasterio.open("data/processed/final_tiles/tile_2.tif") as src:
    image = src.read()
    meta = src.meta.copy()
    crs = src.crs
    nodata = src.nodata
    transform = src.transform
with rasterio.open("data/cloud_mask.tif") as mask_src:
    cloud_mask = mask_src.read(1)

# create a boolean mask where cloud pixels are marked as True
cloud_pixels = cloud_mask == 1

masked_image = np.where(cloud_pixels, nodata, image)
utils.write_multiband_tiff(
    "data/processed/final_tiles/tile_2.tif", masked_image, transform, crs
)

"""Creating Patches from tiles and GroundTruth"""
for i in range(4):
    utils.create_patches(
        final_dir + f"/tile_{i}.tif",
        "data/ground_truth/GBDA24_ex2_ref_data.tif",
        "data/patches/masks",
        "data/patches/images",
    )

"""Perform pansharpening on pred. image"""
path_pr="/home/ubuntu/bdrs_exercise_1/data/prediction/S2A_MSIL1C_20210727T092031_N0500_R093_T34SEH_20230131T010144.SAFE/S2A_MSIL1C_20210727T092031_N0500_R093_T34SEH_20230131T010144.SAFE/GRANULE/L1C_T34SEH_A031836_20210727T092343/IMG_DATA"
utils.pansharpening(path_pr, "/home/ubuntu/bdrs_exercise_1/data/prediction/predict_pan.tif")


"""Reproject Pred Image"""
gt_info = utils.get_tiff_info(
    "/home/ubuntu/bdrs_exercise_1/data/prediction/GBDA24_ex2_34SEH_ref_data.tif"
) 
print(gt_info)
gt_box = box(*gt_info["bounds"])
gt_crs = gt_info["crs"]
gsd = gt_info["gsd"]
utils.reproject_raster(
    src_path="/home/ubuntu/bdrs_exercise_1/data/prediction/predict_pan.tif",
            dst_path="/home/ubuntu/bdrs_exercise_1/data/prediction/rep_predict_pan.tif",
            dst_crs=gt_crs,
            target_bounds=gt_box.bounds,
            target_resolution=gsd
            )


