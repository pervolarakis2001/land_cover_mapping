import utils
from rasterio.warp import transform_bounds
from shapely.geometry import box
import numpy as np
from shapely.ops import unary_union


SATELITE_IMG_1 = "/home/ubuntu/bdrs_ex_1/data/raw/S2A_MSIL1C_20210727T092031_N0500_R093_T34SEJ_20230131T010144.SAFE/S2A_MSIL1C_20210727T092031_N0500_R093_T34SEJ_20230131T010144.SAFE/GRANULE/L1C_T34SEJ_A031836_20210727T092343/IMG_DATA"
SATELITE_IMG_2 = "/home/ubuntu/bdrs_ex_1/data/raw/S2A_MSIL1C_20210727T092031_N0500_R093_T34SFJ_20230131T010144.SAFE/S2A_MSIL1C_20210727T092031_N0500_R093_T34SFJ_20230131T010144.SAFE/GRANULE/L1C_T34SFJ_A031836_20210727T092343/IMG_DATA"
SATELITE_IMG_3 = "/home/ubuntu/bdrs_ex_1/data/raw/S2A_MSIL1C_20210727T092031_N0500_R093_T34TEK_20230131T010144.SAFE/S2A_MSIL1C_20210727T092031_N0500_R093_T34TEK_20230131T010144.SAFE/GRANULE/L1C_T34TEK_A031836_20210727T092343/IMG_DATA"
SATELITE_IMG_4 = "/home/ubuntu/bdrs_ex_1/data/raw/S2A_MSIL1C_20210727T092031_N0500_R093_T34TFK_20230131T010144.SAFE/S2A_MSIL1C_20210727T092031_N0500_R093_T34TFK_20230131T010144.SAFE/GRANULE/L1C_T34TFK_A031836_20210727T092343/IMG_DATA"

paths = [SATELITE_IMG_1, SATELITE_IMG_2, SATELITE_IMG_3, SATELITE_IMG_4]

# panshapening raw data
# for i, path in enumerate(paths):
#     utils.pansharpen(
#         path,
#         f"/home/ubuntu/bdrs_ex_1/data/processed/pansharpened_tiles/pansharpened_tile_{i}.tif",
#     )
#     utils.cloud_cleaning(
#         f"/home/ubuntu/bdrs_ex_1/data/processed/pansharpened_tiles/pansharpened_{i}.tif",
#         f"/home/ubuntu/bdrs_ex_1/data/processed/cloud_cleaned_tiles/cleaned_tile_{i}.tif",
#     )

# reprojecting data
covered_area = []
gt_info = utils.get_tiff_info(
    "/home/ubuntu/bdrs_ex_1/data/ground_truth/GBDA24_ex2_ref_data.tif"
)
for i in range(4):
    sat_info = utils.get_tiff_info(
        f"data/processed/pansharpened_tiles/pansharpened_tile_{i}.tif"
    )
    sat_bounds = transform_bounds(sat_info["crs"], gt_info["crs"], *sat_info["bounds"])
    gt_box = box(*gt_info["bounds"])
    sat_box = box(*sat_bounds)

    # Compute new area not yet covered
    already_covered = unary_union(covered_area) if covered_area else None
    new_area = (
        sat_box if already_covered is None else sat_box.difference(already_covered)
    )

    intersection_box = gt_box.intersection(sat_box)
    covered_area.append(intersection_box)
    utils.reproject_raster(
        src_path=f"data/processed/pansharpened_tiles/pansharpened_tile_{i}.tif",
        dst_path=f"data/processed/reprojected_tiles/reprojected_tile_{i}.tif",
        dst_crs=gt_info["crs"],
        target_bounds=intersection_box.bounds,  # crop satelite image based on GT
        target_resolution=gt_info["gsd"],
    )


# # creating patches
# PATCH_SIZE = (256, 256)
