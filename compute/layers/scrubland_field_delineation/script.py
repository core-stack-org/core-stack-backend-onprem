import sys
from PIL import Image
import PIL
import re
from multiprocessing import cpu_count
from glob import glob
from pathlib import Path
from mxnet import gluon
from mxnet import image
from skimage import measure
import pickle
from commons import tms_to_geotiff
import math
from itertools import product
import ee
import geopandas as gpd
from ultralytics import YOLO
import ast
import time

PIL.Image.MAX_IMAGE_PIXELS = 2000000000

# module_paths = [
#     "decode/FracTAL_ResUNet/models/semanticsegmentation",
#     "decode/FracTAL_ResUNet/nn/loss",
# ]
module_paths = [
    "scrubland_field_delineation/decode/FracTAL_ResUNet/models/semanticsegmentation",
    "scrubland_field_delineation/decode/FracTAL_ResUNet/nn/loss",
]
for module_path in module_paths:
    if module_path not in sys.path:
        sys.path.append(module_path)
from FracTAL_ResUNet import FracTAL_ResUNet_cmtsk
from datasets import *
from instance_segment import InstSegm
from skimage import data
from skimage.filters.rank import entropy
from skimage.morphology import disk, ball
import multiprocessing as mp
from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_bilateral,
    denoise_wavelet,
    estimate_sigma,
)
from osgeo import gdal
from commons import raster_to_shp
import zipfile
import pandas as pd
from itertools import combinations
from utils import (
    ee_initialize,
    get_gee_asset_path,
    valid_gee_text,
    upload_shp_to_gee,
    is_gee_asset_exists,
    upload_file_to_gcs,
    check_task_status,
    sync_fc_to_gee,
)
from constants import GCS_SHAPEFILE_BUCKET
from misc import get_points, download


original_image, min_j, min_i, max_j, max_i, instances_predicted = (0, 0, 0, 0, 0, 0)
mapping = {"farm": 1, "plantation": 2, "scrubland": 3, "rest": 0}


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


"""
    Helper functions to Download High Resolution Images
"""


# # Function to convert latitude to pixel Y at a given zoom level
# def lat_to_pixel_y(lat, zoom):
#     sin_lat = math.sin(math.radians(lat))
#     pixel_y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * (
#         2 ** (zoom + 8)
#     )
#     return pixel_y
#
#
# # Function to convert longitude to pixel X at a given zoom level
# def lon_to_pixel_x(lon, zoom):
#     pixel_x = ((lon + 180) / 360) * (2 ** (zoom + 8))
#     return pixel_x

#
# # Function to convert pixel X to longitude
# def pixel_x_to_lon(pixel_x, zoom):
#     lon = (pixel_x / (2 ** (zoom + 8))) * 360 - 180
#     return lon
#
#
# # Function to convert pixel Y to latitude
# def pixel_y_to_lat(pixel_y, zoom):
#     n = math.pi - 2 * math.pi * pixel_y / (2 ** (zoom + 8))
#     lat = math.degrees(math.atan(math.sinh(n)))
#     return lat


# def lat_lon_from_pixel(lat, lon, zoom, scale):
#     """
#     Given a starting latitude and longitude, calculate the latitude and longitude
#     of the opposite corner of a 256x256 image at a given zoom level.
#     """
#     pixel_x = lon_to_pixel_x(lon, zoom)
#     pixel_y = lat_to_pixel_y(lat, zoom)
#
#     new_lon = pixel_x_to_lon(pixel_x + 256 * scale, zoom)
#     new_lat = pixel_y_to_lat(pixel_y + 256 * scale, zoom)
#
#     return new_lat, new_lon


# def divide_tiff_into_chunks(chunk_size, output_dir):
#     # Load the large TIFF image
#     input_image_path = output_dir + "/field.tif"
#     image = Image.open(input_image_path)
#     # Get image dimensions
#     width, height = image.size
#
#     # Iterate over the image to create 256x256 chunks
#     ind_i = 0
#     ind_j = 0
#     for i in range(0, width, chunk_size):
#         for j in range(0, height, chunk_size):
#             # Define the box to crop
#             box = (i, j, i + chunk_size, j + chunk_size)
#
#             # Crop the image using the defined box
#             chunk = image.crop(box)
#             # Save each chunk as a separate TIFF file
#             chunk.save(
#                 os.path.join(output_dir + "/chunks/", f"chunk_{ind_i}_{ind_j}.tif")
#             )
#             ind_j += 1
#         ind_i += 1
#         ind_j = 0
#
#     print("Image has been split into 256x256 chunks and saved successfully.")


# def download(bbox, output_dir, row, index, directory, blocks_df):
#     if row["download_status_" + str(zoom)] == True:
#         return
#
#     # scale = 16
#     # zoom = 17
#     chunk_size = 256
#
#     (lat1, lon1), (lat2, lon2) = bbox
#     # new_lat, new_lon = lat_lon_from_pixel(lat, lon, zoom, scale)
#     # print(lat,",",lon,new_lat,",",new_lon)
#     # os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(output_dir + "/chunks", exist_ok=True)
#     tms_to_geotiff(
#         output=output_dir + "/field.tif",
#         bbox=[lon1, lat1, lon2, lat2],
#         zoom=zoom,
#         source="Satellite",
#         overwrite=True,
#         threads=1,
#     )
#     divide_tiff_into_chunks(chunk_size, output_dir)
#     mark_done(index, directory, blocks_df, "download_status_" + str(zoom))


"""

Helper functions for Inference

"""

# Function to extract the i, j values from the file name


def extract_indices(file_name):
    match = re.search(r"chunk_(\d+)_(\d+)\.tif", file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def load_model():
    # hyperparameters for model architecture
    n_filters = 32
    depth = 6
    n_classes = 1
    ctx_name = "gpu"
    gpu_id = 0
    trained_model = "scrubland_field_delineation/india_Airbus_SPOT_model.params"
    if ctx_name == "cpu":
        ctx = mx.cpu()
    elif ctx_name == "gpu":
        ctx = mx.gpu(gpu_id)

    # initialise model
    model = FracTAL_ResUNet_cmtsk(
        nfilters_init=n_filters, depth=depth, NClasses=n_classes
    )
    model.load_parameters(trained_model, ctx=ctx)
    return model, ctx


def run_inference(test_dataloader, model, ctx):
    logits_array = []
    bound_array = []
    dist_array = []
    for batch_i, img_data in enumerate(test_dataloader):
        # extract batch data
        imgs = img_data
        imgs = imgs.as_in_context(ctx)
        logits, bound, dist = model(imgs)
        logits_array.append(logits)
        bound_array.append(bound)
        dist_array.append(dist)

    def flatten(array):
        array_new = []
        for batch, arr in enumerate(array):
            for a in arr.asnumpy():
                array_new.append(a[0])
            print("Batch Done", batch + 1, "/", len(array))
        return array_new

    logits_array = flatten(logits_array)
    bound_array = flatten(bound_array)
    print(len(logits_array))
    return logits_array, bound_array


def run_model(output_dir, row, index, directory, blocks_df):
    if row["model_status"] == True:
        return
    batch_size = 32
    CPU_COUNT = cpu_count()
    # extract chunk ids of validation data
    gt_bound_names = glob(output_dir + "/chunks/*.tif")
    gt_bound_names = [i for i in gt_bound_names if "chunk_" in i]
    print("Found {} groundtruth chunks".format(len(gt_bound_names)))

    # Sort the file list based on the i, j values
    image_names = sorted(gt_bound_names, key=extract_indices)
    # Extract the maximum i and j values to determine the final stitched image size
    max_i = max(extract_indices(file)[0] for file in image_names)
    max_j = max(extract_indices(file)[1] for file in image_names)
    print("Max i and Max j are ", max_i, max_j)

    # Load dataset
    test_dataset = Planet_Dataset_No_labels(image_names=image_names)
    test_dataloader = gluon.data.DataLoader(
        test_dataset, batch_size=batch_size
    )  # , num_workers=CPU_COUNT)

    model, ctx = load_model()
    logits_array, bound_array = run_inference(test_dataloader, model, ctx)

    with open(output_dir + "/logits_bounds.pickle", "wb") as handle:
        pickle.dump(
            [logits_array, bound_array], handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    mark_done(index, directory, blocks_df, "model_status")


"""
    Helper functions for watershed algorithm
"""


def get_segmentation(output_dir, row, index, directory, blocks_df):
    if row["segmentation_status"] == True:
        return

    image_size = 256
    gt_bound_names = glob(output_dir + "/chunks/*.tif")
    gt_bound_names = [i for i in gt_bound_names if "chunk_" in i]
    print("Found {} groundtruth chunks".format(len(gt_bound_names)))
    # Sort the file list based on the i, j values
    image_names = sorted(gt_bound_names, key=extract_indices)
    # Extract the maximum i and j values to determine the final stitched image size
    max_i = max(extract_indices(file)[0] for file in image_names)
    max_j = max(extract_indices(file)[1] for file in image_names)
    print("Loading Pickle")
    with open(output_dir + "/logits_bounds.pickle", "rb") as handle:
        logits_array, bound_array = pickle.load(handle)
    print("Pickle Loaded")

    # Stitch image
    print("stiching image")
    stitched_image_array = np.zeros(
        ((max_i + 1) * image_size, (max_j + 1) * image_size), dtype=np.float32
    )
    stitched_bound_array = np.zeros(
        ((max_i + 1) * image_size, (max_j + 1) * image_size), dtype=np.float32
    )
    for ind, name in enumerate(image_names):
        img_arr = logits_array[ind].T
        bound_arr = bound_array[ind].T
        i, j = extract_indices(name)
        y_offset = i * image_size
        x_offset = j * image_size
        stitched_image_array[
            y_offset : y_offset + image_size, x_offset : x_offset + image_size
        ] = img_arr
        stitched_bound_array[
            y_offset : y_offset + image_size, x_offset : x_offset + image_size
        ] = bound_arr

    print("deleting pickle")
    del logits_array
    del bound_array
    # t_ext_best=0.3
    # t_bnd_best=0.1

    t_ext_best = 0.3
    t_bnd_best = 0.4
    # do segmentation
    print("Doing segmentation")
    instances_predicted = InstSegm(
        stitched_image_array, stitched_bound_array, t_ext=t_ext_best, t_bound=t_bnd_best
    )
    # label connected regions, non-field (-1) will be labelled as 0
    print("Doing measure")
    instances_predicted = measure.label(
        instances_predicted, background=-1, return_num=False
    )
    segments = instances_predicted.max()
    print("Max segments are ", segments)
    with open(output_dir + "/instance_predicted.pickle", "wb") as handle:
        pickle.dump(instances_predicted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mark_done(index, directory, blocks_df, "segmentation_status")


def crop_image_by_mask(image, index):
    # Find the bounding box of the True values in the mask
    min_x, min_y, max_x, max_y = min_j[index], min_i[index], max_j[index], max_i[index]
    # Crop the image using the bounding box
    cropped_image = image.crop((min_x, min_y, max_x + 1, max_y + 1))
    # Crop the mask as well
    cropped_mask = instances_predicted[min_y : max_y + 1, min_x : max_x + 1]
    cropped_mask = cropped_mask == index
    # Convert the cropped image to RGBA (if not already)
    cropped_image = cropped_image.convert("RGBA")
    # Get pixel data
    pixels = np.array(cropped_image)
    # Set pixels where the mask is False to transparent
    pixels[~cropped_mask] = [0, 0, 0, 0]  # Set to transparent
    # Convert back to an image
    masked_image = Image.fromarray(pixels)
    return masked_image, cropped_mask


def get_entropy(img, mask):
    ent = entropy(np.asarray(img.convert("L")).copy(), disk(5), mask=mask)
    ent = ent[ent > 5.2]
    ent = sum(ent) / (sum(sum(mask)))
    return ent


def get_entropy_plantation(img, mask):
    ent = entropy(np.asarray(img.convert("L")).copy(), disk(30), mask=mask)
    ent = ent[ent > 0]
    ent = sum(ent) / (sum(sum(mask)))
    return ent


def get_lines_by_hough(img, mask):
    masked_image_np = np.array(img.convert("L"))
    # _, binary_image = cv2.threshold(masked_image_np, 50, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(masked_image_np, 50, 150)
    erosion_size = 1
    # Define the kernel for erosion
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    # Erode the mask to remove edge pixels
    eroded_mask = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=1)
    # print(index)
    edges = cv2.bitwise_and(edges, edges, mask=eroded_mask)
    # Perform Hough Line Transformation
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=30
    )
    if lines is not None and len(lines) > 2:
        return 1
    else:
        return 0


def get_perimeter_area_fractal_dimension(mask):
    # Load the image
    _, binary = cv2.threshold(mask.astype(np.uint8) * 255, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # No shape detected

    # Assume the largest contour is the shape
    contour = max(contours, key=cv2.contourArea)

    # Compute area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Compute fractal dimension
    if area > 0:
        D = 2 * (np.log(perimeter) / np.log(area))
        return D
    else:
        return None


def get_rectangularity(mask):
    """
    Compute how rectangular a given binary mask is.

    Args:
        mask (np.ndarray): Binary mask (1 for object, 0 for background)

    Returns:
        float: Rectangularity score (1.0 is a perfect rectangle, lower means less rectangular)
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0  # No object found

    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Compute contour area
    contour_area = cv2.contourArea(contour)

    # Compute minimum area rectangle (rotated)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    bounding_rect_area = cv2.contourArea(box)

    if bounding_rect_area == 0:
        return 0  # Avoid division by zero

    # Compute rectangularity score
    rectangularity_score = contour_area / bounding_rect_area

    return rectangularity_score


def get_ht_lines(img, mask):
    masked_image_np = np.array(img.convert("L"))
    # masked_image_np = cv2.bilateralFilter(masked_image_np,3,75,75)
    # _, binary_image = cv2.threshold(masked_image_np, 50, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(masked_image_np, 50, 150)
    erosion_size = 3
    # Define the kernel for erosion
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    # Erode the mask to remove edge pixels
    eroded_mask = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=1)
    # print(index)
    edges = cv2.bitwise_and(edges, edges, mask=eroded_mask)
    # Perform Hough Line Transformation
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=30
    )
    detected_lines = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            detected_lines.append(((x1, y1), (x2, y2)))

    return detected_lines


def calculate_angle(line1, line2):
    def line_angle(line):
        (x1, y1), (x2, y2) = line
        return np.degrees(np.arctan2(y2 - y1, x2 - x1))

    angle1 = line_angle(line1)
    angle2 = line_angle(line2)

    # Calculate absolute angle difference
    angle_diff = abs(angle1 - angle2)

    # Ensure the angle is within 0-180 degrees range
    return min(angle_diff, 180 - angle_diff)


def check_right_angles(lines, tolerance=5):
    right_angle_pairs = []
    for line1, line2 in combinations(lines, 2):
        angle = calculate_angle(line1, line2)
        if abs(angle - 90) <= tolerance:
            right_angle_pairs.append((line1, line2, angle))

    return right_angle_pairs


def map_entropy_(index):
    img, mask = crop_image_by_mask(original_image, index)
    ent = get_entropy(img, mask)
    rectangularity = get_rectangularity(mask)
    color_map = 0
    # if index == 5341:
    #    print(sum(sum(mask)))
    size_of_segment = sum(sum(mask))
    if size_of_segment > 80000:
        color_map = 0
    elif ent < 1 and rectangularity > 0.6:  # or ent > 5:
        color_map = 1
    else:
        ent_plantation = get_entropy_plantation(img, mask)
        if ent_plantation < 8.5 and rectangularity > 0.67 and size_of_segment < 20000:
            color_map = 2
        lines = get_lines_by_hough(img, mask)
        if lines == 1 and rectangularity > 0.67 and size_of_segment < 20000:
            color_map = 2
    return (index, color_map)


def map_entropy(index):
    img, mask = crop_image_by_mask(original_image, index)
    ent = get_entropy(img, mask)
    ent_plantation = get_entropy_plantation(img, mask)
    rectangularity = get_rectangularity(mask)
    fractal_dimension = get_perimeter_area_fractal_dimension(mask)
    size = sum(sum(mask))
    lines = get_ht_lines(img, mask)
    right_angles = check_right_angles(lines)
    # print(fractal_dimension)
    blueness = np.mean(cv2.split(np.array(img))[2])
    greeness = np.mean(cv2.split(np.array(img))[1])
    redness = np.mean(cv2.split(np.array(img))[0])
    red = redness / greeness

    easy_farm = rectangularity >= 0.67 and size > 500 and size < 2000 and ent < 1
    # easy_plantation =  rectangularity>=0.7 and size>500 and size<20000 and ent>4 and len(right_angles)>5
    easy_scrub = (
        ent > 2.5
        and len(lines) <= 1
        and size > 2000
        and rectangularity < 0.67
        and red > 1
    ) or size > 100000
    class_ = "rest"
    if easy_farm:
        class_ = "farm"
    # elif easy_plantation:
    #    color_map = 2
    elif easy_scrub:
        class_ = "scrubland"
    return (
        index,
        class_,
        ent,
        ent_plantation,
        rectangularity,
        fractal_dimension,
        size,
        len(lines),
        len(right_angles),
        blueness,
        greeness,
        redness,
        red,
    )


def set_global_for_multiprocessing(oi, mnj, mni, mxj, mxi, ip):
    global original_image
    global min_j
    global min_i
    global max_j
    global max_i
    global instances_predicted
    original_image, min_j, min_i, max_j, max_i, instances_predicted = (
        oi,
        mnj,
        mni,
        mxj,
        mxi,
        ip,
    )


def process_in_chunks(number, chunk_size):
    total_chunks = (
        number + chunk_size - 1
    ) // chunk_size  # Round up to the next whole number
    results = []
    for i in range(total_chunks):
        start = i * chunk_size
        if start == 0:
            start += 1
        end = min(start + chunk_size, number)  # Make sure not to exceed the number
        print(f"Processing chunk: {start} to {end - 1}")
        with mp.Pool(12) as p:
            results += p.map(map_entropy, list(range(start, end)))
    return results


def get_color(color_dict, index):
    def color(ind):
        label = color_dict.get(ind)
        if label == index:
            label = ind
        else:
            label = 0
        return label

    return color


def get_min_max_array(instances_predicted):
    min_i = {}
    min_j = {}
    max_i = {}
    max_j = {}
    printcounter = 0
    for i in range(instances_predicted.shape[0]):
        if printcounter == 1000:
            print(i)
            printcounter = 0
        printcounter += 1
        for j in range(instances_predicted.shape[0]):
            val = instances_predicted[i][j]
            if val not in min_i:
                min_i[val] = i
            if val not in max_i:
                max_i[val] = i
            if val not in min_j:
                min_j[val] = j
            if val not in max_j:
                max_j[val] = j
            min_i[val] = min(i, min_i[val])
            min_j[val] = min(j, min_j[val])
            max_i[val] = max(i, max_i[val])
            max_j[val] = max(j, max_j[val])
    return min_i, min_j, max_i, max_j


def save_field_boundaries(output_dir, instances_predicted, df=None, others=None):
    # if there is no boundary save empty field boundary
    ds = gdal.Open(output_dir + "/field.tif")

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [rows, cols] = arr.shape

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_dir + "/out.tif", cols, rows, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  ##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(instances_predicted)
    # outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!
    outdata = None
    band = None
    ds = None
    # import ipdb
    # ipdb.set_trace()
    if df is not None:
        raster_to_shp(tiff_path=output_dir + "/out.tif", output=output_dir + "/all.shp")
        gdf = gpd.read_file(output_dir + "/all.shp")
        gdf = gdf.set_crs("epsg:3857", allow_override=True)
        print(f"Original CRS: {gdf.crs}")
        gdf = gdf.to_crs(epsg=4326)
        print(f"Reprojected CRS: {gdf.crs}")
        gdf = gdf.merge(df, on="value")
        gdf.to_file(output_dir + "/all.shp")
    if "plantation" == others:
        if sum(sum(instances_predicted)) == 0:
            gdf = gpd.GeoDataFrame(
                columns=["id", "geometry"], geometry="geometry", crs="EPSG:4326"
            )
            gdf.to_file(output_dir + "/" + others + ".shp")
        else:
            raster_to_shp(
                tiff_path=output_dir + "/out.tif", output=output_dir + "/plantation.shp"
            )
        gdf = gpd.read_file(output_dir + "/plantation.shp")
        gdf = gdf.set_crs("epsg:3857", allow_override=True)
        print(f"Original CRS: {gdf.crs}")
        gdf = gdf.to_crs(epsg=4326)
        print(f"Reprojected CRS: {gdf.crs}")
        gdf["class"] = "plantation"
        gdf.to_file(output_dir + "/" + others + ".shp")
    for file in ["out.tif"]:
        os.remove(output_dir + "/" + file)


def run_postprocessing(output_dir, row, index, directory, blocks_df):
    if row["postprocessing_status"] == True:
        return
    input_image_path = output_dir + "/field.tif"
    image = Image.open(input_image_path)

    with open(output_dir + "/instance_predicted.pickle", "rb") as handle:
        instances_predicted = pickle.load(handle)
        instances_predicted = instances_predicted.T

    segments = instances_predicted.max()
    print("Max segments are ", segments)

    original_image = image
    original_image = original_image.crop(
        (
            0,
            0,
        )
        + instances_predicted.shape
    )

    min_i, min_j, max_i, max_j = get_min_max_array(instances_predicted)
    set_global_for_multiprocessing(
        original_image, min_j, min_i, max_j, max_i, instances_predicted
    )
    results = process_in_chunks(segments + 1, 12000)
    df = pd.DataFrame(
        results,
        columns=[
            "value",
            "class",
            "ent",
            "ent_pl",
            "rect",
            "frct_dim",
            "size",
            "num_lines",
            "num_rt_ang",
            "blueness",
            "greeness",
            "redness",
            "red",
        ],
    )
    save_field_boundaries(output_dir, instances_predicted, df=df)
    mark_done(index, directory, blocks_df, "postprocessing_status")


def zip_vector(output_dir, vector_name):
    zip = zipfile.ZipFile(
        output_dir + "/" + vector_name + ".zip", "w", zipfile.ZIP_DEFLATED
    )
    files = [vector_name + i for i in [".shp", ".cpg", ".dbf", ".prj", ".shx"]]
    for file in files:
        zip.write(output_dir + "/" + file)
    zip.close()


def join_boundaries_for_domain(output_dir, blocks_count, domain):
    gdf = None
    for i in range(0, blocks_count):
        gdf_new = gpd.read_file(output_dir + "/" + str(i) + "/" + domain + ".shp")
        if i == 0:
            gdf = gdf_new
        else:
            gdf = pd.concat([gdf, gdf_new])
    gdf.to_file(output_dir + "/" + domain + ".shp")
    zip_vector(output_dir, domain)


def join_boundaries_for_domain_chunks(output_dir, block_start, block_end, domain):
    gdf = None
    for i in range(block_start, block_end):
        gdf_new = gpd.read_file(output_dir + "/" + str(i) + "/" + domain + ".shp")
        if i == 0:
            gdf = gdf_new
        else:
            gdf = pd.concat([gdf, gdf_new])
    chunk_name = f"{domain}_{block_start}_{block_end}"
    gdf.to_file(f"{directory}/{chunk_name}.shp")
    zip_vector(directory, chunk_name)
    return chunk_name


def join_boundaries(output_dir, blocks_count):
    if os.path.exists(output_dir + "/all_done"):
        print("Everything already done")
        return
    chunk_names = []
    chunk_size = 200
    if blocks_count > chunk_size:
        num_chunks = math.ceil(blocks_count / chunk_size)
        for i in range(num_chunks):
            block_end = chunk_size + (i * chunk_size)
            block_start = block_end - chunk_size
            block_end = blocks_count if block_end > blocks_count else block_end
            print(block_start, block_end)
            gdf = None
            for ind, domain in enumerate(["all", "plantation"]):
                chunk_name = join_boundaries_for_domain_chunks(
                    directory, block_start, block_end, domain
                )
                gdf_new = gpd.read_file(output_dir + "/" + chunk_name + ".shp")
                if ind == 0:
                    gdf = gdf_new
                else:
                    gdf = pd.concat([gdf, gdf_new])

            description = f"lulc_v4_{valid_gee_text(district)}_{valid_gee_text(block)}_boundaries_{block_start}_{block_end}"
            chunk_names.append(description)
            gdf.to_file(directory + f"/{description}.shp")
            zip_vector(directory, description)
    else:
        gdf = None
        for ind, domain in enumerate(["all", "plantation"]):
            join_boundaries_for_domain(output_dir, blocks_count, domain)
            gdf_new = gpd.read_file(output_dir + "/" + domain + ".shp")
            if ind == 0:
                gdf = gdf_new
            else:
                gdf = pd.concat([gdf, gdf_new])
        description = (
            f"lulc_v4_{valid_gee_text(district)}_{valid_gee_text(block)}_boundaries"
        )
        chunk_names.append(description)
        gdf.to_file(output_dir + f"/{description}.shp")
        zip_vector(output_dir, description)

    with open(output_dir + "/all_done", "w") as f:
        f.write("all done")

    return chunk_names


def export_to_gee(chunk_names):
    task_ids = []
    asset_ids = []
    for chunk_name in chunk_names:
        print("chunk_name=", chunk_name)
        asset_id = get_gee_asset_path(state, district, block) + chunk_name
        asset_ids.append(asset_id)
        # if is_gee_asset_exists(asset_id):
        #     return
        path = directory + "/" + chunk_name + ".shp"
        task_id = upload_shp_to_gee(path, chunk_name, asset_id)
        task_ids.append(task_id)
    check_task_status(task_ids, 200)

    if len(asset_ids) > 1:
        assets = []
        for asset_id in asset_ids:
            assets.append(ee.FeatureCollection(asset_id))

        fc = ee.FeatureCollection(assets).flatten()

        description = (
            f"lulc_v4_{valid_gee_text(district)}_{valid_gee_text(block)}_boundaries"
        )
        asset_id = get_gee_asset_path(state, district, block) + description
        sync_fc_to_gee(fc, description, asset_id)


"""

Helper function for dividing an roi into blocks

"""


def process_image(image_path, model, conf_thresholds, class_names):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return None, None, None, None, None

    results = model.predict(img)

    pred_classes = []
    conf_scores = []
    masks = []

    if results[0].masks is not None:
        for i, (mask, cls, conf) in enumerate(
            zip(
                results[0].masks.data.cpu().numpy(),
                results[0].boxes.cls.cpu().numpy(),
                results[0].boxes.conf.cpu().numpy(),
            )
        ):
            class_name = class_names[int(cls)]
            if conf >= conf_thresholds[class_name]:
                pred_classes.append(class_name)
                conf_scores.append(conf)
                masks.append(mask)
    if masks == []:
        binary_array = np.zeros((256, 256), dtype=np.uint8)
    else:
        sum_array = np.sum(masks, axis=0)
        binary_array = (sum_array > 0).astype(np.uint8)
    return image_path, binary_array, pred_classes, conf_scores


def stitch_masks(masks, output_dir):
    image_size = 256
    gt_bound_names = glob(output_dir + "/chunks/*.tif")
    gt_bound_names = [i for i in gt_bound_names if "chunk_" in i]
    print("Found {} groundtruth chunks".format(len(gt_bound_names)))
    # Sort the file list based on the i, j values
    image_names = sorted(gt_bound_names, key=extract_indices)
    # Extract the maximum i and j values to determine the final stitched image size
    max_i = max(extract_indices(file)[0] for file in image_names)
    max_j = max(extract_indices(file)[1] for file in image_names)

    # Stitch image
    print("stiching mask")
    stitched_image_array = np.zeros(
        ((max_i + 1) * image_size, (max_j + 1) * image_size), dtype=np.float32
    )
    for ind, name in enumerate(image_names):
        img_arr = masks[ind].T
        i, j = extract_indices(name)
        y_offset = i * image_size
        x_offset = j * image_size
        stitched_image_array[
            y_offset : y_offset + image_size, x_offset : x_offset + image_size
        ] = img_arr

    with open(output_dir + "/plantations_predicted.pickle", "wb") as handle:
        pickle.dump(instances_predicted, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stitched_image_array


def run_plantation_model(output_dir, row, index, directory, blocks_df):
    if row["plantation_status"] == True:
        return
    model_path = "scrubland_field_delineation/plantation_model.pt"
    conf_thresholds = {
        "plantations": 0.5,
    }
    class_names = [
        "plantations",
    ]
    model = YOLO(model_path)
    gt_bound_names = glob(output_dir + "/chunks/*.tif")
    gt_bound_names = [i for i in gt_bound_names if "chunk_" in i]
    print("Found {} groundtruth chunks".format(len(gt_bound_names)))

    # Sort the file list based on the i, j values
    image_names = sorted(gt_bound_names, key=extract_indices)
    masks = []
    for image in image_names:
        _, mask, _, _ = process_image(image, model, conf_thresholds, class_names)
        masks.append(mask)
    mask = stitch_masks(masks, output_dir)
    save_field_boundaries(output_dir, mask.T, others="plantation")
    mark_done(index, directory, blocks_df, "plantation_status")
    return


def mark_done(index, output_dir, df, label):
    df = pd.read_csv(output_dir + "/status.csv", index_col=False)
    df.loc[df["index"] == index, label] = True
    df.to_csv(output_dir + "/status.csv", index=False)


def run(roi, directory, max_tries=5, delay=1):
    attempt = 0
    complete = False

    while attempt < max_tries + 1 and not complete:
        try:
            blocks_df = get_points(roi, directory, zoom, scale)
            gcs_blob_name = f"{GCS_SHAPEFILE_BUCKET}/{district}_{block}/status.csv"
            upload_file_to_gcs(directory + "/status.csv", gcs_blob_name)

            for _, row in blocks_df[blocks_df["overall_status"] == False].iterrows():
                index = row["index"]
                point = row["points"]

                # import ipdb
                # ipdb.set_trace()
                output_dir = f"{directory}/{zoom}/{str(index)}"
                download(
                    point, output_dir, row, index, directory, blocks_df, zoom, scale
                )
                run_model(output_dir, row, index, directory, blocks_df)
                get_segmentation(output_dir, row, index, directory, blocks_df)
                run_postprocessing(output_dir, row, index, directory, blocks_df)
                run_plantation_model(output_dir, row, index, directory, blocks_df)
                mark_done(index, directory, blocks_df, "overall_status")
                attempt = 0

            chunk_names = join_boundaries(directory, len(blocks_df))
            # Export final shape files to GEE
            export_to_gee(chunk_names)
            complete = True
        except Exception as e:
            if attempt == max_tries:
                print(f"Run failed after {max_tries} retries. Aborting.")
                return
            print(f"Retrying: Attempt {attempt + 1} failed at run {e}")
            attempt += 1
            time.sleep(delay)


if __name__ == "__main__":
    ee_initialize()
    state = sys.argv[1]
    district = sys.argv[2]
    block = sys.argv[3]

    roi = ee.FeatureCollection(
        get_gee_asset_path(state, district, block)
        + "filtered_mws_"
        + valid_gee_text(district.lower())
        + "_"
        + valid_gee_text(block.lower())
        + "_uid"
    ).union()

    zoom = 17
    scale = 16
    directory = f"data/{state}/{district}/{block}"

    os.makedirs(directory, exist_ok=True)
    sys.stdout = Logger(directory + "/output.log")
    print("Area of the Rectangle is ", roi.geometry().area().getInfo() / 1e6)

    # print("Running for " + str(len(blocks_df)) + " points...")

    run(roi, directory)
