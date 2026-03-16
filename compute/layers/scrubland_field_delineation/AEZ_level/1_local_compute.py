import os, sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
print(SCRIPT_DIR)
print(SCRIPT_DIR.parent)
PROJECT_ROOT = SCRIPT_DIR.parent

for search_path in (PROJECT_ROOT, SCRIPT_DIR):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

module_paths = [
    PROJECT_ROOT / "decode/FracTAL_ResUNet/models/semanticsegmentation",
    PROJECT_ROOT / "decode/FracTAL_ResUNet/nn/loss",
]


for module_path in module_paths:
    module_path_str = str(module_path)
    if module_path_str not in sys.path:
        sys.path.append(module_path_str)

from PIL import Image
import PIL
import re
from multiprocessing import cpu_count
from glob import glob
from mxnet import gluon
from mxnet import image
from skimage import measure
import pickle
from commons import tms_to_geotiff
import math
from datasets import *
from itertools import product
import ee
import geopandas as gpd
from ultralytics import YOLO
import ast
import time

PIL.Image.MAX_IMAGE_PIXELS = 2000000000
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
from osgeo import gdal, ogr, osr
from commons import raster_to_shp
import zipfile
import pandas as pd
from itertools import combinations
from datetime import datetime
from scipy.spatial.distance import jensenshannon

# from pydrive2.auth import GoogleAuth
# from pydrive2.drive import GoogleDrive

from utils import (
    ee_initialize,
    get_gee_asset_path,
    get_gee_dir_path,
    valid_gee_text,
    upload_shp_to_gee,
    is_gee_asset_exists,
    upload_file_to_gcs,
    check_task_status,
    sync_fc_to_gee,
    make_asset_public,
    download_csv_from_gcs,
)

from constants import GCS_BUCKET_NAME


original_image, min_j, min_i, max_j, max_i, instances_predicted = (0, 0, 0, 0, 0, 0)
mapping = {"farm": 1, "plantation": 2, "scrubland": 3, "rest": 0}

STATUS_COLUMNS = [
    "overall_status",
    "download_status",
    "model_status",
    "segmentation_status",
    "postprocessing_status",
    "plantation_status",
]

MODEL_DIR = PROJECT_ROOT / "models"
SEGMENTATION_MODEL_PATH = MODEL_DIR / "india_Airbus_SPOT_model.params"
PLANTATION_MODEL_PATH = MODEL_DIR / "plantation_model.pt"


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


# Function to convert latitude to pixel Y at a given zoom level
def lat_to_pixel_y(lat, zoom):
    sin_lat = math.sin(math.radians(lat))
    pixel_y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * (
        2 ** (zoom + 8)
    )
    return pixel_y


# Function to convert longitude to pixel X at a given zoom level
def lon_to_pixel_x(lon, zoom):
    pixel_x = ((lon + 180) / 360) * (2 ** (zoom + 8))
    return pixel_x


# Function to convert pixel X to longitude
def pixel_x_to_lon(pixel_x, zoom):
    lon = (pixel_x / (2 ** (zoom + 8))) * 360 - 180
    return lon


# Function to convert pixel Y to latitude
def pixel_y_to_lat(pixel_y, zoom):
    n = math.pi - 2 * math.pi * pixel_y / (2 ** (zoom + 8))
    lat = math.degrees(math.atan(math.sinh(n)))
    return lat


def lat_lon_from_pixel(lat, lon, zoom, scale):
    """
    Given a starting latitude and longitude, calculate the latitude and longitude
    of the opposite corner of a 256x256 image at a given zoom level.
    """
    pixel_x = lon_to_pixel_x(lon, zoom)
    pixel_y = lat_to_pixel_y(lat, zoom)

    new_lon = pixel_x_to_lon(pixel_x + 256 * scale, zoom)
    new_lat = pixel_y_to_lat(pixel_y + 256 * scale, zoom)

    return new_lat, new_lon


def divide_tiff_into_chunks(chunk_size, output_dir):
    # Load the large TIFF image
    input_image_path = output_dir + "/field.tif"
    image = Image.open(input_image_path)
    # Get image dimensions
    width, height = image.size

    # Iterate over the image to create 256x256 chunks
    ind_i = 0
    ind_j = 0
    for i in range(0, width, chunk_size):
        for j in range(0, height, chunk_size):
            # Define the box to crop
            box = (i, j, i + chunk_size, j + chunk_size)

            # Crop the image using the defined box
            chunk = image.crop(box)
            # Save each chunk as a separate TIFF file
            chunk.save(
                os.path.join(output_dir + "/chunks/", f"chunk_{ind_i}_{ind_j}.tif")
            )
            ind_j += 1
        ind_i += 1
        ind_j = 0

    print("Image has been split into 256x256 chunks and saved successfully.")


def download(bbox, output_dir, row, index, directory, blocks_df):
    if row["download_status"] == True:
        return

    # scale = 16
    zoom = 17
    chunk_size = 256

    (lat1, lon1), (lat2, lon2) = bbox
    # new_lat, new_lon = lat_lon_from_pixel(lat, lon, zoom, scale)
    # print(lat,",",lon,new_lat,",",new_lon)
    # os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + "/chunks", exist_ok=True)
    tms_to_geotiff(
        output=output_dir + "/field.tif",
        bbox=[lon1, lat1, lon2, lat2],
        zoom=zoom,
        source="Satellite",
        overwrite=True,
        threads=1,
    )
    divide_tiff_into_chunks(chunk_size, output_dir)
    mark_done(index, directory, "download_status")


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
    trained_model = SEGMENTATION_MODEL_PATH
    if not trained_model.exists():
        raise FileNotFoundError(f"Segmentation model not found: {trained_model}")
    if ctx_name == "cpu":
        ctx = mx.cpu()
    elif ctx_name == "gpu":
        ctx = mx.gpu(gpu_id)

    # initialise model
    model = FracTAL_ResUNet_cmtsk(
        nfilters_init=n_filters, depth=depth, NClasses=n_classes
    )
    model.load_parameters(str(trained_model), ctx=ctx)
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

    mark_done(index, directory, "model_status")


"""
    Helper functions for watershed algorithm
"""


# Function to extract the i, j values from the file name
def extract_indices(file_name):
    match = re.search(r"chunk_(\d+)_(\d+)\.tif", file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


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
    mark_done(index, directory, "segmentation_status")


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


def map_entropy_old(index):
    img, mask = crop_image_by_mask(original_image, index)
    ent = get_entropy(img, mask)
    # ent_plantation = get_entropy_plantation(img, mask)
    rectangularity = get_rectangularity(mask)
    # fractal_dimension = get_perimeter_area_fractal_dimension(mask)
    size = sum(sum(mask))
    # lines = get_ht_lines(img, mask)
    # right_angles = check_right_angles(lines)
    # print(fractal_dimension)
    # blueness = np.mean(cv2.split(np.array(img))[2])
    # greeness = np.mean(cv2.split(np.array(img))[1])
    # redness = np.mean(cv2.split(np.array(img))[0])
    # red = redness/greeness

    # easy_farm = rectangularity>=0.67 and size>500 and size <2000 and ent<1
    # easy_plantation =  rectangularity>=0.7 and size>500 and size<20000 and ent>4 and len(right_angles)>5
    # easy_scrub = (ent>2.5 and len(lines)<=1 and size>2000 and rectangularity<0.67 and red>1) or size>100000
    # class_ = "rest"
    # if easy_farm:
    #    class_ = "farm"
    # elif easy_plantation:
    #    color_map = 2
    # elif easy_scrub:
    #    class_ = "scrubland"
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


def map_entropy(index):
    img, mask = crop_image_by_mask(original_image, index)
    ent = get_entropy(img, mask)
    rectangularity = get_rectangularity(mask)
    size = sum(sum(mask))
    return (
        index,
        ent,
        rectangularity,
        size,
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
            # "class",
            "ent",
            # "ent_pl",
            "rect",
            # "frct_dim",
            "size",
            # "num_lines",
            # "num_rt_ang",
            # "blueness",
            # "greeness",
            # "redness",
            # "red"
        ],
    )
    save_field_boundaries(output_dir, instances_predicted, df=df)
    mark_done(index, directory, "postprocessing_status")


def zip_vector(output_dir, vector_name):
    zip = zipfile.ZipFile(
        output_dir + "/" + vector_name + ".zip", "w", zipfile.ZIP_DEFLATED
    )
    files = [vector_name + i for i in [".shp", ".cpg", ".dbf", ".prj", ".shx"]]
    for file in files:
        zip.write(output_dir + "/" + file)
    zip.close()


def divide_into_chunks(block_size, chunk_size=100):
    chunks = []
    for start in range(0, block_size, chunk_size):
        end = min(start + chunk_size - 1, block_size - 1)
        chunks.append((start, end))
    return chunks


def join_boundaries_for_domain(output_dir, ind, block_start, block_end, domain):
    gdf = None
    for i in range(block_start, block_end):
        gdf_new = gpd.read_file(output_dir + "/" + str(i) + "/" + domain + ".shp")
        if i == 0:
            gdf = gdf_new
        else:
            gdf = pd.concat([gdf, gdf_new])
    gdf.to_file(output_dir + "/" + domain + "_" + str(ind) + ".shp")
    # zip_vector(output_dir, domain)


def join_boundaries(output_dir, blocks_count):
    if os.path.exists(output_dir + "/all_done"):
        print("Everything already done")
        return
    output_name = os.path.basename(os.path.normpath(output_dir))
    chunks = divide_into_chunks(blocks_count)
    for index, (block_start, block_end) in enumerate(chunks):
        gdf = None
        for ind, domain in enumerate(["all", "plantation"]):
            join_boundaries_for_domain(
                output_dir, index, block_start, block_end, domain
            )
            gdf_new = gpd.read_file(
                output_dir + "/" + domain + "_" + str(index) + ".shp"
            )
            if ind == 0:
                gdf = gdf_new
            else:
                gdf = pd.concat([gdf, gdf_new])
        gdf.to_file(
            output_dir + "/" + output_name + "_boundaries_" + str(index) + ".shp"
        )
        zip_vector(output_dir, output_name + "_boundaries_" + str(index))
    with open(output_dir + "/all_done", "w") as f:
        f.write("all done")


"""

Helper function for dividing an roi into blocks

"""


def get_n_boxes(lat, lon, n, zoom, scale):
    diagonal_lat_lon = [
        (lat, lon),
    ]
    for i in range(n):
        new_lat_lon = lat_lon_from_pixel(lat, lon, zoom, scale)
        diagonal_lat_lon.append(new_lat_lon)
        lat, lon = new_lat_lon
    lats = [i[0] for i in diagonal_lat_lon]
    longs = [i[1] for i in diagonal_lat_lon]
    first_points = list(product(lats, longs))
    boxes = []
    for point in first_points:
        lat, lon = point
        new_lat, new_lon = lat_lon_from_pixel(lat, lon, zoom, scale)
        boxes.append([(lat, lon), (lat, new_lon), (new_lat, new_lon), (new_lat, lon)])
    return boxes


def latlon_to_tile_xy(lat, lon, zoom):
    """Converts lat/lon to tile x/y at given zoom level"""
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    tile_x = int((lon + 180.0) / 360.0 * n)
    tile_y = int(
        (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    )
    return tile_x, tile_y


def tile_xy_to_latlon(tile_x, tile_y, zoom):
    """Converts top-left corner of tile x/y at given zoom level to lat/lon"""
    n = 2.0**zoom
    lon_deg = tile_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def get_points(roi, directory):
    df = pd.read_csv(directory + "/status.csv", index_col=False)
    df["points"] = df["points"].apply(ast.literal_eval)
    return df


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
    model_path = PLANTATION_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Plantation model not found: {model_path}")
    conf_thresholds = {
        "plantations": 0.5,
    }
    class_names = [
        "plantations",
    ]
    model = YOLO(str(model_path))
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
    mark_done(index, directory, "plantation_status")
    return


from filelock import FileLock


def mark_done(index, output_dir, label):
    csv_path = os.path.join(output_dir, "status.csv")
    lock_path = csv_path + ".lock"
    with FileLock(lock_path):
        df = pd.read_csv(csv_path)
        df.loc[df["index"] == index, label] = True
        tmp = csv_path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, csv_path)


def get_csv(points):
    first_points = [i[0] for i in points]
    tiles = []
    for index, point in enumerate(first_points):
        four_first_point = get_n_boxes(point[0], point[1], 1, 17, 16)
        four_first_point = [(i[0], i[2]) for i in four_first_point]
        for top_left, bottom_right in four_first_point:
            tiles.append((index, (top_left, bottom_right)))
    return tiles


def get_tiles(roi):
    zoom = 17
    scale = 32
    bounds = roi.bounds().coordinates().get(0).getInfo()
    lons = sorted([i[0] for i in bounds])
    lats = sorted([i[1] for i in bounds])

    tile_x, tile_y = latlon_to_tile_xy(lats[-1], lons[0], zoom)
    top_left_lat, top_left_lon = tile_xy_to_latlon(tile_x, tile_y, zoom)

    starting_point = top_left_lat, top_left_lon
    min_, max_ = (
        [lon_to_pixel_x(top_left_lon, zoom), lat_to_pixel_y(lats[0], zoom)],
        [lon_to_pixel_x(lons[-1], zoom), lat_to_pixel_y(top_left_lat, zoom)],
    )
    iterations = math.ceil(
        max(abs(min_[0] - max_[0]), abs(min_[1] - max_[1])) / 256 / scale
    )
    tiles = get_n_boxes(starting_point[0], starting_point[1], iterations, zoom, scale)
    points = get_csv(tiles)
    features = []
    for i, tile in enumerate(tiles):
        coords = [[lon, lat] for lat, lon in tile]  # Convert to [lon, lat]
        # coords.append(coords[0])  # Close the polygon
        polygon = ee.Geometry.Polygon([coords])
        feature = ee.Feature(polygon, {"grid_id": i})
        features.append(feature)
    tiles = ee.FeatureCollection(features)
    return tiles, points


def write_to_gee(fc, asset_name):
    task = ee.batch.Export.table.toAsset(
        collection=fc.filterBounds(roi),
        description="exportToTableAssetExample",
        assetId=asset_name,
    )
    task.start()
    print(f"Started export to GEE asset: {asset_name}")
    while True:
        status = task.status()
        state = status.get("state", "UNKNOWN")
        print(f"Export status: {state}")
        if state in ["COMPLETED", "FAILED", "CANCELLED"]:
            break
        time.sleep(30)
    if state == "COMPLETED":
        print(f"Export to GEE asset {asset_name} completed successfully.")
    else:
        print(f"Export to GEE asset {asset_name} failed with status: {state}")


def write_to_drive(fc, folder_name, file_name):
    task = ee.batch.Export.table.toDrive(
        collection=fc.filterBounds(roi),
        description="exportToDriveTableExample",
        folder=folder_name,
        fileNamePrefix=file_name,
        fileFormat="CSV",
    )
    task.start()
    print(f"Started export to Google Drive: {folder_name}/{file_name}")
    while True:
        status = task.status()
        state = status.get("state", "UNKNOWN")
        print(f"Export status: {state}")
        if state in ["COMPLETED", "FAILED", "CANCELLED"]:
            break
        time.sleep(30)
    if state == "COMPLETED":
        print(
            f"Export to Google Drive {folder_name}/{file_name}.csv completed successfully."
        )
    else:
        print(
            f"Export to Google Drive {folder_name}/{file_name}.csv failed with status: {state}"
        )


def write_to_gcs(fc, folder_name, file_name):
    task = ee.batch.Export.table.toCloudStorage(
        collection=fc.filterBounds(roi),
        description="gcs_" + file_name,
        bucket=GCS_BUCKET_NAME,
        fileNamePrefix="lulc_v4/" + file_name,
        fileFormat="CSV",
    )
    task.start()
    print(f"Started export to GCS: {folder_name}/{file_name}")
    while True:
        status = task.status()
        state = status.get("state", "UNKNOWN")
        print(f"Export status: {state}")
        if state in ["COMPLETED", "FAILED", "CANCELLED"]:
            break
        time.sleep(30)
    if state == "COMPLETED":
        print(f"Export to GCS {folder_name}/{file_name}.csv completed successfully.")
    else:
        print(
            f"Export to GCS {folder_name}/{file_name}.csv failed with status: {state}"
        )


def get_prev_and_curr_year_dates():
    """Returns start_date (first day of previous year) and end_date (first day of current year) as strings."""
    now = datetime.now()
    curr_year = now.year
    start_date = f"{curr_year-1}-01-01"
    end_date = f"{curr_year}-01-01"
    return start_date, end_date


def get_histogram(roi, tiles_path):
    start_date, end_date = get_prev_and_curr_year_dates()
    tiles = ee.FeatureCollection(tiles_path)
    lulc = ee.Image(
        "projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/pan_india_lulc_v3_2023_2024"
    )
    green = lulc.gte(6).And(lulc.lte(12))
    emb = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .mosaic()
        .updateMask(green)
    )

    # KMeans clustering as before
    samples = emb.sample(region=roi, scale=10, numPixels=50000)
    clusterer = ee.Clusterer.wekaKMeans(32).train(samples)
    clusters = emb.cluster(clusterer)

    # Use reduceRegions for efficiency
    tiles_with_hist = clusters.reduceRegions(
        collection=tiles,
        reducer=ee.Reducer.frequencyHistogram(),
        scale=10,
        crs=None,
        crsTransform=None,
        tileScale=2,  # Increase tileScale if you still hit memory issues
    )
    return tiles_with_hist


def initialize_status_dataframe(df_points):
    df_status = df_points.reset_index(drop=True).copy()
    df_status["index"] = range(len(df_status))
    for column in STATUS_COLUMNS:
        df_status[column] = False
    return df_status


def save_status_for_representative_tiles(directory, representative_tiles):
    points_path = os.path.join(directory, "points.csv")
    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing points file: {points_path}")

    df_points = pd.read_csv(points_path)
    if "index" not in df_points.columns:
        raise ValueError(f"'index' column missing in {points_path}")

    df_points["index"] = pd.to_numeric(df_points["index"], errors="coerce")
    if df_points["index"].isna().any():
        raise ValueError(
            f"Non-numeric values found in points index column at {points_path}"
        )
    df_points["index"] = df_points["index"].astype(int)

    if not representative_tiles:
        raise ValueError("No representative grid IDs were provided.")
    representative_tiles = [int(grid_id) for grid_id in representative_tiles]

    unique_available_ids = set(df_points["index"].tolist())
    missing_ids = sorted(set(representative_tiles) - unique_available_ids)
    if missing_ids:
        print(
            f"Warning: these grid_ids were not found in points.csv and will be ignored: {missing_ids}"
        )

    filtered_points = df_points[
        df_points["index"].isin(representative_tiles)
    ].reset_index(drop=True)
    if filtered_points.empty:
        raise ValueError(
            "No matching rows found in points.csv for the supplied grid_ids."
        )

    status_df = initialize_status_dataframe(filtered_points)
    status_path = os.path.join(directory, "status.csv")
    status_df.to_csv(status_path, index=False)
    print(f"Saved status.csv with {len(status_df)} rows at {status_path}")


def read_grid_ids(grid_file_path):
    if not os.path.exists(grid_file_path):
        raise FileNotFoundError(f"Grid ID file not found: {grid_file_path}")

    grid_ids = []
    with open(grid_file_path, "r") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            tokens = [token.strip() for token in line.split(",") if token.strip()]
            for token in tokens:
                if token.lower() in {"grid_id", "grid_ids", "tile_no", "index"}:
                    continue
                try:
                    grid_ids.append(int(float(token)))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid grid_id '{token}' in {grid_file_path} at line {line_no}"
                    ) from exc

    if not grid_ids:
        raise ValueError(f"No grid_ids found in {grid_file_path}")

    # Preserve order while removing duplicates.
    unique_grid_ids = list(dict.fromkeys(grid_ids))
    return unique_grid_ids


def get_representative_tiles(directory):
    # this code first download csv from google drive as a df
    directory_name = os.path.basename(os.path.normpath(directory))

    download_csv_from_gcs(
        "lulc_v4",
        f"{directory_name}_tiles_with_hist.csv",
        f"{directory}/histogram_data.csv",
    )
    points_file = Path(directory + "/histogram_data.csv")
    if points_file.is_file():
        df = pd.read_csv(directory + "/histogram_data.csv", index_col=False)
        # df["points"] = df["points"].apply(ast.literal_eval)
    else:
        return

    # Convert histograms into aligned probability vectors
    all_keys = set()
    print(df)
    for h in df["histogram"]:
        hist_dict = ast.literal_eval(h.replace("=", ":"))
        all_keys.update(hist_dict.keys())

    all_keys = sorted([int(k) for k in all_keys])
    K = len(all_keys)

    def normalize(hist_dict):
        vec = np.array([hist_dict.get(k, 0) for k in all_keys])
        if vec.sum() == 0:
            return np.zeros_like(vec)
        return vec / vec.sum()

    df["vector"] = df["histogram"].apply(lambda h: normalize(eval(h.replace("=", ":"))))

    # Compute full area distribution
    full_dist = df["vector"].sum()
    full_dist = full_dist / full_dist.sum()

    # Greedy selection
    selected = []
    remaining = df.index.tolist()
    p = 60

    for _ in range(p):
        best_score = float("inf")
        best_idx = None

        for idx in remaining:
            subset = selected + [idx]
            combined = df.loc[subset, "vector"].sum()
            combined = combined / combined.sum()

            score = jensenshannon(combined, full_dist)
            if score < best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)
    representative_tiles = [int(i) for i in list(df.loc[selected]["grid_id"])]
    save_status_for_representative_tiles(directory, representative_tiles)


def pre_process(roi, directory):
    # Add any pre-processing steps here
    directory_name = os.path.basename(os.path.normpath(directory))
    print(f"{directory_name=}")
    tiles_path = "projects/ee-amanverma/assets/lulc_v4/" + directory_name + "_tiles"
    hist_tile_drive_path = "Scrubland_Field_Delineation/" + directory_name
    print("Getting Tiles")

    tiles, points = get_tiles(roi)
    df = pd.DataFrame(points, columns=["index", "points"])
    df.to_csv(directory + "/points.csv", index=False)
    print("Saving Tiles", tiles_path)
    write_to_gee(tiles, tiles_path)
    print("Doing K-means clustering and computing Histogram")
    tiles_with_hist = get_histogram(roi, tiles_path)
    print("Saving histogram")
    # write_to_drive(
    #     tiles_with_hist, hist_tile_drive_path, directory_name + "_tiles_with_hist"
    # )
    write_to_gcs(
        tiles_with_hist, hist_tile_drive_path, directory_name + "_tiles_with_hist"
    )
    print(
        "Pulling data from drive and getting representative tiles and saving then to status.csv for local compute"
    )
    get_representative_tiles(directory)


def pre_process_from_grid_file(directory, grid_file_path):
    print(f"Preparing status.csv for {directory} from grid-id file: {grid_file_path}")
    representative_tiles = read_grid_ids(grid_file_path)
    print(f"Loaded {len(representative_tiles)} grid_ids from {grid_file_path}")
    save_status_for_representative_tiles(directory, representative_tiles)


def parse_int_list(values, field_name):
    parsed_values = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                parsed_values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Invalid integer '{token}' in {field_name}") from exc
    if not parsed_values:
        raise ValueError(f"No values provided for {field_name}")
    return parsed_values


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run AEZ-level local compute either for full AEZs or selected grid_ids in one AEZ."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    aez_parser = subparsers.add_parser(
        "aez", help="Run full AEZ preprocessing + run for one or more AEZs."
    )
    aez_parser.add_argument(
        "--aezs",
        nargs="+",
        required=True,
        help="List of AEZ numbers. Supports space-separated or comma-separated values, e.g. --aezs 6 7 or --aezs 6,7",
    )
    aez_parser.add_argument(
        "--base-dir",
        default=".",
        help="Base folder where AEZ_<no> directories are stored.",
    )
    aez_parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip pre_process and directly run using existing status.csv.",
    )
    aez_parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only run pre_process.",
    )

    grid_parser = subparsers.add_parser(
        "grid", help="Run one AEZ using a custom grid-id file."
    )
    grid_parser.add_argument("--aez", type=int, required=True, help="AEZ number.")
    grid_parser.add_argument(
        "--grid-file",
        required=True,
        help="Path to grid-id CSV/text file (for example: <AEZ_no>_grids.csv).",
    )
    grid_parser.add_argument(
        "--base-dir",
        default=".",
        help="Base folder where AEZ_<no> directory is stored.",
    )
    grid_parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only create status.csv from grid IDs.",
    )
    return parser


def resolve_grid_file_path(grid_file, directory):
    if os.path.isabs(grid_file):
        return grid_file

    local_candidate = os.path.join(directory, grid_file)
    if os.path.exists(local_candidate):
        return local_candidate
    return grid_file


def run(roi, directory, max_tries=5, delay=1):
    attempt = 0
    complete = False
    while attempt < max_tries + 1 and not complete:
        try:
            blocks_df = get_points(roi, directory)
            for r_i, row in blocks_df[blocks_df["overall_status"] == False].iterrows():
                print("Working on AEZ" + directory + " point " + str(r_i))
                index = row["index"]
                point = row["points"]
                output_dir = directory + "/" + str(index)
                download(point, output_dir, row, index, directory, blocks_df)
                print(index, point)
                run_model(output_dir, row, index, directory, blocks_df)
                get_segmentation(output_dir, row, index, directory, blocks_df)
                run_postprocessing(output_dir, row, index, directory, blocks_df)
                run_plantation_model(output_dir, row, index, directory, blocks_df)
                mark_done(index, directory, "overall_status")
                attempt = 0
            print("Working on " + directory)
            join_boundaries(directory, len(blocks_df))
            complete = True
        except Exception as e:
            time.sleep(30)
            if attempt == max_tries:
                print(f"Run failed after {max_tries} retries. Aborting.")
                return
            print(f"Retrying: Attempt {attempt + 1} failed at run {e}")
            attempt += 1
            time.sleep(delay)


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.mode == "aez":
        aez_numbers = parse_int_list(args.aezs, "--aezs")
        ee_initialize("aman")

        for aez_no in aez_numbers:
            roi = ee.FeatureCollection("users/mtpictd/agro_eco_regions").filter(
                ee.Filter.eq("ae_regcode", aez_no)
            )
            directory = os.path.join(args.base_dir, f"AEZ_{aez_no}")
            os.makedirs(directory, exist_ok=True)

            if not args.skip_preprocess:
                pre_process(roi, directory)
            if not args.skip_run:
                run(roi, directory)

    elif args.mode == "grid":
        directory = os.path.join(args.base_dir, f"AEZ_{args.aez}")
        os.makedirs(directory, exist_ok=True)

        grid_file_path = resolve_grid_file_path(args.grid_file, directory)
        pre_process_from_grid_file(directory, grid_file_path)
        if not args.skip_run:
            run(None, directory)
