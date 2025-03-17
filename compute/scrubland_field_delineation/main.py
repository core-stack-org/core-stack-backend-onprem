import os, sys
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
from samgeo import tms_to_geotiff
import math
from datasets import *
from itertools import product
import ee
import geopandas as gpd

from utilities.utils import ee_initialize, get_gee_asset_path, valid_gee_text

PIL.Image.MAX_IMAGE_PIXELS = 2000000000

module_paths=['decode/FracTAL_ResUNet/models/semanticsegmentation', 'decode/FracTAL_ResUNet/nn/loss']
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
from osgeo import gdal, ogr, osr
from samgeo.common import raster_to_shp
import zipfile
import pandas as pd

original_image, min_j, min_i, max_j, max_i, instances_predicted = (0,0,0,0,0,0)

"""
    Helper functions to Download High Resolution Images
"""
# Function to convert latitude to pixel Y at a given zoom level
def lat_to_pixel_y(lat, zoom):
    sin_lat = math.sin(math.radians(lat))
    pixel_y = ((0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * (2 ** (zoom + 8)))
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
    
    new_lon = pixel_x_to_lon(pixel_x + 256*scale, zoom)
    new_lat = pixel_y_to_lat(pixel_y + 256*scale, zoom)

    return new_lat, new_lon

def divide_tiff_into_chunks(chunk_size, output_dir):
    # Load the large TIFF image
    input_image_path = output_dir + '/field.tif'
    image = Image.open(input_image_path)
    # Get image dimensions
    width, height = image.size
    
    # Iterate over the image to create 256x256 chunks
    ind_i=0
    ind_j=0
    for i in range(0, width, chunk_size):
        for j in range(0, height, chunk_size):
            # Define the box to crop
            box = (i, j, i + chunk_size, j + chunk_size)
            
            # Crop the image using the defined box
            chunk = image.crop(box)
            # Save each chunk as a separate TIFF file
            chunk.save(os.path.join(output_dir + "/chunks/", f'chunk_{ind_i}_{ind_j}.tif'))
            ind_j+=1
        ind_i+=1
        ind_j=0

    print("Image has been split into 256x256 chunks and saved successfully.")

def download(bbox, output_dir):
    scale = 16
    zoom = 17
    chunk_size = 256
    
    lat, lon = bbox[0], bbox[1]
    new_lat, new_lon = lat_lon_from_pixel(lat, lon, zoom, scale)
    print(lat,",",lon,new_lat,",",new_lon)
    
    #os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+"/chunks", exist_ok=True)
    tms_to_geotiff(output=output_dir + "/field.tif", bbox=[bbox[1], bbox[0], new_lon, new_lat], zoom=zoom, source='Satellite', overwrite=True)
    divide_tiff_into_chunks(chunk_size, output_dir)


"""

Helper functions for Inference

""" 

# Function to extract the i, j values from the file name
    
def extract_indices(file_name):
    match = re.search(r'chunk_(\d+)_(\d+)\.tif', file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def load_model():
    # hyperparameters for model architecture
    n_filters = 32
    depth = 6
    n_classes = 1
    ctx_name = 'gpu'
    gpu_id = 0
    trained_model='india_Airbus_SPOT_model.params'
    if ctx_name == 'cpu':
        ctx = mx.cpu()
    elif ctx_name == 'gpu':
        ctx = mx.gpu(gpu_id)

    # initialise model
    model = FracTAL_ResUNet_cmtsk(nfilters_init=n_filters, depth=depth, NClasses=n_classes)
    model.load_parameters(trained_model, ctx=ctx)
    return model, ctx

def run_inference(test_dataloader, model, ctx):
    logits_array = []
    bound_array = []
    dist_array = []
    for batch_i, img_data in enumerate(test_dataloader):
        # extract batch data
        imgs=img_data
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
            print("Batch Done", batch+1,"/",len(array))
        return array_new
    logits_array = flatten(logits_array)
    bound_array = flatten(bound_array)
    print(len(logits_array))
    return logits_array, bound_array
    
def run_model(output_dir):
    batch_size = 32  
    CPU_COUNT = cpu_count()
    # extract chunk ids of validation data
    gt_bound_names = glob(output_dir + "/chunks/*.tif")
    gt_bound_names = [i for i in gt_bound_names if "chunk_" in i]
    print('Found {} groundtruth chunks'.format(len(gt_bound_names)))
    
    # Sort the file list based on the i, j values
    image_names = sorted(gt_bound_names, key=extract_indices)
    # Extract the maximum i and j values to determine the final stitched image size
    max_i = max(extract_indices(file)[0] for file in image_names)
    max_j = max(extract_indices(file)[1] for file in image_names)
    print("Max i and Max j are ", max_i, max_j)

    # Load dataset
    test_dataset = Planet_Dataset_No_labels(image_names=image_names)
    test_dataloader = gluon.data.DataLoader(test_dataset, batch_size=batch_size)#, num_workers=CPU_COUNT)

    model, ctx = load_model()
    logits_array, bound_array = run_inference(test_dataloader, model, ctx)


    with open(output_dir + '/logits_bounds.pickle', 'wb') as handle:
        pickle.dump([logits_array, bound_array], handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
    Helper functions for watershed algorithm
"""

# Function to extract the i, j values from the file name
def extract_indices(file_name):
    match = re.search(r'chunk_(\d+)_(\d+)\.tif', file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def get_segmentation(output_dir):
    image_size = 256
    gt_bound_names=glob(output_dir + '/chunks/*.tif')
    gt_bound_names = [i for i in gt_bound_names if "chunk_" in i]
    print('Found {} groundtruth chunks'.format(len(gt_bound_names)))
    # Sort the file list based on the i, j values
    image_names = sorted(gt_bound_names, key=extract_indices)
    # Extract the maximum i and j values to determine the final stitched image size
    max_i = max(extract_indices(file)[0] for file in image_names)
    max_j = max(extract_indices(file)[1] for file in image_names)
    print("Loading Pickle")
    with open(output_dir + '/logits_bounds.pickle', 'rb') as handle:
        logits_array, bound_array = pickle.load(handle)
    print("Pickle Loaded")

    #Stitch image
    print("stiching image")
    stitched_image_array = np.zeros(((max_i + 1) * image_size, (max_j + 1) * image_size), dtype=np.float32)
    stitched_bound_array = np.zeros(((max_i + 1) * image_size, (max_j + 1) * image_size), dtype=np.float32)
    for ind, name in enumerate(image_names):
        img_arr = logits_array[ind].T
        bound_arr = bound_array[ind].T
        i, j = extract_indices(name)
        y_offset = i * image_size
        x_offset = j * image_size
        stitched_image_array[y_offset:y_offset + image_size, x_offset:x_offset + image_size] = img_arr
        stitched_bound_array[y_offset:y_offset + image_size, x_offset:x_offset + image_size] = bound_arr

    print("deleting pickle")
    del logits_array
    del bound_array
    #t_ext_best=0.3
    #t_bnd_best=0.1

    t_ext_best=0.3
    t_bnd_best=0.4
    # do segmentation
    print("Doing segmentation")
    instances_predicted=InstSegm(stitched_image_array, stitched_bound_array, t_ext=t_ext_best, t_bound=t_bnd_best)    
    # label connected regions, non-field (-1) will be labelled as 0
    print("Doing measure")
    instances_predicted= measure.label(instances_predicted, background=-1,return_num=False)
    segments = instances_predicted.max()
    print("Max segments are ", segments)
    with open(output_dir + '/instance_predicted.pickle', 'wb') as handle:
        pickle.dump(instances_predicted, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def crop_image_by_mask(image, index):
    # Find the bounding box of the True values in the mask
    min_x, min_y, max_x, max_y = min_j[index], min_i[index], max_j[index], max_i[index]
    # Crop the image using the bounding box
    cropped_image = image.crop((min_x, min_y, max_x + 1, max_y + 1))
    # Crop the mask as well
    cropped_mask = instances_predicted[min_y:max_y + 1, min_x:max_x + 1]
    cropped_mask = cropped_mask==index
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
    ent = entropy(np.asarray(img.convert('L')).copy(), disk(5), mask=mask)
    ent = ent[ent>5.2]
    ent = sum(ent)/(sum(sum(mask)))
    return ent

def get_entropy_plantation(img, mask):
    ent = entropy(np.asarray(img.convert('L')).copy(), disk(30), mask=mask)
    ent = ent[ent>0]
    ent = sum(ent)/(sum(sum(mask)))
    return ent

def get_lines_by_hough(img, mask):
    masked_image_np = np.array(img.convert("L"))
    #_, binary_image = cv2.threshold(masked_image_np, 50, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(masked_image_np, 50, 150)
    erosion_size = 1
    # Define the kernel for erosion
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    # Erode the mask to remove edge pixels
    eroded_mask = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=1)
    #print(index)
    edges = cv2.bitwise_and(edges, edges, mask=eroded_mask)
    # Perform Hough Line Transformation
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=30)
    if lines is not None and len(lines)>2:
        return 1
    else:
        return 0

def get_rectangularity(mask):
    """
    Compute how rectangular a given binary mask is.
    
    Args:
        mask (np.ndarray): Binary mask (1 for object, 0 for background)
    
    Returns:
        float: Rectangularity score (1.0 is a perfect rectangle, lower means less rectangular)
    """
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
    
def map_entropy(index):
    img, mask = crop_image_by_mask(original_image, index)
    ent = get_entropy(img, mask)
    rectangularity = get_rectangularity(mask)
    color_map = 0
    #if index == 5341:
    #    print(sum(sum(mask)))
    size_of_segment = sum(sum(mask))
    if size_of_segment > 80000:
        color_map = 0 
    elif ent < 1 and rectangularity>0.6:# or ent > 5:
        color_map = 1
    else:
        ent_plantation = get_entropy_plantation(img, mask)
        if ent_plantation < 8.5 and rectangularity>0.67 and size_of_segment<20000:
            color_map = 2
        lines = get_lines_by_hough(img, mask)
        if lines == 1 and rectangularity>0.67 and size_of_segment<20000:
            color_map = 2
    return (ent, index, color_map)

def set_global_for_multiprocessing(oi, mnj, mni, mxj, mxi, ip):
    global original_image
    global min_j
    global min_i
    global max_j
    global max_i
    global instances_predicted
    original_image, min_j, min_i, max_j, max_i, instances_predicted = oi, mnj, mni, mxj, mxi, ip
    
def process_in_chunks(number, chunk_size):
    total_chunks = (number + chunk_size - 1) // chunk_size  # Round up to the next whole number
    results = []
    for i in range(total_chunks):
        start = i * chunk_size
        if start==0:
            start+=1
        end = min(start + chunk_size, number)  # Make sure not to exceed the number
        print(f"Processing chunk: {start} to {end - 1}")
        with mp.Pool(12) as p:
            results += p.map(map_entropy,list(range(start,end)))
    return results

def get_color(color_dict, index):
    def color(ind):
        label = color_dict.get(ind)
        if label==index:
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
            printcounter=0
        printcounter+=1
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

def save_field_boundaries(output_dir, instances_predicted, vector_name):
    ds = gdal.Open(output_dir + "/field.tif")

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [rows, cols] = arr.shape
    arr_min = arr.min()
    arr_max = arr.max()

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_dir + "/out.tif", cols, rows, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(instances_predicted)
    #outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    band=None
    ds=None
    raster_to_shp(tiff_path=output_dir + "/out.tif", output=output_dir + "/" +vector_name+ ".shp")
    gdf = gpd.read_file(output_dir + "/" +vector_name+ ".shp")
    gdf = gdf.set_crs('epsg:3857', allow_override=True)
    print(f"Original CRS: {gdf.crs}")
    gdf = gdf.to_crs(epsg=4326)
    print(f"Reprojected CRS: {gdf.crs}")
    gdf.to_file(output_dir + "/" + vector_name + ".shp")
    #zip = zipfile.ZipFile(output_dir + "/"+vector_name+".zip", "w", zipfile.ZIP_DEFLATED)
    #files = ["out.shp", "out.cpg", "out.dbf", "out.prj", "out.shx"]
    #for file in files:
    #    zip.write(output_dir + "/" + file)
    #zip.close()
    for file in ["out.tif"]:
        os.remove(output_dir + "/" + file)

def run_postprocessing(output_dir):
    input_image_path = output_dir + '/field.tif'
    image = Image.open(input_image_path)

    with open(output_dir + '/instance_predicted.pickle', 'rb') as handle:
        instances_predicted = pickle.load(handle)
        instances_predicted = instances_predicted.T
        
    segments = instances_predicted.max()
    print("Max segments are ", segments)

    original_image = image
    original_image = original_image.crop((0,0,) + instances_predicted.shape)

    min_i, min_j, max_i, max_j = get_min_max_array(instances_predicted)
    set_global_for_multiprocessing(original_image, min_j, min_i, max_j, max_i, instances_predicted)
    print("Haha")
    results = process_in_chunks(segments+1, 12000)
    color_dict = {i[1]:i[2] for i in results}
    color_dict[0] = 0
    
    farm_color = get_color(color_dict, 1)
    plantation_color = get_color(color_dict, 2)
    instances_predicted_farm = np.vectorize(farm_color)(instances_predicted)
    instances_predicted_plantation = np.vectorize(plantation_color)(instances_predicted)
    
    save_field_boundaries(output_dir, instances_predicted_farm, "farm")
    save_field_boundaries(output_dir, instances_predicted_plantation, "plantation")
    
def zip_vector(output_dir, vector_name):
    zip = zipfile.ZipFile(output_dir + "/"+vector_name+".zip", "w", zipfile.ZIP_DEFLATED)
    files = [vector_name+i for i in [".shp", ".cpg", ".dbf", ".prj", ".shx"]]
    for file in files:
        zip.write(output_dir + "/" + file)
    zip.close()
    
def join_boundaries(output_dir, blocks_count):
    gdf_farm = None
    gdf_plantation = None
    for i in range(0, blocks_count):
        gdf_farm_new = gpd.read_file(output_dir+"/"+str(i)+"/farm.shp")
        gdf_plantation_new = gpd.read_file(output_dir+"/"+str(i)+"/plantation.shp")
        if i == 0:
            gdf_farm = gdf_farm_new
            gdf_plantation = gdf_plantation_new
        else:
            gdf_farm = pd.concat([gdf_farm, gdf_farm_new])
            gdf_plantation = pd.concat([gdf_plantation, gdf_plantation_new])

    gdf_farm.to_file(output_dir+"/farm.shp")
    gdf_plantation.to_file(output_dir+"/plantation.shp")
    
    zip_vector(output_dir, "farm")
    zip_vector(output_dir, "plantation")
    
"""

Helper function for dividing an roi into blocks

"""
def get_n_boxes(lat, lon, n, zoom, scale):
    diagonal_lat_lon = [(lat, lon),]
    for i in range(n):
        new_lat_lon = lat_lon_from_pixel(lat, lon, zoom, scale)
        diagonal_lat_lon.append(new_lat_lon)
        lat, lon = new_lat_lon
    lats = [i[0] for i in diagonal_lat_lon]
    longs = [i[1] for i in diagonal_lat_lon]
    return list(product(lats, longs))

def get_points(roi):
    zoom = 17
    scale = 16
    bounds = roi.bounds().coordinates().get(0).getInfo()
    lons = sorted([i[0] for i in bounds])
    lats = sorted([i[1] for i in bounds])
    starting_point = lats[-1], lons[0]
    min_, max_ = (
        [lon_to_pixel_x(lons[0], zoom), lat_to_pixel_y(lats[0], zoom) ],
        [lon_to_pixel_x(lons[-1], zoom), lat_to_pixel_y(lats[-1], zoom)]
        )
    iterations = math.ceil(max(abs(min_[0] -  max_[0]), abs(min_[1] - max_[1]))/256/16)
    points = get_n_boxes(starting_point[0], starting_point[1], iterations, zoom, scale)
    intersect_list = []
    print(len(points))
    for point in points:
        top_left = point
        bottom_right = lat_lon_from_pixel(top_left[0], top_left[1], zoom, scale)
        rectangle = ee.Geometry.Rectangle([(top_left[1], top_left[0]), (bottom_right[1], bottom_right[0])])
        print(top_left, bottom_right)
        intersects = roi.geometry().intersects(rectangle, ee.ErrorMargin(1)).getInfo()
        if intersects:
            intersect_list.append(top_left)
        print(intersects)
    return intersect_list


def scrubland_field_delineation(state, district, block):
    #
    # ee.Authenticate()
    # ee.Initialize(project='ee-raman')

    ee_initialize()
    # Set ROI and directory name below
    roi = ee.FeatureCollection(
        get_gee_asset_path(state, district, block)
        + "filtered_mws_"
        + valid_gee_text(district.lower())
        + "_"
        + valid_gee_text(block.lower())
        + "_uid"
    )
    # roi = ee.FeatureCollection("projects/df-project-iit/assets/core-stack/tamil_nadu/theni/periyakulam/filtered_mws_theni_periyakulam_uid").filter(ee.Filter.stringContains("uid", "2_25099"))
    directory = "tamil_nadu"

    #Boiler plate code to run for a rectangle
    
    #top_left = [19.61346189, 75.38005825]  # Replace lon1 and lat1 with actual values
    #bottom_right = [19.44480749, 75.53687926]  # Replace lon2 and lat2 with actual values
    #directory = "Area_paithan"
    
    # Create a rectangle geometry using the defined corners
    #rectangle = ee.Geometry.Rectangle([top_left[1], bottom_right[0], bottom_right[1], top_left[0]])
    #print("Area of the Rectangle is ", rectangle.area().getInfo()/1e6)
    
    # Create a feature collection with the rectangle as a boundary
    #roi = ee.FeatureCollection([ee.Feature(rectangle)])
    
    points = get_points(roi)
    print("Running for " + str(len(points)) + " points...")
    for index, point in enumerate(points):
        output_dir = directory + "/" + str(index)
        download(point, output_dir)
        run_model(output_dir)
        get_segmentation(output_dir)
        run_postprocessing(output_dir)
    join_boundaries(directory, len(points))
