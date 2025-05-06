import os
import ee
import sys
import cv2
import numpy as np

import csv
import zipfile
import pandas as pd
import glob
import math
import geopandas as gpd

import time
from ultralytics import YOLO
import skimage
from shapely.geometry import Point
from constants import PONDS_WELLS_DATA_PATH, PONDS_WELLS_MODEL_PATH, GEE_HELPER_PATH
from utils import (
    ee_initialize,
    get_gee_asset_path,
    valid_gee_text,
    upload_shp_to_gee,
    is_gee_asset_exists,
)

from misc import get_points, download

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

def inference_wells():
    # load trained model
    model_path = "ponds_and_wells/Models/Wells_best.pt"

    # CSV file name where masks of detected object will be saved
    csv_file = f"{directory}/TRY_Wells.csv"

    # Define class-specific confidence thresholds
    conf_thresholds = {
        'Wells': 0.71,
    }

    # Class names (ensure these match the order used in your model training)
    class_names = [
        'Wells',

    ]

    # Mapping of class names to abbreviations
    class_abbreviations = {
        'Wells': 'W',

    }
    my_new_model = YOLO(model_path)

    def process_image(image_path, conf_thresholds):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image {image_path}")
            return None, None, None, None, None

        results = my_new_model.predict(img)

        polygons = []
        pred_classes = []
        conf_scores = []

        if results[0].masks is not None:
            for i, (polygon, cls, conf) in enumerate(
                    zip(results[0].masks.xy, results[0].boxes.cls.cpu().numpy(), results[0].boxes.conf.cpu().numpy())):
                class_name = class_names[int(cls)]
                if conf >= conf_thresholds[class_name]:
                    polygons.append(polygon)
                    pred_classes.append(class_name)
                    conf_scores.append(conf)

        # Print only when detections are made
        if len(polygons) == 0:
            print(f"No detections in {image_path}")
        else:
            print(f"Detections found in {image_path}: {len(polygons)} polygons")

        return image_path, len(polygons), polygons, pred_classes, conf_scores

    def extract_xtile_ytile_from_tile_name(tile_name):
        try:
            parts = tile_name.replace('.tif', '').split('_')
            if len(parts) == 4:
                xtile = int(parts[2])
                ytile = int(parts[3])
                return xtile, ytile
            else:
                raise ValueError("Tile name does not contain valid coordinates")
        except Exception as e:
            raise ValueError(f"Invalid tile name {tile_name}: {e}")

    def tile_corners_to_latlon(xtile, ytile, zoom):
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad_nw = math.atan(math.sinh(math.pi * (1 - 2 * (ytile / n))))
        lat_deg_nw = math.degrees(lat_rad_nw)

        lat_rad_se = math.atan(math.sinh(math.pi * (1 - 2 * ((ytile + 1) / n))))
        lat_deg_se = math.degrees(lat_rad_se)

        lat_deg_nw = max(min(lat_deg_nw, 85.0511), -85.0511)
        lat_deg_se = max(min(lat_deg_se, 85.0511), -85.0511)

        top_left = (lat_deg_nw, lon_deg)
        top_right = (lat_deg_nw, lon_deg + (360.0 / n))
        bottom_right = (lat_deg_se, lon_deg + (360.0 / n))
        bottom_left = (lat_deg_se, lon_deg)

        return top_left, top_right, bottom_left, bottom_right

    def calculate_tile_center(top_left, top_right, bottom_left, bottom_right):
        center_lat = (top_left[0] + bottom_left[0]) / 2
        center_lon = (top_left[1] + top_right[1]) / 2
        return (center_lat, center_lon)

    image_data = []
    max_vertices = 0

    # Traverse subfolders like TRY/0, TRY/1, ...
    for subfolder in sorted(os.listdir(directory)):
        subfolder_path = os.path.join(directory, subfolder)

        chunk_dir = os.path.join(subfolder_path, "chunks")
        tile_map_path = os.path.join(subfolder_path, "tile_mapping.csv")

        if not os.path.isdir(chunk_dir) or not os.path.exists(tile_map_path):
            continue

        # Load mapping: {chunk_0_0.tif: tile_17_96446_56783.tif}
        tile_mapping = {}
        with open(tile_map_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                chunk_name, tile_name = line.strip().split(',')
                tile_mapping[chunk_name] = tile_name

        # Process each chunk image
        image_files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".tif")]

        for image_path in image_files:
            chunk_name = os.path.basename(image_path)
            if chunk_name not in tile_mapping:
                print(f"Warning: {chunk_name} not in mapping. Skipping.")
                continue

            tile_name = tile_mapping[chunk_name]
            _, _, polygons, pred_classes, conf_scores = process_image(image_path, conf_thresholds)

            if polygons:
                max_vertices = max(max_vertices, max(len(polygon) for polygon in polygons))

            image_data.append({
                'tile_name': tile_name,
                'polygons': polygons,
                'pred_classes': pred_classes,
            })

    # ===================== SAVE TO CSV =====================
    coordinate_headers = []
    for i in range(1, max_vertices + 1):
        coordinate_headers.extend([f"X_{i}", f"Y_{i}"])

    header = ["Tile Name", "Predicted Class", "Center Latitude", "Center Longitude",
              "Top Left Latitude", "Top Left Longitude", "Top Right Latitude", "Top Right Longitude",
              "Bottom Left Latitude", "Bottom Left Longitude", "Bottom Right Latitude",
              "Bottom Right Longitude"] + coordinate_headers

    start_time = time.time()
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)

        for data in image_data:
            tile_name = data['tile_name']
            polygons = data['polygons']
            pred_classes = data['pred_classes']

            xtile, ytile = extract_xtile_ytile_from_tile_name(tile_name)
            top_left, top_right, bottom_left, bottom_right = tile_corners_to_latlon(xtile, ytile, zoom)
            latitude, longitude = calculate_tile_center(top_left, top_right, bottom_left, bottom_right)

            for pred_class, polygon in zip(pred_classes, polygons):
                row = [tile_name, pred_class, latitude, longitude, top_left[0], top_left[1],
                       top_right[0], top_right[1], bottom_left[0], bottom_left[1], bottom_right[0], bottom_right[1]]

                flat_polygon = [coord for point in polygon for coord in point]
                flat_polygon += [None] * (2 * max_vertices - len(flat_polygon))
                row.extend(flat_polygon)
                csvwriter.writerow(row)

    end_time = time.time()
    print(f"CSV file '{csv_file}' saved successfully.")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")

    EARTH_CIRCUMFERENCE_DEGREES = 360

    df = pd.read_csv(csv_file)
    df.rename(columns={'Predicted Class': 'Class'}, inplace=True)

    # Extract the base name dynamically (e.g., "TRY" from "TRY.csv")
    csv_basename = os.path.splitext(os.path.basename(csv_file))[0]

    def degrees_per_pixel(zoom):
        total_pixels = 256 * (2 ** zoom)
        return EARTH_CIRCUMFERENCE_DEGREES / total_pixels

    def pixel_to_geo(x, y, lat_top_left, lon_top_left, lat_bottom_right, lon_bottom_right, img_width, img_height):
        lon_range = lon_bottom_right - lon_top_left
        lat_range = lat_top_left - lat_bottom_right  # Note: latitude decreases as you go south
        lon = lon_top_left + (x / img_width) * lon_range
        lat = lat_top_left - (y / img_height) * lat_range  # y increases downward in image coordinates
        return lon, lat

    # Find the maximum index for X_n and Y_n columns
    x_columns = [col for col in df.columns if col.startswith('X_')]
    if x_columns:
        max_index = max(int(col.split('_')[1]) for col in x_columns)
    else:
        max_index = 0  # If no X_n columns are found

    # Initialize an empty list to store GeoJSON features
    geojson_features = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        lat_top_left = row['Top Left Latitude']
        lon_top_left = row['Top Left Longitude']
        lat_bottom_right = row['Bottom Right Latitude']
        lon_bottom_right = row['Bottom Right Longitude']

        tile_width, tile_height = 256, 256
        object_coords = []

        for i in range(1, max_index + 1):  # Dynamically set the range
            x_col = f'X_{i}'
            y_col = f'Y_{i}'
            if x_col in row and y_col in row:
                x = row[x_col]
                y = row[y_col]
                if pd.notna(x) and pd.notna(y):
                    lon, lat = pixel_to_geo(x, y, lat_top_left, lon_top_left, lat_bottom_right, lon_bottom_right,
                                            tile_width, tile_height)
                    if np.isfinite(lon) and np.isfinite(lat):
                        object_coords.append((lon, lat))

            if pd.isna(x) or pd.isna(y) or i == max_index:
                if object_coords:
                    centroid_lon = np.mean([coord[0] for coord in object_coords])
                    centroid_lat = np.mean([coord[1] for coord in object_coords])
                    centroid = Point(centroid_lon, centroid_lat)

                    feature = {
                        "type": "Feature",
                        "geometry": centroid,
                        "properties": {"Class": row['Class']}
                    }
                    geojson_features.append(feature)
                object_coords = []

    # Create a GeoDataFrame
    geometries = [feature['geometry'] for feature in geojson_features]
    properties = [feature['properties'] for feature in geojson_features]
    gdf_final = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")

    # Transform and save
    gdf_final = gdf_final.to_crs(epsg=3857)

    output_shapefile = f"{csv_basename}.shp"

    # Ensure the output folder exists
    output_folder = directory + "/wells_output"
    os.makedirs(output_folder, exist_ok=True)

    description = f"wells_{valid_gee_text(district)}_{valid_gee_text(block)}"

    # Save the final shapefile in the 'Shapefile_Output' folder
    shapefile_path = os.path.join(output_folder, f"{description}.shp")
    gdf_final.to_file(shapefile_path)

    # Create the ZIP file path
    zip_filename = os.path.join(output_folder, f"{description}.zip")

    # Find all shapefile components (exclude any CSVs just in case)
    shapefile_files = glob.glob(os.path.join(output_folder, f"{description}.*"))
    shapefile_files = [file for file in shapefile_files if not file.endswith('.csv')]

    # Add shapefile components to ZIP
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for file in shapefile_files:
            zipf.write(file, os.path.basename(file))
            print(f"Added to ZIP: {os.path.basename(file)}")

    print(f"Created ZIP file: {zip_filename}")

    # # Delete shapefile components after zipping
    # for file in shapefile_files:
    #     os.remove(file)
    #     print(f"Deleted: {file}")

    print("All shapefile components have been deleted after zipping, except for the .csv file.")


def export_to_gee():
    description = f"wells_{valid_gee_text(district)}_{valid_gee_text(block)}"
    asset_id = get_gee_asset_path(state, district, block) + description
    # if is_gee_asset_exists(asset_id):
    #     return
    path = directory + "/wells_output/" + description + ".shp"
    upload_shp_to_gee(path, description, asset_id)


def run(roi, directory, max_tries=5, delay=1):
    attempt = 0
    complete = False

    while attempt < max_tries + 1 and not complete:
        try:
            blocks_df = get_points(roi, directory, zoom, scale)
            for _, row in blocks_df[
                blocks_df["download_status_" + str(zoom)] == False
            ].iterrows():
                print("Index>>>>", row["index"])
                index = row["index"]
                point = row["points"]

                # import ipdb
                # ipdb.set_trace()
                output_dir = directory + "/" + str(index)
                download(point, output_dir, row, index, directory, blocks_df, zoom)
                # mark_done(index, directory, blocks_df, "overall_status")
                attempt = 0
            print("Download Completed")
            inference_wells()
            export_to_gee()
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
    zoom = 18
    scale = 16
    directory = f"data/{state}/{district}/{block}/{zoom}"

    os.makedirs(directory, exist_ok=True)
    sys.stdout = Logger(directory + "/output.log")
    print("Area of the Rectangle is ", roi.geometry().area().getInfo() / 1e6)

    # print("Running for " + str(len(blocks_df)) + " points...")

    run(roi, directory)
