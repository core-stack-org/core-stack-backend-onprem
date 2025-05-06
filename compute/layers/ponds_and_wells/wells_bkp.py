import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
from random import choice
import re
import csv
import zipfile
import pandas as pd
import glob
import math
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry import Polygon
from ultralytics import YOLO
from PIL import Image
import requests
import time
import os
from ultralytics import YOLO
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from shapely.geometry import box
import ee
from constants import PONDS_WELLS_DATA_PATH, PONDS_WELLS_MODEL_PATH
from utils import (
    ee_initialize,
    get_gee_asset_path,
    valid_gee_text,
    ee_to_gdf,
    export_gdf_to_gee,
)
import sys


def inference_wells(state, district, block):
    # ### 1. Specify input region
    #         a. geojson, or
    #         b. bounding box coordinates

    # gdf = gpd.read_file("Shapefiles/Masalia_mws.geojson")
    ee_initialize()
    roi = ee.FeatureCollection(
        get_gee_asset_path(state, district, block)
        + "filtered_mws_"
        + valid_gee_text(district.lower())
        + "_"
        + valid_gee_text(block.lower())
        + "_uid"
    )
    gdf = ee_to_gdf(roi)

    # %%
    # Compute the bounding box
    minx, miny, maxx, maxy = gdf.total_bounds

    # # Create a bounding box polygon
    # bounding_box = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=gdf.crs)

    # # Plot the geometry and the updated bounding box
    # fig, ax = plt.subplots(figsize=(10, 6))
    # gdf.plot(ax=ax, color="blue", alpha=0.5, edgecolor="black")
    # bounding_box.plot(ax=ax, edgecolor="red", facecolor="none", linewidth=2)

    # plt.show()

    # %%
    # # Get the bounding box coordinates
    # minx, miny, maxx, maxy = gdf.total_bounds

    # Define bounding box points
    topLeft = [minx, maxy]
    topRight = [maxx, maxy]
    bottomRight = [maxx, miny]
    bottomLeft = [minx, miny]

    # # Print the coordinates
    # print(f"topLeft = {topLeft}")
    # print(f"topRight = {topRight}")
    # print(f"bottomRight = {bottomRight}")
    # print(f"bottomLeft = {bottomLeft}")

    # %% [markdown]
    # NOT NEEDED ANYMORE - Give coordinated of the bounding box drawn on GEE

    # %%
    """
    #coords for the bounding box (boipariguda)
    topLeft = [82.07347488402685,18.57110542069261]
    topRight = [82.60081863402685,18.57110542069261]
    bottomRight = [82.60081863402685,18.9572939770273]
    bottomLeft = [82.07347488402685,18.9572939770273]
    """

    # topLeft = [87.17428199198116,24.153542533932317]
    # topRight = [87.1763419285046,24.153542533932317]
    # bottomRight = [87.1763419285046,24.15520674659833]
    # bottomLeft = [87.17428199198116,24.15520674659833] #no wells or ponds

    # topLeft = [87.17053768466987, 24.139038123063]
    # topRight = [87.17703935932197, 24.139038123063]
    # bottomRight = [87.17703935932197, 24.143874618138323]
    # bottomLeft = [87.17053768466987, 24.143874618138323]  # one pond atleast

    # Zoom Level and folder name, where the output (image tiles) will be saved
    zoom_level = 18
    # block_name = "around_masalia"  # 'masalia_subset'

    data_download_folder = os.path.join(
        PONDS_WELLS_DATA_PATH,
        "input",
        str(zoom_level),
        block,
    )

    def deg2num(lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0**zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int(
            (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
            / 2.0
            * n
        )
        return (xtile, ytile)

    def getCoords(zoomLevel):
        topleft = deg2num(topLeft[1], topLeft[0], zoomLevel)  # (c,b)
        topright = deg2num(topRight[1], topRight[0], zoomLevel)  # (a,b)
        bottomright = deg2num(bottomRight[1], bottomRight[0], zoomLevel)  # (a,d)
        bottomleft = deg2num(bottomLeft[1], bottomLeft[0], zoomLevel)  # (c,d)
        xmin = min(topleft[0], topright[0], bottomleft[0], bottomright[0])
        xmax = max(topleft[0], topright[0], bottomleft[0], bottomright[0])
        ymin = min(topleft[1], topright[1], bottomleft[1], bottomright[1])
        ymax = max(topleft[1], topright[1], bottomleft[1], bottomright[1])
        return (xmin, xmax, ymin, ymax)

    def download_map_tiles(base_url, output_folder, zoom_level, scale):
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Get start time
        start_time = time.time()

        # Iterate over each tile within the specified range
        xmin, xmax, ymin, ymax = getCoords(zoom_level)
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                # Construct the URL for the current tile with scale=3 for 640x640
                tile_url = f"{base_url}&x={x}&y={y}&z={zoom_level}&scale={scale}"
                print(tile_url)
                try:
                    # Send HTTP GET request to fetch the tile
                    response = requests.get(tile_url)

                    # Check if request was successful (status code 200)
                    if response.status_code == 200:
                        # Save the tile image to a file in the output folder
                        filename = f"tile_{zoom_level}_{x}_{y}.png"
                        filepath = os.path.join(output_folder, filename)
                        with open(filepath, "wb") as f:
                            f.write(response.content)
                        print(f"Downloaded: {filename}")
                    else:
                        print(
                            f"Failed to download tile ({x}, {y}), HTTP status code: {response.status_code}"
                        )

                except Exception as e:
                    print(f"Error downloading tile ({x}, {y}): {e}")

        # Get end time
        end_time = time.time()

        # Print the total execution time
        print(f"Total time taken: {end_time - start_time} seconds")

    base_url = "https://mt1.google.com/vt/lyrs=s"
    scale = 1  # scale of 1 = 256*256 dimensional image

    if not os.path.exists(data_download_folder):
        os.makedirs(data_download_folder)
        print(f"Created the folder: {data_download_folder}")
    else:
        print(f"The folder already exists: {data_download_folder}")

    download_map_tiles(base_url, data_download_folder, zoom_level, scale)

    model_path = f"{PONDS_WELLS_MODEL_PATH}/Wells_best.pt"

    output_folder = os.path.join(
        f"{PONDS_WELLS_DATA_PATH}/output/{str(zoom_level)}",
        block,
    )

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    csv_file = os.path.join(output_folder, block + ".csv")
    print(csv_file)

    # Define class-specific confidence thresholds
    conf_thresholds = {
        "Wells": 0.71,
    }

    # Class names (ensure these match the order used in your model training)
    class_names = [
        "Wells",
    ]

    # Mapping of class names to abbreviations
    class_abbreviations = {
        "Wells": "W",
    }

    # Load the model
    my_new_model = YOLO(model_path)

    # FUNCTIONS
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
                zip(
                    results[0].masks.xy,
                    results[0].boxes.cls.cpu().numpy(),
                    results[0].boxes.conf.cpu().numpy(),
                )
            ):
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

    def extract_xtile_ytile(image_path):
        try:
            basename = os.path.basename(image_path)
            parts = basename.split("_")
            if len(parts) >= 4:
                xtile = int(parts[2])
                ytile = int(parts[3].split(".")[0].split()[0])
                return xtile, ytile
            else:
                raise ValueError("Filename does not contain valid tile coordinates")
        except Exception as e:
            raise ValueError(
                f"Filename {image_path} does not contain valid tile coordinates: {e}"
            )

    def tile_corners_to_latlon(xtile, ytile, zoom):
        n = 2.0**zoom
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

    # def visualize_polygons(image_path, polygons, pred_classes, conf_scores):
    #     img = cv2.imread(image_path)
    #     if img is None:
    #         print(f"Error: Unable to load image {image_path}")
    #         return
    #
    #     if len(polygons) == 0:
    #         print(f"No predictions for image {image_path}, not saving.")
    #         return
    #
    #     for i, polygon in enumerate(polygons):
    #         polygon = np.array(polygon, dtype=np.int32)
    #         polygon = polygon.reshape((-1, 1, 2))
    #         cv2.polylines(img, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
    #
    #         # Calculate the centroid of the polygon for placing the text
    #         M = cv2.moments(polygon)
    #         if M["m00"] != 0:
    #             cX = int(M["m10"] / M["m00"])
    #             cY = int(M["m01"] / M["m00"])
    #         else:
    #             cX, cY = polygon[0][0]
    #
    #         # Put class abbreviation and confidence score text on the image
    #         class_abbr = class_abbreviations[pred_classes[i]]
    #         conf_text = f"{class_abbr}: {conf_scores[i]:.2f}"
    #         cv2.putText(
    #             img, conf_text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
    #         )
    #
    #     output_path = os.path.join(annotated_images_dir, os.path.basename(image_path))
    #     cv2.imwrite(output_path, img)
    #     print(f"Annotated image saved to {output_path}")

    # Predictions
    image_files = [
        os.path.join(data_download_folder, f)
        for f in os.listdir(data_download_folder)
        if os.path.isfile(os.path.join(data_download_folder, f))
    ]

    len(image_files)

    # Start timing
    start_time = time.time()

    max_vertices = 0
    for image_path in image_files:
        _, _, polygons, _, _ = process_image(image_path, conf_thresholds)
        for polygon in polygons:
            max_vertices = max(max_vertices, len(polygon))

    with open(csv_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        header = [
            "Image Path",
            "Predicted Class",
            "Center Latitude",
            "Center Longitude",
            "Top Left Latitude",
            "Top Left Longitude",
            "Top Right Latitude",
            "Top Right Longitude",
            "Bottom Left Latitude",
            "Bottom Left Longitude",
            "Bottom Right Latitude",
            "Bottom Right Longitude",
        ]
        for i in range(1, max_vertices + 1):
            header.extend([f"X_{i}", f"Y_{i}"])
        csvwriter.writerow(header)

        for image_path in image_files:
            try:
                image_path, num_polygons, polygons, pred_classes, _ = process_image(
                    image_path, conf_thresholds
                )
                if image_path is None:
                    continue

                xtile, ytile = extract_xtile_ytile(image_path)
                top_left, top_right, bottom_left, bottom_right = tile_corners_to_latlon(
                    xtile, ytile, zoom_level
                )
                latitude, longitude = calculate_tile_center(
                    top_left, top_right, bottom_left, bottom_right
                )

                for pred_class, polygon in zip(pred_classes, polygons):
                    row = [
                        image_path,
                        pred_class,
                        latitude,
                        longitude,
                        top_left[0],
                        top_left[1],
                        top_right[0],
                        top_right[1],
                        bottom_left[0],
                        bottom_left[1],
                        bottom_right[0],
                        bottom_right[1],
                    ]
                    for point in polygon:
                        row.extend([point[0], point[1]])
                    csvwriter.writerow(row)
            except ValueError as e:
                print(e)
                continue

    # End timing
    end_time = time.time()

    # Print the time taken
    time_taken = end_time - start_time
    print(f"CSV file '{csv_file}' saved successfully.")
    print(f"Time taken to complete: {time_taken:.2f} seconds.")

    # Convert to Geometry

    # Constants
    EARTH_CIRCUMFERENCE_DEGREES = 360  # degrees

    # Load the CSV file
    df = pd.read_csv(csv_file)

    df.rename(columns={"Predicted Class": "Class"}, inplace=True)

    from shapely.geometry import Polygon, Point

    def degrees_per_pixel(zoom):
        total_pixels = 256 * (2**zoom)
        degrees_per_pixel = EARTH_CIRCUMFERENCE_DEGREES / total_pixels
        return degrees_per_pixel

    def pixel_to_geo(
        x,
        y,
        lat_top_left,
        lon_top_left,
        lat_bottom_right,
        lon_bottom_right,
        img_width,
        img_height,
    ):
        lon_range = lon_bottom_right - lon_top_left
        lat_range = (
            lat_top_left - lat_bottom_right
        )  # Note: latitude decreases as you go south
        lon = lon_top_left + (x / img_width) * lon_range
        lat = (
            lat_top_left - (y / img_height) * lat_range
        )  # y increases downward in image coordinates
        return lon, lat

    # Initialize an empty list to store GeoJSON features
    geojson_features = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Extract bounding coordinates of the image tile
        lat_top_left = row["Top Left Latitude"]
        lon_top_left = row["Top Left Longitude"]
        lat_bottom_right = row["Bottom Right Latitude"]
        lon_bottom_right = row["Bottom Right Longitude"]

        tile_width, tile_height = 256, 256

        # Initialize an empty list to store coordinates of the current object
        object_coords = []

        for i in range(1, 8759):  # Adjust this range according to your data
            x_col = f"X_{i}"
            y_col = f"Y_{i}"
            if x_col in row and y_col in row:
                x = row[x_col]
                y = row[y_col]
                if pd.notna(x) and pd.notna(y):
                    # Convert pixel coordinates to geographic coordinates
                    lon, lat = pixel_to_geo(
                        x,
                        y,
                        lat_top_left,
                        lon_top_left,
                        lat_bottom_right,
                        lon_bottom_right,
                        tile_width,
                        tile_height,
                    )
                    if np.isfinite(lon) and np.isfinite(lat):
                        object_coords.append((lon, lat))

            # If it's the end of the coordinates or NaN is encountered, calculate the center
            if pd.isna(x) or pd.isna(y) or i == 8758:
                if object_coords:  # If there are valid coordinates, compute the center
                    # Calculate the centroid
                    centroid_lon = np.mean([coord[0] for coord in object_coords])
                    centroid_lat = np.mean([coord[1] for coord in object_coords])
                    centroid = Point(centroid_lon, centroid_lat)

                    # Create the GeoJSON feature for the point
                    feature = {
                        "type": "Feature",
                        "geometry": centroid,
                        "properties": {"Class": row["Class"]},
                    }
                    geojson_features.append(feature)
                object_coords = []

    # Extract geometries and properties for GeoDataFrame
    geometries = [feature["geometry"] for feature in geojson_features]
    properties = [feature["properties"] for feature in geojson_features]

    # Create a GeoDataFrame using the geometries and properties
    gdf_final = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")

    # Transform the GeoDataFrame to EPSG:3857 (Web Mercator)
    gdf_final = gdf_final.to_crs(epsg=3857)

    # Save the GeoDataFrame as a shapefile
    gdf_final.to_file(
        os.path.join(
            os.path.dirname(csv_file),
            str.split(os.path.basename(csv_file), ".")[0] + ".shp",
        )
    )

    # Optional: Print the first 5 rows of the GeoDataFrame to verify
    print(gdf_final.head())

    print(degrees_per_pixel(17))

    os.path.join(
        os.path.dirname(csv_file),
        str.split(os.path.basename(csv_file), ".")[0] + ".shp",
    )
    description = f"wells_{district}_{block}"
    export_gdf_to_gee(gdf_final, roi, description, state, district, block)


def export_wells_to_gee(state, district, block):
    ee_initialize()
    path = "data/ponds_and_wells/output/18/gobindpur/gobindpur.geojson"

    gdf = gpd.read_file(path)

    print(gdf)
    roi = ee.FeatureCollection(
        get_gee_asset_path(state, district, block)
        + "filtered_mws_"
        + valid_gee_text(district.lower())
        + "_"
        + valid_gee_text(block.lower())
        + "_uid"
    )
    description = f"wells_{district}_{block}"
    export_gdf_to_gee(gdf, roi, description, state, district, block)


if __name__ == "__main__":
    print("Hi, I'm in wells!!!")
    print(sys.argv)
    state = sys.argv[1]
    district = sys.argv[2]
    block = sys.argv[3]
    inference_wells(state, district, block)
    # export_wells_to_gee(state, district, block)
    print("Processing Done")
