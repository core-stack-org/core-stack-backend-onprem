import json
import os
import cv2
import numpy as np
import shutil

# import re
import csv
import zipfile
import pandas as pd
import glob
import math
import geopandas as gpd
import shapely
from shapely.geometry import Polygon

# from PIL import Image
import requests
import time
from ultralytics import YOLO
import skimage
import ee
import sys
from utils import (
    get_gee_asset_path,
    valid_gee_text,
    ee_to_gdf,
    ee_initialize,
    export_multipolygon_to_gee,
)
from constants import PONDS_WELLS_DATA_PATH, PONDS_WELLS_MODEL_PATH, GEE_HELPER_PATH


def inference_ponds(state, district, block):
    # Load GoeJSON of the block on which you want to compute the ponds
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

    # Zoom level
    zoom_level = 17

    # Folder paths where you want to save image tiles
    # block_name = "around_masalia"  # 'masalia_subset'
    data_download_folder = os.path.join(
        PONDS_WELLS_DATA_PATH,
        "input",
        str(zoom_level),
        block,
    )
    # data_download_folder

    # Scale of image tile
    scale = 1  # scale of 1 = 256*256 dimensional image

    # Entropy threshold needed to calculate entropy (only in wet ponds case)
    entropy_threshold = 2.5

    # model_path = os.path.join(os.environ["MODELS"], "ponds_and_wells", "Ponds_best.pt")
    model_path = f"{PONDS_WELLS_MODEL_PATH}/Ponds_best.pt"

    output_folder = os.path.join(
        f"{PONDS_WELLS_DATA_PATH}/output/{str(zoom_level)}",
        block,
    )

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    csv_file = os.path.join(output_folder, block + ".csv")

    print(csv_file)

    # Download Data for Inference

    # Get Bounding boxes automatically from GeoJSON instead of manually drawing on GEE

    # Get the bounding box coordinates
    minx, miny, maxx, maxy = gdf.total_bounds

    # Define bounding box points
    topLeft = [minx, maxy]
    topRight = [maxx, maxy]
    bottomRight = [maxx, miny]
    bottomLeft = [minx, miny]

    # topLeft = [87.17053768466987, 24.139038123063]
    # topRight = [87.17703935932197, 24.139038123063]
    # bottomRight = [87.17703935932197, 24.143874618138323]
    # bottomLeft = [87.17053768466987, 24.143874618138323]

    # Helper function to convert latitude and longitude to tile numbers
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

    # Function to download map tiles
    def download_map_tiles(
        base_url,
        image_dir,
        zoom_level,
        scale,
        topLeft,
        topRight,
        bottomLeft,
        bottomRight,
    ):
        # Ensure output folder exists

        # os.makedirs(image_dir, exist_ok=True)

        # Convert lat/lon to tile numbers and get bounding box
        topleft = deg2num(topLeft[1], topLeft[0], zoom_level)
        topright = deg2num(topRight[1], topRight[0], zoom_level)
        bottomleft = deg2num(bottomLeft[1], bottomLeft[0], zoom_level)
        bottomright = deg2num(bottomRight[1], bottomRight[0], zoom_level)

        xmin = min(topleft[0], topright[0], bottomleft[0], bottomright[0])
        xmax = max(topleft[0], topright[0], bottomleft[0], bottomright[0])
        ymin = min(topleft[1], topright[1], bottomleft[1], bottomright[1])
        ymax = max(topleft[1], topright[1], bottomleft[1], bottomright[1])

        # Get start time
        start_time = time.time()

        # Iterate over each tile within the specified range
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
                        filepath = os.path.join(image_dir, filename)
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

    if not os.path.exists(data_download_folder):
        os.makedirs(data_download_folder)
        print(f"Created the folder: {data_download_folder}")

        # Call the function to download map tiles
        download_map_tiles(
            base_url,
            data_download_folder,
            zoom_level,
            scale,
            topLeft,
            topRight,
            bottomLeft,
            bottomRight,
        )
    else:
        print(f"data already downloaded: {data_download_folder}")

    # SAVE PREDICTIONS IN CSV
    conf_thresholds = {"Dry": 0.75, "Wet": 0.6}

    class_names = ["Dry", "Wet"]

    class_abbreviations = {"Dry": "D", "Wet": "W"}

    # Load the model
    my_new_model = YOLO(model_path)

    # Function to calculate entropy

    # %%
    def get_entropy(img, mask):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )  # Convert to grayscale
        mask[mask > 0] = 1  # Ensure mask is binary
        if mask.shape[:2] != img_gray.shape:
            print("Mask shape does not match image shape")
            return 0

        # Normalize the grayscale image to [0, 1]
        img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())

        # Convert to uint8 after normalization
        img_gray = skimage.util.img_as_ubyte(img_gray)

        ent = skimage.filters.rank.entropy(
            img_gray.copy(), skimage.morphology.disk(5), mask=mask
        )
        ent = ent[ent > 5.2]
        if np.sum(mask) > 0:
            # Average entropy based on the mask area
            ent = np.sum(ent) / np.sum(mask)
        else:
            ent = 0
        return ent

    # Functions to save segmented/detected object in a CSV
    def process_image(image_path, conf_thresholds):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image {image_path}")
            return None, None, None, None, None

        results = my_new_model.predict(img)

        polygons = []
        pred_classes = []
        conf_scores = []
        entropies = []

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

                    if class_name == "Wet":  # Only compute entropy for Wet class
                        predicted_mask = (
                            results[0].masks.data.cpu().numpy()[i].astype(np.uint8)
                        )  # Convert to uint8
                        # Calculate entropy for the masked region
                        entropy_value = get_entropy(img, predicted_mask)
                        entropies.append(entropy_value)
                        print(f"Entropy for {class_name}: {entropy_value:.2f}")
                    else:
                        # Skip entropy calculation for Dry class
                        entropies.append(None)

        return image_path, len(polygons), polygons, pred_classes, conf_scores, entropies

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

    # Predictions
    image_files = []
    for file in os.listdir(data_download_folder):
        if file.endswith(".png"):
            image_files.append(os.path.join(data_download_folder, file))

    max_vertices = 0

    # To store processed data from process_image
    image_data = []

    for current_image in image_files:
        _, _, polygons, pred_classes, conf_scores, entropies = process_image(
            current_image, conf_thresholds
        )
        if polygons:
            max_vertices = max(max_vertices, max(len(polygon) for polygon in polygons))
        image_data.append(
            {
                "image_path": current_image,
                "polygons": polygons,
                "pred_classes": pred_classes,
                "entropies": entropies,
            }
        )

    # Create dynamic column headers for X/Y coordinates
    coordinate_headers = []
    for i in range(1, max_vertices + 1):
        coordinate_headers.append(f"X_{i}")
        coordinate_headers.append(f"Y_{i}")

    # Full header row
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
    ] + coordinate_headers

    # Process and save to CSV
    start_time = time.time()

    with open(csv_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)  # Write dynamic header row

        for data in image_data:
            image_path = data["image_path"]
            polygons = data["polygons"]
            pred_classes = data["pred_classes"]
            entropies = data["entropies"]

            if image_path is None:
                continue

            xtile, ytile = extract_xtile_ytile(image_path)
            top_left, top_right, bottom_left, bottom_right = tile_corners_to_latlon(
                xtile, ytile, zoom_level
            )
            latitude, longitude = calculate_tile_center(
                top_left, top_right, bottom_left, bottom_right
            )

            for pred_class, polygon, entropy_value in zip(
                pred_classes, polygons, entropies
            ):
                if entropy_value is None or entropy_value > entropy_threshold:
                    print(
                        f"Skipped {pred_class} due to high entropy: {entropy_value if entropy_value is not None else 'None'}"
                    )
                    continue

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

                # Flatten polygon coordinates while ensuring it matches max_vertices
                flat_polygon = [coord for point in polygon for coord in point]
                # Fill missing values with None
                flat_polygon += [None] * (2 * max_vertices - len(flat_polygon))

                row.extend(flat_polygon)
                csvwriter.writerow(row)

    end_time = time.time()

    print(f"CSV file '{csv_file}' saved successfully.")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")

    # %% [markdown]
    # # Add Buffer to combine nearby predicted objects

    # %%
    EARTH_CIRCUMFERENCE_DEGREES = 360  # degrees

    # %%
    # Load the CSV file
    df = pd.read_csv(csv_file)
    df.rename(columns={"Predicted Class": "Class"}, inplace=True)

    # %%
    # Extract the base name dynamically (e.g., "TRY" from "TRY.csv")
    csv_basename = os.path.splitext(os.path.basename(csv_file))[0]

    # %%
    # Function to convert pixel coordinates to geo-coordinates
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
        lat_range = lat_top_left - lat_bottom_right  # Latitude decreases southward
        lon = lon_top_left + (x / img_width) * lon_range
        lat = lat_top_left - (y / img_height) * lat_range
        return lon, lat

    # Initialize an empty list for GeoJSON features
    geojson_features = []

    # Iterate through each row
    for _, row in df.iterrows():
        lat_top_left = row["Top Left Latitude"]
        lon_top_left = row["Top Left Longitude"]
        lat_bottom_right = row["Bottom Right Latitude"]
        lon_bottom_right = row["Bottom Right Longitude"]

        tile_width, tile_height = 256, 256

        object_coords = []

        # Iterate over all possible coordinate columns dynamically
        i = 1
        while True:
            x_col = f"X_{i}"
            y_col = f"Y_{i}"

            if x_col not in row or y_col not in row:
                break  # Stop if columns don't exist

            x = row[x_col]
            y = row[y_col]

            if pd.notna(x) and pd.notna(y):
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
            else:
                break  # Stop when NaN values appear

            i += 1  # Move to the next set of coordinates

        # Ensure the polygon has at least 3 points before adding
        if len(object_coords) >= 3:
            object_coords.append(object_coords[0])  # Close the polygon
            polygon_geometry = shapely.geometry.Polygon(object_coords)
            feature = {
                "type": "Feature",
                "geometry": polygon_geometry,
                "properties": {"Class": row["Class"]},
            }
            geojson_features.append(feature)

    # Convert to GeoDataFrame
    gdf_final = gpd.GeoDataFrame(
        [feature["properties"] for feature in geojson_features],
        geometry=[feature["geometry"] for feature in geojson_features],
        crs="EPSG:4326",
    )

    # Merge overlapping geometries
    buffer_distance = 0.0005
    gdf_final["Buffered"] = gdf_final.geometry.buffer(buffer_distance)
    combined_polygons = shapely.ops.unary_union(gdf_final["Buffered"])
    combined_polygons = combined_polygons.buffer(-buffer_distance)

    # Convert back to GeoDataFrame
    gdf_combined = gpd.GeoDataFrame(geometry=[combined_polygons], crs="EPSG:4326")

    # Save the final shapefile in the 'Shapefile_Output' folder
    shapefile_path = os.path.join(
        output_folder, f"{csv_basename}_COMBINED_GEOMETRY.shp"
    )
    gdf_combined.to_file(shapefile_path)
    # Create the ZIP file in the 'Shapefile_Output' folder
    zip_filename = os.path.join(output_folder, f"{csv_basename}_COMBINED_GEOMETRY.zip")
    shapefile_files = glob.glob(
        os.path.join(output_folder, f"{csv_basename}_COMBINED_GEOMETRY.*")
    )

    # Exclude the .csv file from the list of files to be zipped
    shapefile_files = [file for file in shapefile_files if not file.endswith(".csv")]

    # Add the shapefile files to the zip archive
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for file in shapefile_files:
            zipf.write(file, os.path.basename(file))

    print(f"Created ZIP file: {zip_filename}")

    # Delete the original shapefile files after zipping, but keep the .csv file
    for file in shapefile_files:
        os.remove(file)
        print(f"Deleted: {file}")

    print(
        "All shapefile components have been deleted after zipping, except for the .csv file."
    )

    # Export gdf to GEE
    fc_description = f"ponds_{district}_{block}"
    export_ponds_to_gee(gdf_combined, roi, fc_description, state, district, block)


def export_ponds_to_gee(gdf, roi, description, state, district, block):
    # ee_initialize()
    # path = "data/ponds_and_wells/output/17/gobindpur.geojson"
    #
    # gdf = gpd.read_file(path)
    multipolygon = gdf.iloc[0].geometry  # TODO: Fix in main script instead of here
    # Extract all the constituent polygons
    polygons = [Polygon(p.exterior, p.interiors) for p in multipolygon.geoms]

    # Create a new geodataframe with individual polygons
    # Copy all other columns from the original dataframe
    new_gdf = gpd.GeoDataFrame(
        [gdf.iloc[0].to_dict() for _ in range(len(polygons))],
        geometry=polygons,
        crs=gdf.crs,
    )
    #roi = ee.FeatureCollection(
    #    get_gee_asset_path(state, district, block)
    #    + "filtered_mws_"
    #    + valid_gee_text(district.lower())
    #    + "_"
    #    + valid_gee_text(block.lower())
    #    + "_uid"
    #)
    #description = f"ponds_{district}_{block}"
    export_multipolygon_to_gee(new_gdf, roi, description, state, district, block)


if __name__ == "__main__":
    print("Hi, I'm in ponds!!!")
    print(sys.argv)
    state = sys.argv[1]
    district = sys.argv[2]
    block = sys.argv[3]
    # export_ponds_to_gee()
    inference_ponds(state, district, block)
    print("Processing Done for Ponds")
