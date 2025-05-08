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

from PIL import Image
import requests
import time
from ultralytics import YOLO
import skimage
from commons import tms_to_geotiff
import ee
import sys
from utils import (
    get_gee_asset_path,
    valid_gee_text,
    ee_to_gdf,
    ee_initialize,
    export_gdf_to_gee,
)


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
    ).union()
    # gdf = ee_to_gdf(roi)

    # Zoom level
    zoom = 17

    # Folder paths where you want to save image tiles
    # block_name = "around_masalia"  # 'masalia_subset'
    # data_download_folder = os.path.join(
    #     PONDS_WELLS_DATA_PATH,
    #     "input",
    #     str(zoom),
    #     block,
    # )
    # data_download_folder
    directory = f"data/{state}/{district}/{block}/{zoom}"
    os.makedirs(directory, exist_ok=True)
    sys.stdout = Logger(directory + "/output.log")

    # Scale of image tile
    # scale = 1  # scale of 1 = 256*256 dimensional image

    # Entropy threshold needed to calculate entropy (only in wet ponds case)
    entropy_threshold = 2.5

    # model_path = os.path.join(os.environ["MODELS"], "ponds_and_wells", "Ponds_best.pt")
    model_path = "ponds_and_wells/Models/Ponds_best.pt"

    # output_folder = os.path.join(
    #     f"{PONDS_WELLS_DATA_PATH}/output/{str(zoom)}",
    #     block,
    # )
    #
    # if not os.path.exists(output_folder):
    #     os.mkdir(output_folder)

    csv_file = os.path.join(directory, block + ".csv")

    print(csv_file)

    def lat_to_pixel_y(lat, zoom):
        sin_lat = math.sin(math.radians(lat))
        pixel_y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * (
            2 ** (zoom + 8)
        )
        return int(round(pixel_y))

    def lon_to_pixel_x(lon, zoom):
        pixel_x = ((lon + 180) / 360) * (2 ** (zoom + 8))
        return int(round(pixel_x))

    def pixel_x_to_lon(pixel_x, zoom):
        return (pixel_x / (2 ** (zoom + 8))) * 360 - 180

    def pixel_y_to_lat(pixel_y, zoom):
        n = math.pi - 2 * math.pi * pixel_y / (2 ** (zoom + 8))
        return math.degrees(math.atan(math.sinh(n)))

    def lat_lon_from_pixel(lat, lon, zoom, scale):
        tile_width_deg = pixel_x_to_lon(256, zoom) - pixel_x_to_lon(0, zoom)
        new_lon = lon + (tile_width_deg * scale)
        new_pixel_y = lat_to_pixel_y(lat, zoom) + (256 * scale)
        new_lat = pixel_y_to_lat(new_pixel_y, zoom)
        return new_lat, new_lon

    def divide_tiff_into_chunks_with_mapping(
        chunk_size,
        output_dir,
        zoom,
        top_left_lat,
        top_left_lon,
        mapping_file="tile_mapping.csv",
    ):
        """Splits a TIFF into chunks and saves a mapping of chunk names to tile names."""
        input_image_path = os.path.join(output_dir, "field.tif")
        image = Image.open(input_image_path)
        width, height = image.size

        top_left_x_tile = int(lon_to_pixel_x(top_left_lon, zoom) / 256)
        top_left_y_tile = int(lat_to_pixel_y(top_left_lat, zoom) / 256)

        os.makedirs(os.path.join(output_dir, "chunks"), exist_ok=True)

        mappings = []

        for row, i in enumerate(range(0, width, chunk_size)):
            for col, j in enumerate(range(0, height, chunk_size)):
                box = (i, j, i + chunk_size, j + chunk_size)
                chunk = image.crop(box)

                chunk_filename = f"chunk_{row}_{col}.tif"
                chunk.save(os.path.join(output_dir, "chunks", chunk_filename))

                x_tile = top_left_x_tile + (i // chunk_size)
                y_tile = top_left_y_tile + (j // chunk_size)
                tile_filename = f"tile_{zoom}_{x_tile}_{y_tile}.tif"

                mappings.append([chunk_filename, tile_filename])

        print("Image has been split into chunks correctly.")

        # Save mapping to a CSV file
        mapping_path = os.path.join(output_dir, mapping_file)
        with open(mapping_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Chunk Name", "Tile Name"])
            writer.writerows(mappings)

        print(f"Mapping file saved: {mapping_path}")

    def download(coords, output_dir, zoom, scale):
        """Improved download function that handles both points and bboxes"""
        # Check if input is a single point (lat, lon) or bbox [min_lat, min_lon, max_lat, max_lon]
        if len(coords) == 2:  # It's a single point
            lat, lon = coords
            # Create a bbox around the point using the scale
            new_lat, new_lon = lat_lon_from_pixel(lat, lon, zoom, scale)
            bbox = [
                min(lat, new_lat),
                min(lon, new_lon),
                max(lat, new_lat),
                max(lon, new_lon),
            ]
        else:  # It's already a bbox
            bbox = coords

        # Rest of your download function remains the same
        min_lon = min(bbox[1], bbox[3])
        max_lon = max(bbox[1], bbox[3])
        min_lat = min(bbox[0], bbox[2])
        max_lat = max(bbox[0], bbox[2])

        # Calculate the exact tile boundaries (fixed missing parentheses)
        tile_x_min = math.floor(lon_to_pixel_x(min_lon, zoom) / 256)
        tile_x_max = math.ceil(lon_to_pixel_x(max_lon, zoom) / 256)
        tile_y_min = math.floor(
            lat_to_pixel_y(max_lat, zoom) / 256
        )  # Note: y tiles increase downward
        tile_y_max = math.ceil(lat_to_pixel_y(min_lat, zoom) / 256)

        # Calculate the exact geographic bounds of the tile area
        exact_min_lon = pixel_x_to_lon(tile_x_min * 256, zoom)
        exact_max_lon = pixel_x_to_lon(tile_x_max * 256, zoom)
        exact_min_lat = pixel_y_to_lat(tile_y_max * 256, zoom)
        exact_max_lat = pixel_y_to_lat(tile_y_min * 256, zoom)

        corrected_bbox = [exact_min_lon, exact_min_lat, exact_max_lon, exact_max_lat]
        print(f"Downloading tiles for exact bbox: {corrected_bbox}")

        os.makedirs(os.path.join(output_dir, "chunks"), exist_ok=True)
        tms_to_geotiff(
            output=os.path.join(output_dir, "field.tif"),
            bbox=corrected_bbox,
            zoom=zoom,
            source="Satellite",
            overwrite=True,
        )

        # Modify the divide function to use exact coordinates
        divide_tiff_into_chunks_with_mapping(
            256, output_dir, zoom, exact_max_lat, exact_min_lon
        )

    # def get_n_boxes(lat, lon, n, zoom, scale):
    #     diagonal_lat_lon = [(lat, lon)]
    #     for _ in range(n):
    #         pixel_x = lon_to_pixel_x(lon, zoom) + (256 * scale)
    #         new_lon = pixel_x_to_lon(pixel_x, zoom)
    #         pixel_y = lat_to_pixel_y(lat, zoom) + (256 * scale)
    #         new_lat = pixel_y_to_lat(pixel_y, zoom)
    #         diagonal_lat_lon.append((new_lat, new_lon))
    #         lat, lon = new_lat, new_lon
    #     return diagonal_lat_lon
    from itertools import product

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
        return list(product(lats, longs))

    def get_points(roi, zoom, scale):
        bounds = roi.bounds().coordinates().get(0).getInfo()
        lons = sorted([coord[0] for coord in bounds])
        lats = sorted([coord[1] for coord in bounds])
        starting_point = (lats[-1], lons[0])
        min_pixel = [lon_to_pixel_x(lons[0], zoom), lat_to_pixel_y(lats[0], zoom)]
        max_pixel = [lon_to_pixel_x(lons[-1], zoom), lat_to_pixel_y(lats[-1], zoom)]
        iterations = math.ceil(
            max(abs(min_pixel[0] - max_pixel[0]), abs(min_pixel[1] - max_pixel[1]))
            / (256 * scale)
        )
        points = get_n_boxes(
            starting_point[0], starting_point[1], iterations, zoom, scale
        )
        intersect_list = []
        print(f"Generated {len(points)} points")
        for point in points:
            bottom_right = lat_lon_from_pixel(point[0], point[1], zoom, scale)
            rectangle = ee.Geometry.Rectangle(
                [(point[1], point[0]), (bottom_right[1], bottom_right[0])]
            )
            if roi.geometry().intersects(rectangle, ee.ErrorMargin(1)).getInfo():
                intersect_list.append(point)
        return intersect_list

    # top_left = [23.531653742232088, 85.82344897858329]
    # bottom_right = [23.64099145921938, 85.99871524445243]
    # rectangle = ee.Geometry.Rectangle(
    #     [top_left[1], bottom_right[0], bottom_right[1], top_left[0]]
    # )
    # roi = ee.FeatureCollection([ee.Feature(rectangle)])
    #
    # print("Area of the Rectangle: ", rectangle.area().getInfo() / 1e6, "sq km")
    points = get_points(roi, zoom, 16)
    print(f"Running for {len(points)} points...")

    for index, point in enumerate(points):
        print(f"Iterating index={index}, point= {point}")
        output_dir = os.path.join(directory, str(index))
        download(point, output_dir, zoom, 16)

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

    def extract_xtile_ytile_from_tile_name(image_path):
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
        with open(tile_map_path, "r") as f:
            next(f)  # Skip header
            for line in f:
                chunk_name, tile_name = line.strip().split(",")
                tile_mapping[chunk_name] = tile_name

        # Process each chunk image
        image_files = [
            os.path.join(chunk_dir, f)
            for f in os.listdir(chunk_dir)
            if f.endswith(".tif")
        ]

        for image_path in image_files:
            chunk_name = os.path.basename(image_path)
            if chunk_name not in tile_mapping:
                print(f"Warning: {chunk_name} not in mapping. Skipping.")
                continue

            tile_name = tile_mapping[chunk_name]
            _, _, polygons, pred_classes, conf_scores, entropies = process_image(
                image_path, conf_thresholds
            )

            if polygons:
                max_vertices = max(
                    max_vertices, max(len(polygon) for polygon in polygons)
                )

            image_data.append(
                {
                    "tile_name": tile_name,
                    "polygons": polygons,
                    "pred_classes": pred_classes,
                    "entropies": entropies,
                }
            )

    # ===================== SAVE TO CSV =====================
    coordinate_headers = []
    for i in range(1, max_vertices + 1):
        coordinate_headers.extend([f"X_{i}", f"Y_{i}"])

    header = [
        "Tile Name",
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

    start_time = time.time()
    with open(csv_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)

        for data in image_data:
            tile_name = data["tile_name"]
            polygons = data["polygons"]
            pred_classes = data["pred_classes"]
            entropies = data["entropies"]

            xtile, ytile = extract_xtile_ytile_from_tile_name(tile_name)
            top_left, top_right, bottom_left, bottom_right = tile_corners_to_latlon(
                xtile, ytile, zoom
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
                    tile_name,
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

                flat_polygon = [coord for point in polygon for coord in point]
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

    ############ SEPARTE MULTIPOLYGONS

    multipolygon = gdf_combined.iloc[0].geometry
    # Extract all the constituent polygons
    polygons = [Polygon(p.exterior, p.interiors) for p in multipolygon.geoms]

    # Create a new geodataframe with individual polygons
    # Copy all other columns from the original dataframe
    gdf_combined = gpd.GeoDataFrame(
        [gdf_combined.iloc[0].to_dict() for _ in range(len(polygons))],
        geometry=polygons,
        crs=gdf_combined.crs,
    )

    ########### SAVE AS GEOJSON

    # Save as GeoJSON in a separate output folder
    output_folder = directory + "/ponds_output"
    os.makedirs(output_folder, exist_ok=True)

    description = f"ponds_{valid_gee_text(district)}_{valid_gee_text(block)}"

    # Path for the GeoJSON file
    geojson_path = os.path.join(output_folder, f"{description}.geojson")

    # Save GeoDataFrame as GeoJSON
    gdf_combined.to_file(geojson_path, driver="GeoJSON")

    print(f"GeoJSON file saved to: {geojson_path}")

    ########## SAVE AS SHAPEFILE

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the final shapefile in the 'Shapefile_Output' folder
    shapefile_path = os.path.join(output_folder, f"{description}.shp")
    gdf_combined.to_file(shapefile_path)

    # Create the ZIP file in the 'Shapefile_Output' folder
    zip_filename = os.path.join(output_folder, f"{description}.zip")
    shapefile_files = glob.glob(os.path.join(output_folder, f"{description}.*"))

    # Exclude the .csv file from the list of files to be zipped
    shapefile_files = [file for file in shapefile_files if not file.endswith(".csv")]

    # Add the shapefile files to the zip archive
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for file in shapefile_files:
            zipf.write(file, os.path.basename(file))

    print(f"Created ZIP file: {zip_filename}")

    # # Delete the original shapefile files after zipping, but keep the .csv file
    # for file in shapefile_files:
    #     os.remove(file)
    #     print(f"Deleted: {file}")
    #
    # print(
    #     "All shapefile components have been deleted after zipping, except for the .csv file."
    # )

    # Export gdf to GEE
    fc_description = f"ponds_{district}_{block}"
    export_gdf_to_gee(gdf_combined, roi, fc_description, state, district, block)


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
    # roi = ee.FeatureCollection(
    #    get_gee_asset_path(state, district, block)
    #    + "filtered_mws_"
    #    + valid_gee_text(district.lower())
    #    + "_"
    #    + valid_gee_text(block.lower())
    #    + "_uid"
    # )
    # description = f"ponds_{district}_{block}"
    export_gdf_to_gee(new_gdf, roi, description, state, district, block)


if __name__ == "__main__":
    print("Hi, I'm in ponds!!!")
    print(sys.argv)
    state = sys.argv[1]
    district = sys.argv[2]
    block = sys.argv[3]
    # export_ponds_to_gee()
    inference_ponds(state, district, block)
    print("Processing Done for Ponds")
