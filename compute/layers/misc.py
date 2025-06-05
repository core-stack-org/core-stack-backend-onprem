from pathlib import Path
import pandas as pd
import ast
import math
import ee
from itertools import product
import os
import csv
from PIL import Image
from commons import tms_to_geotiff


def get_points(roi, directory, zoom, scale):
    points_file = Path(directory + "/status.csv")
    if points_file.is_file():
        df = pd.read_csv(directory + "/status.csv", index_col=False)
        df["points"] = df["points"].apply(ast.literal_eval)
        return df
    # zoom = 17
    # scale = 16
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
        max(abs(min_[0] - max_[0]), abs(min_[1] - max_[1])) / 256 / 16
    )
    points = get_n_boxes(starting_point[0], starting_point[1], iterations, zoom, scale)
    intersect_list = []
    print(f"Generated {len(points)} points")
    index = 0
    for point in points:
        top_left = point
        bottom_right = lat_lon_from_pixel(top_left[0], top_left[1], zoom, scale)
        rectangle = ee.Geometry.Rectangle(
            [(top_left[1], top_left[0]), (bottom_right[1], bottom_right[0])]
        )
        print(top_left, bottom_right)
        intersects = roi.geometry().intersects(rectangle, ee.ErrorMargin(1)).getInfo()
        if intersects:
            intersect_list.append((index, (top_left, bottom_right)))
            index += 1
        print(intersects)
    df = pd.DataFrame(intersect_list, columns=["index", "points"])
    df["overall_status"] = False
    df["download_status_17"] = False
    df["download_status_18"] = False
    df["model_status"] = False
    df["segmentation_status"] = False
    df["postprocessing_status"] = False
    df["plantation_status"] = False
    df.to_csv(directory + "/status.csv", index=False)
    return df


# def get_points(roi, directory, zoom, scale):
#     points_file = Path(directory + "/status.csv")
#     if points_file.is_file():
#         df = pd.read_csv(directory + "/status.csv", index_col=False)
#         df["points"] = df["points"].apply(ast.literal_eval)
#         return df
#     bounds = roi.bounds().coordinates().get(0).getInfo()
#     lons = sorted([coord[0] for coord in bounds])
#     lats = sorted([coord[1] for coord in bounds])
#
#     # Convert lat/lon to pixel coords
#     min_pixel_x = lon_to_pixel_x(lons[0], zoom)
#     max_pixel_x = lon_to_pixel_x(lons[-1], zoom)
#     min_pixel_y = lat_to_pixel_y(lats[0], zoom)
#     max_pixel_y = lat_to_pixel_y(lats[-1], zoom)
#
#     # Tile index ranges
#     x_start = min_pixel_x // (256 * scale)
#     x_end = max_pixel_x // (256 * scale)
#     y_start = max_pixel_y // (256 * scale)
#     y_end = min_pixel_y // (256 * scale)
#
#     points = []
#     index = 0
#     for y_tile in range(y_start, y_end + 1):
#         for x_tile in range(x_start, x_end + 1):
#             pixel_x = x_tile * 256 * scale
#             pixel_y = y_tile * 256 * scale
#             lon = pixel_x_to_lon(pixel_x, zoom)
#             lat = pixel_y_to_lat(pixel_y, zoom)
#             bottom_right = lat_lon_from_pixel(lat, lon, zoom, scale)
#             tile_geom = ee.Geometry.Rectangle(
#                 [lon, bottom_right[0], bottom_right[1], lat]
#             )
#             if roi.geometry().intersects(tile_geom, ee.ErrorMargin(1)).getInfo():
#                 # points.append((lat, lon))
#                 points.append((index, (lat, lon)))
#                 index += 1
#
#     print(f"Generated {len(points)} grid-aligned points")
#     # return points
#     df = pd.DataFrame(points, columns=["index", "points"])
#     df["overall_status"] = False
#     df["download_status_17"] = False
#     df["download_status_18"] = False
#     df["model_status"] = False
#     df["segmentation_status"] = False
#     df["postprocessing_status"] = False
#     df["plantation_status"] = False
#     df.to_csv(directory + "/status.csv", index=False)
#     return df


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


# Function to convert latitude to pixel Y at a given zoom level
def lat_to_pixel_y(lat, zoom):
    print("Inside lat_to_pixel_y 2", lat)
    sin_lat = math.sin(math.radians(lat))
    pixel_y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * (
        2 ** (zoom + 8)
    )
    print("Inside lat_to_pixel_y 2")
    return int(round(pixel_y))


# Function to convert longitude to pixel X at a given zoom level
def lon_to_pixel_x(lon, zoom):
    print("Inside lon_to_pixel_x 1", lon)
    pixel_x = ((lon + 180) / 360) * (2 ** (zoom + 8))
    print("Inside lon_to_pixel_x 2")
    return int(round(pixel_x))


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
    print("Inside lat_lon_from_pixel 1")
    pixel_x = lon_to_pixel_x(lon, zoom)
    pixel_y = lat_to_pixel_y(lat, zoom)
    print("Inside lat_lon_from_pixel 2")
    new_lon = pixel_x_to_lon(pixel_x + 256 * scale, zoom)
    new_lat = pixel_y_to_lat(pixel_y + 256 * scale, zoom)
    print("Inside lat_lon_from_pixel 3")
    return new_lat, new_lon


def divide_tiff_into_chunks(chunk_size, output_dir, zoom, top_left_lat, top_left_lon):
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

    # top_left_x_tile = int(lon_to_pixel_x(top_left_lon, zoom) / 256)
    # top_left_y_tile = int(lat_to_pixel_y(top_left_lat, zoom) / 256)
    #
    # os.makedirs(os.path.join(output_dir, "chunks"), exist_ok=True)
    #
    # mappings = []
    #
    # for row, i in enumerate(range(0, width, chunk_size)):
    #     for col, j in enumerate(range(0, height, chunk_size)):
    #         box = (i, j, i + chunk_size, j + chunk_size)
    #         chunk = image.crop(box)
    #
    #         chunk_filename = f"chunk_{row}_{col}.tif"
    #         chunk.save(os.path.join(output_dir, "chunks", chunk_filename))
    #
    #         x_tile = top_left_x_tile + (i // chunk_size)
    #         y_tile = top_left_y_tile + (j // chunk_size)
    #         tile_filename = f"tile_{zoom}_{x_tile}_{y_tile}.tif"
    #
    #         mappings.append([chunk_filename, tile_filename])
    #
    # print("Image has been split into chunks correctly.")

    # # Save mapping to a CSV file
    # mapping_path = os.path.join(output_dir, "tile_mapping.csv")
    # with open(mapping_path, mode="w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Chunk Name", "Tile Name"])
    #     writer.writerows(mappings)
    #
    # print(f"Mapping file saved: {mapping_path}")


# def download(bbox, output_dir, row, index, directory, blocks_df, zoom):
#     if row["download_status_" + str(zoom)]:
#         return
#
#     chunk_size = 256
#
#     (lat1, lon1), (lat2, lon2) = bbox
#
#     os.makedirs(output_dir + "/chunks", exist_ok=True)
#     tms_to_geotiff(
#         output=output_dir + "/field.tif",
#         bbox=[lon1, lat1, lon2, lat2],
#         zoom=zoom,
#         source="Satellite",
#         overwrite=True,
#         threads=1,
#     )
#     divide_tiff_into_chunks(chunk_size, output_dir, zoom, lat2, lon1)
#     mark_done(index, directory, blocks_df, "download_status_" + str(zoom))


def divide_tiff_into_chunks_with_mapping(
    chunk_size,
    output_dir,
    zoom,
    top_left_lat,
    top_left_lon,
    mapping_file="tile_mapping.csv",
):
    print("Inside divide_tiff_into_chunks_with_mapping function")
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


def download(coords, output_dir, row, index, directory, blocks_df, zoom, scale):
    """Improved download function that handles both points and bboxes"""
    print("Inside download function")
    if row["download_status_" + str(zoom)]:
        print(f"Already downloaded for chunk {index}")
        return
    # Check if input is a single point (lat, lon) or bbox [min_lat, min_lon, max_lat, max_lon]
    if len(coords) == 2:  # It's a single point
        print("Inside download function 1", coords)
        lat, lon = coords[0]
        print("lat=", lat)
        print("long=", lon)
        # Create a bbox around the point using the scale
        new_lat, new_lon = lat_lon_from_pixel(lat, lon, zoom, scale)
        bbox = [
            min(lat, new_lat),
            min(lon, new_lon),
            max(lat, new_lat),
            max(lon, new_lon),
        ]
    else:  # It's already a bbox
        print("Inside download function 2")
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
    mark_done(index, directory, blocks_df, "download_status_" + str(zoom))


def mark_done(index, output_dir, df, label):
    df = pd.read_csv(output_dir + "/status.csv", index_col=False)
    df.loc[df["index"] == index, label] = True
    df.to_csv(output_dir + "/status.csv", index=False)
