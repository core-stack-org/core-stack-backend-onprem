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
    starting_point = lats[-1], lons[0]
    min_, max_ = (
        [lon_to_pixel_x(lons[0], zoom), lat_to_pixel_y(lats[0], zoom)],
        [lon_to_pixel_x(lons[-1], zoom), lat_to_pixel_y(lats[-1], zoom)],
    )
    iterations = math.ceil(
        max(abs(min_[0] - max_[0]), abs(min_[1] - max_[1])) / 256 / 16
    )
    points = get_n_boxes(starting_point[0], starting_point[1], iterations, zoom, scale)
    intersect_list = []
    print(len(points))
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


def divide_tiff_into_chunks(chunk_size, output_dir, zoom, top_left_lat, top_left_lon):
    # Load the large TIFF image
    input_image_path = output_dir + "/field.tif"
    image = Image.open(input_image_path)
    # Get image dimensions
    width, height = image.size

    # # Iterate over the image to create 256x256 chunks
    # ind_i = 0
    # ind_j = 0
    # for i in range(0, width, chunk_size):
    #     for j in range(0, height, chunk_size):
    #         # Define the box to crop
    #         box = (i, j, i + chunk_size, j + chunk_size)
    #
    #         # Crop the image using the defined box
    #         chunk = image.crop(box)
    #         # Save each chunk as a separate TIFF file
    #         chunk.save(
    #             os.path.join(output_dir + "/chunks/", f"chunk_{ind_i}_{ind_j}.tif")
    #         )
    #         ind_j += 1
    #     ind_i += 1
    #     ind_j = 0
    #
    # print("Image has been split into 256x256 chunks and saved successfully.")

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
    mapping_path = os.path.join(output_dir, "tile_mapping.csv")
    with open(mapping_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Chunk Name", "Tile Name"])
        writer.writerows(mappings)

    print(f"Mapping file saved: {mapping_path}")


def download(bbox, output_dir, row, index, directory, blocks_df, zoom):
    if row["download_status_" + str(zoom)]:
        return

    chunk_size = 256

    (lat1, lon1), (lat2, lon2) = bbox

    os.makedirs(output_dir + "/chunks", exist_ok=True)
    tms_to_geotiff(
        output=output_dir + "/field.tif",
        bbox=[lon1, lat1, lon2, lat2],
        zoom=zoom,
        source="Satellite",
        overwrite=True,
        threads=1,
    )
    divide_tiff_into_chunks(chunk_size, output_dir, zoom, lat2, lon1)
    mark_done(index, directory, blocks_df, "download_status_" + str(zoom))


def mark_done(index, output_dir, df, label):
    df = pd.read_csv(output_dir + "/status.csv", index_col=False)
    df.loc[df["index"] == index, label] = True
    df.to_csv(output_dir + "/status.csv", index=False)
