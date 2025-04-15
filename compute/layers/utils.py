import ee, geetools
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from google.cloud import storage
from google.api_core import retry
import re
import json
import time
from constants import (
    GEE_ASSET_PATH,
    GEE_HELPER_PATH,
    GCS_BUCKET_NAME,
)


def ee_initialize(project=None):
    try:
        if project == "helper":
            service_account = (
                "corestack-helper@ee-corestack-helper.iam.gserviceaccount.com"
            )
            credentials = ee.ServiceAccountCredentials(
                service_account,
                "ee-corestack-helper-a7148ade82db.json",
                # "/home/core-stack/Code/GitClones/core-stack-backend-onprem/compute/layers/ee-corestack-helper-a7148ade82db.json",
            )
        # elif project == "aman":
        #     service_account = "amanverma@ee-amanverma.iam.gserviceaccount.com"
        #     credentials = ee.ServiceAccountCredentials(
        #         service_account,
        #         "/home/core-stack/Code/GitClones/core-stack-backend-onprem/compute/layers/ee-amanverma.json",
        #     )
        else:
            service_account = "core-stack-dev@ee-corestackdev.iam.gserviceaccount.com"
            credentials = ee.ServiceAccountCredentials(
                service_account,
                "ee-corestackdev-b1cf638f9352.json",
                # "/home/core-stack/Code/GitClones/core-stack-backend-onprem/compute/layers/ee-corestackdev-b1cf638f9352.json",
            )
        ee.Initialize(credentials)
        print("ee initialized", project)
    except Exception as e:
        print("Exception in gee connection", e)


def gcs_config():
    from google.oauth2 import service_account

    # Authenticate Earth Engine
    ee_initialize()

    # Authenticate Google Cloud Storage
    credentials = service_account.Credentials.from_service_account_file(
        "ee-corestack-helper-a7148ade82db.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    # Create Storage Client
    storage_client = storage.Client(credentials=credentials)

    # Verify access
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    return bucket

    # print(list(bucket.list_blobs()))


def valid_gee_text(description):
    description = re.sub(r"[^a-zA-Z0-9 .,:;_-]", "", description)
    return description.replace(" ", "_")


def create_gee_folder(folder_path, gee_project_path=GEE_ASSET_PATH):
    try:
        res = ee.data.createAsset(
            {"type": "Folder"},
            gee_project_path + folder_path,
        )
        print(res)
        time.sleep(10)
    except Exception as e:
        print("Error:", e)


def create_gee_directory(state, district, block, gee_project_path=GEE_ASSET_PATH):
    folder_path = valid_gee_text(state.lower()) + "/" + valid_gee_text(district.lower())
    create_gee_folder(folder_path, gee_project_path)

    folder_path = (
        valid_gee_text(state.lower())
        + "/"
        + valid_gee_text(district.lower())
        + "/"
        + valid_gee_text(block.lower())
    )
    create_gee_folder(folder_path, gee_project_path)


def get_gee_asset_path(state, district=None, block=None, asset_path=GEE_ASSET_PATH):
    gee_path = asset_path + valid_gee_text(state.lower()) + "/"
    if district:
        gee_path += valid_gee_text(district.lower()) + "/"
    if block:
        gee_path += valid_gee_text(block.lower()) + "/"
    return gee_path


def gdf_to_ee_fc(gdf):
    features = []
    for i, row in gdf.iterrows():
        properties = row.drop("geometry").to_dict()
        geometry = ee.Geometry(row.geometry.__geo_interface__)
        feature = ee.Feature(geometry, properties)
        features.append(feature)

    return ee.FeatureCollection(features)


def sync_fc_to_gee(fc, description, asset_id):
    try:
        task = ee.batch.Export.table.toAsset(
            **{"collection": fc, "description": description, "assetId": asset_id}
        )
        task.start()
        print("Successfully started task", task.status())
        return task.status()["id"]
    except Exception as e:
        print(f"Error in task: {e}")


def make_asset_public(asset_id):
    try:
        # Get the ACL of the asset
        acl = ee.data.getAssetAcl(asset_id)

        # Add 'all_users' to readers
        acl["all_users_can_read"] = True

        # Update the ACL
        @retry.Retry()
        def update_acl():
            ee.data.setAssetAcl(asset_id, acl)

        update_acl()

        # Verify the change
        updated_acl = ee.data.getAssetAcl(asset_id)
        if updated_acl.get("all_users_can_read"):
            print(f"Successfully made asset {asset_id} public")
            return True
        else:
            print(f"Failed to make asset {asset_id} public")
            return False
    except Exception as e:
        print(f"Error making asset public: {str(e)}")
        return False


def ee_to_gdf(fc):
    # Get the feature collection as a list of dictionaries
    features = fc.getInfo()["features"]

    # Create lists to store properties and geometries
    properties_list = []
    geometry_list = []

    for f in features:
        # Get properties
        properties = f["properties"]
        properties_list.append(properties)

        # Get geometry
        geometry = None
        if "geometry" in f:
            geom_type = f["geometry"]["type"]
            coords = f["geometry"]["coordinates"]

            if geom_type == "Polygon":
                geometry = Polygon(coords[0])  # First ring is exterior
            # Add more geometry types as needed

        geometry_list.append(geometry)

    # Create GeoDataFrame
    df = pd.DataFrame(properties_list)
    gdf = gpd.GeoDataFrame(df, geometry=geometry_list, crs="EPSG:4326")

    return gdf


def check_task_status(task_id_list, sleep_time=60):
    if len(task_id_list) > 0:
        time.sleep(sleep_time)
        tasks = ee.data.listOperations()
        # tasks = check_gee_task_status(task_id_list[0])
        # print("tasks>>>", tasks)
        if tasks:
            for task in tasks:
                task_id = task["name"].split("/")[-1]
                if task_id in task_id_list and task["metadata"]["state"] in (
                    "SUCCEEDED",
                    "COMPLETED",
                    "FAILED",
                    "CANCELLED",
                ):
                    task_id_list.remove(task_id)
        print("task_id_list after", task_id_list)

        if len(task_id_list) > 0:
            print("Tasks not completed yet...")
            check_task_status(task_id_list)
    return task_id_list


def export_multipolygon_to_gee(gdf, roi, description, state, district, block):
    df_size = gdf.shape[0]
    chunk_size = 2000
    if df_size > chunk_size:
        asset_ids = []
        assets = []
        task_ids = []
        ee_initialize("helper")
        create_gee_directory(state, district, block, GEE_HELPER_PATH)
        for i in range(0, df_size, chunk_size):
            chunk = gdf.iloc[i : i + chunk_size]
            fc = gdf_to_ee_fc(chunk)
            chunk_description = f"{description}_{i}_{i + chunk_size}"
            asset_id = (
                get_gee_asset_path(state, district, block, GEE_HELPER_PATH)
                + chunk_description
            )
            asset_ids.append(asset_id)
            assets.append(ee.FeatureCollection(asset_id))

            task_id = sync_fc_to_gee(fc, chunk_description, asset_id)
            task_ids.append(task_id)

        check_task_status(task_ids)
        for asset_id in asset_ids:
            make_asset_public(asset_id)

        final_asset = ee.FeatureCollection(assets).flatten()

        ee_initialize()
        final_asset = final_asset.filterBounds(roi.geometry())
        sync_fc_to_gee(final_asset, description, GEE_ASSET_PATH + description)
    else:
        fc = gdf_to_ee_fc(gdf)
        asset_id = (
            get_gee_asset_path(state, district, block, GEE_ASSET_PATH) + description
        )
        sync_fc_to_gee(fc, description, asset_id)


def geojson_to_ee_featurecollection(geojson_path):
    """
    Convert a GeoJSON FeatureCollection to an Earth Engine FeatureCollection
    """
    # # Read the GeoJSON file
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)

    # Convert GeoJSON features to Earth Engine features
    ee_features = []
    for feature in geojson_data["features"]:
        # Convert the feature to a GeoJSON string
        feature_geojson = json.dumps(feature)

        # Create an Earth Engine Feature using ee.Geometry.coordinates()
        geometry = ee.Geometry(feature["geometry"])
        ee_feature = ee.Feature(geometry)

        # Add properties from the original feature
        if "properties" in feature:
            ee_feature = ee_feature.set(feature["properties"])

        ee_features.append(ee_feature)

    # Create an Earth Engine FeatureCollection
    return ee.FeatureCollection(ee_features)
