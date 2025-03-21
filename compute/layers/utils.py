from constants import (
    GEE_ASSET_PATH,
    GCS_BUCKET_NAME,
)
import ee, geetools

from google.cloud import storage
import re


def ee_initialize(project=None):
    try:
        if project == "helper":
            service_account = (
                "corestack-helper@ee-corestack-helper.iam.gserviceaccount.com"
            )
            credentials = ee.ServiceAccountCredentials(
                service_account,
                "ee-corestack-helper-a7148ade82db.json",
            )
        else:
            service_account = "core-stack-dev@ee-corestackdev.iam.gserviceaccount.com"
            credentials = ee.ServiceAccountCredentials(
                service_account, "ee-corestackdev-b1cf638f9352.json"
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


def get_gee_asset_path(state, district=None, block=None, asset_path=GEE_ASSET_PATH):
    gee_path = asset_path + valid_gee_text(state.lower()) + "/"
    if district:
        gee_path += valid_gee_text(district.lower()) + "/"
    if block:
        gee_path += valid_gee_text(block.lower()) + "/"
    return gee_path
