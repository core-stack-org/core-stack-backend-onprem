import os
import re
from core_stack_backend_onprem.celery import app
import subprocess


def valid_gee_text(description):
    description = re.sub(r"[^a-zA-Z0-9 .,:;_-]", "", description)
    return description.replace(" ", "_")


@app.task(bind=True)
def scrubland_field_delineation(self, state, district, block):
    print("In scrubland_field_delineation")
    pwd = os.getcwd()

    cmd = [
        "sudo",
        "docker",
        "run",
        "--shm-size=60gb",
        "--gpus",
        "all",
        "--init",
        "-v",
        f"{pwd}/compute/layers:/app",
        "-e",
        "http_proxy=http://corestk.visitor:1BDdklnP@xen03.iitd.ernet.in:3128",
        "-e",
        "https_proxy=http://corestk.visitor:1BDklnP@xen03.iitd.ernet.in:3128",
        "-e",
        "no_proxy=localhost,127.0.0.1,::1",
        "farms",
        "bash",
        "-c",
        (
            f"cd /app && "
            f"PYTHONUNBUFFERED=1 PYTHONPATH=/app conda run -n myenv python scrubland_field_delineation/script.py "
            f"{valid_gee_text(state)} {valid_gee_text(district)} {valid_gee_text(block)}"
        ),
    ]
    try:
        # Run the command and capture output
        # response = os.system(docker_cmd)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        # print(response)

        for line in process.stdout:
            print(f"[Docker log] {line.strip()}")
    except Exception as e:
        print(f"Error running Docker container: {e}")


@app.task(bind=True)
def compute_ponds_detection(self, state, district, block):
    print("In compute_ponds_detection")

    pwd = os.getcwd()

    cmd = [
        "sudo",
        "docker",
        "run",
        "--shm-size=60gb",
        "--gpus",
        "all",
        "--init",
        "-v",
        f"{pwd}/compute/layers:/app",
        "pondswell:1.3",
        "bash",
        "-c",
        (
            f"cd /app && "
            f"PYTHONUNBUFFERED=1 PYTHONPATH=/app conda run -n myenv python ponds_and_wells/ponds.py "
            f"{valid_gee_text(state)} {valid_gee_text(district)} {valid_gee_text(block)}"
        ),
    ]
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        for line in process.stdout:
            print(f"[Docker log] {line.strip()}")

    except Exception as e:
        print(f"Error running compute_ponds_detection Docker container: {e}")


@app.task(bind=True)
def compute_wells_detection(self, state, district, block):
    print("In compute_wells_detection")
    pwd = os.getcwd()

    cmd = [
        "sudo",
        "docker",
        "run",
        "--shm-size=60gb",
        "--gpus",
        "all",
        "--init",
        "-v",
        f"{pwd}/compute/layers:/app",
        "pondswell:1.3",
        "bash",
        "-c",
        (
            f"cd /app && "
            f"PYTHONUNBUFFERED=1 PYTHONPATH=/app conda run -n myenv python ponds_and_wells/wells.py "
            f"{valid_gee_text(state)} {valid_gee_text(district)} {valid_gee_text(block)}"
        ),
    ]
    try:
        # Run the command and capture output
        # response = os.system(docker_cmd)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        # print(response)

        for line in process.stdout:
            print(f"[Docker log] {line.strip()}")
    except Exception as e:
        print(f"Error running Docker container: {e}")
