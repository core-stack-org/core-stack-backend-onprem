import os
import re
from core_stack_backend_onprem.celery import app


def valid_gee_text(description):
    description = re.sub(r"[^a-zA-Z0-9 .,:;_-]", "", description)
    return description.replace(" ", "_")


@app.task(bind=True)
def scrubland_field_delineation(self, state, district, block):
    # docker_cmd = f"sudo docker run --shm-size=60gb --gpus all --init -it -v $(pwd)/compute/scrubland_field_delineation_1:/app farm bash -c 'conda run -n myenv python main.py {valid_gee_text(state)} {valid_gee_text(district)} {valid_gee_text(block)}'"
    docker_cmd = (
        f"sudo docker run --shm-size=60gb --gpus all --init -it -v $(pwd)/compute/layers:/app farm bash -c "
        f"'cd /app && PYTHONPATH=/app conda run -n myenv python scrubland_field_delineation/main.py {valid_gee_text(state)} {valid_gee_text(district)} {valid_gee_text(block)}'"
    )
    try:
        # Run the command and capture output
        response = os.system(docker_cmd)
        print(response)
    except Exception as e:
        print(f"Error running Docker container: {e}")


def compute_wells_ponds_detection():
    pass
