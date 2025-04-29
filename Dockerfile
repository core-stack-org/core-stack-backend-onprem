# Use Ubuntu 22.04.1 as the base image
FROM ubuntu:22.04

# Set the environment variable to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=interactive

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    curl \
    bzip2 \
    ca-certificates \
    git \
    nvidia-cuda-toolkit \
    apt-transport-https \
    ca-certificates \
    gnupg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#install google cloud
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y
    
# Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O anaconda.sh && \
    bash anaconda.sh -b -p /opt/anaconda && \
    rm anaconda.sh

# Set environment variables for Anaconda
ENV PATH=/opt/anaconda/bin:$PATH

# Create a new conda environment with Python 3.10.13
RUN conda create -y -n myenv python=3.11.11

# Activate the environment by default
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

RUN conda info --env
RUN pip install -q pillow
RUN pip install -q scikit-image
RUN pip install -q higra
RUN pip install -q earthengine-api
RUN pip install -q opencv-python==4.11.0.86
# RUN pip install -q segment-geospatial
RUN pip install -q shapely==2.0.7
RUN pip install -q rasterio==1.4.3
RUN pip install -q geopandas==1.0.1
RUN pip install -q pyproj==3.7.1
RUN pip install -q xyzservices==2025.1.0
RUN conda install -y -c conda-forge cudatoolkit=11.7 cudnn=8.1.0
RUN conda install -y -c conda-forge nccl
RUN conda install -y -c conda-forge gdal==3.6.2
RUN pip install -q mxnet-cu117
RUN pip install -q ultralytics
RUN pip uninstall -y numpy
RUN pip install -q numpy==1.23.1

# Set the working directory (will be the mounted GitHub repo)
WORKDIR /app

# Mount the current repo from the host system to the container's /app directory
VOLUME ["/app"]

# Set the default command to bash with the environment activated
CMD ["conda", "run", "-n", "myenv", "bash"]
RUN conda init bash
RUN echo "conda activate myenv" >> ~/.bashrc
RUN source ~/.bashrc
