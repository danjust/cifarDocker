FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

# Dockerfile is based on tensorflow/tensorflow:1.4.1-gpu

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py


# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# These lines will be edited automatically by parameterized_docker_build.sh. #
# COPY _PIP_FILE_ /
# RUN pip --no-cache-dir install /_PIP_FILE_
# RUN rm -f /_PIP_FILE_

# Install TensorFlow GPU version.
RUN pip --no-cache-dir install \
    http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-1.4.1-cp36-cp36m-manylinux1_x86_64.whl
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #


# RUN ln -s /usr/bin/python3 /usr/bin/python#


# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888


RUN apt-get update && apt-get install -y \
  git


# Copy the current directory contents into the container at /model
# and make this the work directory
RUN mkdir /model
WORKDIR /model
ADD . /model


# pip install requirements
RUN pip --no-cache-dir install requirements.txt && \
    python -m ipykernel.kernelspec


# clone the model from git
RUN git clone https://github.com/tensorflow/models

RUN cd /model/models/research/slim


# train model
CMD python /model/models/research/slim/train_image_classifier.py \
    --train_dir=/traindir \
    --dataset_dir=/data \
    --dataset_name=cifar10 \
    --dataset_split_name=train \
    --clone_on_cpu=True \
    --model_name=inception_v3
