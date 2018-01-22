FROM nvcr.io/nvidia/tensorflow:17.12


RUN apt-get update && apt-get install -y \
  git


# Copy the current directory contents into the container at /model
# and make this the work directory
RUN mkdir /model
WORKDIR /model
ADD . /model


# pip install requirements
RUN pip install -r requirements.txt


# clone the model from git (forked from tensorflow/models)
RUN git clone https://github.com/danjust/models


WORKDIR /model/models/research/slim


# port for TensorBoard
EXPOSE 6006


# train model
CMD python /model/models/research/slim/benchmark_train_image_classifier.py \
    --train_dir=/traindir \
    --dataset_dir=/data \
    --dataset_name=cifar10 \
    --dataset_split_name=train \
    --clone_on_cpu=False \
    --model_name=inception_v3 \
    --benchmark_steps=500 \
    --num_clones=8 \
