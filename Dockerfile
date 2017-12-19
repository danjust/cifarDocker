FROM tensorflow/tensorflow:1.4.1-gpu


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


# train model
CMD python /model/models/research/slim/train_image_classifier.py \
    --train_dir=/traindir \
    --dataset_dir=/data \
    --dataset_name=cifar10 \
    --dataset_split_name=train \
    --clone_on_cpu=False \
    --model_name=inception_v3
