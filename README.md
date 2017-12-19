# cifarDocker
Putting the training of inceptionV3 using cifar10 in a docker

- Build a docker image called cifardocker from dockerfile
- Download the cifar10 dataset
- create docker volume that will be populated with training checkpoints "docker volume create trainresults"
- Run dockerrun with the path to the cifar10 dataset as argument
