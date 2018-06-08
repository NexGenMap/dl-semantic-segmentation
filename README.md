# nextgenmap

Planet Data: https://www.lapig.iesa.ufg.br/lapig/nextgenmap-data/

## Proposed Estructure to Integrate GEE and Tensorflow

### Minimal Requirements
* Google Cloud Account
* Create an Computer Engine Instance:
    > 1 core
    > 2gb RAM
    > 30gb of Storage (hdd or ssd)
    > CentOS 7 system installed
    > 1 GPU Nvidia K80

### Steps of cloud configuration:
* Install Nvidia/CUDA driver
* Install Docker
* Install Nvidia-Docker
* Deploy Portainer.io: Web-Based Docker Manager
* Deploy TensorFlow: CPU
* Install GEE library
    > Deploy TensorFlow: GPU
* Install GEE library



## Install  NVIDIA/CUDA Driver
```sh
$ sudo yum install wget
$ sudo yum install cc gcc
$ sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
$ wget https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux
$ sudo sh cuda_9.1.85_387.26_linux  # dont install the nvidia-xconfig
```

### CentOs install Docker
```sh
$ sudo yum install docker
```

### CentOs install Nvidia-Docker
```sh
$ curl -s -L https://nvidia.github.io/nvidia-docker/centos7/x86_64/nvidia-docker.repo | \ 
  sudo tee /etc/yum.repos.d/nvidia-docker.repo

$ sudo yum install -y yum-utils \
  device-mapper-persistent-data \
  lvm2

$ sudo yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo

$ sudo yum install docker-ce        
$ sudo yum install -y nvidia-docker2
$ sudo pkill -SIGHUP dockerd
```

### Start Docker Again
```sh
$ sudo systemctl start docker
```

### Centos Deploing "portainer.io"
``` sh
$ docker volume create portainer_data
$ docker run --runtime=nvidia -d -p 80:9000 -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer
# Now, the portainer will run, you can access throught putting you server address on browser url.
# On main page you will click on 

## Deploy TensorFlow:GPU

# Dependencias
$ pip install opencv-python
$ apt update && apt install -y libsm6 libxext6
$ apt install libxext6
$ pip install tqdm
```
