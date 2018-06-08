#!/bin/bash

echo "Setup OS packages"

sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get -y update
sudo apt install -y git wget python3 python3-gdal gdal-bin

echo "Setup Python dependencies packages"

sudo pip3 install tensorflow-gpu==1.5.0
sudo pip3 install pillow
sudo pip install download https://bitbucket.org/chchrsc/rios/downloads/rios-1.4.5.tar.gz

echo "Installation Done"
