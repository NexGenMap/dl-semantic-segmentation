#!/bin/bash

echo "00) ------------------ Downloading and extracting forest_toy.zip ------------------"
rm -R forest_toy forest_toy.zip
wget https://storage.googleapis.com/nextgenmap-dataset/dl-semantic-segmentation/forest_toy.zip
unzip forest_toy.zip

echo "01) ------------------ Running standardize_imgs.py ------------------"
./standardize_imgs.py -n 0 -b 1 2 3 4 -i forest_toy/raw_data/mosaic_201709.tif forest_toy/raw_data/mosaic_201801.tif -o forest_toy/stand_data

echo "02) ------------------ Running stack_imgs.py ------------------"
./stack_imgs.py -i forest_toy/stand_data/mosaic_201709_stand.tif -r forest_toy/raw_data/forest_201709.tif -o forest_toy/stand_data/forest_201709_model_input.vrt

echo "03) ------------------ Running generate_chips.py ------------------"
./generate_chips.py -f 0,0 -r -l -u -i forest_toy/stand_data/forest_201709_model_input.vrt -o forest_toy/chips

echo "04) ------------------ Running train_model.py ------------------"
./train_model.py -e 20 -i forest_toy/chips -o forest_toy/model/

echo "05) ------------------ Running evaluate_model.py ------------------"
./evaluate_model.py -m forest_toy/model

echo "06) ------------------ Running classify_imgs.py ------------------"
./classify_imgs.py -m forest_toy/model -i forest_toy/raw_data/mosaic_201801.tif -o forest_toy/result

echo "Check the the raster file forest_toy/result/mosaic_201801_pred.tif ! :)"