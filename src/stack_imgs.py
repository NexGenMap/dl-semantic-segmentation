#!/usr/bin/python3

import os
import ntpath
from osgeo import gdal
import argparse
import dl_utils

import subprocess

def parse_args():
	parser = argparse.ArgumentParser(description='STEP 02/06 - ' + \
		'Stack multiple images into a sigle Virtual Dataset-VRT image. If informed,' + \
		' the reference image will the last band.')
	parser.add_argument("-i", "--images", nargs='+', help='<Required> List of input images.', required=True)
	parser.add_argument("-b", "--bands", nargs='+', type=int, help='The bands that should be stacked. [DEFAULT=All]', default=None)
	parser.add_argument("-r", "--reference", help=' Image with reference data, that should have only these pixel values:' + \
		' 0=without information, 1=object of interest, 2=not an object of interest.')
	parser.add_argument("-o", "--output", help='<Required> The name of VRT output image', required=True)
	return parser.parse_args()

def reference_params(img_path):
	image_ds = gdal.Open(img_path, gdal.GA_ReadOnly)
	
	xmin, pixel_width, _, ymax, _, pixel_height = image_ds.GetGeoTransform()
	xmax = xmin + pixel_width * image_ds.RasterXSize
	ymin = ymax + pixel_height * image_ds.RasterYSize
	
	return [str(xmin), str(ymin), str(xmax), str(ymax)], [str(pixel_width), str(pixel_width)]

def create_vrt_bands(img_path, output_vrt, bands):
	
	image_ds = gdal.Open(img_path, gdal.GA_ReadOnly)

	vrt_bands = []
	if bands is None:
		bands = range(1, (image_ds.RasterCount+1) )

	for band in bands:
		vrt_filepath = dl_utils.new_filepath(img_path, suffix = str(band), ext='vrt', 
			directory=dl_utils.basedir(output_vrt))

		command = ["gdalbuildvrt"]
		command += ["-b", str(band)]
		command += [vrt_filepath]
		command += [img_path]
		
		subprocess.call(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		vrt_bands += [vrt_filepath]

	return vrt_bands

def create_separate_bands(images, output_vrt, bands):
	separate_bands = []

	for img_path in images:
		vrt_bands = create_vrt_bands(img_path, output_vrt, bands)
		separate_bands += vrt_bands

	return separate_bands

def create_vrt_output(input_imgs, output_vrt, ref_img = None, bands = None):
	separate_bands = create_separate_bands(input_imgs, output_vrt, bands)

	command = ["gdalbuildvrt"]
	command += ["-separate"]

	if ref_img is not None:
		separate_bands += [ref_img]

		ref_extent, ref_pixel_size = reference_params(ref_img)

		command += ["-te"]
		command += ref_extent
		command += ['-tr']
		command += ref_pixel_size

	command += [output_vrt]
	command += separate_bands

	print('Creating vrt file ' + output_vrt)

	subprocess.call(command, stdout=subprocess.PIPE)

if __name__ == "__main__":
	args = parse_args()

	bands = args.bands
	images = args.images
	output = args.output
	reference = args.reference

	create_vrt_output(images, output, reference, bands)
