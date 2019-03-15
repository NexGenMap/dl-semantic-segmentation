#!/usr/bin/python3

import os
from osgeo import gdal
import argparse
import dl_utils
import pickle
import time
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser(description='STEP 03/06 - ' + \
		' Generate a several chips (i.e. a set of pixels with regular squared size) ' + \
		' considerering the input image. The last band will be used' + \
		' as expected output result, and should have only these pixel values:' + \
		' 0=without information, 1=object of interest, 2=not an object of interest.' + \
		' If a chip has only pixel values equal to 0, into reference band, the chip will discarded.')
	parser.add_argument("-i", "--image", help='<Required> Input image' + \
		' that will be used by chip generation process.', required=True)
	parser.add_argument("-o", "--output-dir", help='<Required> The output directory that' + \
		' will have the generated chips.', required=True)
	parser.add_argument("-n", "--nodata", help='Nodata value of input image. [DEFAULT=-50]', type=int, default=-50)
	parser.add_argument("-s", "--chip-size", help='Size of the chip with output result.' + \
		' A chip always will be a square. [DEFAULT=100]', type=int, default=100)
	parser.add_argument("-p", "--pad-size", help='Padding size that will establish the size of input chip, with spectral data.' + \
		 ' A padding size of 93px and a chip size of 100px will result in a input chip of 286px. [DEFAULT=93]', type=int, default=93)
	parser.add_argument("-f", "--offset", help='As a data augmentation option, ' + \
		' offset argument will be used to produce chips with a percentage of overlap.' + \
		' An offset 0,50 will generate chips with 50 percent of overlap in the axis y. [DEFAULT=0,0]', nargs='+', default=['0,0'])
	parser.add_argument("-r", "--rotate", help='As a data augmentation option, ' + \
		' rotate argument will rotate all the chips at angles 90, 180 and 270 degrees. [DEFAULT=false]', action='store_true')
	parser.add_argument("-u", "--shuffle", help='Shuffle generated chips. ' + \
		' If the generated chips is only for test propose, you should set false here. [DEFAULT=true]', action='store_true')
	parser.add_argument("-l", "--flip", help='As a data augmentation option, ' + \
		' flip argument will flip, in the left/right direction, all the chips. [DEFAULT=false]', action='store_true')
	parser.add_argument("-d", "--discard-nodata", help='Chips with nodata values will be discard by' + \
		' chip generation process. You shouldn\'t considerer put true here. [DEFAULT=false]', action='store_true')
	
	return parser.parse_args()

def parse_offset(offset):
	offset_list = []
	for offset in args.offset:
		offset_aux = offset.split(',')
		offset_list.append([ int(offset_aux[0]), int(offset_aux[1]) ])

	return offset_list

def shuffle_chips(dat_ndarray, exp_ndarray, nsamples):
	np.random.seed(int(time.time()))

	half_size = int(nsamples / 2)
	firt_half = np.random.choice(nsamples, half_size)
	second_half = np.random.choice(nsamples, half_size)

	for idx in range(0,half_size):
		f1 = firt_half[idx]
		f2 = second_half[idx]
		dat_ndarray[f1,:,:,:], dat_ndarray[f2,:,:,:] = dat_ndarray[f2,:,:,:], dat_ndarray[f1,:,:,:]
		exp_ndarray[f1,:,:,:], exp_ndarray[f2,:,:,:] = exp_ndarray[f2,:,:,:], exp_ndarray[f1,:,:,:]
	
def exec(img_path, output_dir, chip_size, pad_size,	flip,	rotate, shuffle = True, offset_list = [[0,0]], nodata_value = -50.0, discard_nodata = False):
	print("Analyzing " + img_path + " image.")
	dat_path, exp_path, mtd_path = dl_utils.chips_data_files(output_dir)

	chips_info = dl_utils.chips_info(img_path, nodata_value, chip_size, pad_size, offset_list, rotate, flip, discard_nodata)

	dl_utils.save_object(mtd_path, chips_info)

	dat_ndarray = np.memmap(dat_path, dtype=chips_info['dat_dtype'], mode='w+', shape=chips_info['dat_shape'])
	exp_ndarray = np.memmap(exp_path, dtype=chips_info['exp_dtype'], mode='w+', shape=chips_info['exp_shape'])

	print("Generating " + str(chips_info['dat_shape'][0]) + " chips into " + output_dir + " directory.")
	dl_utils.generate_chips(img_path, dat_ndarray, exp_ndarray, nodata_value, chip_size, pad_size, offset_list, rotate, flip, discard_nodata)

	if shuffle:
		print("Shuffling generated chips.")
		shuffle_chips(dat_ndarray, exp_ndarray, chips_info['n_chips'])

	dat_ndarray.flush()
	exp_ndarray.flush()

	return dat_ndarray, exp_ndarray

if __name__ == "__main__":
	args = parse_args()
	start_time = time.time()

	img_path = args.image
	output_dir = args.output_dir

	flip = args.flip
	rotate = args.rotate
	shuffle = args.shuffle
	pad_size = args.pad_size
	chip_size = args.chip_size
	nodata_value = args.nodata
	discard_nodata = args.discard_nodata
	offset_list = parse_offset(args.offset)

	exec(img_path, output_dir, chip_size, pad_size,	flip,	rotate, shuffle, offset_list, nodata_value, discard_nodata)

	elapsed_time = time.time() - start_time
	print('Time elapsed ' + str(elapsed_time) + ' seg.');
