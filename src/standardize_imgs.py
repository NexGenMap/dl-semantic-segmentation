#!/usr/bin/python3

import math
import gdal
import numpy as np
import multiprocessing
import os
import csv
import time
import osr
from pathlib import Path
import image_utils

import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='STEP 01/06 - ' + \
		'Standardize multiple images using the formula: (value - median) / std_dev.' + \
		' The median and std_dev will be calculate by band (e.g. blue, red) considering all images.')
	parser.add_argument("-i", "--images", nargs='+', help='<Required> List of input images.', \
		required=True)
	parser.add_argument("-b", "--bands", nargs='+', type=int, help='<Required> The image bands' + \
		' that will be standardized.', required=True)
	parser.add_argument("-n", "--in-nodata", help='<Required> Nodata value of input images.' \
		, type=float, required=True)
	parser.add_argument("-d", "--out-nodata", help='Nodata value of standardized images.' + \
		' It will be ignored when convert-int16 is enabled. [DEFAULT=-50]', type=float, default=-50)
	parser.add_argument("-t", "--convert-int16", help='Convert the standardized images to ' + \
		' int16, multiply its pixel values by scale factor 10000. It will reduce the size of' + \
		' the output files and use -32767 as nodata value. [DEFAULT=No]', action='store_true')
	parser.add_argument("-o", "--output-dir", help='<Required> Output directory that' + \
		' will have the standardized images.', required=True)
	parser.add_argument("-c", "--chunk-size", help='The amount of data that will be processed,' + \
		' per time, by standardization process. In case of memory error you should decrease this ' + \
		' argument. [DEFAULT=1000]', type=int, default=1000)
	return parser.parse_args()

def merge_unique_values(result, uniq_vals, count_vals):
	for i in range(0, len(uniq_vals)):
		uniq_val = uniq_vals[i]
		if uniq_val not in result:
			result[uniq_val] = 0
		result[uniq_val] = result[uniq_val] + count_vals[i]

def unique_values(chunk):

	result = {}
	
	image_ds = gdal.Open(chunk['image_file'], gdal.GA_ReadOnly)

	band_ds = image_ds.GetRasterBand(chunk['band'])
	band_data = band_ds.ReadAsArray(chunk['xoff'], chunk['yoff'], chunk['win_xsize'], chunk['win_ysize']);

	validPixels = (band_data != chunk['nodata'])

	uniq_vals, count_vals = np.unique(band_data[validPixels], return_counts=True)
	merge_unique_values(result, uniq_vals, count_vals)
	
	print('Processing ' + chunk['id'])

	return result
		
def prepare_chunks(image_file, band, chunk_x_size, in_nodata):
	image_ds = gdal.Open(image_file, gdal.GA_ReadOnly)
	
	x_size = image_ds.RasterXSize
	y_Size = image_ds.RasterYSize

	indexes = []

	for xoff in range(0,x_size,chunk_x_size):
		if (xoff+chunk_x_size) > x_size:
			chunk_x_size = x_size - xoff

		suffix = 'b'+str(band) +'_' +'x'+str(xoff)
		chunk_id = image_utils.new_filepath(image_file, suffix = suffix, ext='', directory='')
		indexes.append({
			'id':chunk_id, 
			'image_file':image_file, 
			'band':band, 
			'xoff': xoff, 
			'yoff': 0, 
			'win_xsize': chunk_x_size, 
			'win_ysize': y_Size,
			'nodata': in_nodata
		})

	return indexes

def export_csv(csv_path, orig_image_name, freq_data):
	with open(csv_path, 'a') as csv_file: 
		for uniq_val in freq_data.keys():
			csv_file.write(str(uniq_val) + ';' + str(freq_data[uniq_val]) + ';' + str(orig_image_name) + '\n')

def calc_stats(freq_data, in_nodata):
	
	values = np.array(list(freq_data.keys()))
	frequencies = np.array(list(freq_data.values()))
	
	total = np.sum(frequencies)
	
	max = np.max(values)
	min = np.min(values)
	mean = np.sum(values * frequencies) / total
	
	val_distance = (values - mean)
	val_distance_weighted = val_distance * val_distance * frequencies
	variance = np.sum(val_distance_weighted) / total
	std = np.sqrt(variance)

	median_pos = math.ceil((total + 1) / 2)
	indices = np.arange(len(values))
	cumsum_freq = np.cumsum(frequencies)
		
	median_pos = np.min(indices[cumsum_freq >= median_pos])
	median = (values[median_pos-1] + values[median_pos])/2

	return {
		'max': max,
		'min': min,
		'mean': mean,
		'variance': variance,
		'std': std,
		'median': median,
		'nodata': in_nodata
	}

def calc_freq_histogram(images, band, in_nodata, output_dir, chunk_x_size):

	input_images = []
	freq_histogram = None

	pool = multiprocessing.Pool()

	for image_path in images:
		
		chunks = prepare_chunks(image_path, band, chunk_x_size, in_nodata)
		chunks_result = pool.map(unique_values, chunks)
		
		freq_histogram_aux = chunks_result[0]

		for i in range(1, len(chunks_result)):
			chunk_uniq = list(chunks_result[i].keys())
			chunk_count = list(chunks_result[i].values())
			merge_unique_values(freq_histogram_aux, chunk_uniq, chunk_count)
		
		input_images.append(image_path)

		csvSuffix = 'b'+str(band)+'_byimgs'
		csvFreqFile = image_utils.new_filepath(image_path, suffix = csvSuffix, \
			ext='csv', directory=output_dir)

		export_csv(csvFreqFile, image_path, freq_histogram_aux)

		if freq_histogram is None:
			freq_histogram = freq_histogram_aux
		else:
			band_uniq_vals = list(freq_histogram_aux.keys())
			band_count_vals = list(freq_histogram_aux.values())
			merge_unique_values(freq_histogram, band_uniq_vals, band_count_vals)
	
	csvFreqFile = image_utils.new_filepath('band'+str(band), suffix = 'all', \
		ext='csv', directory=output_dir)

	export_csv(csvFreqFile, input_images, freq_histogram)

	pool.terminate()

	return freq_histogram;

def standardize(images, band, stats, output_dir, convert_int16, bands, chunk_x_size):
	
	for image_path in images:

		output_image_path = image_utils.new_filepath(image_path, suffix = 'stand', \
			directory=output_dir)

		print("Standardizing band " + str(band) + ' ' + image_path + " => " + output_image_path)

		if not Path(output_image_path).is_file():
			dataType = gdal.GDT_Float32
			nbands = len(bands)
			
			if convert_int16:
				dataType = gdal.GDT_Int16

			output_ds = image_utils.create_output_file(image_path, output_image_path, \
				nbands, dataType)

		else:
			output_ds = gdal.Open(output_image_path, gdal.GA_Update)
		
		input_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
		
		x_size = input_ds.RasterXSize
		y_Size = input_ds.RasterYSize

		for xoff in range(0,x_size,chunk_x_size):
			if (xoff+chunk_x_size) > x_size:
				chunk_x_size = x_size - xoff

			output_band_ds = output_ds.GetRasterBand(band)

			intput_band_ds = input_ds.GetRasterBand(band)
			band_data = intput_band_ds.ReadAsArray(xoff, 0, chunk_x_size, y_Size);
			band_data = band_data.astype('Float32')

			validPixels = (band_data != stats['nodata'])
			band_data[validPixels] = (band_data[validPixels] - stats['median']) / stats['std']
			band_data[np.logical_not(validPixels)] = output_nodata

			if convert_int16:
				positive_outliers = (band_data >= 3.2760)
				negative_outliers = (band_data <= -3.2760)

				band_data[positive_outliers] = 3.2760
				band_data[negative_outliers] = -3.2760

				band_data[np.logical_not(validPixels)] = -3.2767

				band_data = band_data * 10000
				band_data = band_data.astype('Int16')

			output_band_ds.WriteArray(band_data, xoff, 0)

if __name__ == "__main__":
	args = parse_args()

	images = args.images
	bands = args.bands
	chunk_x_size = args.chunk_size
	output_nodata = args.out_nodata
	in_nodata = args.in_nodata
	convert_int16 = args.convert_int16
	output_dir = args.output_dir

	start_time = time.time()
	image_utils.mkdirp(output_dir)

	for band in bands:
		freq_histogram = calc_freq_histogram(images, band, in_nodata, output_dir, chunk_x_size)
		stats = calc_stats(freq_histogram, in_nodata)
		standardize(images, band, stats, output_dir, convert_int16, bands, chunk_x_size)

	elapsed_time = time.time() - start_time
	print('Time elapsed ', elapsed_time)