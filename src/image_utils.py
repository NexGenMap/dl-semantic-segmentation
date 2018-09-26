import os
import gdal
import osr
import numpy as np
import psutil
import gc
import json

import math

def pad_index(index, dim_size, chip_size, pad_size):

	i0 = (index - pad_size)
	i1 = (index + chip_size + pad_size)

	if i1 > (dim_size + pad_size):
		i1 = (dim_size + pad_size)
		i0 = i1 - chip_size - 2*pad_size

	return i0, i1

def split_data(data, pad_size):
	xsize, ysize, nbands = data.shape

	last_band = (nbands-1)
	
	x0 = pad_size
	x1 = xsize-pad_size

	y0 = pad_size
	y1 = ysize-pad_size

	chip_expect = data[x0:x1, y0:y1, last_band:nbands]
	chip_data = data[:, :, 0:last_band]

	return chip_data, chip_expect

def chip_augmentation(data, rotate = True, flip = True):
	result = [ data ]

	if rotate:
		result = result + [ np.rot90(data, k=k, axes=(0,1)) for k in [1,2,3] ]

	if flip:
		result = result + [np.flip(data, axis=k) for k in [0,1] ]

	return result

def chip_generation(img_path, nodata_value, chip_size, pad_size, offset_list=[(0,0)], \
							rotate=False, flip=False, remove_chips_wnodata=True, 
							chips_data_np = None, chips_expect_np = None):
	
	index = 0
	chip_data_list = []
	chip_expect_list = []
	input_img_ds = gdal.Open(img_path)

	for x_offset_percent, y_offset_percent in offset_list:
		x_offset = int(chip_size * (x_offset_percent / 100.0))
		y_offset = int(chip_size * (y_offset_percent / 100.0))

		input_positions = get_predict_positions(input_img_ds.RasterXSize, input_img_ds.RasterYSize, \
																						chip_size, pad_size, x_offset, y_offset)

		for input_position in input_positions:
			chip_data, _ = get_predict_data(input_img_ds, input_position, pad_size)

			chip_data, chip_expect = split_data(chip_data, pad_size)
			xsize, ysize, _ = chip_expect.shape
			
			if (chip_size == xsize and chip_size == ysize) and (not remove_chips_wnodata or float(np.min(chip_expect)) != float(nodata_value)):
				if(float(np.max(chip_expect)) > float(0.0)): # Only include chips with some object
					chip_expect[ chip_expect != 1] = 0 # convert all other class to pixel == 0
					chip_data_aux = chip_augmentation(chip_data, rotate, flip)
					chip_expect_aux = chip_augmentation(chip_expect, rotate, flip)
					nchips = len(chip_data_aux)

					if (chips_data_np is not None):
						chips_data_np[index:index+nchips,:,:,:] = np.stack(chip_data_aux)
						chips_expect_np[index:index+nchips,:,:,:] = np.stack(chip_expect_aux)
					else:
						chip_data_list = chip_data_list + chip_data_aux
						chip_expect_list = chip_expect_list + chip_expect_aux

					index = index + nchips

	return chip_data_list, chip_expect_list

def create_memmap_np(memmap_path, chips_list):
	
	nsamples = len(chips_list)
	width, height, nbands = chips_list[0].shape

	del chips_list
	gc.collect()

	return np.memmap(memmap_path, dtype='float32', mode='w+', shape=(nsamples, width, height, nbands))

def train_test_split(data_path, expect_path, metadata_path, test_size=0.2):
	
	chips_mtl = json.load(open(metadata_path, 'r'))
	nsamples = chips_mtl['nsamples']

	nsamples_test = int(nsamples * test_size)
	nsamples_train = nsamples - nsamples_test
	
	shape_train_data = (nsamples_train, chips_mtl['data_size'], chips_mtl['data_size'], chips_mtl['data_nbands'])
	shape_train_expect = (nsamples_train, chips_mtl['expe_size'], chips_mtl['expe_size'], chips_mtl['expe_nbands'])

	shape_test_data = (nsamples_test, chips_mtl['data_size'], chips_mtl['data_size'], chips_mtl['data_nbands'])
	shape_test_expect = (nsamples_test, chips_mtl['expe_size'], chips_mtl['expe_size'], chips_mtl['expe_nbands'])

	offset_test_data = 4*nsamples_train*chips_mtl['data_size']*chips_mtl['data_size']*chips_mtl['data_nbands']
	offset_test_expect = 4*nsamples_train*chips_mtl['expe_size']*chips_mtl['expe_size']*chips_mtl['expe_nbands']

	train_data = np.memmap(data_path, dtype='float32', mode='r', shape=shape_train_data)
	train_expect = np.memmap(expect_path, dtype='float32', mode='r', shape=shape_train_expect)

	test_data = np.memmap(data_path, dtype='float32', mode='r', offset=offset_test_data, shape=shape_test_data)
	test_expect = np.memmap(expect_path, dtype='float32', mode='r', offset=offset_test_expect, shape=shape_test_expect)

	return train_data, test_data, train_expect, test_expect

def get_train_test_data(img_path, nodata_value, ninput_bands, chip_size, pad_size, seed, \
													offset_list=[(0,0)], rotate=False, flip=False, remove_chips_wnodata=True):
	
	npz_path = os.path.splitext(img_path)[0] + '.npz'
	data_path = os.path.splitext(img_path)[0] + '_data.dat'
	expect_path = os.path.splitext(img_path)[0] + '_exp.dat'
	metadata_path = os.path.splitext(img_path)[0] + '_data.json'

	if not os.path.isfile(data_path):
		print("Creating chips from image " + img_path + "...")
	
		chip_data_list, chip_expect_list = chip_generation(img_path, nodata_value, chip_size, pad_size, offset_list, rotate, flip, remove_chips_wnodata)

		chips_data = create_memmap_np(data_path, chip_data_list)
		chips_expect = create_memmap_np(expect_path, chip_expect_list)

		chip_generation(img_path, nodata_value, chip_size, pad_size, offset_list, rotate, flip, remove_chips_wnodata, chips_data, chips_expect)

		chips_mtl = {
			"nsamples": chips_data.shape[0],
			"data_size": chips_data.shape[1],
			"data_nbands": chips_data.shape[3],
			"expe_size": chips_expect.shape[1],
			"expe_nbands": chips_expect.shape[3]
		}

		json.dump(chips_mtl, open(metadata_path, 'w'))
		chips_data.flush()
		chips_expect.flush()

		del chips_data
		del chips_expect

	train_data, test_data, train_expect, test_expect = train_test_split(data_path, expect_path, metadata_path)

	print('Train samples: ', len(train_data))
	print('Test samples: ', len(test_data))

	return train_data, test_data, train_expect, test_expect

def get_predict_data(input_img_ds, input_position, pad_size):
	inp_x0 = input_position[0]
	inp_x1 = input_position[1]
	inp_y0 = input_position[2]
	inp_y1 = input_position[3]

	inp_xlen = inp_x1 - inp_x0
	inp_ylen = inp_y1 - inp_y0

	inp_x0pad = 0
	inp_y0pad = 0
	inp_x1pad = 0
	inp_y1pad = 0
	out_x0 = inp_x0 + pad_size
	out_y0 = inp_y0 + pad_size

	if (inp_x0 == 0):
		inp_xlen = inp_xlen-pad_size
		inp_x0pad = pad_size
		out_x0 = 0

	if (inp_y0 == 0):
		inp_ylen = inp_ylen-pad_size
		inp_y0pad = pad_size
		out_y0 = 0

	if (inp_x1 > input_img_ds.RasterXSize):
		inp_xlen = inp_xlen-pad_size
		inp_x1pad = pad_size
	
	if inp_y1 > input_img_ds.RasterYSize:
		inp_ylen = inp_ylen-pad_size
		inp_y1pad = pad_size

	chip_data = input_img_ds.ReadAsArray(inp_x0, inp_y0, inp_xlen, inp_ylen)
	chip_data = np.pad(chip_data, [(0,0), (inp_y0pad, inp_y1pad), (inp_x0pad, inp_x1pad)], mode='reflect')
	chip_data = np.transpose(chip_data, [1,2,0])

	return chip_data, [out_x0, out_y0]

def get_predict_positions(x_size, y_size, chip_size = 388, pad_size = 92, x_offset = 0, y_offset = 0):

	x_start = pad_size + x_offset
	x_end = x_size - pad_size

	y_start = pad_size + y_offset
	y_end = y_size - pad_size

	input_positions = []

	for x0 in range(x_start, x_end, chip_size):
		
		x0_pad, x1_pad = pad_index(x0, x_size, chip_size, pad_size)

		for y0 in range(y_start, y_end, chip_size):
			
			y0_pad, y1_pad = pad_index(y0, y_size, chip_size, pad_size)
			input_positions.append([x0_pad, x1_pad, y0_pad, y1_pad])
				
	return input_positions

def memory_percentage():
	memory = psutil.virtual_memory()
	return memory[2]

def discretize_values(data, numberClass, startValue = 0):
	for clazz in range(startValue, (numberClass + 1) ):
		if clazz == startValue:
			classFilter = (data <= clazz + 0.5)
		elif  clazz == numberClass:
			classFilter = (data > clazz - 0.5)
		else:
			classFilter = np.logical_and(data > clazz - 0.5, data <= clazz + 0.5) 
		data[classFilter] = clazz

	return data.astype(np.uint8)

def create_output_file(base_filepath, out_filepath, dataType = gdal.GDT_Int16, imageFormat = 'GTiff'):
    
  driver = gdal.GetDriverByName(imageFormat)
  base_ds = gdal.Open(base_filepath)

  x_start, pixel_width, _, y_start, _, pixel_height = base_ds.GetGeoTransform()
  x_size = base_ds.RasterXSize 
  y_size = base_ds.RasterYSize
  
  out_srs = osr.SpatialReference()
  out_srs.ImportFromWkt(base_ds.GetProjectionRef())

  output_img_ds = driver.Create(out_filepath, x_size, y_size, 1, dataType, ['COMPRESS=LZW'])
  output_img_ds.SetGeoTransform((x_start, pixel_width, 0, y_start, 0, pixel_height))
  output_img_ds.SetProjection(out_srs.ExportToWkt())

  return output_img_ds