import os
import gdal
import osr
import numpy as np
import psutil

import math
from sklearn.model_selection import train_test_split

def pad_index(index, dim_size, chip_size, pad_size):

	i0 = (index - pad_size)
	i1 = (index + chip_size + pad_size)

	if i1 > (dim_size + pad_size):
		i1 = (dim_size + pad_size)
		i0 = i1 - chip_size - 2*pad_size

	return i0, i1

def split_data(data, pad_size):
	nbands, xsize, ysize = data.shape

	last_band = (nbands-1)
	
	x0 = pad_size
	x1 = xsize-pad_size

	y0 = pad_size
	y1 = ysize-pad_size

	chip_expect = data[last_band:nbands, x0:x1, y0:y1]
	chip_data = data[0:last_band, :, :]

	return chip_data, chip_expect

def get_chips(data, chip_size = 388, pad_size = 92, x_offset = 0, y_offset = 0, nodata_value = 0, remove_chips_wnodata=False):

	data = np.pad(data, [(0,0), (pad_size, pad_size), (pad_size, pad_size)], mode='reflect')
	_, x_size, y_size = data.shape

	x_start = pad_size + x_offset
	x_end = x_size - pad_size

	y_start = pad_size + y_offset
	y_end = y_size - pad_size

	chip_data_list = []
	chip_expect_list = []

	for x0 in range(x_start, x_end, chip_size):
		
		x0_pad, x1_pad = pad_index(x0, x_size, chip_size, pad_size)

		for y0 in range(y_start, y_end, chip_size):
			
			y0_pad, y1_pad = pad_index(y0, y_size, chip_size, pad_size)
			chip_data, chip_expect = split_data( data[:, x0_pad:x1_pad, y0_pad:y1_pad], pad_size )
			
			_, xsize, ysize = chip_expect.shape

			if (chip_size == xsize and chip_size == ysize) and (not remove_chips_wnodata or float(np.min(chip_expect)) != float(nodata_value)):
				if(float(np.max(chip_expect)) == float(1.0)): # Only include chips with some object (pixels == 1)
					chip_data_list.append(chip_data)
					chip_expect_list.append(chip_expect)

	return np.stack(chip_data_list), np.stack(chip_expect_list)

def rotate_flip_chips(data, rotate = True, flip = True):
	result = [ data ]

	if rotate:
		result = result + [ np.rot90(data, k=k, axes=(2,3)) for k in [1,2,3] ]

	if flip:
		result = result + [np.fliplr(r_data) for r_data in result] # Band flip

	return np.concatenate(result)

def get_chips_with_augmentation(data, chip_size = 572, pad_size = 388, offset_list = [(0,0)], rotate = True, flip = True, nodata_value = 0, remove_chips_wnodata=False):
	
	chips_data_list = []
	chips_expect_list = []

	for x_offset_percent, y_offset_percent in offset_list:
		x_offset = int(chip_size * (x_offset_percent / 100.0))
		y_offset = int(chip_size * (y_offset_percent / 100.0))
		
		chips_data, chips_expect = get_chips(data, x_offset = x_offset, y_offset = y_offset, chip_size = chip_size, pad_size = pad_size,
																		nodata_value=nodata_value, remove_chips_wnodata=remove_chips_wnodata)

		chips_data_list.append(chips_data)
		chips_expect_list.append(chips_expect)

	data = None
	chips_data = rotate_flip_chips( np.concatenate(chips_data_list), rotate, flip)
	chips_expect = rotate_flip_chips( np.concatenate(chips_expect_list), rotate, flip)

	return np.transpose(chips_data, [0,2,3,1]), np.transpose(chips_expect, [0,2,3,1])

def get_train_test_data(img_path, nodata_value, ninput_bands, chip_size, pad_size, seed, \
													offset_list=[(0,0)], rotate=False, flip=False):
	
	npz_path = os.path.splitext(img_path)[0] + '.npz'

	if os.path.isfile(npz_path):
		print("Reading from cache " + npz_path + "...")
		data = np.load(npz_path)
		chips_data = data['chips_data'].reshape(-1, (chip_size+2*pad_size), (chip_size+2*pad_size), ninput_bands-1).astype(np.float32)
		chips_expect = data['chips_expect'].reshape(-1, chip_size, chip_size, 1).astype(np.float32)
	else:
		print("Reading image " + img_path + "...")
		gtif = gdal.Open(img_path)
		img_data = gtif.ReadAsArray(0, 0, gtif.RasterXSize, gtif.RasterYSize)

		chips_data, chips_expect = get_chips_with_augmentation(img_data, chip_size=chip_size, pad_size = pad_size, \
														offset_list=offset_list, rotate=rotate, flip=flip, \
														nodata_value=nodata_value, remove_chips_wnodata=True)

		print("Saving in cache " + npz_path + "...")
		np.savez_compressed(npz_path, chips_data=chips_data, chips_expect=chips_expect)

	train_data, test_data, train_expect, test_expect = train_test_split(chips_data, chips_expect, test_size=0.2, random_state=seed)

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