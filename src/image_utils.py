import os
import rasterio
import numpy as np

import math
from rasterio import windows
from sklearn.model_selection import train_test_split

def get_chips(path_image, size = 100, pad_x = 0, pad_y = 0, nodata_value = 0, remove_chips_wnodata=False):
	indexes = None
	dataset = rasterio.open(path_image, dtype=rasterio.float32)
	height = dataset.height
	width = dataset.width
	n_grid_height = math.ceil(dataset.height/float(size)) - 1
	n_grid_width = math.ceil(dataset.width/float(size)) - 1

	data_result = []
	windows_result = []

	for i in range(n_grid_height):
		for j in range(n_grid_width):
			
			row_start = i*size + pad_y
			col_start = j*size + pad_x
			
			window_aux = windows.Window(col_start, row_start, size, size)
			data_aux = dataset.read(indexes, window=window_aux, masked=False, boundless=True)

			if(not remove_chips_wnodata or float(np.min(data_aux)) != float(nodata_value)):
				data_result.append(data_aux)
				windows_result.append(window_aux)

			if (row_start + size*2) > height or (col_start + size*2) > width:
				break
	
	dataset.close()

	return data_result, windows_result

def get_chips_padding(path_image, size = 100, start_perc_positions = [(0,0)], rotate = True, flip = True, nodata_value = 0, remove_chips_wnodata=False):
	data_result = []
	window_result = []

	for start_perc_x, start_perc_y in start_perc_positions:
		pad_x = int(size * (start_perc_x / 100.0))
		pad_y = int(size * (start_perc_y / 100.0))
		
		data_aux, windows_aux = get_chips(path_image, pad_x = pad_x, pad_y = pad_y, size = size, nodata_value=nodata_value, remove_chips_wnodata=remove_chips_wnodata)
		
		data_result = data_result + data_aux
		window_result = window_result + windows_aux

	if(rotate):
		listas_resultado_090 = [np.rot90(m, k=1, axes=(1,2)) for m in data_result]
		listas_resultado_180 = [np.rot90(m, k=2, axes=(1,2)) for m in data_result]
		listas_resultado_270 = [np.rot90(m, k=3, axes=(1,2)) for m in data_result]
		
		data_result = data_result + listas_resultado_090 + listas_resultado_180 + listas_resultado_270
		
	if(flip):
		data_result = [np.fliplr(m) for m in data_result]
		
	return np.transpose(np.stack(data_result), [0,2,3,1]), window_result

def get_input_data(img_path, nodata_value, ninput_bands, chip_size, padding_offset, seed):
	
	npz_path = os.path.splitext(img_path)[0] + '.npz'

	#padding_list = [(0,0), (0,padding_offset), (padding_offset, 0), (padding_offset,padding_offset)]
	padding_list = [(0,0), (padding_offset,padding_offset)]

	if os.path.isfile(npz_path):
		print("Reading from cache " + npz_path + "...")
		data = np.load(npz_path)
		input_data = data['input'].reshape(-1, chip_size, chip_size, ninput_bands)
	else:
		print("Reading image " + img_path + "...")
		input_data, _ = get_chips_padding(img_path, size=chip_size, start_perc_positions=padding_list, rotate=False, flip=True, \
															nodata_value=nodata_value, remove_chips_wnodata=True)
		print(input_data.shape)
		print("Saving in cache " + npz_path + "...")
		np.savez_compressed(npz_path, input=input_data)

	x_train = input_data[:,:,:, 0:(ninput_bands-1)]
	y_train = input_data[:,:,:, (ninput_bands-1):ninput_bands]

	nsamples, _, _, channels = x_train.shape
	img_size = chip_size

	train_data, test_data, train_labels, test_labels = train_test_split(x_train, y_train, test_size=0.3, random_state=seed)
	test_data, val_data, test_labels, val_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=seed)

	print('Train samples: ', len(train_data))
	print('Test samples: ', len(test_data))
	print('Validation samples: ', len(val_data))

	return train_data, test_data, val_data, train_labels, test_labels, val_labels

def write_data(in_path_image, out_path_image, data, input_windows):
	
	in_dataset = dataset = rasterio.open(in_path_image)
	out_dataset = rasterio.open(out_path_image, 'w', driver='GTiff', height=in_dataset.height, width=in_dataset.width, 
										dtype=rasterio.ubyte, crs=in_dataset.crs, transform=in_dataset.transform, count=1)
	
	print("Writing " + str( len(input_windows) ) + " chips in file" + out_path_image)
	for i in range(0, len(input_windows)):
		chip = np.transpose(data[i], [2,0,1])
		out_dataset.write(chip, window=input_windows[i])

def discretize_values(data, numberClass):
	for clazz in range(1, (numberClass + 1) ):
		if clazz == 1:
			classFilter = (data <= clazz + 0.5)
		elif  clazz == numberClass:
			classFilter = (data > clazz - 0.5)
		else:
			classFilter = np.logical_and(data > clazz - 0.5, data <= clazz + 0.5) 
		data[classFilter] = clazz

	return data.astype(np.uint8)