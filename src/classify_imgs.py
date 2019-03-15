#!/usr/bin/python3
import argparse

from dl_models import unet as md
import tensorflow as tf
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import time
import gc
from osgeo import gdal
import dl_utils

def parse_args():
	parser = argparse.ArgumentParser(description='STEP 06/06 - Classify a list of images' + \
		' using a trained model.')
	parser.add_argument("-i", "--images", nargs='+', help='<Required> List of input images' + \
		' that will be classified.', required=True)
	parser.add_argument("-m", "--model-dir", help='<Required> Input directory with' + \
		' the trained model and the tensorboard logs.', required=True)
	parser.add_argument("-o", "--output-dir", help='<Required> The output directory that will ' + \
		' that will have the classification output.', required=True)
	parser.add_argument("-p", "--memory-percentage", help='Reading the input image until' + \
		' memory percentage reach the value defined by this argument. After that, the classification' + \
		' will execute for readed data. [DEFAULT=40.0]', default=40.0, type=float)

	return parser.parse_args()

def exec(images, model_dir, output_dir, memory_percentage = 40):
	tf.logging.set_verbosity(tf.logging.INFO)

	dl_utils.mkdirp(output_dir)

	param_path = dl_utils.new_filepath('train_params.dat', directory=model_dir)
	params = dl_utils.load_object(param_path)

	chips_info_path = dl_utils.new_filepath('chips_info.dat', directory=model_dir)
	chips_info = dl_utils.load_object(chips_info_path)

	for in_image in images:

		in_image_ds = gdal.Open(in_image)
		out_image = dl_utils.new_filepath(in_image, suffix='pred', ext='tif' , directory=output_dir)
		out_image_ds = dl_utils.create_output_file(in_image, out_image)
		out_band = out_image_ds.GetRasterBand(1)

		estimator = tf.estimator.Estimator(model_fn=md.description, params=params, model_dir=model_dir)

		print(chips_info)
		_, dat_xsize, dat_ysize, dat_nbands = chips_info['dat_shape']
		_, exp_xsize, exp_ysize, _ = chips_info['exp_shape']
		pad_size = int( (dat_xsize - exp_xsize) / 2 )

		input_positions = dl_utils.get_predict_positions(in_image_ds.RasterXSize, in_image_ds.RasterYSize, exp_xsize, pad_size)

		cache_chip_data = []
		cache_out_position = []

		count = 0
		for i in range(len(input_positions)):
			input_position = input_positions[i]
			
			try:
				chip_data, out_position = dl_utils.get_predict_data(in_image_ds, input_position, pad_size)
			except IOError as error:
				print(error)
				print('Ignoring this data block')
				continue;

			cache_chip_data.append(chip_data)
			cache_out_position.append(out_position)

			print("Reading image " + in_image + ": memory percentage " + str(dl_utils.memory_percentage()) + "%" )		

			if (dl_utils.memory_percentage() > memory_percentage) or i == (len(input_positions)-1):
				input_data = np.stack(cache_chip_data)

				del cache_chip_data
				cache_chip_data = []

				input_data = input_data[:,:,:,0:dat_nbands]

				tensors_to_log = {}

				print("Classifying image " + in_image + ": progress " + str(float(i)/len(input_positions)*100) + "%" )
				predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"data": input_data}, batch_size=params['batch_size'], shuffle=False)
				predict_results = estimator.predict(input_fn=predict_input_fn)

				print("Writing classification result in " + out_image)
				for chip_predict, out_position in zip(predict_results, cache_out_position):
					out_predict = dl_utils.discretize_values(chip_predict, 1, 0)

					out_x0 = out_position[0]
					out_xy = out_position[1]
					count = count + 1
					out_band.WriteArray(out_predict[:,:,0], out_x0, out_xy)

				out_band.FlushCache()

				del input_data
				del predict_results
				cache_out_position = [] 
				gc.collect()


if __name__ == "__main__":

	args = parse_args()

	model_dir = args.model_dir
	images = args.images
	output_dir = args.output_dir
	memory_percentage = args.memory_percentage

	start_time = time.time()

	exec(images, model_dir, output_dir, memory_percentage)

	elapsed_time = time.time() - start_time
	print('Time elapsed ', elapsed_time)
