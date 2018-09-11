import os
import sys
import scipy.misc
import numpy as np
import gdal
import gc

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import tensorflow as tf

from models import unet as md
import image_utils

def do_evaluation(estimator, input_data, input_expected, category, params):
	predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"data": input_data}, batch_size=params['batch_size'], shuffle=False)
	predict_results = estimator.predict(input_fn=predict_input_fn)
	
	output_dir = 'results/'+category
	
	try:
		os.makedirs(output_dir)
	except:
		pass

	pred_flat_array = []
	ref_flat_array = []
	mean_acc = []

	i = 0
	print("Saving images in " + output_dir + "...")
	for predict, expect in zip(predict_results, input_expected):
		
		predict_vis = image_utils.discretize_values(predict, params['n_classes'], 0)

		predict_vis[predict_vis == 1] = 255

		scipy.misc.imsave(output_dir+'/'+str(i)+'_predict.jpg', predict_vis[:,:,0])
		scipy.misc.imsave(output_dir+'/'+str(i)+'_expect.jpg', expect[:,:,0])

		pre_flat = predict.reshape(-1)
		exp_flat = expect.reshape(-1)

		pred_flat_array = np.append(pred_flat_array, pre_flat)
		ref_flat_array = np.append(ref_flat_array, exp_flat)

		mean_acc.append( accuracy_score(exp_flat, pre_flat) )

		i = i + 1

	print(category + ' accurancy:',np.mean(mean_acc))
	print('\n--------------------------------------------------')
	print('------------- '+category+' METRICS -------------')
	print('--------------------------------------------------')
	print(classification_report(ref_flat_array, pred_flat_array))

def predict(img_input_path, params, img_output_path):

	apply_seed(params)
	
	input_img_ds = gdal.Open(img_input_path)
	output_img_ds = image_utils.create_output_file(img_input_path, img_output_path)
	output_band = output_img_ds.GetRasterBand(1)
	
	estimator = tf.estimator.Estimator(model_fn=md.description, params=params, model_dir=params['model_dir'])

	input_positions = image_utils.get_predict_positions(input_img_ds.RasterXSize, input_img_ds.RasterYSize, params['chip_size'], params['pad_size'])
	
	cache_chip_data = []
	cache_out_position = []

	count = 0
	for i in range(len(input_positions)):
		input_position = input_positions[i]
		chip_data, out_position = image_utils.get_predict_data(input_img_ds, input_position, params['pad_size'])

		cache_chip_data.append(chip_data)
		cache_out_position.append(out_position)

		if (image_utils.memory_percentage() > 40) or i == (len(input_positions)-1):
			input_data = np.stack(cache_chip_data)

			del cache_chip_data
			cache_chip_data = []

			input_data = input_data[:,:,:,0:(params['n_input_bands']-1)]

			tensors_to_log = {}
			data_size, _, _, _ =  input_data.shape

			predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"data": input_data}, batch_size=params['batch_size'], shuffle=False)
			predict_results = estimator.predict(input_fn=predict_input_fn)

			print("Predicting image " + img_input_path + " " + str(float(i)/len(input_positions)) + "%" )

			for chip_predict, out_position in zip(predict_results, cache_out_position):
				out_predict = image_utils.discretize_values(chip_predict, params['n_classes'], 0)

				out_x0 = out_position[0]
				out_xy = out_position[1]
				count = count + 1
				output_band.WriteArray(out_predict[:,:,0], out_x0, out_xy)

			output_band.FlushCache()

			del input_data
			del predict_results
			cache_out_position = [] 
			gc.collect()

def evaluate(img_path, params):
	
	apply_seed(params)
	
	train_data, test_data, train_labels, test_labels = image_utils.get_train_test_data(img_path, params['nodata_value'], \
																												params['n_input_bands'], params['chip_size'], params['pad_size'], params['seed'])
	data_size, _, _, _ =  train_data.shape

	estimator = tf.estimator.Estimator(model_fn=md.description, params=params, model_dir=params['model_dir'])
	logging_hook = tf.train.LoggingTensorHook(tensors={}, every_n_iter=data_size)

	print("Evaluating results for image " + img_path + "...")

	do_evaluation(estimator, train_data, train_labels, 'train', params)
	do_evaluation(estimator, test_data, test_labels, 'test', params)

def train(img_path, params):
	
	apply_seed(params)
	
	train_data, test_data, train_labels, test_labels = image_utils.get_train_test_data(img_path, params['nodata_value'], \
																												params['n_input_bands'], params['chip_size'], params['pad_size'], params['seed'])
	data_size, _, _, _ =  train_data.shape
	
	print("Memory size: %d Mb" % ( ((train_data.size * train_data.itemsize) + (test_data.size * test_data.itemsize))*0.000001 ))
	print("Train data shape: " + str(train_data.shape))
	print("Train label shape: " + str(train_labels.shape))
	print("Train params: " + str(params))

	estimator = tf.estimator.Estimator(model_fn=md.description, params=params, model_dir=params['model_dir'])
	logging_hook = tf.train.LoggingTensorHook(tensors={'loss': 'cost/loss'}, every_n_iter=params['batch_size']*4)

	for i in range(0, params['num_epochs']):
		print(train_data.shape)
		train_input = tf.estimator.inputs.numpy_input_fn(x={"data": train_data}, y=train_data, batch_size=params['batch_size'], num_epochs=1, shuffle=True)
		train_results = estimator.train(input_fn=train_input, steps=None, hooks=[logging_hook])

		test_input = tf.estimator.inputs.numpy_input_fn(x={"data": test_data}, y=test_labels, batch_size=params['batch_size'], num_epochs=1, shuffle=False)
		test_results = estimator.evaluate(input_fn=test_input)

def apply_seed(params):
	tf.set_random_seed(params['seed'])
	tf.logging.set_verbosity(tf.logging.INFO)

def print_usage():
	print("Usage: run.py <MODE> image_path image_nodata\n")
	print("Avaiable modes: train, evaluate, predict \n")
	print("For train mode check the parameters in the code")
	exit(0)

####################################
### Parameters
####################################
params = {
	'n_classes': 2,
	'n_input_bands': 9,
	'nodata_value': -10,
	'pad_size': 93,
	'chip_size': 100,
	
	'seed': 1989,
	'batch_size': 32,
	'num_epochs': 200,
	'model_dir': 'logs_bebedouros',

	'dropout_rate': 0.5,
	'learning_rate': 0.00001,
	'tensorboard_maxoutput': 2,
	'l2_regularizer': 0.5
}

if __name__ == "__main__":
	if(len(sys.argv) != 4):
		print_usage()

	mode = sys.argv[1]
	img_input_path = sys.argv[2]
	img_input_nodata = sys.argv[3]

	params['nodata_value'] = img_input_nodata

	if mode == 'evaluate':

		evaluate(img_input_path, params)

	elif mode == 'predict':
		
		img_output_path = os.path.splitext(img_input_path)[0] + '_predict.tif'
		predict(img_input_path, params, img_output_path)

	elif mode == 'train':
		
		train(img_input_path, params)

	else:
		print('Invalid mode !')
		print_usage()