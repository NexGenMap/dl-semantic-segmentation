#!/usr/bin/python3

import argparse

from models import unet as md
import tensorflow as tf
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import time
import gc
import gdal
import image_utils

def do_evaluation(estimator, input_data, input_expected, category, params):
	predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"data": input_data}, batch_size=params['batch_size'], shuffle=False)
	predict_results = estimator.predict(input_fn=predict_input_fn)

	pred_flat_array = []
	ref_flat_array = []
	mean_acc = []

	for predict, expect in zip(predict_results, input_expected):
		
		predict_out = image_utils.discretize_values(predict, 1, 0)

		pre_flat = predict_out.reshape(-1)
		exp_flat = expect.reshape(-1)

		pred_flat_array = np.append(pred_flat_array, pre_flat)
		ref_flat_array = np.append(ref_flat_array, exp_flat)

		mean_acc.append( accuracy_score(exp_flat, pre_flat) )

	print(category + ' accurancy:',np.mean(mean_acc))
	print('\n--------------------------------------------------')
	print('------------- '+category+' METRICS -------------')
	print('--------------------------------------------------')
	print(classification_report(ref_flat_array, pred_flat_array))

def parse_args():
	parser = argparse.ArgumentParser(description='STEP 05/06 - Evaluate a ' + \
		' trained model.')
	parser.add_argument("-m", "--model-dir", help='<Required> Input directory with' + \
		' the trained model and the tensorboard logs.', required=True)

	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	model_dir = args.model_dir

	tf.logging.set_verbosity(tf.logging.INFO)

	start_time = time.time()

	param_path = image_utils.new_filepath('train_params.dat', directory=model_dir)
	params = image_utils.load_object(param_path)

	tf.set_random_seed(params['seed'])

	dat_path, exp_path, mtd_path = image_utils.chips_data_files(params['chips_dir'])
	train_data, test_data, train_expect, test_expect, chips_info = image_utils.train_test_split(dat_path, exp_path, mtd_path, params['test_size'])

	print("Evaluating the model stored into " + model_dir)

	estimator = tf.estimator.Estimator(model_fn=md.description, params=params, model_dir=model_dir)
	do_evaluation(estimator, test_data, test_expect, 'test', params)