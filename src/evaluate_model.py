#!/usr/bin/python3

import argparse

from dl_models import unet as md
import tensorflow as tf
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import time
import gc
import gdal
import dl_utils

def do_evaluation(estimator, input_data, input_expected, category, params):
	predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"data": input_data}, batch_size=params['batch_size'], shuffle=False)
	predict_results = estimator.predict(input_fn=predict_input_fn)

	pred_flat_list = []
	ref_flat_list = []

	for predict, expect in zip(predict_results, input_expected):
		
		predict_out = dl_utils.discretize_values(predict, 1, 0)

		pre_flat = predict_out.reshape(-1)
		exp_flat = expect.reshape(-1)

		pred_flat_list.append(pre_flat)
		ref_flat_list.append(exp_flat)

		print("Evaluating process " + str( len(ref_flat_list)/len(input_expected) * 100 ) + "%")

	pred_flat_arr = np.concatenate(pred_flat_list)
	ref_flat_arr = np.concatenate(ref_flat_list)

	print(pred_flat_arr.shape)

	accuracy = accuracy_score(ref_flat_arr, pred_flat_arr)
	print(category + ' accurancy:', accuracy)
	print('\n--------------------------------------------------')
	print('------------- '+category+' METRICS -------------')
	print('--------------------------------------------------')
	print(classification_report(ref_flat_arr, pred_flat_arr))

	print('\n--------------------------------------------------')
	print('------------- CONFUSION MATRIX -------------')
	print('--------------------------------------------------')
	print(confusion_matrix(ref_flat_arr, pred_flat_arr))

def parse_args():
	parser = argparse.ArgumentParser(description='STEP 05/06 - Evaluate a ' + \
		' trained model.')
	parser.add_argument("-m", "--model-dir", help='<Required> Input directory with' + \
		' the trained model and the tensorboard logs.', required=True) 
	parser.add_argument("-s", "--eval-size", help='Percentage size of chips that will be' + \
		' used in the evaluation [DEFAULT=Value defined by train_model.py]', type=float, default=0)
	parser.add_argument("-i", "--chips-dir", help='Input directory of chips' + \
		' that will be used by evaluation process [DEFAULT=Value defined by train_model.py]', \
		 default=None)

	return parser.parse_args()

def exec(model_dir, chips_dir, eval_size):
	tf.logging.set_verbosity(tf.logging.INFO)

	start_time = time.time()

	param_path = dl_utils.new_filepath('train_params.dat', directory=model_dir)
	params = dl_utils.load_object(param_path)
	tf.set_random_seed(params['seed'])

	if eval_size <= 0:
		eval_size = params['eval_size']

	if chips_dir is None:
		chips_dir = params['chips_dir']

	dat_path, exp_path, mtd_path = dl_utils.chips_data_files(chips_dir)
	train_data, eval_data, train_expect, eval_expect, chips_info = dl_utils.train_test_split(dat_path, exp_path, mtd_path, eval_size)

	print("Evaluating the model stored into " + model_dir)

	estimator = tf.estimator.Estimator(model_fn=md.description, params=params, model_dir=model_dir)
	do_evaluation(estimator, eval_data, eval_expect, 'EVALUATING', params)

if __name__ == "__main__":
	args = parse_args()

	model_dir = args.model_dir
	eval_size = args.eval_size
	chips_dir = args.chips_dir

	exec(model_dir, chips_dir, eval_size)	