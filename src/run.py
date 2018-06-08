import os
import sys
import scipy.misc
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import tensorflow as tf

from models import unet as md
import image_utils

def do_evaluation(estimator, input_data, input_expected, category):
	predict_input = tf.estimator.inputs.numpy_input_fn(x={"data": input_data}, batch_size=BATCH_SIZE, shuffle=False)
	predict_output = estimator.predict(input_fn=predict_input)
	
	output_dir = 'results/'+category
	
	try:
		os.makedirs(output_dir)
	except:
		pass

	mean_acc = []
	mean_precision = []
	mean_recall = []

	i = 0
	print("Saving images in " + output_dir + "...")
	for predict, expect in zip(predict_output, input_expected):
		
		predict = image_utils.discretize_values(predict, NUMBER_CLASS)

		scipy.misc.imsave(output_dir+'/'+str(i)+'_predict.jpg', predict[:,:,0])
		scipy.misc.imsave(output_dir+'/'+str(i)+'_expect.jpg', expect[:,:,0])

		pre_flat = predict.reshape(-1)
		exp_flat = expect.reshape(-1)

		mean_acc.append( accuracy_score(exp_flat, pre_flat) )

		# Works only for binary classifications
		#mean_precision.append( precision_score(exp_flat, pre_flat) ) 
		#mean_recall.append( recall_score(exp_flat, pre_flat) )

		i = i + 1

	print(category + ' accurancy:',np.mean(mean_acc))
	#print(category + ' precision:',np.mean(mean_precision))
	#print(category + ' recall:',np.mean(mean_recall))

def predict(img_input_path, img_output_path, img_input_nodata):

	input_data, input_windows = image_utils.get_chips_padding(img_input_path, nodata_value=img_input_nodata, size=CHIP_SIZE, start_perc_positions=[(0,0)], rotate=False, flip=False)
	estimator = tf.estimator.Estimator(model_fn=md.description, model_dir=MODEL_DIR)

	# Remove the reference map (labels) in last band
	input_data = input_data[:,:,:,0:(NUMBER_INPUT_BANDS-1)]

	tensors_to_log = {}
	data_size, _, _, _ =  input_data.shape

	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=data_size)

	predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"data": input_data}, batch_size=BATCH_SIZE, shuffle=False)
	predict_results = estimator.predict(input_fn=predict_input_fn)

	print("Predicting image " + img_input_path + "...")

	predict_array = []
	for predict, dummy in zip(predict_results, input_data):
		predict_array.append( image_utils.discretize_values(predict, NUMBER_CLASS) )
	
	image_utils.write_data(img_input_path, img_output_path, predict_array, input_windows)

def evaluate(img_path, nodata_value):

	train_data, test_data, val_data, train_labels, test_labels, val_labels = image_utils.get_input_data(img_path, nodata_value, \
																																													NUMBER_INPUT_BANDS, CHIP_SIZE, PADDING_OFFSET, SEED)
	data_size, _, _, _ =  train_data.shape

	estimator = tf.estimator.Estimator(model_fn=md.description, model_dir=MODEL_DIR)
	logging_hook = tf.train.LoggingTensorHook(tensors={}, every_n_iter=data_size)

	print("Evaluating results from image " + img_input_path + "...")

	do_evaluation(estimator, train_data, train_labels, 'train')
	do_evaluation(estimator, test_data, test_labels, 'test')
	do_evaluation(estimator, val_data, val_labels, 'validation')

def train(img_path, nodata_value):

	train_data, test_data, val_data, train_labels, test_labels, val_labels = image_utils.get_input_data(img_path, nodata_value, \
																																													NUMBER_INPUT_BANDS, CHIP_SIZE, PADDING_OFFSET, SEED)
	data_size, _, _, _ =  train_data.shape
	
	print("Memory size: %d Mb" % ( ((train_data.size * train_data.itemsize) + (test_data.size * test_data.itemsize) + (val_data.size * val_data.itemsize))*0.000001 ))

	estimator = tf.estimator.Estimator(model_fn=md.description, model_dir=MODEL_DIR)
	logging_hook = tf.train.LoggingTensorHook(tensors={}, every_n_iter=data_size)

	for i in range(0,NUMBER_EPOCHS):
		train_input = tf.estimator.inputs.numpy_input_fn(x={"data": train_data}, y=train_labels, batch_size=BATCH_SIZE, num_epochs=1, shuffle=True)
		train_results = estimator.train(input_fn=train_input, steps=None, hooks=[logging_hook])

		test_input = tf.estimator.inputs.numpy_input_fn(x={"data": test_data}, y=test_labels, num_epochs=1, shuffle=False)
		test_results = estimator.evaluate(input_fn=test_input)

def print_usage():
	print("Usage: run.py <MODE> image_path image_nodata\n")
	print("Avaiable modes: train, evaluate, predict \n")
	print("For train mode check the parameters in the code")
	exit(0)

####################################
### Classification Parameters
####################################
NUMBER_CLASS = 4
NUMBER_INPUT_BANDS = 5

####################################
### Data Augmentation
####################################
CHIP_SIZE = 256
PADDING_OFFSET = 50

####################################
### Hyperparameters
####################################
SEED = 1989
BATCH_SIZE = 50
NUMBER_EPOCHS = 100

MODEL_DIR = 'logs_planet'

if(len(sys.argv) != 4):
	print_usage()

mode = sys.argv[1]
img_input_path = sys.argv[2]
img_input_nodata = sys.argv[3]

tf.set_random_seed(SEED)
tf.logging.set_verbosity(tf.logging.INFO)

if mode == 'evaluate':

	evaluate(img_input_path, img_input_nodata)

elif mode == 'predict':
	
	img_output_path = os.path.splitext(img_input_path)[0] + '_predict.tif'
	predict(img_input_path, img_output_path, img_input_nodata)

elif mode == 'train':
	
	train(img_input_path, img_input_nodata)

else:
	print('Invalid mode !')
	print_usage()
