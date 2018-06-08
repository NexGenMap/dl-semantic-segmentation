import tensorflow as tf

def conv_conv_pool(input_, n_filters, mode, flags, name, pool=True, activation=tf.nn.relu):
	
	net = input_

	with tf.variable_scope("layer{}".format(name)):
			for i, F in enumerate(n_filters):
					net = tf.layers.conv2d(
							inputs=net,
							filters=F, 
							kernel_size=(3, 3),
							activation=None,
							padding='same',
							kernel_regularizer=tf.contrib.layers.l2_regularizer(0.30),
							kernel_initializer=tf.initializers.variance_scaling(scale=0.01, distribution="normal"),
							name="conv_{}".format(i + 1))
					net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN), name="bn_{}".format(i + 1))
					net = activation(net, name="relu{}_{}".format(name, i + 1))

			if pool is False:
					return net

			pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

			return net, pool

def upconv_concat(inputA, input_B, n_filter, flags, name):
	
	up_conv = upconv_2D(inputA, n_filter, flags, name)
	return tf.concat([up_conv, input_B], axis=-1, name="concat_{}".format(name))

def upconv_2D(tensor, n_filter, flags, name):
	return tf.layers.conv2d_transpose(
			tensor,
			filters=n_filter,
			kernel_size=2,
			strides=2,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(0.30),
			kernel_initializer=tf.initializers.variance_scaling(scale=0.01, distribution="normal"),
			name="upsample_{}".format(name))

def twoclass_cost(y_pred, y_true):
	with tf.name_scope("cost"):
		logits = tf.reshape(y_pred, [-1])
		trn_labels = tf.reshape(y_true, [-1])

		intersection = tf.reduce_sum( tf.multiply(logits,trn_labels) )
		union = tf.reduce_sum( tf.subtract( tf.add(logits,trn_labels) , tf.multiply(logits,trn_labels) ) )
		loss = tf.subtract( tf.constant(1.0, dtype=tf.float32), tf.divide(intersection,union) )

		return loss

def multiclass_cost(y_pred, y_true):
	with tf.name_scope("cost"):
		loss = tf.losses.mean_squared_error(y_true, y_pred) 

		return loss

def description(features, labels, mode, params, config):
	
	input_data = features['data']
	flags = params

	conv1, pool1 = conv_conv_pool(input_data, [32, 32], mode, flags, name=1)
	conv2, pool2 = conv_conv_pool(pool1, [64, 64], mode, flags, name=2)
	conv3, pool3 = conv_conv_pool(pool2, [128, 128], mode, flags, name=3)
	conv4 = conv_conv_pool(pool3, [256, 256], mode, flags, name=4, pool=False)

	up7 = upconv_concat(conv4, conv3, 128, flags, name=7)
	conv7 = conv_conv_pool(up7, [128, 128], mode, flags, name=7, pool=False)

	up8 = upconv_concat(conv7, conv2, 64, flags, name=8)
	conv8 = conv_conv_pool(up8, [64, 64], mode, flags, name=8, pool=False)

	up9 = upconv_concat(conv8, conv1, 32, flags, name=9)
	conv9 = conv_conv_pool(up9, [32, 32], mode, flags, name=9, pool=False)

	output = tf.layers.conv2d(conv9, 1, (1, 1), name='output', activation=tf.nn.relu, padding='same',
									kernel_initializer=tf.initializers.variance_scaling(scale=0.01, distribution="normal"))

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=output)

	loss = multiclass_cost(output, labels)
	
	optimizer = tf.contrib.opt.NadamOptimizer(0.00005, name='optimizer')

	with tf.name_scope("img_metrics"):
		input_data_viz = ((input_data[:,:,:,0:3]) + 20)
		input_data_viz = tf.image.convert_image_dtype(input_data_viz, tf.uint8)

		output_viz = tf.image.grayscale_to_rgb(output)
		labels_viz = tf.image.grayscale_to_rgb(labels)
	
		tf.summary.image('img',  input_data_viz, max_outputs=2)
		tf.summary.image('output', output_viz, max_outputs=2)
		tf.summary.image('labels', labels_viz, max_outputs=2)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		
	eval_summary_hook = tf.train.SummarySaverHook(save_steps=1, output_dir=config.model_dir+"/eval", summary_op=tf.summary.merge_all())
	return tf.estimator.EstimatorSpec(mode=mode, predictions=output, loss=loss, train_op=train_op, evaluation_hooks=[eval_summary_hook])