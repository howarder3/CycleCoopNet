import tensorflow as tf


class batch_norm(object):
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)


def gen_encode_conv2d(input_image, output_dim, name="gen_encode_conv2d"):
	with tf.variable_scope(name):
		# input_image shape = [batch_size, 256, 256, 3]
		# weight shape = [filter_width, filter_height, input_channels, output_channels(num_filters)] = [5,5,3,64] 
		weight = tf.get_variable('gen_encode_weight',[5,5,input_image.get_shape()[-1],output_dim],
								initializer=tf.truncated_normal_initializer(stddev=0.02))

		# do convolution
		# weight shape = [filter_width, filter_height, input_channels, output_channels(num_filters)] = [5,5,3,64] 
		# filter = 64 5*5 filters (3 channels)
		# stride = 2*2 moving steps
		# padding "same" = zero_padding
		encode_conv_result = tf.nn.conv2d(input_image, weight, strides=[1,2,2,1], padding='SAME')

		# output_dim: how many pictures output
		encode_bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))

		# add bias
		encode_conv_result_add_bias = tf.nn.bias_add(encode_conv_result, encode_bias)

		encode_conv_output = tf.reshape(encode_conv_result_add_bias, encode_conv_result.get_shape())

		print("{} shape: {}".format(name, encode_conv_output.shape))

		return encode_conv_output


def gen_decode_conv2d(input_image, output_dim, name="gen_decode_conv2d"):
	with tf.variable_scope(name):
		# input_image shape = [batch_size, 1, 1, num_filter*8 *2(if concat)]
		previous_layer_shape = input_image.get_shape()

		# weight shape = [filter_width, filter_height, output_channels, input_channels] = [5,5,,] 
		weight = tf.get_variable('gen_decode_weight',[5, 5, output_dim, previous_layer_shape[-1]],
								initializer=tf.random_normal_initializer(stddev=0.02))

		# output shape = [batch_size, input_shape*2, input_shape*2, output_dim]
		output_shape = [int(previous_layer_shape[0]), int(previous_layer_shape[1])*2, int(previous_layer_shape[2])*2, output_dim]

		# do deconvolution
		# weight shape = [filter_width, filter_height, output_channels, input_channels] = [5,5,,] 
		# filter = input_channels 5*5 filters (output_channels)
		# stride = 2*2 moving steps
		decode_conv_result = tf.nn.conv2d_transpose(input_image, weight, strides=[1,2,2,1], output_shape=output_shape)

		# output_dim: how many pictures output
		decode_bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))

		# add bias
		decode_conv_result_add_bias = tf.nn.bias_add(decode_conv_result, decode_bias)
		
		decode_conv_output = tf.reshape(decode_conv_result_add_bias, decode_conv_result.get_shape())

		print("{} shape: {}".format(name, decode_conv_output.shape))

		return decode_conv_output


def des_conv2d(input_image, output_dim, name="des_conv2d"):
	with tf.variable_scope(name):
		# input_image shape = [batch_size, 256, 256, 3]
		# weight shape = [filter_width, filter_height, input_channels, output_channels(num_filters)] = [5,5,3,64] 
		weight = tf.get_variable('descriptor_weight',[5,5,input_image.get_shape()[-1],output_dim],
								initializer=tf.random_normal_initializer(stddev=0.01))

		# do convolution
		# weight shape = [filter_width, filter_height, input_channels, output_channels(num_filters)] = [5,5,3,64] 
		# filter = 64 5*5 filters (3 channels)
		# stride = 2*2 moving steps
		# padding "same" = zero_padding
		descriptor_conv_result = tf.nn.conv2d(input_image, weight, strides=[1,2,2,1], padding='SAME')

		# output_dim: how many pictures output
		descriptor_bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))

		# add bias
		descriptor_conv_result_add_bias = tf.nn.bias_add(descriptor_conv_result, descriptor_bias)

		descriptor_conv_output = tf.reshape(descriptor_conv_result_add_bias, descriptor_conv_result.get_shape())

		print("{} shape: {}".format(name, descriptor_conv_output.shape))

		return descriptor_conv_output


def des_fully_connected(input_image, output_dim, name="des_fully_connected"):

	with tf.variable_scope(name):

		# weight shape = [filter_width, filter_height, output_channels, input_channels] = [5,5,,] 
		weight = tf.get_variable('fully_connected_weight', [input_image.get_shape()[1], input_image.get_shape()[2], input_image.get_shape()[-1], output_dim],
		   				 initializer=tf.random_normal_initializer(stddev=0.01))

		# do deconvolution
		# weight shape = [filter_width, filter_height, output_channels, input_channels] = [size,size,,] 
		# filter = input_channels size * size filters (output_channels)
		# stride = 1*1 moving steps
		# padding "same" = zero_padding
		fully_connected_conv_result = tf.nn.conv2d(input_image, weight, strides=[1,1,1,1], padding='VALID')

		# output_dim: how many pictures output
		fully_connected_bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))

		# add bias
		fully_connected_conv_result_add_bias = tf.nn.bias_add(fully_connected_conv_result, fully_connected_bias)

		fully_connected_conv_output = tf.reshape(fully_connected_conv_result_add_bias, fully_connected_conv_result.get_shape())

		print("{} shape: {}".format(name, fully_connected_conv_output.shape))

		return fully_connected_conv_output


# leaky_relu
def leaky_relu(x, leak=0.2, name="leaky_relu"):
	return tf.maximum(x, leak*x)		


# relu
def relu(x, name="relu"):
	return tf.nn.relu(x)








