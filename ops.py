import tensorflow as tf








def encode_conv2d(input_image, output_dim, name="conv2d"):
	with tf.variable_scope(name):
		# input_image shape = [batch_size, 256, 256, 3]
		# weight shape = [filter_width, filter_height, dimension, num_filters] = [5,5,3,64] 
		weight = tf.get_variable('weight',[5,5,input_image.get_shape()[-1],output_dim],
								initializer=tf.truncated_normal_initializer(stddev=0.02))

		# do convolution
		# weight shape = [filter_width, filter_height, dimension, num_filters] = [5,5,3,64] 
		# filter = 64 5*5 filters (3 dimensions)
		# stride = 2*2 moving steps
		# padding "same" = zero_padding
		conv_result = tf.nn.conv2d(input_image,weight,strides=[1,2,2,1],padding='SAME')

		# output_dim: how many pictures output
		bias = tf.get_variable('bias',[output_dim],initializer=tf.constant_initializer(0.0))

		# add bias
		conv_result_add_bias = tf.nn.bias_add(conv_result, bias)

		print("{} shape: {}".format(name,conv_result.shape))

		return conv_result_add_bias


# lrelu
def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)		


# relu
def relu(x, name="relu"):
	return tf.nn.relu(x)
	