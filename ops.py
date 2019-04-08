import tensorflow as tf




class batch_norm(object):
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)



def encode_conv2d(input_image, output_dim, name="encode_conv2d"):
	with tf.variable_scope(name):
		# input_image shape = [batch_size, 256, 256, 3]
		# weight shape = [filter_width, filter_height, input_channels, output_channels(num_filters)] = [5,5,3,64] 
		weight = tf.get_variable('weight',[5,5,input_image.get_shape()[-1],output_dim],
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

		print("{} shape: {}".format(name, encode_conv_result.shape))

		return encode_conv_result_add_bias


def decode_conv2d(input_image, output_dim, name="decode_conv2d"):
	with tf.variable_scope(name):
		print("{} shape: {}".format(name, input_image.shape))

		# input_image shape = [batch_size, 1, 1, num_filter*8 *2(if concat)]
		previous_layer_shape = input_image.get_shape()

		# weight shape = [filter_width, filter_height, output_channels, input_channels] = [5,5,,] 
		weight = tf.get_variable('weight',[5, 5, output_dim, previous_layer_shape[-1]],
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
		
		print("{} shape: {}".format(name, decode_conv_result_add_bias.shape))

		return decode_conv_result_add_bias













'''
def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
'''


# lrelu
def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)		


# relu
def relu(x, name="relu"):
	return tf.nn.relu(x)
	