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


def descriptor_conv2d(input_image, output_dim, name="descriptor_conv2d"):
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

# leaky_relu
def leaky_relu(x, leak=0.2, name="leaky_relu"):
	return tf.maximum(x, leak*x)		


# relu
def relu(x, name="relu"):
	return tf.nn.relu(x)
	
# linearization	
def linearization(input_image, output_size, name="des_linear"):
    shape = input_image.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(0.0))

        descriptor_linear_output = tf.matmul(input_image, matrix) + bias

        print("{} shape: {}".format(name, descriptor_linear_output.shape))

        return descriptor_linear_output


def conv2d(input_, output_dim, kernal=(5, 5), strides=(2, 2), padding='SAME', activate_fn=None, name="conv2d"):
    if type(kernal) == list or type(kernal) == tuple:
        [k_h, k_w] = list(kernal)
    else:
        k_h = k_w = kernal
    if type(strides) == list or type(strides) == tuple:
        [d_h, d_w] = list(strides)
    else:
        d_h = d_w = strides

    with tf.variable_scope(name):
        if type(padding) == list or type(padding) == tuple:
            padding = [0] + list(padding) + [0]
            input_ = tf.pad(input_, [[p, p] for p in padding], "CONSTANT")
            padding = 'VALID'

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        if activate_fn:
            conv = activate_fn(conv)
        return conv


def fully_connected(input_, output_dim, name="des_fully_connected"):
    shape = input_.shape
    return conv2d(input_, output_dim, kernal=list(shape[1:3]), strides=(1, 1), padding="VALID", name=name)











