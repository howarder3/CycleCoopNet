import tensorflow as tf

def conv2d(input_image, output_dim, kernal=(5, 5), strides=(2, 2), padding='SAME', activate_func=None, name="conv2d"):
	with tf.variable_scope(name):  

		weight = tf.get_variable('weight',[kernal[0], kernal[1], input_image.get_shape()[-1],output_dim], 	
						initializer=tf.random_normal_initializer(stddev=0.01))

		conv_result = tf.nn.conv2d(input_image, weight, strides=[1, strides[0], strides[1], 1], padding=padding)

		bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))

		result = tf.nn.bias_add(conv_result, bias)

		if activate_func == "leaky_relu":
			result = leaky_relu(result)
		elif activate_func == "relu":
			result = relu(result)

		return result

def transpose_conv2d(input_image, output_shape, kernal=(5, 5), strides=(2, 2), padding='SAME', activate_func=None, name="transpose_conv2d"):
	with tf.variable_scope(name):

		output_shape = list(output_shape)
		output_shape[0] = tf.shape(input_image)[0]
		# output_shape[0] = input_image.get_shape()[0]
		# output_shape[0] = 1

		# output_shape = [int(input_image.get_shape()[0]), int(output_shape[1]), int(output_shape[2]), int(output_shape[-1])]

		[strides_x, strides_y] = [2,2] #list(strides)
		# strides = [1, 2, 2, 1]


		# previous_layer_shape = input_image.get_shape()

		# output shape = [batch_size, input_shape*2, input_shape*2, output_dim]
		# output_shape = [int(previous_layer_shape[0]), int(previous_layer_shape[1])*2, int(previous_layer_shape[2])*2, int(output_shape[-1])]

		weight = tf.get_variable('weight',[5, 5, int(output_shape[-1]), input_image.get_shape()[-1]],
								initializer=tf.random_normal_initializer(stddev=0.02))


		# weight = tf.get_variable('weight', [kernal[0], kernal[1], output_shape[-1], input_image.get_shape()[-1]],
		#                     initializer=tf.random_normal_initializer(stddev=0.005))

		print(output_shape)
		# print(strides)

		# x = (tf.stack(output_shape, axis=0))
		# print(str(x))

		# trans_conv_result = tf.nn.conv2d_transpose(input_image, weight, output_shape=output_shape, strides=strides, padding=padding)
		trans_conv_result = tf.nn.conv2d_transpose(input_image, weight, output_shape=output_shape, strides=[1, strides_x, strides_y, 1])

		bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

		result = tf.nn.bias_add(trans_conv_result, bias)

		if activate_func == "leaky_relu":
			result = leaky_relu(result)
		elif activate_func == "relu":
			result = relu(result)

		return result


def fully_connected(input_image, output_dim, name="fc"):
	with tf.variable_scope(name):

		weight = tf.get_variable('fc_weight', [input_image.get_shape()[1], input_image.get_shape()[2], input_image.get_shape()[-1], output_dim],
		   				 initializer=tf.random_normal_initializer(stddev=0.01))

		conv_result = tf.nn.conv2d(input_image, weight, strides=[1,1,1,1], padding='VALID')

		bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))

		result = tf.nn.bias_add(conv_result, bias)

		return result


# leaky_relu
def leaky_relu(x, leak=0.2, name="leaky_relu"):
	return tf.maximum(x, leak*x)		


# relu
def relu(x, name="relu"):
	return tf.nn.relu(x)


def L1_distance(input_data, target):
    return tf.reduce_mean(tf.abs(input_data - target))


def L2_distance(input_data, target):
    return tf.reduce_mean((input_data-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def convt2d(input_, output_shape, kernal=(5, 5), strides=(2, 2), padding='SAME', activate_fn=None, name="convt2d"):
    assert type(kernal) in [list, tuple, int]
    assert type(strides) in [list, tuple, int]
    assert type(padding) in [list, tuple, int, str]
    if type(kernal) == list or type(kernal) == tuple:
        [k_h, k_w] = list(kernal)
    else:
        k_h = k_w = kernal
    if type(strides) == list or type(strides) == tuple:
        [d_h, d_w] = list(strides)
    else:
        d_h = d_w = strides
    output_shape = list(output_shape)
    output_shape[0] = tf.shape(input_)[0]
    with tf.variable_scope(name):
        if type(padding) in [tuple, list, int]:
            if type(padding) == int:
                p_h = p_w = padding
            else:
                [p_h, p_w] = list(padding)
            pad_ = [0, p_h, p_w, 0]
            input_ = tf.pad(input_, [[p, p] for p in pad_], "CONSTANT")
            padding = 'VALID'

        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=0.005))
        convt = tf.nn.conv2d_transpose(input_, w, output_shape=tf.stack(output_shape, axis=0), strides=[1, 2, 2, 1],
                                       padding=padding)
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.bias_add(convt, biases)
        if activate_fn:
            convt = activate_fn(convt)
        return convt


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


class batch_norm(object):
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
		                  decay=self.momentum, 
		                  updates_collections=None,
		                  epsilon=self.epsilon,
		                  scale=True,
		                  is_training=train,
		                  scope=self.name)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		try:
			matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
				tf.random_normal_initializer(stddev=stddev))
		except ValueError as err:
			msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
			err.args = err.args + (msg,)
			raise
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias