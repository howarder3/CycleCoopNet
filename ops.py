import tensorflow as tf








def conv2d(input_image, output_dim, name="conv2d"):
	with tf.variable_scope(name):
		# input_image shape = [batch_size, 256, 256, 3]
		# weight shape = [filter_width, filter_height, dimension, num_filters] = [5,5,3,64] 
		weight = tf.get_variable('weight',[5,5,input_image.get_shape()[-1],output_dim],
								initializer=tf.truncated_normal_initializer(stddev=0.02))

		print("input_image shape = {}".format(input_image.shape))
		print("weight shape = {}".format(weight.shape))

		# do convolution
		# weight shape = [filter_width, filter_height, dimension, num_filters] = [5,5,3,64] 
		# filter = 64 5*5 filters (3 dimensions)
		# stride = 2*2 moving steps
		# padding "same" = zero_padding
		conv2d_result = tf.nn.conv2d(input_image,weight,strides=[1,2,2,1],padding='SAME')

		# output_dim: how many pictures output
		bias = tf.get_variable('bias',[output_dim],initializer=tf.constant_initializer(0.0))

		conv_result = tf.nn.bias_add(conv2d_result, bias)

		print(conv_result)

		return conv_result