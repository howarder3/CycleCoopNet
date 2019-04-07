import tensorflow as tf








def conv2d(input_image, output_dim, name="conv2d"):
	with tf.variable_scope(name):
		# weight shape = [filter_width, filter_height, dimension, num_filters] = [5,5,3,64] 
		weight = tf.get_variable('weight',[5,5,input_image.get_shape()[-1],output_dim],
								initializer=tf.truncated_normal_initializer(stddev=0.02))

		print("input_image shape = {}".format(input_image.shape))
		print("weight shape = {}".format(weight.shape))

		# do convolution
		# stride = 2*2 moving steps
		# padding "same" = zero_padding
		conv = tf.nn.conv2d(input_image,weight,strides=[1,2,2,1],padding='SAME')
		conv_result = weight


		return conv_result
