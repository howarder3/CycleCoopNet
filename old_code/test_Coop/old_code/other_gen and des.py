# 4+4 generator


	# def generator(self, input_image, reuse=False):
	# 	with tf.variable_scope("gen", reuse=reuse):

	# 		print("\n------  generator layers  ------\n")

	# 		num_filter = 64

	# 		# ---------- encoder part ----------
	# 		# gen_encode_conv2d(input_image, output_dimension (by how many filters), scope_name)
	# 		# input image = [batch_size, 256, 256, input_pic_dim]

	# 		# gen_encode_layer_1_output = (batch_size, 128, 128, num_filter)
	# 		gen_encode_layer_1_conv = gen_encode_conv2d(input_image, num_filter, name='gen_encode_layer_1_conv') 

	# 		# gen_encode_layer_2_output = (batch_size, 64, 64, num_filter*2)
	# 		gen_encode_layer_2_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_1_conv), num_filter*2, name='gen_encode_layer_2_conv') 
	# 		gen_encode_layer_2_batchnorm = self.gen_encode_batchnorm_layer2(gen_encode_layer_2_conv)
			
	# 		# gen_encode_layer_3_output = (batch_size, 32, 32, num_filter*4)
	# 		gen_encode_layer_3_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_2_batchnorm), num_filter*4, name='gen_encode_layer_3_conv')
	# 		gen_encode_layer_3_batchnorm = self.gen_encode_batchnorm_layer3(gen_encode_layer_3_conv)

	# 		# gen_encode_layer_8_output = (batch_size, 1, 1, num_filter*8)
	# 		gen_encode_layer_4_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_3_batchnorm), num_filter*8, name='gen_encode_layer_4_conv') 
	# 		gen_encode_layer_4_batchnorm = self.gen_encode_batchnorm_layer8(gen_encode_layer_4_conv)

	# 		# ---------- decoder part ----------
	# 		# gen_decode_conv2d(input_image, output_dimension (by how many filters), scope_name)
	# 		# input image = [batch_size, 1, 1, num_filter*8]

	# 		# gen_decode_layer_1_output = (batch_size, 2, 2, num_filter*8*2)
	# 		gen_decode_layer_1_deconv = gen_decode_conv2d(relu(gen_encode_layer_4_batchnorm), num_filter*8, name='gen_decode_layer_1_deconv') 
	# 		gen_decode_layer_1_batchnorm = self.gen_decode_batchnorm_layer1(gen_decode_layer_1_deconv)
	# 		gen_decode_layer_1_dropout = tf.nn.dropout(gen_decode_layer_1_batchnorm, rate=0.5)
	# 		gen_decode_layer_1_concat = tf.concat([gen_decode_layer_1_dropout, gen_encode_layer_3_batchnorm], 3)

	# 		# gen_decode_layer_2_output = (batch_size, 4, 4, num_filter*8*2)
	# 		gen_decode_layer_2_deconv = gen_decode_conv2d(relu(gen_decode_layer_1_concat), num_filter*4, name='gen_decode_layer_2_deconv') 
	# 		gen_decode_layer_2_batchnorm = self.gen_decode_batchnorm_layer2(gen_decode_layer_2_deconv)
	# 		gen_decode_layer_2_dropout = tf.nn.dropout(gen_decode_layer_2_batchnorm, rate=0.5)
	# 		gen_decode_layer_2_concat = tf.concat([gen_decode_layer_2_dropout, gen_encode_layer_2_batchnorm], 3)

	# 		# gen_decode_layer_3_output = (batch_size, 8, 8, num_filter*8*2)
	# 		gen_decode_layer_3_deconv = gen_decode_conv2d(relu(gen_decode_layer_2_concat), num_filter*2, name='gen_decode_layer_3_deconv') 
	# 		gen_decode_layer_3_batchnorm = self.gen_decode_batchnorm_layer3(gen_decode_layer_3_deconv)
	# 		gen_decode_layer_3_dropout = tf.nn.dropout(gen_decode_layer_3_batchnorm, rate=0.5)
	# 		gen_decode_layer_3_concat = tf.concat([gen_decode_layer_3_dropout, gen_encode_layer_1_conv], 3)

	# 		# gen_decode_layer_8_output = (batch_size, 256, 256, output_pic_dim)
	# 		gen_decode_layer_4_deconv = gen_decode_conv2d(relu(gen_decode_layer_3_concat), self.output_pic_dim, name='gen_decode_layer_4_deconv') 
	# 		generator_output = tf.nn.tanh(gen_decode_layer_4_deconv)

	# 		return generator_output

