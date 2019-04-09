import tensorflow as tf
import numpy as np
import time
import os

from glob import glob
from six.moves import xrange

# --------- self define function ---------
# ops: layers structure
from ops import *
# utils: for loading data, model
from utils import *
# data_io: for data_io
from data_io import *

class Coop_pix2pix(object):
	def __init__(self, sess, 
				epoch=2000, 
				batch_size=1,
				picture_amount=99999,
				image_size = 256,
				output_size = 256,
				input_pic_dim = 3, 
				output_pic_dim = 3,				
				dataset_name='facades', dataset_dir ='./test_datasets', 
				output_dir='./output_dir', checkpoint_dir='./checkpoint_dir', log_dir='./log_dir'):
		"""
		args:
			sess: tensorflow session
			batch_size: how many pic in one group(batch), iteration(num_batch) = picture_amount/batch_size
			input_pic_dim: input picture dimension : colorful = 3, grayscale = 1
			output_pic_dim: output picture dimension : colorful = 3, grayscale = 1 

		"""


		self.sess = sess
		self.epoch = epoch		
		self.batch_size = batch_size
		self.picture_amount = picture_amount
		self.image_size = image_size
		self.output_size = output_size
		self.input_pic_dim = input_pic_dim
		self.output_pic_dim = output_pic_dim
		

		self.dataset_dir = dataset_dir
		self.dataset_name = dataset_name

		self.output_dir = output_dir
		self.checkpoint_dir = checkpoint_dir
		self.log_dir = log_dir


		self.gen_encode_batchnorm_layer2 = batch_norm(name='gen_encode_batchnorm_layer2')
		self.gen_encode_batchnorm_layer3 = batch_norm(name='gen_encode_batchnorm_layer3')
		self.gen_encode_batchnorm_layer4 = batch_norm(name='gen_encode_batchnorm_layer4')
		self.gen_encode_batchnorm_layer5 = batch_norm(name='gen_encode_batchnorm_layer5')
		self.gen_encode_batchnorm_layer6 = batch_norm(name='gen_encode_batchnorm_layer6')
		self.gen_encode_batchnorm_layer7 = batch_norm(name='gen_encode_batchnorm_layer7')
		self.gen_encode_batchnorm_layer8 = batch_norm(name='gen_encode_batchnorm_layer8')

		self.gen_decode_batchnorm_layer1 = batch_norm(name='gen_decode_batchnorm_layer1')
		self.gen_decode_batchnorm_layer2 = batch_norm(name='gen_decode_batchnorm_layer2')
		self.gen_decode_batchnorm_layer3 = batch_norm(name='gen_decode_batchnorm_layer3')
		self.gen_decode_batchnorm_layer4 = batch_norm(name='gen_decode_batchnorm_layer4')
		self.gen_decode_batchnorm_layer5 = batch_norm(name='gen_decode_batchnorm_layer5')
		self.gen_decode_batchnorm_layer6 = batch_norm(name='gen_decode_batchnorm_layer6')
		self.gen_decode_batchnorm_layer7 = batch_norm(name='gen_decode_batchnorm_layer7')

		self.descriptor_batchnorm_layer1 = batch_norm(name='descriptor_batchnorm_layer1')
		self.descriptor_batchnorm_layer2 = batch_norm(name='descriptor_batchnorm_layer2')
		self.descriptor_batchnorm_layer3 = batch_norm(name='descriptor_batchnorm_layer3')

		# descriptor langevin steps
		self.descriptor_langevin_steps = 10 

		self.descriptor_step_size = 0.002
		self.sigma1 = 0.016
		self.beta1 = 0.5
		self.L1_lambda = 100

		# learning rate
		self.descriptor_learning_rate = 0.01
		self.generator_learning_rate = 0.0001

	def build_model(self):
		self.input_data_A = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_images_A')
		self.input_generated_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_generated_B')
		self.input_revised_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_revised_B')
		self.real_data_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='real_data_B')

		# self.input_data = tf.placeholder(tf.float32,
		# 		[self.batch_size, self.image_size, self.image_size, self.input_pic_dim+self.output_pic_dim],
		# 		name='input_A_and_B_images')

		# # data domain A and data domain B
		# self.data_A = self.input_data[:, :, :, :self.input_pic_dim]
		# self.data_B = self.input_data[:, :, :, self.input_pic_dim:self.input_pic_dim+self.output_pic_dim]

		# print("data_A shape = {}".format(self.data_A.shape)) # data_A shape = (1, 256, 256, 3)
		# print("data_B shape = {}".format(self.data_B.shape)) # data_B shape = (1, 256, 256, 3)


		self.generated_B = self.generator(self.input_data_A, reuse = False)

		descripted_real_data_B = self.descriptor(self.real_data_B, reuse = False)
		descripted_generated_B = self.descriptor(self.input_generated_B, reuse = True)
		descripted_revised_B = self.descriptor(self.input_revised_B, reuse = True)


		# symbolic langevins
		self.langevin_descriptor = self.langevin_dynamics_descriptor(self.input_generated_B)

		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if var.name.startswith('des')]
		self.g_vars = [var for var in t_vars if var.name.startswith('gen')]

		print("\n------  self.d_vars  ------\n")
		for var in self.d_vars:
			print(var)

		print("\n------  self.g_vars  ------\n")
		for var in self.g_vars:
			print(var)



		# descriptor variables
		# self.d_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(descripted_real_data_B, axis=0), tf.reduce_mean(descripted_revised_B, axis=0)))
		self.d_loss = self.L1_lambda * tf.reduce_mean(tf.abs(descripted_real_data_B - descripted_revised_B))

		d_optim = tf.train.AdamOptimizer(self.descriptor_learning_rate, beta1=self.beta1)
		des_grads_vars = d_optim.compute_gradients(self.d_loss, var_list=self.d_vars)

		# update by mean of gradients
		self.apply_d_grads = d_optim.apply_gradients(des_grads_vars)



		# # generator variables
		self.g_loss = self.L1_lambda * tf.reduce_mean(tf.abs(self.input_revised_B - self.generated_B))

		# self.gen_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(self.real_data_B, axis=0), tf.reduce_mean(self.generated_B, axis=0)))

		# gen_optim = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=self.beta1) #.minimize(self.gen_loss, var_list=self.gen_vars)
		# gen_grads_vars = gen_optim.compute_gradients(self.gen_loss, var_list=self.gen_vars)
		# # gen_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in gen_grads_vars if '/w' in var.name]
		# self.apply_g_grads = gen_optim.apply_gradients(gen_grads_vars)


		# Compute Mean square error(MSE) for generator
		self.mse_loss = tf.reduce_mean(
			tf.pow(tf.subtract(tf.reduce_mean(descripted_generated_B, axis=0), tf.reduce_mean(descripted_revised_B, axis=0)), 2))

		# self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
		# self.real_data_B ,  self.input_data_B:
		# self.d_loss = self.L1_lambda * tf.reduce_mean(tf.abs(descripted_real_data_B - descripted_revised_B))
		# self.d_loss = self.L1_lambda * tf.reduce_mean(tf.abs(self.real_data_B - self.input_data_B))

		

		self.saver = tf.train.Saver(max_to_keep=50)	


	def train(self,sess):
		# start learning
		start_time = time.time()
		print("time: {:.4f} , Learning start!!! ".format(0))
		

		# build model
		self.build_model()

		"""Train pix2pix"""
		# d_optim = tf.train.AdamOptimizer(self.descriptor_learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)


		# prepare training data
		training_data = glob('{}/{}/train/*.jpg'.format(self.dataset_dir, self.dataset_name))

		# iteration(num_batch) = picture_amount/batch_size
		num_batch = min(len(training_data), self.picture_amount) // self.batch_size


		# initialize training
		sess.run(tf.global_variables_initializer())

		sample_results = np.random.randn(num_batch, self.image_size, self.image_size, 3)

		counter = 0

		# start training	
		print("time: {:.4f} , Start training model......".format(time.time()-start_time))
		

		for epoch in xrange(self.epoch): # how many epochs to train

			for index in xrange(num_batch): # num_batch
				# find picture list index*self.batch_size to (index+1)*self.batch_size (one batch)
				# if batch_size = 2, get one batch = batch[0], batch[1]
				batch_files = training_data[index*self.batch_size:(index+1)*self.batch_size] 

				# load data : list format, amount = one batch
				batch = [load_data(batch_file) for batch_file in batch_files]
				batch_images = np.array(batch).astype(np.float32)

				# data domain A and data domain B
				data_A = batch_images[:, :, :, :self.input_pic_dim]
				data_B = batch_images[:, :, :, self.input_pic_dim:self.input_pic_dim+self.output_pic_dim]


				# step G1: try to generate B domain(target domain) picture
				generated_B = sess.run(self.generated_B, feed_dict={self.input_data_A: data_A})
				# print(generated_B.shape) # (1, 256, 256, 3)

				# step D1: descriptor try to revised image:"generated_B"
				revised_B = sess.run(self.langevin_descriptor, feed_dict={self.input_generated_B: generated_B})

				# print(generated_B.shape) # (1, 256, 256, 3)
				# print(revised_B.shape) # (1, 256, 256, 3)

				# step D2: update descriptor net
				_ , descriptor_loss = sess.run([self.apply_d_grads, self.d_loss],
                                  feed_dict={self.real_data_B: data_B, self.input_revised_B: revised_B})

				print(descriptor_loss)

				# # step G2: update generator net
				# generator_loss = sess.run([self.gen_loss, self.apply_g_grads],
				# 					feed_dict={self.real_data_B: revised_B, self.input_data_A: data_A})[0]

				# # Update D network
				# _ , descriptor_loss = self.sess.run([d_optim, self.d_loss],
				# 					feed_dict={self.real_data_B: data_B, self.input_data_B: revised_B})

				# Update G network
				_ , generator_loss = self.sess.run([g_optim, self.g_loss],
									feed_dict={self.input_revised_B: revised_B, self.input_data_A: data_A})

				# Compute Mean square error(MSE) for generator
				mse_loss = sess.run(self.mse_loss, feed_dict={self.input_revised_B: revised_B, self.input_generated_B: generated_B})

				sample_results[index : (index + 1)] = revised_B
				# print(sample_results.shape)

				print("Epoch: [{:4d}] [{:4d}/{:4d}] time: {:.4f}, avg_d_loss: {:.4f}, avg_g_loss: {:.4f}, avg_mse: {:.4f}"
					.format(epoch, index, num_batch, time.time() - start_time, descriptor_loss, generator_loss, mse_loss))

				if np.mod(counter, 100) == 0:
					save_images(generated_B, [self.batch_size, 1],
						'./{}/train_generator_{:02d}_{:04d}.png'.format(self.output_dir, epoch, index))
					save_images(revised_B, [self.batch_size, 1],
						'./{}/train_descriptor_{:02d}_{:04d}.png'.format(self.output_dir, epoch, index))

					saveSampleResults(revised_B, "%s/des_%03d.png" % (self.output_dir, epoch), col_num=1)
					saveSampleResults(generated_B, "%s/gen_%03d.png" % (self.output_dir, epoch), col_num=1)



				if np.mod(counter, 500) == 0:
					self.save(self.checkpoint_dir, counter)

				counter += 1



				# if index == 0 and epoch % 1 == 0:
				# 	if not os.path.exists(self.output_dir):
				# 		os.makedirs(self.output_dir)
				# 	saveSampleResults(revised_B, "%s/des%03d.png" % (self.output_dir, epoch), col_num=self.nTileCol)
				# 	saveSampleResults(generated_B, "%s/gen%03d.png" % (self.output_dir, epoch), col_num=self.nTileCol)

			# # print("time: {:.4f} , Epoch: {} ".format(time.time() - start_time, epoch))
			# print('Epoch #{:d}, avg.descriptor loss: {:.4f}, avg.generator loss: {:.4f}, avg.L2 distance: {:4.4f}, '
			# 	'time: {:.2f}s'.format(epoch, np.mean(des_loss_avg), np.mean(gen_loss_avg), np.mean(mse_avg), time.time() - start_time))


			# if epoch % 1 == 0:
			# 	if not os.path.exists(self.checkpoint_dir):
			# 		os.makedirs(self.checkpoint_dir)
			# 	saver.save(sess, "%s/%s" % (self.checkpoint_dir, 'model.ckpt'), global_step=epoch)

			# 	if not os.path.exists(self.log_dir):
			# 		os.makedirs(self.log_dir)


	def generator(self, input_image, reuse=False):
		with tf.variable_scope("gen", reuse=reuse):

			num_filter = 64

			# ---------- encoder part ----------
			# gen_encode_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 256, 256, input_pic_dim]

			# gen_encode_layer_1_output = (batch_size, 128, 128, num_filter)
			gen_encode_layer_1_conv = gen_encode_conv2d(input_image, num_filter, name='gen_encode_layer_1_conv') 

			# gen_encode_layer_2_output = (batch_size, 64, 64, num_filter*2)
			gen_encode_layer_2_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_1_conv), num_filter*2, name='gen_encode_layer_2_conv') 
			gen_encode_layer_2_batchnorm = self.gen_encode_batchnorm_layer2(gen_encode_layer_2_conv)
			
			# gen_encode_layer_3_output = (batch_size, 32, 32, num_filter*4)
			gen_encode_layer_3_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_2_batchnorm), num_filter*4, name='gen_encode_layer_3_conv')
			gen_encode_layer_3_batchnorm = self.gen_encode_batchnorm_layer3(gen_encode_layer_3_conv)

			# gen_encode_layer_4_output = (batch_size, 16, 16, num_filter*8)
			gen_encode_layer_4_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_3_batchnorm), num_filter*8, name='gen_encode_layer_4_conv') 
			gen_encode_layer_4_batchnorm = self.gen_encode_batchnorm_layer4(gen_encode_layer_4_conv)

			# gen_encode_layer_5_output = (batch_size, 8, 8, num_filter*8)
			gen_encode_layer_5_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_4_batchnorm), num_filter*8, name='gen_encode_layer_5_conv') 
			gen_encode_layer_5_batchnorm = self.gen_encode_batchnorm_layer5(gen_encode_layer_5_conv)

			# gen_encode_layer_6_output = (batch_size, 4, 4, num_filter*8)
			gen_encode_layer_6_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_5_batchnorm), num_filter*8, name='gen_encode_layer_6_conv') 
			gen_encode_layer_6_batchnorm = self.gen_encode_batchnorm_layer6(gen_encode_layer_6_conv)

			# gen_encode_layer_7_output = (batch_size, 2, 2, num_filter*8)
			gen_encode_layer_7_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_6_batchnorm), num_filter*8, name='gen_encode_layer_7_conv') 
			gen_encode_layer_7_batchnorm = self.gen_encode_batchnorm_layer7(gen_encode_layer_7_conv)

			# gen_encode_layer_8_output = (batch_size, 1, 1, num_filter*8)
			gen_encode_layer_8_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_7_batchnorm), num_filter*8, name='gen_encode_layer_8_conv') 
			gen_encode_layer_8_batchnorm = self.gen_encode_batchnorm_layer8(gen_encode_layer_8_conv)

			# ---------- decoder part ----------
			# gen_decode_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 1, 1, num_filter*8]

			# gen_decode_layer_1_output = (batch_size, 2, 2, num_filter*8*2)
			gen_decode_layer_1_deconv = gen_decode_conv2d(relu(gen_encode_layer_8_batchnorm), num_filter*8, name='gen_decode_layer_1_deconv') 
			gen_decode_layer_1_batchnorm = self.gen_decode_batchnorm_layer1(gen_decode_layer_1_deconv)
			gen_decode_layer_1_dropout = tf.nn.dropout(gen_decode_layer_1_batchnorm, rate=0.5)
			gen_decode_layer_1_concat = tf.concat([gen_decode_layer_1_dropout, gen_encode_layer_7_batchnorm], 3)

			# gen_decode_layer_2_output = (batch_size, 4, 4, num_filter*8*2)
			gen_decode_layer_2_deconv = gen_decode_conv2d(relu(gen_decode_layer_1_concat), num_filter*8, name='gen_decode_layer_2_deconv') 
			gen_decode_layer_2_batchnorm = self.gen_decode_batchnorm_layer2(gen_decode_layer_2_deconv)
			gen_decode_layer_2_dropout = tf.nn.dropout(gen_decode_layer_2_batchnorm, rate=0.5)
			gen_decode_layer_2_concat = tf.concat([gen_decode_layer_2_dropout, gen_encode_layer_6_batchnorm], 3)

			# gen_decode_layer_3_output = (batch_size, 8, 8, num_filter*8*2)
			gen_decode_layer_3_deconv = gen_decode_conv2d(relu(gen_decode_layer_2_concat), num_filter*8, name='gen_decode_layer_3_deconv') 
			gen_decode_layer_3_batchnorm = self.gen_decode_batchnorm_layer3(gen_decode_layer_3_deconv)
			gen_decode_layer_3_dropout = tf.nn.dropout(gen_decode_layer_3_batchnorm, rate=0.5)
			gen_decode_layer_3_concat = tf.concat([gen_decode_layer_3_dropout, gen_encode_layer_5_batchnorm], 3)

			# gen_decode_layer_4_output = (batch_size, 16, 16, num_filter*8*2)
			gen_decode_layer_4_deconv = gen_decode_conv2d(relu(gen_decode_layer_3_concat), num_filter*8, name='gen_decode_layer_4_deconv') 
			gen_decode_layer_4_batchnorm = self.gen_decode_batchnorm_layer4(gen_decode_layer_4_deconv)
			gen_decode_layer_4_dropout = tf.nn.dropout(gen_decode_layer_4_batchnorm, rate=0.5)
			gen_decode_layer_4_concat = tf.concat([gen_decode_layer_4_dropout, gen_encode_layer_4_batchnorm], 3)

			# gen_decode_layer_5_output = (batch_size, 32, 32, num_filter*4*2)
			gen_decode_layer_5_deconv = gen_decode_conv2d(relu(gen_decode_layer_4_concat), num_filter*4, name='gen_decode_layer_5_deconv') 
			gen_decode_layer_5_batchnorm = self.gen_decode_batchnorm_layer5(gen_decode_layer_5_deconv)
			gen_decode_layer_5_dropout = tf.nn.dropout(gen_decode_layer_5_batchnorm, rate=0.5)
			gen_decode_layer_5_concat = tf.concat([gen_decode_layer_5_dropout, gen_encode_layer_3_batchnorm], 3)

			# gen_decode_layer_6_output = (batch_size, 64, 64, num_filter*2*2)
			gen_decode_layer_6_deconv = gen_decode_conv2d(relu(gen_decode_layer_5_concat), num_filter*2, name='gen_decode_layer_6_deconv') 
			gen_decode_layer_6_batchnorm = self.gen_decode_batchnorm_layer6(gen_decode_layer_6_deconv)
			gen_decode_layer_6_dropout = tf.nn.dropout(gen_decode_layer_6_batchnorm, rate=0.5)
			gen_decode_layer_6_concat = tf.concat([gen_decode_layer_6_dropout, gen_encode_layer_2_batchnorm], 3)

			# gen_decode_layer_7_output = (batch_size, 128, 128, num_filter*1*2)
			gen_decode_layer_7_deconv = gen_decode_conv2d(relu(gen_decode_layer_6_concat), num_filter, name='gen_decode_layer_7_deconv') 
			gen_decode_layer_7_batchnorm = self.gen_decode_batchnorm_layer7(gen_decode_layer_7_deconv)
			gen_decode_layer_7_dropout = tf.nn.dropout(gen_decode_layer_7_batchnorm, rate=0.5)
			gen_decode_layer_7_concat = tf.concat([gen_decode_layer_7_dropout, gen_encode_layer_1_conv], 3)


			# gen_decode_layer_8_output = (batch_size, 256, 256, output_pic_dim)
			gen_decode_layer_8_deconv = gen_decode_conv2d(relu(gen_decode_layer_7_concat), self.output_pic_dim, name='gen_decode_layer_8_deconv') 
			generator_output = tf.nn.tanh(gen_decode_layer_8_deconv)

			return generator_output


	# def descriptor(self, input_image, reuse=False):
	# 	with tf.variable_scope('des', reuse=reuse):

	# 		num_filter = 64

	# 		# ---------- descriptor part ----------
	# 		# descriptor_conv2d(input_image, output_dimension (by how many filters), scope_name)
	# 		# input image = [batch_size, 256, 256, input_pic_dim]

	# 		# des_layer_0_conv = (batch_size, 128, 128, num_filter)
	# 		des_layer_0_conv = descriptor_conv2d(input_image, num_filter, name='des_layer_0_conv')

	# 		# des_layer_1_conv = (batch_size, 64, 64, num_filter*2)
	# 		des_layer_1_conv = descriptor_conv2d(leaky_relu(des_layer_0_conv), num_filter*2, name='des_layer_1_conv')
	# 		des_layer_1_batchnorm = self.descriptor_batchnorm_layer1(des_layer_1_conv)

	# 		# des_layer_2_conv = (batch_size, 32, 32, num_filter*4)
	# 		des_layer_2_conv = descriptor_conv2d(leaky_relu(des_layer_1_batchnorm), num_filter*4, name='des_layer_2_conv')
	# 		des_layer_2_batchnorm = self.descriptor_batchnorm_layer2(des_layer_2_conv)
			
	# 		# des_layer_3_conv = (batch_size, 16, 16, num_filter*8)
	# 		des_layer_3_conv = descriptor_conv2d(leaky_relu(des_layer_2_batchnorm), num_filter*8, name='des_layer_3_conv')
	# 		des_layer_3_batchnorm = self.descriptor_batchnorm_layer3(des_layer_3_conv)

	# 		# linearization the descriptor result
	# 		des_layer_3_reshape = tf.reshape(leaky_relu(des_layer_3_batchnorm), [self.batch_size, -1])
	# 		# des_layer_3_linearization = linearization(des_layer_3_reshape, 1, 'des_layer_3_linearization')
	# 		# print(des_layer_3_batchnorm.shape) # (1, 16, 16, 512)
	# 		# print(des_layer_3_reshape.shape) # (1, 131072)


	# 		# input image = [batch_size, 256, 256, input_pic_dim]

	# 		return des_layer_3_reshape


	def descriptor(self, input_image, reuse=False):
		with tf.variable_scope('des', reuse=reuse):

			conv1 = conv2d(input_image, 64, kernal=(5, 5), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
				name="conv1")

			conv2 = conv2d(conv1, 128, kernal=(3, 3), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
				name="conv2")

			conv3 = conv2d(conv2, 256, kernal=(3, 3), strides=(1, 1), padding="SAME", activate_fn=leaky_relu,
				name="conv3")

			# conv3_reshape = tf.reshape(conv3, [self.batch_size, 1, 1, -1])
			# print(conv3_reshape.shape)
			# conv3_reshape.get_shape()[-1] = (1, 1, 1, 1048576)

			fc = fully_connected(conv3, 100, name="fc")

			return fc

	def langevin_dynamics_descriptor(self, input_image_arg):

		def cond(i, input_image):
			return tf.less(i, self.descriptor_langevin_steps)

		def body(i, input_image):
			noise = tf.random_normal(shape=[1, self.image_size, self.image_size, 3], name='noise')
			descripted_input_image = self.descriptor(input_image, reuse=True)

			print("descripted_input_image:",descripted_input_image.shape)

			grad = tf.gradients(descripted_input_image, input_image, name='grad_des')[0]
			input_image = input_image - 0.5 * self.descriptor_step_size * self.descriptor_step_size * (input_image / self.sigma1 / self.sigma1 - grad) + self.descriptor_step_size * noise
			return tf.add(i, 1), input_image

		with tf.name_scope("langevin_dynamics_descriptor"):
			i = tf.constant(0)
			i, input_image = tf.while_loop(cond, body, [i, input_image_arg])
			return input_image




	def save(self, checkpoint_dir, step):
		model_name = "pix2pix.model"
		model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
			os.path.join(checkpoint_dir, model_name),
			global_step=step)





