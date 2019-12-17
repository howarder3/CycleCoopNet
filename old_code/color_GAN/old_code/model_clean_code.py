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
				epoch = 1000, 
				batch_size = 1,
				picture_amount = 99999,
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


		self.gen_encode_layer2_batchnorm = batch_norm(name='gen_encode_layer_2_batchnorm')
		self.gen_encode_layer3_batchnorm = batch_norm(name='gen_encode_layer_3_batchnorm')
		self.gen_encode_layer4_batchnorm = batch_norm(name='gen_encode_layer_4_batchnorm')
		self.gen_encode_layer5_batchnorm = batch_norm(name='gen_encode_layer_5_batchnorm')
		self.gen_encode_layer6_batchnorm = batch_norm(name='gen_encode_layer_6_batchnorm')
		self.gen_encode_layer7_batchnorm = batch_norm(name='gen_encode_layer_7_batchnorm')
		self.gen_encode_layer8_batchnorm = batch_norm(name='gen_encode_layer_8_batchnorm')

		self.gen_decode_layer1_batchnorm = batch_norm(name='gen_decode_layer_1_batchnorm')
		self.gen_decode_layer2_batchnorm = batch_norm(name='gen_decode_layer_2_batchnorm')
		self.gen_decode_layer3_batchnorm = batch_norm(name='gen_decode_layer_3_batchnorm')
		self.gen_decode_layer4_batchnorm = batch_norm(name='gen_decode_layer_4_batchnorm')
		self.gen_decode_layer5_batchnorm = batch_norm(name='gen_decode_layer_5_batchnorm')
		self.gen_decode_layer6_batchnorm = batch_norm(name='gen_decode_layer_6_batchnorm')
		self.gen_decode_layer7_batchnorm = batch_norm(name='gen_decode_layer_7_batchnorm')

		self.des_layer_1_batchnorm = batch_norm(name='des_layer_1_batchnorm')
		self.des_layer_2_batchnorm = batch_norm(name='des_layer_2_batchnorm')
		self.des_layer_3_batchnorm = batch_norm(name='des_layer_3_batchnorm')

		# descriptor langevin steps
		self.descriptor_sample_steps = 10 

		self.descriptor_step_size = 0.002
		self.sigma1 = 0.016
		self.sigma2 = 0.3
		self.beta1 = 0.5

		# learning rate
		self.descriptor_learning_rate = 0.007 # 1e-6 # 0.01 # 0.001 # 1e-6 # 0.01 # 0.007
		self.generator_learning_rate  = 0.0001 # 1e-5 # 0.0001 # 1e-4 # 0.0001 # 0.0001
		# print(1e-5) # 0.00001

		self.input_revised_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_revised_B')
		self.input_generated_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_generated_B')
		self.input_real_data_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_real_data_B')
		self.input_real_data_A = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_real_data_A')


	def build_model(self):
		# self.input_data = tf.placeholder(tf.float32,
		# 		[self.batch_size, self.image_size, self.image_size, self.input_pic_dim+self.output_pic_dim],
		# 		name='input_A_and_B_images')

		# # data domain A and data domain B
		# self.data_A = self.input_data[:, :, :, :self.input_pic_dim]
		# self.data_B = self.input_data[:, :, :, self.input_pic_dim:self.input_pic_dim+self.output_pic_dim]

		# print("data_A shape = {}".format(self.data_A.shape)) # data_A shape = (1, 256, 256, 3)
		# print("data_B shape = {}".format(self.data_B.shape)) # data_B shape = (1, 256, 256, 3)


		self.generated_B = self.generator(self.input_real_data_A, reuse = False)

		# descripted_real_data_B = self.descriptor(self.input_real_data_B, reuse = False)
		# descripted_generated_B = self.descriptor(self.input_generated_B, reuse = True)
		# descripted_revised_B = self.descriptor(self.input_revised_B, reuse = True)

		described_real_data_B = self.descriptor(self.input_real_data_B, reuse=False)
		described_revised_B = self.descriptor(self.input_revised_B, reuse=True)
		descripted_generated_B = self.descriptor(self.input_generated_B, reuse=True)


		# symbolic langevins
		self.langevin_descriptor = self.langevin_dynamics_descriptor(self.input_revised_B)

		t_vars = tf.trainable_variables()
		self.des_vars = [var for var in t_vars if var.name.startswith('des')]
		self.gen_vars = [var for var in t_vars if var.name.startswith('gen')]

		# descriptor variables
		print("\n------  self.des_vars  ------\n")
		for var in self.des_vars:
			print(var)


		# # generator variables
		print("\n------  self.gen_vars  ------\n")
		for var in self.gen_vars:
			print(var)

		print("")


		# descriptor loss functions
		self.des_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(described_revised_B, axis=0), tf.reduce_mean(described_real_data_B, axis=0)))


		# self.d_loss = tf.reduce_mean(tf.abs(descripted_real_data_B - descripted_revised_B))
		# # self.d_loss = tf.reduce_mean(tf.abs(self.input_real_data_B - self.input_revised_B))
		## self.des_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(descripted_real_data_B, axis=0), tf.reduce_mean(descripted_revised_B, axis=0)))
		# self.d_loss = self.L1_lambda * tf.reduce_mean(tf.abs(tf.subtract(descripted_real_data_B, descripted_revised_B)))
		# # self.d_loss = tf.reduce_mean(tf.subtract(descripted_real_data_B, descripted_revised_B))
		# self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        # self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

		self.des_optim = tf.train.AdamOptimizer(self.descriptor_learning_rate, beta1=self.beta1).minimize(self.des_loss, var_list=self.des_vars)





		# generator loss functions
		# self.g_loss = tf.reduce_mean(tf.abs(tf.subtract(self.input_revised_B, self.generated_B)))
		# self.gen_loss = tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.input_revised_B - self.generated_B), axis=0))
		## self.gen_loss = tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.input_real_data_B - self.generated_B), axis=0))
		# self.g_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(self.input_real_data_B, axis=0), tf.reduce_mean(self.generated_B, axis=0)))
		# self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
        #                 + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

		self.gen_loss = tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.input_revised_B - self.generated_B), axis=0))
		
		self.gen_optim = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=self.beta1).minimize(self.gen_loss, var_list=self.gen_vars)



		# Compute Mean square error(MSE) for generated data and real data
		# self.mse_loss = tf.reduce_mean(
		# 	tf.pow(tf.subtract(tf.reduce_mean(self.input_real_data_B, axis=0), tf.reduce_mean(self.generated_B, axis=0)), 2))
		self.mse_loss = tf.reduce_mean(
            tf.pow(tf.subtract(tf.reduce_mean(self.input_generated_B, axis=0), tf.reduce_mean(self.input_revised_B, axis=0)), 2))

		self.saver = tf.train.Saver()	


	def train(self,sess):

		# start learning
		start_time = time.time()
		print("time: {:.4f} , Learning start!!! ".format(0))		

		# build model
		self.build_model()

		# prepare training data
		training_data = glob('{}/{}/train/*.jpg'.format(self.dataset_dir, self.dataset_name))

		# iteration(num_batch) = picture_amount/batch_size
		num_batch = min(len(training_data), self.picture_amount) // self.batch_size

		# initialize training
		sess.run(tf.global_variables_initializer())

		# sample picture initialize
		# sample_results = np.random.randn(num_batch, self.image_size, self.image_size, 3)

		# counter initialize
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
				data_B = batch_images[:, :, :, :self.input_pic_dim]
				data_A = batch_images[:, :, :, self.input_pic_dim:self.input_pic_dim+self.output_pic_dim]


				# step G1: try to generate B domain(target domain) picture
				generated_B = sess.run(self.generated_B, feed_dict={self.input_real_data_A: data_A})
				# print(generated_B.shape) # (1, 256, 256, 3)

				# step D1: descriptor try to revised image:"generated_B"
				# revised_B = sess.run(self.langevin_descriptor, feed_dict={self.input_generated_B: generated_B})

				revised_B = sess.run(self.langevin_descriptor, feed_dict={self.input_revised_B: generated_B})

				# print(generated_B.shape) # (1, 256, 256, 3)
				# print(revised_B.shape) # (1, 256, 256, 3)

				# step D2: update descriptor network
				descriptor_loss , _ = sess.run([self.des_loss, self.des_optim],
                                  		feed_dict={self.input_real_data_B: data_B, self.input_revised_B: revised_B})


				# # step G2: update generator network
				generator_loss , _ = sess.run([self.gen_loss, self.gen_optim],
                                  		feed_dict={self.input_revised_B: revised_B, self.input_real_data_A: data_A})


				# Compute Mean square error(MSE) for generated data and real data
				mse_loss = sess.run(self.mse_loss, feed_dict={self.input_revised_B: revised_B, self.input_generated_B: generated_B})


				# put picture in sample picture
				# sample_results[index : (index + 1)] = revised_B


				print("Epoch: [{:4d}] [{:4d}/{:4d}] time: {:.4f}, d_loss: {:.4f}, g_loss: {:.4f}, mse_loss: {:.4f}"
					.format(epoch, index, num_batch, time.time() - start_time, descriptor_loss, generator_loss, mse_loss))

				# if need calculate time interval
				# start_time = time.time()

				# if index == 0:
				if np.mod(counter, 10) == 0:
					save_images(data_A, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_01_input_data_A.png'.format(self.output_dir, epoch, index))
					save_images(generated_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_02_output_generator.png'.format(self.output_dir, epoch, index))
					save_images(revised_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_03_output_descriptor.png'.format(self.output_dir, epoch, index))
					save_images(data_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_04_input_data_B.png'.format(self.output_dir, epoch, index))

				if np.mod(counter, 500) == 0:
					self.save(self.checkpoint_dir, counter)

				counter += 1

	def generator(self, input_image, reuse=False):
		with tf.variable_scope("gen", reuse=reuse):

			print("\n------  generator layers shape  ------\n")
			print("input_image shape: {}".format(input_image.shape))


			num_filter = 64

			# ---------- encoder part ----------
			# gen_encode_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 256, 256, input_pic_dim]

			# gen_encode_layer_1_output = (batch_size, 128, 128, num_filter)
			gen_encode_layer_1_conv = gen_encode_conv2d(input_image, num_filter, name='gen_encode_layer_1_conv') 

			# gen_encode_layer_2_output = (batch_size, 64, 64, num_filter*2)
			gen_encode_layer_2_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_1_conv), num_filter*2, name='gen_encode_layer_2_conv') 
			gen_encode_layer_2_batchnorm = self.gen_encode_layer2_batchnorm(gen_encode_layer_2_conv)
			
			# gen_encode_layer_3_output = (batch_size, 32, 32, num_filter*4)
			gen_encode_layer_3_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_2_batchnorm), num_filter*4, name='gen_encode_layer_3_conv')
			gen_encode_layer_3_batchnorm = self.gen_encode_layer3_batchnorm(gen_encode_layer_3_conv)

			# gen_encode_layer_4_output = (batch_size, 16, 16, num_filter*8)
			gen_encode_layer_4_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_3_batchnorm), num_filter*8, name='gen_encode_layer_4_conv') 
			gen_encode_layer_4_batchnorm = self.gen_encode_layer4_batchnorm(gen_encode_layer_4_conv)

			# gen_encode_layer_5_output = (batch_size, 8, 8, num_filter*8)
			gen_encode_layer_5_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_4_batchnorm), num_filter*8, name='gen_encode_layer_5_conv') 
			gen_encode_layer_5_batchnorm = self.gen_encode_layer5_batchnorm(gen_encode_layer_5_conv)

			# gen_encode_layer_6_output = (batch_size, 4, 4, num_filter*8)
			gen_encode_layer_6_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_5_batchnorm), num_filter*8, name='gen_encode_layer_6_conv') 
			gen_encode_layer_6_batchnorm = self.gen_encode_layer6_batchnorm(gen_encode_layer_6_conv)

			# gen_encode_layer_7_output = (batch_size, 2, 2, num_filter*8)
			gen_encode_layer_7_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_6_batchnorm), num_filter*8, name='gen_encode_layer_7_conv') 
			gen_encode_layer_7_batchnorm = self.gen_encode_layer7_batchnorm(gen_encode_layer_7_conv)

			# gen_encode_layer_8_output = (batch_size, 1, 1, num_filter*8)
			gen_encode_layer_8_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_7_batchnorm), num_filter*8, name='gen_encode_layer_8_conv') 
			gen_encode_layer_8_batchnorm = self.gen_encode_layer8_batchnorm(gen_encode_layer_8_conv)

			# ---------- decoder part ----------
			# gen_decode_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 1, 1, num_filter*8]

			# gen_decode_layer_1_output = (batch_size, 2, 2, num_filter*8*2)
			gen_decode_layer_1_deconv = gen_decode_conv2d(relu(gen_encode_layer_8_batchnorm), num_filter*8, name='gen_decode_layer_1_deconv') 
			gen_decode_layer_1_batchnorm = self.gen_decode_layer1_batchnorm(gen_decode_layer_1_deconv)
			gen_decode_layer_1_dropout = tf.nn.dropout(gen_decode_layer_1_batchnorm, rate=0.5)
			gen_decode_layer_1_concat = tf.concat([gen_decode_layer_1_dropout, gen_encode_layer_7_batchnorm], 3)

			# gen_decode_layer_2_output = (batch_size, 4, 4, num_filter*8*2)
			gen_decode_layer_2_deconv = gen_decode_conv2d(relu(gen_decode_layer_1_concat), num_filter*8, name='gen_decode_layer_2_deconv') 
			gen_decode_layer_2_batchnorm = self.gen_decode_layer2_batchnorm(gen_decode_layer_2_deconv)
			gen_decode_layer_2_dropout = tf.nn.dropout(gen_decode_layer_2_batchnorm, rate=0.5)
			gen_decode_layer_2_concat = tf.concat([gen_decode_layer_2_dropout, gen_encode_layer_6_batchnorm], 3)

			# gen_decode_layer_3_output = (batch_size, 8, 8, num_filter*8*2)
			gen_decode_layer_3_deconv = gen_decode_conv2d(relu(gen_decode_layer_2_concat), num_filter*8, name='gen_decode_layer_3_deconv') 
			gen_decode_layer_3_batchnorm = self.gen_decode_layer3_batchnorm(gen_decode_layer_3_deconv)
			gen_decode_layer_3_dropout = tf.nn.dropout(gen_decode_layer_3_batchnorm, rate=0.5)
			gen_decode_layer_3_concat = tf.concat([gen_decode_layer_3_dropout, gen_encode_layer_5_batchnorm], 3)

			# gen_decode_layer_4_output = (batch_size, 16, 16, num_filter*8*2)
			gen_decode_layer_4_deconv = gen_decode_conv2d(relu(gen_decode_layer_3_concat), num_filter*8, name='gen_decode_layer_4_deconv') 
			gen_decode_layer_4_batchnorm = self.gen_decode_layer4_batchnorm(gen_decode_layer_4_deconv)
			gen_decode_layer_4_dropout = tf.nn.dropout(gen_decode_layer_4_batchnorm, rate=0.5)
			gen_decode_layer_4_concat = tf.concat([gen_decode_layer_4_dropout, gen_encode_layer_4_batchnorm], 3)

			# gen_decode_layer_5_output = (batch_size, 32, 32, num_filter*4*2)
			gen_decode_layer_5_deconv = gen_decode_conv2d(relu(gen_decode_layer_4_concat), num_filter*4, name='gen_decode_layer_5_deconv') 
			gen_decode_layer_5_batchnorm = self.gen_decode_layer5_batchnorm(gen_decode_layer_5_deconv)
			gen_decode_layer_5_dropout = tf.nn.dropout(gen_decode_layer_5_batchnorm, rate=0.5)
			gen_decode_layer_5_concat = tf.concat([gen_decode_layer_5_dropout, gen_encode_layer_3_batchnorm], 3)

			# gen_decode_layer_6_output = (batch_size, 64, 64, num_filter*2*2)
			gen_decode_layer_6_deconv = gen_decode_conv2d(relu(gen_decode_layer_5_concat), num_filter*2, name='gen_decode_layer_6_deconv') 
			gen_decode_layer_6_batchnorm = self.gen_decode_layer6_batchnorm(gen_decode_layer_6_deconv)
			gen_decode_layer_6_dropout = tf.nn.dropout(gen_decode_layer_6_batchnorm, rate=0.5)
			gen_decode_layer_6_concat = tf.concat([gen_decode_layer_6_dropout, gen_encode_layer_2_batchnorm], 3)

			# gen_decode_layer_7_output = (batch_size, 128, 128, num_filter*1*2)
			gen_decode_layer_7_deconv = gen_decode_conv2d(relu(gen_decode_layer_6_concat), num_filter, name='gen_decode_layer_7_deconv') 
			gen_decode_layer_7_batchnorm = self.gen_decode_layer7_batchnorm(gen_decode_layer_7_deconv)
			gen_decode_layer_7_dropout = tf.nn.dropout(gen_decode_layer_7_batchnorm, rate=0.5)
			gen_decode_layer_7_concat = tf.concat([gen_decode_layer_7_dropout, gen_encode_layer_1_conv], 3)


			# gen_decode_layer_8_output = (batch_size, 256, 256, output_pic_dim)
			gen_decode_layer_8_deconv = gen_decode_conv2d(relu(gen_decode_layer_7_concat), self.output_pic_dim, name='gen_decode_layer_8_deconv') 
			generator_output = tf.nn.tanh(gen_decode_layer_8_deconv)

			return generator_output

	def descriptor(self, input_image, reuse=False):
		with tf.variable_scope('des', reuse=reuse):

			print("\n------  descriptor layers shape  ------\n")
			print("input_image shape: {}".format(input_image.shape))

			num_filter = 64

			# ---------- descriptor part ----------
			# descriptor_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 256, 256, input_pic_dim]

			# des_layer_0_conv = (batch_size, 128, 128, num_filter)
			des_layer_0_conv = descriptor_conv2d(input_image, num_filter, name='des_layer_0_conv')

			# des_layer_1_conv = (batch_size, 64, 64, num_filter*2)
			des_layer_1_conv = descriptor_conv2d(leaky_relu(des_layer_0_conv), num_filter*2, name='des_layer_1_conv')
			des_layer_1_batchnorm = self.des_layer_1_batchnorm(des_layer_1_conv)

			# des_layer_2_conv = (batch_size, 32, 32, num_filter*4)
			des_layer_2_conv = descriptor_conv2d(leaky_relu(des_layer_1_batchnorm), num_filter*4, name='des_layer_2_conv')
			des_layer_2_batchnorm = self.des_layer_2_batchnorm(des_layer_2_conv)
			
			# des_layer_3_conv = (batch_size, 16, 16, num_filter*8)
			des_layer_3_conv = descriptor_conv2d(leaky_relu(des_layer_2_batchnorm), num_filter*8, name='des_layer_3_conv')
			des_layer_3_batchnorm = self.des_layer_3_batchnorm(des_layer_3_conv)

			# linearization the descriptor result
			# des_layer_3_reshape = tf.reshape(leaky_relu(des_layer_3_batchnorm), [self.batch_size, -1])
			# des_layer_3_linearization = linearization(des_layer_3_reshape, 100, 'des_layer_3_linearization')
			# # print(des_layer_3_batchnorm.shape) # (1, 16, 16, 512)
			# # print(des_layer_3_reshape.shape) # (1, 131072)
			# print("des_layer_3_linearization: ",des_layer_3_linearization.shape)


			des_layer_3_fully_connected = fully_connected(leaky_relu(des_layer_3_batchnorm), 100, name="des_fully_connected")

			# input image = [batch_size, 256, 256, input_pic_dim]

			return des_layer_3_fully_connected # des_layer_3_linearization


	def langevin_dynamics_descriptor(self, input_image_arg):

		def cond(i, input_image):
			return tf.less(i, self.descriptor_sample_steps)

		def body(i, input_image):
			noise = tf.random_normal(shape=[1, self.image_size, self.image_size, 3], name='noise')
			descripted_input_image = self.descriptor(input_image, reuse=True)

			# print("descripted_input_image:",descripted_input_image.shape)

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





