import tensorflow as tf
import numpy as np
import time
import datetime
import os

from glob import glob
from six.moves import xrange

# --------- self define function ---------
# ops: layers structure
from ops import *

# utils: for loading data, model
from utils import *

class Coop_pix2pix(object):
	def __init__(self, sess, 
				epoch = 1000, 
				batch_size = 1,
				picture_amount = 99999,
				image_size = 256, output_size = 256,
				input_pic_dim = 3, output_pic_dim = 3,	
				langevin_revision_steps = 1, 
				langevin_step_size = 0.002,
				descriptor_learning_rate = 0.01,
				generator_learning_rate = 0.0001,
				dataset_name='facades', dataset_dir ='./test_datasets', 
				output_dir='./output_dir', checkpoint_dir='./checkpoint_dir', log_dir='./log_dir'):
		"""
		args:
			sess: tensorflow session
			batch_size: how many pic in one group(batch), iteration(num_batch) = picture_amount/batch_size
			input_pic_dim: input picture dimension : colorful = 3, grayscale = 1
			output_pic_dim: output picture dimension : colorful = 3, grayscale = 1 
			langevin_revision_steps = langevin revision steps
			descriptor_learning_rate = descriptor learning rate
			generator_learning_rate = generator learning rate

		"""


		self.sess = sess
		self.epoch = epoch		
		self.batch_size = batch_size
		self.picture_amount = picture_amount
		self.image_size = image_size
		self.output_size = output_size
		self.input_pic_dim = input_pic_dim
		self.output_pic_dim = output_pic_dim

		# descriptor langevin steps
		self.langevin_revision_steps = langevin_revision_steps
		self.langevin_step_size = langevin_step_size

		# learning rate
		self.descriptor_learning_rate = descriptor_learning_rate 
		self.generator_learning_rate  = generator_learning_rate 
		# print(1e-5) # 0.00001
		

		self.dataset_dir = dataset_dir
		self.dataset_name = dataset_name

		self.output_dir = output_dir
		self.checkpoint_dir = checkpoint_dir
		self.log_dir = log_dir
		self.epoch_startpoint = 0

		self.sketcher_layer2_batchnorm = batch_norm(name='sketcher_layer_2_batchnorm')
		self.sketcher_layer3_batchnorm = batch_norm(name='sketcher_layer_3_batchnorm')
		self.sketcher_layer4_batchnorm = batch_norm(name='sketcher_layer_4_batchnorm')
		self.sketcher_layer5_batchnorm = batch_norm(name='sketcher_layer_5_batchnorm')
		self.sketcher_layer6_batchnorm = batch_norm(name='sketcher_layer_6_batchnorm')
		self.sketcher_layer7_batchnorm = batch_norm(name='sketcher_layer_7_batchnorm')
		self.sketcher_layer8_batchnorm = batch_norm(name='sketcher_layer_8_batchnorm')

		self.gen_layer1_batchnorm = batch_norm(name='gen_layer_1_batchnorm')
		self.gen_layer2_batchnorm = batch_norm(name='gen_layer_2_batchnorm')
		self.gen_layer3_batchnorm = batch_norm(name='gen_layer_3_batchnorm')
		self.gen_layer4_batchnorm = batch_norm(name='gen_layer_4_batchnorm')
		self.gen_layer5_batchnorm = batch_norm(name='gen_layer_5_batchnorm')
		self.gen_layer6_batchnorm = batch_norm(name='gen_layer_6_batchnorm')
		self.gen_layer7_batchnorm = batch_norm(name='gen_layer_7_batchnorm')

		self.des_layer_1_batchnorm = batch_norm(name='des_layer_1_batchnorm')
		self.des_layer_2_batchnorm = batch_norm(name='des_layer_2_batchnorm')
		self.des_layer_3_batchnorm = batch_norm(name='des_layer_3_batchnorm')

		self.sigma1 = 0.016
		self.sigma2 = 0.3
		self.beta1 = 0.5

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

		self.input_sketched_A = tf.placeholder(tf.float32,
				[self.batch_size, 1, 1, 64*8],
				name='input_real_data_A')

	def build_model(self):

		# generator 
		self.sketched_A = self.sketcher(self.input_real_data_A, reuse = False)
		self.generated_B = self.generator(self.input_sketched_A, reuse = False)

		# descriptor
		described_real_data_B = self.descriptor(self.input_real_data_B, reuse=False)
		described_revised_B = self.descriptor(self.input_revised_B, reuse=True)

		# symbolic langevins
		self.des_langevin_revision_output = self.des_langevin_revision(self.input_revised_B)
		self.lang_1_output = self.lang_1(self.input_revised_B)
		self.lang_10_output = self.lang_10(self.input_revised_B)
		self.lang_30_output = self.lang_30(self.input_revised_B)
		self.lang_50_output = self.lang_50(self.input_revised_B)
		self.lang_100_output = self.lang_100(self.input_revised_B)
		self.lang_200_output = self.lang_200(self.input_revised_B)

		t_vars = tf.trainable_variables()
		self.des_vars = [var for var in t_vars if var.name.startswith('des')]
		self.gen_vars = [var for var in t_vars if var.name.startswith('gen')]

		# # descriptor variables
		# print("\n------  self.des_vars  ------\n")
		# for var in self.des_vars:
		# 	print(var)


		# # # generator variables
		# print("\n------  self.gen_vars  ------\n")
		# for var in self.gen_vars:
		# 	print(var)

		# print("")
		

		# descriptor loss functions
		self.des_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(described_revised_B, axis=0), tf.reduce_mean(described_real_data_B, axis=0)))

		self.des_optim = tf.train.AdamOptimizer(self.descriptor_learning_rate, beta1=self.beta1).minimize(self.des_loss, var_list=self.des_vars)


		# generator loss functions
		self.gen_loss = tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.input_revised_B - self.generated_B), axis=0))
		
		self.gen_optim = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=self.beta1).minimize(self.gen_loss, var_list=self.gen_vars)


		# Compute Mean square error(MSE) for generated data and real data
		self.mse_loss = tf.reduce_mean(
            tf.pow(tf.subtract(tf.reduce_mean(self.input_generated_B, axis=0), tf.reduce_mean(self.input_revised_B, axis=0)), 2))

		self.saver = tf.train.Saver(max_to_keep=50)	


	def train(self,sess):

		# build model
		self.build_model()

		# prepare training data
		training_data = glob('{}/{}/train/*.jpg'.format(self.dataset_dir, self.dataset_name))

		# iteration(num_batch) = picture_amount/batch_size
		self.num_batch = min(len(training_data), self.picture_amount) // self.batch_size

		# initialize training
		sess.run(tf.global_variables_initializer())

		# sample picture initialize
		# sample_results = np.random.randn(num_batch, self.image_size, self.image_size, 3)

		# counter initialize
		counter = 1
		counter_end = self.epoch * self.num_batch  # 200 * num_batch 

		# load checkpoint
		if self.load(self.checkpoint_dir):
			print(" [v] Loading checkpoint success!!!")
		else:
			print(" [!] Loading checkpoint failed...")

		# start training	
		start_time = time.time()
		print("time: {} , Start training model......".format(str(datetime.timedelta(seconds=int(time.time()-start_time)))))
		

		# print("self.counter = ",self.counter)

		for epoch in xrange(self.epoch_startpoint, self.epoch): # how many epochs to train

			for index in xrange(self.num_batch): # num_batch
				# find picture list index*self.batch_size to (index+1)*self.batch_size (one batch)
				# if batch_size = 2, get one batch = batch[0], batch[1]
				batch_files = training_data[index*self.batch_size:(index+1)*self.batch_size] 

				# load data : list format, amount = one batch
				batch = [load_data(batch_file) for batch_file in batch_files]
				batch_images = np.array(batch).astype(np.float32)

				# data domain A and data domain B
				data_A = batch_images[:, :, :, : self.input_pic_dim] 
				data_B = batch_images[:, :, :, self.input_pic_dim:self.input_pic_dim+self.output_pic_dim] 

				# step G1: try to generate B domain(target domain) picture
				sketched_A = sess.run(self.sketched_A, feed_dict={self.input_real_data_A: data_A})
				generated_B = sess.run(self.generated_B, feed_dict={self.input_sketched_A: sketched_A})

				# step D1: descriptor try to revised image:"generated_B"
				revised_B = sess.run(self.des_langevin_revision_output, feed_dict={self.input_revised_B: generated_B})
				
				# step D2: update descriptor net
				descriptor_loss , _ = sess.run([self.des_loss, self.des_optim],
                                  		feed_dict={self.input_real_data_B: data_B, self.input_revised_B: revised_B})

				# print(descriptor_loss)

				# step G2: update generator net
				generator_loss , _ = sess.run([self.gen_loss, self.gen_optim],
                                  		feed_dict={self.input_revised_B: revised_B, self.input_sketched_A: sketched_A})	

				# Compute Mean square error(MSE) for generated data and revised data
				mse_loss = sess.run(self.mse_loss, feed_dict={self.input_revised_B: revised_B, self.input_generated_B: generated_B})


				# put picture in sample picture
				# sample_results[index : (index + 1)] = revised_B

				print("Epoch: [{:4d}] [{:4d}/{:4d}] time: {}, eta: {}, d_loss: {:.4f}, g_loss: {:.4f}, mse_loss: {:.4f}"
					.format(epoch, index, self.num_batch, 
						str(datetime.timedelta(seconds=int(time.time()-start_time))),
							str(datetime.timedelta(seconds=int((time.time()-start_time)*(counter_end-(self.epoch_startpoint*self.num_batch)-counter)/counter))),
								 descriptor_loss, generator_loss, mse_loss))

				# if need calculate time interval
				# start_time = time.time()

				# print("data_A shape = {}".format(self.data_A.shape)) # data_A shape = (1, 256, 256, 3)
				# print(generated_B.shape) # (1, 256, 256, 3)
				# print(revised_B.shape) # (1, 256, 256, 3)
				# print("data_B shape = {}".format(self.data_B.shape)) # data_B shape = (1, 256, 256, 3)


				if np.mod(counter, 100) == 1:
					lang_1_output = sess.run(self.lang_1_output, feed_dict={self.input_revised_B: generated_B})
					lang_10_output = sess.run(self.lang_10_output, feed_dict={self.input_revised_B: generated_B})
					lang_30_output = sess.run(self.lang_30_output, feed_dict={self.input_revised_B: generated_B})
					lang_50_output = sess.run(self.lang_50_output, feed_dict={self.input_revised_B: generated_B})
					lang_100_output = sess.run(self.lang_100_output, feed_dict={self.input_revised_B: generated_B})
					lang_200_output = sess.run(self.lang_200_output, feed_dict={self.input_revised_B: generated_B})

					save_images(data_A, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_01_input_data_A.png'.format(self.output_dir, epoch, index))
					save_images(data_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_02_input_data_B.png'.format(self.output_dir, epoch, index))
					save_images(generated_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_03_output_generator.png'.format(self.output_dir, epoch, index))
					save_images(revised_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_04_output_descriptor.png'.format(self.output_dir, epoch, index))
					save_images(lang_1_output, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_05_lang_1.png'.format(self.output_dir, epoch, index))
					save_images(lang_10_output, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_06_lang_10.png'.format(self.output_dir, epoch, index))
					save_images(lang_30_output, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_07_lang_30.png'.format(self.output_dir, epoch, index))
					save_images(lang_50_output, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_08_lang_50.png'.format(self.output_dir, epoch, index))
					save_images(lang_100_output, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_09_lang_100.png'.format(self.output_dir, epoch, index))
					save_images(lang_200_output, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_10_lang_200.png'.format(self.output_dir, epoch, index))
					

				counter += 1

			self.save(self.checkpoint_dir, epoch)

			
			# # print("time: {:.4f} , Epoch: {} ".format(time.time() - start_time, epoch))
			# print('Epoch #{:d}, avg.descriptor loss: {:.4f}, avg.generator loss: {:.4f}, avg.L2 distance: {:4.4f}, '
			# 	'time: {:.2f}s'.format(epoch, np.mean(des_loss_avg), np.mean(gen_loss_avg), np.mean(mse_avg), time.time() - start_time))


	def sketcher(self, input_image, reuse=False):
		with tf.variable_scope("sketch", reuse=reuse):

			num_filter = 64

			# sketcher_layer_1_output = (batch_size, 128, 128, num_filter)
			sketcher_layer_1_conv = sketcher_conv2d(input_image, num_filter, name='sketcher_layer_1_conv') 

			# sketcher_layer_2_output = (batch_size, 64, 64, num_filter*2)
			sketcher_layer_2_conv = sketcher_conv2d(leaky_relu(sketcher_layer_1_conv), num_filter*2, name='sketcher_layer_2_conv') 
			sketcher_layer_2_batchnorm = self.sketcher_layer2_batchnorm(sketcher_layer_2_conv)
			
			# sketcher_layer_3_output = (batch_size, 32, 32, num_filter*4)
			sketcher_layer_3_conv = sketcher_conv2d(leaky_relu(sketcher_layer_2_batchnorm), num_filter*4, name='sketcher_layer_3_conv')
			sketcher_layer_3_batchnorm = self.sketcher_layer3_batchnorm(sketcher_layer_3_conv)

			# sketcher_layer_4_output = (batch_size, 16, 16, num_filter*8)
			sketcher_layer_4_conv = sketcher_conv2d(leaky_relu(sketcher_layer_3_batchnorm), num_filter*8, name='sketcher_layer_4_conv') 
			sketcher_layer_4_batchnorm = self.sketcher_layer4_batchnorm(sketcher_layer_4_conv)

			# sketcher_layer_5_output = (batch_size, 8, 8, num_filter*8)
			sketcher_layer_5_conv = sketcher_conv2d(leaky_relu(sketcher_layer_4_batchnorm), num_filter*8, name='sketcher_layer_5_conv') 
			sketcher_layer_5_batchnorm = self.sketcher_layer5_batchnorm(sketcher_layer_5_conv)

			# sketcher_layer_6_output = (batch_size, 4, 4, num_filter*8)
			sketcher_layer_6_conv = sketcher_conv2d(leaky_relu(sketcher_layer_5_batchnorm), num_filter*8, name='sketcher_layer_6_conv') 
			sketcher_layer_6_batchnorm = self.sketcher_layer6_batchnorm(sketcher_layer_6_conv)

			# sketcher_layer_7_output = (batch_size, 2, 2, num_filter*8)
			sketcher_layer_7_conv = sketcher_conv2d(leaky_relu(sketcher_layer_6_batchnorm), num_filter*8, name='sketcher_layer_7_conv') 
			sketcher_layer_7_batchnorm = self.sketcher_layer7_batchnorm(sketcher_layer_7_conv)

			# sketcher_layer_8_output = (batch_size, 1, 1, num_filter*8)
			sketcher_layer_8_conv = sketcher_conv2d(leaky_relu(sketcher_layer_7_batchnorm), num_filter*8, name='sketcher_layer_8_conv') 
			sketcher_layer_8_batchnorm = self.sketcher_layer8_batchnorm(sketcher_layer_8_conv)
			sketcher_output = relu(sketcher_layer_8_batchnorm)

			return sketcher_output



	def generator(self, input_image, reuse=False):
		with tf.variable_scope("gen", reuse=reuse):

			num_filter = 64

			# gen_layer_1_output = (batch_size, 2, 2, num_filter*8*2)
			gen_layer_1_deconv = gen_conv2d(input_image, num_filter*8, name='gen_layer_1_deconv') 
			gen_layer_1_batchnorm = self.gen_layer1_batchnorm(gen_layer_1_deconv)
			gen_layer_1_dropout = tf.nn.dropout(gen_layer_1_batchnorm, rate=0.5)
			# gen_layer_1_concat = tf.concat([gen_layer_1_dropout, sketcher_layer_7_batchnorm], 3)

			# gen_layer_2_output = (batch_size, 4, 4, num_filter*8*2)
			gen_layer_2_deconv = gen_conv2d(relu(gen_layer_1_dropout), num_filter*8, name='gen_layer_2_deconv') 
			gen_layer_2_batchnorm = self.gen_layer2_batchnorm(gen_layer_2_deconv)
			gen_layer_2_dropout = tf.nn.dropout(gen_layer_2_batchnorm, rate=0.5)
			# gen_layer_2_concat = tf.concat([gen_layer_2_dropout, sketcher_layer_6_batchnorm], 3)

			# gen_layer_3_output = (batch_size, 8, 8, num_filter*8*2)
			gen_layer_3_deconv = gen_conv2d(relu(gen_layer_2_dropout), num_filter*8, name='gen_layer_3_deconv') 
			gen_layer_3_batchnorm = self.gen_layer3_batchnorm(gen_layer_3_deconv)
			gen_layer_3_dropout = tf.nn.dropout(gen_layer_3_batchnorm, rate=0.5)
			# gen_layer_3_concat = tf.concat([gen_layer_3_dropout, sketcher_layer_5_batchnorm], 3)

			# gen_layer_4_output = (batch_size, 16, 16, num_filter*8*2)
			gen_layer_4_deconv = gen_conv2d(relu(gen_layer_3_dropout), num_filter*8, name='gen_layer_4_deconv') 
			gen_layer_4_batchnorm = self.gen_layer4_batchnorm(gen_layer_4_deconv)
			gen_layer_4_dropout = tf.nn.dropout(gen_layer_4_batchnorm, rate=0.5)
			# gen_layer_4_concat = tf.concat([gen_layer_4_dropout, sketcher_layer_4_batchnorm], 3)

			# gen_layer_5_output = (batch_size, 32, 32, num_filter*4*2)
			gen_layer_5_deconv = gen_conv2d(relu(gen_layer_4_dropout), num_filter*4, name='gen_layer_5_deconv') 
			gen_layer_5_batchnorm = self.gen_layer5_batchnorm(gen_layer_5_deconv)
			gen_layer_5_dropout = tf.nn.dropout(gen_layer_5_batchnorm, rate=0.5)
			# gen_layer_5_concat = tf.concat([gen_layer_5_dropout, sketcher_layer_3_batchnorm], 3)

			# gen_layer_6_output = (batch_size, 64, 64, num_filter*2*2)
			gen_layer_6_deconv = gen_conv2d(relu(gen_layer_5_dropout), num_filter*2, name='gen_layer_6_deconv') 
			gen_layer_6_batchnorm = self.gen_layer6_batchnorm(gen_layer_6_deconv)
			gen_layer_6_dropout = tf.nn.dropout(gen_layer_6_batchnorm, rate=0.5)
			# gen_layer_6_concat = tf.concat([gen_layer_6_dropout, sketcher_layer_2_batchnorm], 3)

			# gen_layer_7_output = (batch_size, 128, 128, num_filter*1*2)
			gen_layer_7_deconv = gen_conv2d(relu(gen_layer_6_dropout), num_filter, name='gen_layer_7_deconv') 
			gen_layer_7_batchnorm = self.gen_layer7_batchnorm(gen_layer_7_deconv)
			gen_layer_7_dropout = tf.nn.dropout(gen_layer_7_batchnorm, rate=0.5)
			# gen_layer_7_concat = tf.concat([gen_layer_7_dropout, sketcher_layer_1_conv], 3)


			# gen_layer_8_output = (batch_size, 256, 256, output_pic_dim)
			gen_layer_8_deconv = gen_conv2d(relu(gen_layer_7_dropout), self.output_pic_dim, name='gen_layer_8_deconv') 
			generator_output = tf.nn.tanh(gen_layer_8_deconv)

			return generator_output


	def descriptor(self, input_image, reuse=False):
		with tf.variable_scope('des', reuse=reuse):

			# print("\n------  descriptor layers shape  ------\n")
			# print("input_image shape: {}".format(input_image.shape))

			num_filter = 64

			# ---------- descriptor part ----------
			# descriptor_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 256, 256, input_pic_dim]

			# des_layer_0_conv = (batch_size, 128, 128, num_filter)
			des_layer_0_conv = des_conv2d(input_image, num_filter, name='des_layer_0_conv')

			# des_layer_1_conv = (batch_size, 64, 64, num_filter*2)
			des_layer_1_conv = des_conv2d(leaky_relu(des_layer_0_conv), num_filter*2, name='des_layer_1_conv')
			des_layer_1_batchnorm = self.des_layer_1_batchnorm(des_layer_1_conv)

			# des_layer_2_conv = (batch_size, 32, 32, num_filter*4)
			des_layer_2_conv = des_conv2d(leaky_relu(des_layer_1_batchnorm), num_filter*4, name='des_layer_2_conv')
			des_layer_2_batchnorm = self.des_layer_2_batchnorm(des_layer_2_conv)
			
			# des_layer_3_conv = (batch_size, 16, 16, num_filter*8)
			des_layer_3_conv = des_conv2d(leaky_relu(des_layer_2_batchnorm), num_filter*8, name='des_layer_3_conv')
			des_layer_3_batchnorm = self.des_layer_3_batchnorm(des_layer_3_conv)

			# linearization the descriptor result
			# # print(des_layer_3_batchnorm.shape) # (1, 16, 16, 512)
			# # print(des_layer_3_reshape.shape) # (1, 131072)

			des_layer_4_fully_connected = des_fully_connected(leaky_relu(des_layer_3_batchnorm), 100, name="des_layer_4_fully_connected")

			return des_layer_4_fully_connected 

	def des_langevin_revision(self, input_image_arg):
		# print("input_image_arg.shape: ",input_image_arg.shape)
		# self.pic_list = []
		def cond(i, input_image):
			return tf.less(i, self.langevin_revision_steps)

		def body(i, input_image):
			# print("input_image.shape: ",input_image.shape)
			# save_images(input_image, [self.batch_size, 1],
			# 	'./{}/test.png'.format(self.output_dir))
			noise = tf.random_normal(shape=[1, self.image_size, self.image_size, 3], name='noise')
			descripted_input_image = self.descriptor(input_image, reuse=True)

			grad = tf.gradients(descripted_input_image, input_image, name='grad_des')[0]
			input_image = input_image - 0.5 * self.langevin_step_size * self.langevin_step_size * (input_image / self.sigma1 / self.sigma1 - grad) + self.langevin_step_size * noise
			# print("input_image.shape: ",input_image.shape)
			return tf.add(i, 1), input_image

		with tf.name_scope("des_langevin_revision"):
			i = tf.constant(0)
			i, input_image = tf.while_loop(cond, body, [i, input_image_arg])
			return input_image



	def save(self, checkpoint_dir, step):
		saver_name = "epoch"

		model_dir = "{}".format(self.dataset_name)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
					os.path.join(checkpoint_dir, saver_name),
					global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Loading checkpoint...")
		model_dir = "{}".format(self.dataset_name)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
			print(" [v] Found checkpoint name: {}".format(checkpoint_name))
			self.epoch_startpoint = int(checkpoint_name.split("epoch-", 1)[1])+1
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, checkpoint_name))
			return True
		else:
			return False

	# def langevin_revision_controller(self, input_image_arg, langevin_steps):
	# 	def cond(i, input_image):
	# 		return tf.less(i, langevin_steps)

	# 	def body(i, input_image):
	# 		noise = tf.random_normal(shape=[1, self.image_size, self.image_size, 3], name='noise')
	# 		descripted_input_image = self.descriptor(input_image, reuse=True)

	# 		grad = tf.gradients(descripted_input_image, input_image, name='grad_des')[0]
	# 		input_image = input_image - 0.5 * self.langevin_step_size * self.langevin_step_size * (input_image / self.sigma1 / self.sigma1 - grad) + self.langevin_step_size * noise
	# 		return tf.add(i, 1), input_image

	# 	with tf.name_scope("des_langevin_revision"):
	# 		i = tf.constant(0)
	# 		i, input_image = tf.while_loop(cond, body, [i, input_image_arg])
	# 		return input_image

	def lang_1(self, input_image_arg):
		def cond(i, input_image):
			return tf.less(i, 1)

		def body(i, input_image):
			noise = tf.random_normal(shape=[1, self.image_size, self.image_size, 3], name='noise')
			descripted_input_image = self.descriptor(input_image, reuse=True)

			grad = tf.gradients(descripted_input_image, input_image, name='grad_des')[0]
			input_image = input_image - 0.5 * self.langevin_step_size * self.langevin_step_size * (input_image / self.sigma1 / self.sigma1 - grad) + self.langevin_step_size * noise
			return tf.add(i, 1), input_image

		with tf.name_scope("des_langevin_revision"):
			i = tf.constant(0)
			i, input_image = tf.while_loop(cond, body, [i, input_image_arg])
			return input_image

	def lang_10(self, input_image_arg):
		def cond(i, input_image):
			return tf.less(i, 10)

		def body(i, input_image):
			noise = tf.random_normal(shape=[1, self.image_size, self.image_size, 3], name='noise')
			descripted_input_image = self.descriptor(input_image, reuse=True)

			grad = tf.gradients(descripted_input_image, input_image, name='grad_des')[0]
			input_image = input_image - 0.5 * self.langevin_step_size * self.langevin_step_size * (input_image / self.sigma1 / self.sigma1 - grad) + self.langevin_step_size * noise
			return tf.add(i, 1), input_image

		with tf.name_scope("des_langevin_revision"):
			i = tf.constant(0)
			i, input_image = tf.while_loop(cond, body, [i, input_image_arg])
			return input_image

	def lang_30(self, input_image_arg):
		def cond(i, input_image):
			return tf.less(i, 30)

		def body(i, input_image):
			noise = tf.random_normal(shape=[1, self.image_size, self.image_size, 3], name='noise')
			descripted_input_image = self.descriptor(input_image, reuse=True)

			grad = tf.gradients(descripted_input_image, input_image, name='grad_des')[0]
			input_image = input_image - 0.5 * self.langevin_step_size * self.langevin_step_size * (input_image / self.sigma1 / self.sigma1 - grad) + self.langevin_step_size * noise
			return tf.add(i, 1), input_image

		with tf.name_scope("des_langevin_revision"):
			i = tf.constant(0)
			i, input_image = tf.while_loop(cond, body, [i, input_image_arg])
			return input_image

	def lang_50(self, input_image_arg):
		def cond(i, input_image):
			return tf.less(i, 50)

		def body(i, input_image):
			noise = tf.random_normal(shape=[1, self.image_size, self.image_size, 3], name='noise')
			descripted_input_image = self.descriptor(input_image, reuse=True)

			grad = tf.gradients(descripted_input_image, input_image, name='grad_des')[0]
			input_image = input_image - 0.5 * self.langevin_step_size * self.langevin_step_size * (input_image / self.sigma1 / self.sigma1 - grad) + self.langevin_step_size * noise
			return tf.add(i, 1), input_image

		with tf.name_scope("des_langevin_revision"):
			i = tf.constant(0)
			i, input_image = tf.while_loop(cond, body, [i, input_image_arg])
			return input_image

	def lang_100(self, input_image_arg):
		def cond(i, input_image):
			return tf.less(i, 100)

		def body(i, input_image):
			noise = tf.random_normal(shape=[1, self.image_size, self.image_size, 3], name='noise')
			descripted_input_image = self.descriptor(input_image, reuse=True)

			grad = tf.gradients(descripted_input_image, input_image, name='grad_des')[0]
			input_image = input_image - 0.5 * self.langevin_step_size * self.langevin_step_size * (input_image / self.sigma1 / self.sigma1 - grad) + self.langevin_step_size * noise
			return tf.add(i, 1), input_image

		with tf.name_scope("des_langevin_revision"):
			i = tf.constant(0)
			i, input_image = tf.while_loop(cond, body, [i, input_image_arg])
			return input_image


	def lang_200(self, input_image_arg):
		def cond(i, input_image):
			return tf.less(i, 200)

		def body(i, input_image):
			noise = tf.random_normal(shape=[1, self.image_size, self.image_size, 3], name='noise')
			descripted_input_image = self.descriptor(input_image, reuse=True)

			grad = tf.gradients(descripted_input_image, input_image, name='grad_des')[0]
			input_image = input_image - 0.5 * self.langevin_step_size * self.langevin_step_size * (input_image / self.sigma1 / self.sigma1 - grad) + self.langevin_step_size * noise
			return tf.add(i, 1), input_image

		with tf.name_scope("des_langevin_revision"):
			i = tf.constant(0)
			i, input_image = tf.while_loop(cond, body, [i, input_image_arg])
			return input_image




