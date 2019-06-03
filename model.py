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
				cycle_consistency_loss_var = 10000,
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
				cycle_learning_rate = 0.01,
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

		self.A2B_gen_encode_layer2_batchnorm = batch_norm(name='A2B_gen_encode_layer_2_batchnorm')
		self.A2B_gen_encode_layer3_batchnorm = batch_norm(name='A2B_gen_encode_layer_3_batchnorm')
		self.A2B_gen_encode_layer4_batchnorm = batch_norm(name='A2B_gen_encode_layer_4_batchnorm')
		self.A2B_gen_encode_layer5_batchnorm = batch_norm(name='A2B_gen_encode_layer_5_batchnorm')
		self.A2B_gen_encode_layer6_batchnorm = batch_norm(name='A2B_gen_encode_layer_6_batchnorm')
		self.A2B_gen_encode_layer7_batchnorm = batch_norm(name='A2B_gen_encode_layer_7_batchnorm')
		self.A2B_gen_encode_layer8_batchnorm = batch_norm(name='A2B_gen_encode_layer_8_batchnorm')

		self.A2B_gen_decode_layer1_batchnorm = batch_norm(name='A2B_gen_decode_layer_1_batchnorm')
		self.A2B_gen_decode_layer2_batchnorm = batch_norm(name='A2B_gen_decode_layer_2_batchnorm')
		self.A2B_gen_decode_layer3_batchnorm = batch_norm(name='A2B_gen_decode_layer_3_batchnorm')
		self.A2B_gen_decode_layer4_batchnorm = batch_norm(name='A2B_gen_decode_layer_4_batchnorm')
		self.A2B_gen_decode_layer5_batchnorm = batch_norm(name='A2B_gen_decode_layer_5_batchnorm')
		self.A2B_gen_decode_layer6_batchnorm = batch_norm(name='A2B_gen_decode_layer_6_batchnorm')
		self.A2B_gen_decode_layer7_batchnorm = batch_norm(name='A2B_gen_decode_layer_7_batchnorm')

		self.B2A_gen_encode_layer2_batchnorm = batch_norm(name='B2A_gen_encode_layer_2_batchnorm')
		self.B2A_gen_encode_layer3_batchnorm = batch_norm(name='B2A_gen_encode_layer_3_batchnorm')
		self.B2A_gen_encode_layer4_batchnorm = batch_norm(name='B2A_gen_encode_layer_4_batchnorm')
		self.B2A_gen_encode_layer5_batchnorm = batch_norm(name='B2A_gen_encode_layer_5_batchnorm')
		self.B2A_gen_encode_layer6_batchnorm = batch_norm(name='B2A_gen_encode_layer_6_batchnorm')
		self.B2A_gen_encode_layer7_batchnorm = batch_norm(name='B2A_gen_encode_layer_7_batchnorm')
		self.B2A_gen_encode_layer8_batchnorm = batch_norm(name='B2A_gen_encode_layer_8_batchnorm')

		self.B2A_gen_decode_layer1_batchnorm = batch_norm(name='B2A_gen_decode_layer_1_batchnorm')
		self.B2A_gen_decode_layer2_batchnorm = batch_norm(name='B2A_gen_decode_layer_2_batchnorm')
		self.B2A_gen_decode_layer3_batchnorm = batch_norm(name='B2A_gen_decode_layer_3_batchnorm')
		self.B2A_gen_decode_layer4_batchnorm = batch_norm(name='B2A_gen_decode_layer_4_batchnorm')
		self.B2A_gen_decode_layer5_batchnorm = batch_norm(name='B2A_gen_decode_layer_5_batchnorm')
		self.B2A_gen_decode_layer6_batchnorm = batch_norm(name='B2A_gen_decode_layer_6_batchnorm')
		self.B2A_gen_decode_layer7_batchnorm = batch_norm(name='B2A_decode_layer_7_batchnorm')

		self.A2B_des_layer_1_batchnorm = batch_norm(name='A2B_des_layer_1_batchnorm')
		self.A2B_des_layer_2_batchnorm = batch_norm(name='A2B_des_layer_2_batchnorm')
		self.A2B_des_layer_3_batchnorm = batch_norm(name='A2B_des_layer_3_batchnorm')

		self.B2A_des_layer_1_batchnorm = batch_norm(name='B2A_des_layer_1_batchnorm')
		self.B2A_des_layer_2_batchnorm = batch_norm(name='B2A_des_layer_2_batchnorm')
		self.B2A_des_layer_3_batchnorm = batch_norm(name='B2A_des_layer_3_batchnorm')

		self.sigma1 = 0.016
		self.sigma2 = 0.3
		self.beta1 = 0.5
		self.cycle_learning_rate = cycle_learning_rate

		self.input_real_data_A = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_real_data_A')
		self.input_revised_A = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_revised_A')
		self.input_generated_A = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_generated_A')
		self.input_recovered_A = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_recovered_A')
		
		self.input_real_data_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_real_data_B')
		self.input_revised_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_revised_B')
		self.input_generated_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_generated_B')
		self.input_recovered_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_recovered_B')
		
		

	def build_model(self):

		# A2B generator 
		self.generated_A = self.A2B_generator(self.input_real_data_A, reuse = False)

		# A descriptor
		# A2B des : learning B features
		described_data_B = self.A2B_descriptor(self.input_real_data_B, reuse=False)

		described_revised_A = self.A2B_descriptor(self.input_revised_A, reuse=True)
		described_generated_A = self.A2B_descriptor(self.input_generated_A, reuse=True)

		# B2A generator
		self.generated_B = self.B2A_generator(self.input_real_data_B, reuse = False)

		# B descriptor
		# B2A des : learning A features
		described_data_A = self.B2A_descriptor(self.input_real_data_A, reuse=False)

		described_revised_B = self.B2A_descriptor(self.input_revised_B, reuse=True)
		described_generated_B = self.B2A_descriptor(self.input_generated_B, reuse=True)


		# symbolic langevins
		self.revised_A = self.A2B_des_langevin_revision(self.input_generated_A)
		self.revised_B = self.B2A_des_langevin_revision(self.input_generated_B)

		# self.lang_1_output = self.lang_1(self.input_revised_B)
		# self.lang_10_output = self.lang_10(self.input_revised_B)
		# self.lang_30_output = self.lang_30(self.input_revised_B)
		# self.lang_50_output = self.lang_50(self.input_revised_B)
		# self.lang_100_output = self.lang_100(self.input_revised_B)
		# self.lang_200_output = self.lang_200(self.input_revised_B)

		t_vars = tf.trainable_variables()
		self.A2B_des_vars = [var for var in t_vars if var.name.startswith('A2B_des')]
		self.B2A_des_vars = [var for var in t_vars if var.name.startswith('B2A_des')]
		self.A2B_gen_vars = [var for var in t_vars if var.name.startswith('A2B_gen')]
		self.B2A_gen_vars = [var for var in t_vars if var.name.startswith('B2A_gen')]

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
		# A2B des : learning B features
		self.A2B_des_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(described_revised_A, axis=0), tf.reduce_mean(described_data_B, axis=0)))

		self.A2B_des_optim = tf.train.AdamOptimizer(self.descriptor_learning_rate, beta1=self.beta1).minimize(self.A2B_des_loss, var_list=self.A2B_des_vars)

		# B2A des : learning A features
		self.B2A_des_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(described_revised_B, axis=0), tf.reduce_mean(described_data_A, axis=0)))

		self.B2A_des_optim = tf.train.AdamOptimizer(self.descriptor_learning_rate, beta1=self.beta1).minimize(self.B2A_des_loss, var_list=self.B2A_des_vars)


		# A2B generator loss functions
		self.A2B_gen_loss =  tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.input_revised_A - self.generated_A), axis=0)) # + self.cycle_consistency_loss_var * self.cycle_loss 
		
		self.A2B_gen_optim = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=self.beta1).minimize(self.A2B_gen_loss, var_list=self.A2B_gen_vars)

    	# B2A generator loss functions
		self.B2A_gen_loss = tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.input_revised_B - self.generated_B), axis=0)) # + self.cycle_consistency_loss_var * self.cycle_loss 

		self.B2A_gen_optim = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=self.beta1).minimize(self.B2A_gen_loss, var_list=self.B2A_gen_vars)


		# A2B cycle loss (A2 B2A recover part)
		self.A2B_cycle_loss = tf.reduce_mean(tf.abs(self.generated_B - self.input_real_data_A))

		self.A2B_cycle_optim = tf.train.AdamOptimizer(self.cycle_learning_rate, beta1=self.beta1).minimize(self.A2B_cycle_loss, var_list=self.B2A_gen_vars)

		#tf.reduce_mean(
			# tf.pow(tf.subtract(tf.reduce_mean(self.input_recovered_A, axis=0), tf.reduce_mean(self.input_real_data_A, axis=0)), 2))

		# B2A cycle loss (B2 A2B recover part)
		self.B2A_cycle_loss = tf.reduce_mean(tf.abs(self.generated_A - self.input_real_data_B))

		self.B2A_cycle_optim = tf.train.AdamOptimizer(self.cycle_learning_rate, beta1=self.beta1).minimize(self.B2A_cycle_loss, var_list=self.A2B_gen_vars)

		# tf.reduce_mean(
			# tf.pow(tf.subtract(tf.reduce_mean(self.input_recovered_B, axis=0), tf.reduce_mean(self.input_real_data_B, axis=0)), 2))


		self.avg_cycle_loss = (self.A2B_cycle_loss + self.B2A_cycle_loss)/2



		# Compute Mean square error(MSE) for generated data and real data
		# self.mse_loss = tf.reduce_mean(
  #           tf.pow(tf.subtract(tf.reduce_mean(self.input_generated_B, axis=0), tf.reduce_mean(self.input_revised_B, axis=0)), 2))

		# self.rec_optim = tf.train.AdamOptimizer(self.recover_learning_rate, beta1=self.beta1).minimize(self.rec_loss, var_list=self.rec_vars)

		self.saver = tf.train.Saver(max_to_keep=50)

	def train(self,sess):

		# build model
		self.build_model()

		# prepare training data
		# training_data = glob('{}/{}/train/*.jpg'.format(self.dataset_dir, self.dataset_name))

		# iteration(num_batch) = picture_amount/batch_size
		# self.num_batch = min(len(training_data), self.picture_amount) // self.batch_size

		# initialize training
		sess.run(tf.global_variables_initializer())

		# sample picture initialize
		# sample_results = np.random.randn(num_batch, self.image_size, self.image_size, 3)

		# counter initialize
		counter = 1

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

			# prepare training data
			training_dataset_A = glob('{}/{}/trainA/*.jpg'.format(self.dataset_dir, self.dataset_name))
			training_dataset_B = glob('{}/{}/trainB/*.jpg'.format(self.dataset_dir, self.dataset_name))

			# training_data = glob('{}/{}/trainA/*.jpg'.format(self.dataset_dir, self.dataset_name))

			# print("training_dataset_A: {} pictures.".format(len(training_dataset_A)))
			# print("training_dataset_B: {} pictures.".format(len(training_dataset_B)))
			
			np.random.shuffle(training_dataset_A)
			np.random.shuffle(training_dataset_B)
			self.num_batch = min(min(len(training_dataset_A), len(training_dataset_B)), self.picture_amount) // self.batch_size
			# print("num_batch: {} pictures.".format(self.num_batch))

			counter_end = self.epoch * self.num_batch  # 200 * num_batch 

			for index in xrange(self.num_batch): # num_batch

				batch_files = list(zip(training_dataset_A[index * self.batch_size:(index + 1) * self.batch_size],
										training_dataset_B[index * self.batch_size:(index + 1) * self.batch_size]))
				batch_images = [load_train_data(batch_file, 286, 256) for batch_file in batch_files]
				batch_images = np.array(batch_images).astype(np.float32)

				# print(batch_images.shape)

				data_B = batch_images[:, :, :, : self.input_pic_dim] 
				data_A = batch_images[:, :, :, self.input_pic_dim:self.input_pic_dim+self.output_pic_dim] 

				# print("data_A.shape: {} ".format(data_A.shape))
				# print("data_B.shape: {} ".format(data_B.shape))

				# # find picture list index*self.batch_size to (index+1)*self.batch_size (one batch)
				# # if batch_size = 2, get one batch = batch[0], batch[1]
				# batch_files = training_data[index*self.batch_size:(index+1)*self.batch_size] 

				# # load data : list format, amount = one batch
				# batch = [load_data(batch_file) for batch_file in batch_files]
				# batch_images = np.array(batch).astype(np.float32)

				# # data domain A and data domain B
				# data_A = batch_images[:, :, :, : self.input_pic_dim] 
				# data_B = batch_images[:, :, :, self.input_pic_dim:self.input_pic_dim+self.output_pic_dim] 
				
				# A2B

				# step G1: try to generate B domain picture
				generated_A = sess.run(self.generated_A, feed_dict={self.input_real_data_A: data_A})

				# step D1: descriptor try to revised image:"generated_B"
				revised_A = sess.run(self.revised_A, feed_dict={self.input_generated_A: generated_A})

				# step R1: recover origin picture
				recovered_A = sess.run(self.generated_B, feed_dict={self.input_real_data_B: generated_A})
				

				# B2A

				# step G1: try to generate A domain picture
				generated_B = sess.run(self.generated_B, feed_dict={self.input_real_data_B: data_B})

				# step D1: descriptor try to revised image:"generated_A"
				revised_B = sess.run(self.revised_B, feed_dict={self.input_generated_B: generated_B})

				# step R1: recover origin picture
				recovered_B = sess.run(self.generated_A, feed_dict={self.input_real_data_A: generated_B})



				# step D2: update descriptor net

				# A2B des : learning B features
				A2B_descriptor_loss , _ = sess.run([self.A2B_des_loss, self.A2B_des_optim],
                                  		feed_dict={self.input_revised_A: generated_A, self.input_real_data_B: data_B})

				# B2A des : learning A features
				B2A_descriptor_loss , _ = sess.run([self.B2A_des_loss, self.B2A_des_optim],
                                  		feed_dict={self.input_revised_B: generated_B, self.input_real_data_A: data_A})

				# print(descriptor_loss)

				# # step R2: update recover net
				# recover_loss , _ = sess.run([self.rec_loss, self.rec_optim],
    #                               		feed_dict={self.input_generated_B: generated_B, self.input_real_data_A: data_A})

				

				# step R2: A2B cycle loss (A, gen_A2B), (B, gen_B2A)
				A2B_cycle_loss , _ = sess.run([self.A2B_cycle_loss, self.A2B_cycle_optim],
									feed_dict={self.input_real_data_A: data_A, self.input_real_data_B: generated_A})

				B2A_cycle_loss , _ = sess.run([self.B2A_cycle_loss, self.B2A_cycle_optim],
									feed_dict={self.input_real_data_B: data_B, self.input_real_data_A: generated_B})
				


				# step G2: update A2B generator net
				A2B_generator_loss , _ = sess.run([self.A2B_gen_loss, self.A2B_gen_optim],
                                  		feed_dict={self.input_real_data_A: data_A, self.input_real_data_B: data_B,
                                  			self.input_revised_A: revised_A, self.input_recovered_A: recovered_A, self.input_recovered_B: recovered_B}) # self.input_revised_B: revised_B,

				# step G2: update B2A generator net
				B2A_generator_loss , _ = sess.run([self.B2A_gen_loss, self.B2A_gen_optim],
                                  		feed_dict={self.input_real_data_A: data_A, self.input_real_data_B: data_B, 
                                  			self.input_revised_B: revised_B, self.input_recovered_A: recovered_A, self.input_recovered_B: recovered_B}) # self.input_revised_B: revised_B,

<<<<<<< HEAD
=======
				# step G2: update generator net
				generator_loss , _ = sess.run([self.gen_loss, self.gen_optim],
                                  		feed_dict={self.input_generated_B: generated_B, self.input_real_data_A: data_A}) # self.input_revised_B: revised_B,
>>>>>>> parent of b9b3468... update




				# Compute Mean square error(MSE) for generated data and revised data
				# mse_loss = sess.run(self.mse_loss, feed_dict={self.input_revised_B: revised_B, self.input_generated_B: generated_B})


				# put picture in sample picture
				# sample_results[index : (index + 1)] = revised_B

				print("Epoch: [{:4d}] [{:4d}/{:4d}],   time: {},   eta: {}, \n A2B_d_loss: {:>15.4f}, A2B_g_loss: {:>12.4f}, A2B_cycle_loss: {:>8.4f}, \n B2A_d_loss: {:>15.4f}, B2A_g_loss: {:>12.4f}, B2A_cycle_loss: {:>8.4f}"
					.format(epoch, index, self.num_batch, 
						str(datetime.timedelta(seconds=int(time.time()-start_time))),
							str(datetime.timedelta(seconds=int((time.time()-start_time)*(counter_end-(self.epoch_startpoint*self.num_batch)-counter)/counter))),
								 A2B_descriptor_loss, A2B_generator_loss, A2B_cycle_loss, 
								 	B2A_descriptor_loss, B2A_generator_loss, B2A_cycle_loss))
				# if need calculate time interval
				# start_time = time.time()

				# print("data_A shape = {}".format(self.data_A.shape)) # data_A shape = (1, 256, 256, 3)
				# print(generated_B.shape) # (1, 256, 256, 3)
				# print(revised_B.shape) # (1, 256, 256, 3)
				# print("data_B shape = {}".format(self.data_B.shape)) # data_B shape = (1, 256, 256, 3)


				if np.mod(counter, 10) == 1:
					# lang_1_output = sess.run(self.lang_1_output, feed_dict={self.input_revised_B: generated_B})
					# lang_10_output = sess.run(self.lang_10_output, feed_dict={self.input_revised_B: generated_B})
					# lang_30_output = sess.run(self.lang_30_output, feed_dict={self.input_revised_B: generated_B})
					# lang_50_output = sess.run(self.lang_50_output, feed_dict={self.input_revised_B: generated_B})
					# lang_100_output = sess.run(self.lang_100_output, feed_dict={self.input_revised_B: generated_B})
					# lang_200_output = sess.run(self.lang_200_output, feed_dict={self.input_revised_B: generated_B})

					save_images(data_A, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_01_A2B_real_A.png'.format(self.output_dir, epoch, index))
					save_images(data_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_02_A2B_real_B.png'.format(self.output_dir, epoch, index))
					save_images(generated_A, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_03_A2B_gen_A.png'.format(self.output_dir, epoch, index))
					save_images(revised_A, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_04_A2B_revise_A.png'.format(self.output_dir, epoch, index))
					save_images(recovered_A, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_05_A2B_recover_A.png'.format(self.output_dir, epoch, index))

					
					save_images(data_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_11_B2A_real_B.png'.format(self.output_dir, epoch, index))
					save_images(data_A, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_12_B2A_real_A.png'.format(self.output_dir, epoch, index))
					save_images(generated_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_13_B2A_gen_B.png'.format(self.output_dir, epoch, index))
					save_images(revised_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_14_B2A_revise_B.png'.format(self.output_dir, epoch, index))
					save_images(recovered_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_15_B2A_recover_B.png'.format(self.output_dir, epoch, index))

					# save_images(lang_1_output, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_06_lang_001.png'.format(self.output_dir, epoch, index))
					# save_images(lang_10_output, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_06_lang_010.png'.format(self.output_dir, epoch, index))
					# save_images(lang_30_output, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_06_lang_030.png'.format(self.output_dir, epoch, index))
					# save_images(lang_50_output, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_06_lang_050.png'.format(self.output_dir, epoch, index))
					# save_images(lang_100_output, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_06_lang_100.png'.format(self.output_dir, epoch, index))
					# save_images(lang_200_output, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_06_lang_200.png'.format(self.output_dir, epoch, index))
					

				counter += 1

			self.save(self.checkpoint_dir, epoch)

			
			# # print("time: {:.4f} , Epoch: {} ".format(time.time() - start_time, epoch))
			# print('Epoch #{:d}, avg.descriptor loss: {:.4f}, avg.generator loss: {:.4f}, avg.L2 distance: {:4.4f}, '
			# 	'time: {:.2f}s'.format(epoch, np.mean(des_loss_avg), np.mean(gen_loss_avg), np.mean(mse_avg), time.time() - start_time))



	# def generator(self, input_image, reuse=False):
	# 	with tf.variable_scope("gen", reuse=reuse):

	# 		# print("\n------  generator layers shape  ------\n")
	# 		# print("input_image shape: {}".format(input_image.shape))


	# 		num_filter = 64

	# 		# ---------- encoder part ----------
	# 		# gen_encode_conv2d(input_image, output_dimension (by how many filters), scope_name)
	# 		# input image = [batch_size, 256, 256, input_pic_dim]

	# 		# gen_encode_layer_1_output = (batch_size, 128, 128, num_filter)
	# 		gen_encode_layer_1_conv = gen_encode_conv2d(input_image, num_filter, name='gen_encode_layer_1_conv') 

	# 		# gen_encode_layer_2_output = (batch_size, 64, 64, num_filter*2)
	# 		gen_encode_layer_2_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_1_conv), num_filter*2, name='gen_encode_layer_2_conv') 
	# 		gen_encode_layer_2_batchnorm = self.gen_encode_layer2_batchnorm(gen_encode_layer_2_conv)
			
	# 		# gen_encode_layer_3_output = (batch_size, 32, 32, num_filter*4)
	# 		gen_encode_layer_3_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_2_batchnorm), num_filter*4, name='gen_encode_layer_3_conv')
	# 		gen_encode_layer_3_batchnorm = self.gen_encode_layer3_batchnorm(gen_encode_layer_3_conv)

	# 		# gen_encode_layer_4_output = (batch_size, 16, 16, num_filter*8)
	# 		gen_encode_layer_4_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_3_batchnorm), num_filter*8, name='gen_encode_layer_4_conv') 
	# 		gen_encode_layer_4_batchnorm = self.gen_encode_layer4_batchnorm(gen_encode_layer_4_conv)

	# 		# gen_encode_layer_5_output = (batch_size, 8, 8, num_filter*8)
	# 		gen_encode_layer_5_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_4_batchnorm), num_filter*8, name='gen_encode_layer_5_conv') 
	# 		gen_encode_layer_5_batchnorm = self.gen_encode_layer5_batchnorm(gen_encode_layer_5_conv)

	# 		# gen_encode_layer_6_output = (batch_size, 4, 4, num_filter*8)
	# 		gen_encode_layer_6_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_5_batchnorm), num_filter*8, name='gen_encode_layer_6_conv') 
	# 		gen_encode_layer_6_batchnorm = self.gen_encode_layer6_batchnorm(gen_encode_layer_6_conv)

	# 		# gen_encode_layer_7_output = (batch_size, 2, 2, num_filter*8)
	# 		gen_encode_layer_7_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_6_batchnorm), num_filter*8, name='gen_encode_layer_7_conv') 
	# 		gen_encode_layer_7_batchnorm = self.gen_encode_layer7_batchnorm(gen_encode_layer_7_conv)

	# 		# gen_encode_layer_8_output = (batch_size, 1, 1, num_filter*8)
	# 		gen_encode_layer_8_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_7_batchnorm), num_filter*8, name='gen_encode_layer_8_conv') 
	# 		gen_encode_layer_8_batchnorm = self.gen_encode_layer8_batchnorm(gen_encode_layer_8_conv)

	# 		# ---------- decoder part ----------
	# 		# gen_decode_conv2d(input_image, output_dimension (by how many filters), scope_name)
	# 		# input image = [batch_size, 1, 1, num_filter*8]

	# 		# gen_decode_layer_1_output = (batch_size, 2, 2, num_filter*8*2)
	# 		gen_decode_layer_1_deconv = gen_decode_conv2d(relu(gen_encode_layer_8_batchnorm), num_filter*8, name='gen_decode_layer_1_deconv') 
	# 		gen_decode_layer_1_batchnorm = self.gen_decode_layer1_batchnorm(gen_decode_layer_1_deconv)
	# 		gen_decode_layer_1_dropout = tf.nn.dropout(gen_decode_layer_1_batchnorm, rate=0.5)
	# 		gen_decode_layer_1_concat = tf.concat([gen_decode_layer_1_dropout, gen_encode_layer_7_batchnorm], 3)

	# 		# gen_decode_layer_2_output = (batch_size, 4, 4, num_filter*8*2)
	# 		gen_decode_layer_2_deconv = gen_decode_conv2d(relu(gen_decode_layer_1_concat), num_filter*8, name='gen_decode_layer_2_deconv') 
	# 		gen_decode_layer_2_batchnorm = self.gen_decode_layer2_batchnorm(gen_decode_layer_2_deconv)
	# 		gen_decode_layer_2_dropout = tf.nn.dropout(gen_decode_layer_2_batchnorm, rate=0.5)
	# 		gen_decode_layer_2_concat = tf.concat([gen_decode_layer_2_dropout, gen_encode_layer_6_batchnorm], 3)

	# 		# gen_decode_layer_3_output = (batch_size, 8, 8, num_filter*8*2)
	# 		gen_decode_layer_3_deconv = gen_decode_conv2d(relu(gen_decode_layer_2_concat), num_filter*8, name='gen_decode_layer_3_deconv') 
	# 		gen_decode_layer_3_batchnorm = self.gen_decode_layer3_batchnorm(gen_decode_layer_3_deconv)
	# 		gen_decode_layer_3_dropout = tf.nn.dropout(gen_decode_layer_3_batchnorm, rate=0.5)
	# 		gen_decode_layer_3_concat = tf.concat([gen_decode_layer_3_dropout, gen_encode_layer_5_batchnorm], 3)

	# 		# gen_decode_layer_4_output = (batch_size, 16, 16, num_filter*8*2)
	# 		gen_decode_layer_4_deconv = gen_decode_conv2d(relu(gen_decode_layer_3_concat), num_filter*8, name='gen_decode_layer_4_deconv') 
	# 		gen_decode_layer_4_batchnorm = self.gen_decode_layer4_batchnorm(gen_decode_layer_4_deconv)
	# 		gen_decode_layer_4_dropout = tf.nn.dropout(gen_decode_layer_4_batchnorm, rate=0.5)
	# 		gen_decode_layer_4_concat = tf.concat([gen_decode_layer_4_dropout, gen_encode_layer_4_batchnorm], 3)

	# 		# gen_decode_layer_5_output = (batch_size, 32, 32, num_filter*4*2)
	# 		gen_decode_layer_5_deconv = gen_decode_conv2d(relu(gen_decode_layer_4_concat), num_filter*4, name='gen_decode_layer_5_deconv') 
	# 		gen_decode_layer_5_batchnorm = self.gen_decode_layer5_batchnorm(gen_decode_layer_5_deconv)
	# 		gen_decode_layer_5_dropout = tf.nn.dropout(gen_decode_layer_5_batchnorm, rate=0.5)
	# 		gen_decode_layer_5_concat = tf.concat([gen_decode_layer_5_dropout, gen_encode_layer_3_batchnorm], 3)

	# 		# gen_decode_layer_6_output = (batch_size, 64, 64, num_filter*2*2)
	# 		gen_decode_layer_6_deconv = gen_decode_conv2d(relu(gen_decode_layer_5_concat), num_filter*2, name='gen_decode_layer_6_deconv') 
	# 		gen_decode_layer_6_batchnorm = self.gen_decode_layer6_batchnorm(gen_decode_layer_6_deconv)
	# 		gen_decode_layer_6_dropout = tf.nn.dropout(gen_decode_layer_6_batchnorm, rate=0.5)
	# 		gen_decode_layer_6_concat = tf.concat([gen_decode_layer_6_dropout, gen_encode_layer_2_batchnorm], 3)

	# 		# gen_decode_layer_7_output = (batch_size, 128, 128, num_filter*1*2)
	# 		gen_decode_layer_7_deconv = gen_decode_conv2d(relu(gen_decode_layer_6_concat), num_filter, name='gen_decode_layer_7_deconv') 
	# 		gen_decode_layer_7_batchnorm = self.gen_decode_layer7_batchnorm(gen_decode_layer_7_deconv)
	# 		gen_decode_layer_7_dropout = tf.nn.dropout(gen_decode_layer_7_batchnorm, rate=0.5)
	# 		gen_decode_layer_7_concat = tf.concat([gen_decode_layer_7_dropout, gen_encode_layer_1_conv], 3)


	# 		# gen_decode_layer_8_output = (batch_size, 256, 256, output_pic_dim)
	# 		gen_decode_layer_8_deconv = gen_decode_conv2d(relu(gen_decode_layer_7_concat), self.output_pic_dim, name='gen_decode_layer_8_deconv') 
	# 		generator_output = tf.nn.tanh(gen_decode_layer_8_deconv)

	# 		return generator_output

	def A2B_generator(self, input_image, reuse=False):
		with tf.variable_scope("A2B_gen", reuse=reuse):

			# print("\n------  generator layers shape  ------\n")
			# print("input_image shape: {}".format(input_image.shape))


			num_filter = 64

			# ---------- encoder part ----------
			# gen_encode_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 256, 256, input_pic_dim]

			# gen_encode_layer_1_output = (batch_size, 128, 128, num_filter)
			gen_encode_layer_1_conv = gen_encode_conv2d(input_image, num_filter, name='gen_encode_layer_1_conv') 

			# gen_encode_layer_2_output = (batch_size, 64, 64, num_filter*2)
			gen_encode_layer_2_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_1_conv), num_filter*2, name='gen_encode_layer_2_conv') 
			gen_encode_layer_2_batchnorm = self.A2B_gen_encode_layer2_batchnorm(gen_encode_layer_2_conv)

			# gen_encode_layer_3_output = (batch_size, 32, 32, num_filter*4)
			gen_encode_layer_3_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_2_batchnorm), num_filter*4, name='gen_encode_layer_3_conv')
			gen_encode_layer_3_batchnorm = self.A2B_gen_encode_layer3_batchnorm(gen_encode_layer_3_conv)

			# gen_encode_layer_4_output = (batch_size, 16, 16, num_filter*8)
			gen_encode_layer_4_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_3_batchnorm), num_filter*8, name='gen_encode_layer_4_conv') 
			gen_encode_layer_4_batchnorm = self.A2B_gen_encode_layer4_batchnorm(gen_encode_layer_4_conv)

			# gen_encode_layer_5_output = (batch_size, 8, 8, num_filter*8)
			gen_encode_layer_5_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_4_batchnorm), num_filter*8, name='gen_encode_layer_5_conv') 
			gen_encode_layer_5_batchnorm = self.A2B_gen_encode_layer5_batchnorm(gen_encode_layer_5_conv)

			# gen_encode_layer_6_output = (batch_size, 4, 4, num_filter*8)
			gen_encode_layer_6_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_5_batchnorm), num_filter*8, name='gen_encode_layer_6_conv') 
			gen_encode_layer_6_batchnorm = self.A2B_gen_encode_layer6_batchnorm(gen_encode_layer_6_conv)

			# gen_encode_layer_7_output = (batch_size, 2, 2, num_filter*8)
			gen_encode_layer_7_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_6_batchnorm), num_filter*8, name='gen_encode_layer_7_conv') 
			gen_encode_layer_7_batchnorm = self.A2B_gen_encode_layer7_batchnorm(gen_encode_layer_7_conv)

			# gen_encode_layer_8_output = (batch_size, 1, 1, num_filter*8)
			gen_encode_layer_8_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_7_batchnorm), num_filter*8, name='gen_encode_layer_8_conv') 
			gen_encode_layer_8_batchnorm = self.A2B_gen_encode_layer8_batchnorm(gen_encode_layer_8_conv)

			# ---------- decoder part ----------
			# gen_decode_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 1, 1, num_filter*8]

			# decode_layer_1_output = (batch_size, 2, 2, num_filter*8*2)
			decode_layer_1_deconv = gen_decode_conv2d(relu(gen_encode_layer_8_batchnorm), num_filter*8, name='decode_layer_1_deconv') 
			decode_layer_1_batchnorm = self.A2B_gen_decode_layer1_batchnorm(decode_layer_1_deconv)
			decode_layer_1_dropout = tf.nn.dropout(decode_layer_1_batchnorm, rate=0.5)
			decode_layer_1_concat = tf.concat([decode_layer_1_dropout, gen_encode_layer_7_batchnorm], 3)

			# decode_layer_2_output = (batch_size, 4, 4, num_filter*8*2)
			decode_layer_2_deconv = gen_decode_conv2d(relu(decode_layer_1_concat), num_filter*8, name='decode_layer_2_deconv') 
			decode_layer_2_batchnorm = self.A2B_gen_decode_layer2_batchnorm(decode_layer_2_deconv)
			decode_layer_2_dropout = tf.nn.dropout(decode_layer_2_batchnorm, rate=0.5)
			decode_layer_2_concat = tf.concat([decode_layer_2_dropout, gen_encode_layer_6_batchnorm], 3)

			# decode_layer_3_output = (batch_size, 8, 8, num_filter*8*2)
			decode_layer_3_deconv = gen_decode_conv2d(relu(decode_layer_2_concat), num_filter*8, name='decode_layer_3_deconv') 
			decode_layer_3_batchnorm = self.A2B_gen_decode_layer3_batchnorm(decode_layer_3_deconv)
			decode_layer_3_dropout = tf.nn.dropout(decode_layer_3_batchnorm, rate=0.5)
			decode_layer_3_concat = tf.concat([decode_layer_3_dropout, gen_encode_layer_5_batchnorm], 3)

			# decode_layer_4_output = (batch_size, 16, 16, num_filter*8*2)
			decode_layer_4_deconv = gen_decode_conv2d(relu(decode_layer_3_concat), num_filter*8, name='decode_layer_4_deconv') 
			decode_layer_4_batchnorm = self.A2B_gen_decode_layer4_batchnorm(decode_layer_4_deconv)
			decode_layer_4_dropout = tf.nn.dropout(decode_layer_4_batchnorm, rate=0.5)
			decode_layer_4_concat = tf.concat([decode_layer_4_dropout, gen_encode_layer_4_batchnorm], 3)

			# decode_layer_5_output = (batch_size, 32, 32, num_filter*4*2)
			decode_layer_5_deconv = gen_decode_conv2d(relu(decode_layer_4_concat), num_filter*4, name='decode_layer_5_deconv') 
			decode_layer_5_batchnorm = self.A2B_gen_decode_layer5_batchnorm(decode_layer_5_deconv)
			decode_layer_5_dropout = tf.nn.dropout(decode_layer_5_batchnorm, rate=0.5)
			decode_layer_5_concat = tf.concat([decode_layer_5_dropout, gen_encode_layer_3_batchnorm], 3)

			# decode_layer_6_output = (batch_size, 64, 64, num_filter*2*2)
			decode_layer_6_deconv = gen_decode_conv2d(relu(decode_layer_5_concat), num_filter*2, name='decode_layer_6_deconv') 
			decode_layer_6_batchnorm = self.A2B_gen_decode_layer6_batchnorm(decode_layer_6_deconv)
			decode_layer_6_dropout = tf.nn.dropout(decode_layer_6_batchnorm, rate=0.5)
			decode_layer_6_concat = tf.concat([decode_layer_6_dropout, gen_encode_layer_2_batchnorm], 3)

			# decode_layer_7_output = (batch_size, 128, 128, num_filter*1*2)
			decode_layer_7_deconv = gen_decode_conv2d(relu(decode_layer_6_concat), num_filter, name='decode_layer_7_deconv') 
			decode_layer_7_batchnorm = self.A2B_gen_decode_layer7_batchnorm(decode_layer_7_deconv)
			decode_layer_7_dropout = tf.nn.dropout(decode_layer_7_batchnorm, rate=0.5)
			decode_layer_7_concat = tf.concat([decode_layer_7_dropout, gen_encode_layer_1_conv], 3)


			# decode_layer_8_output = (batch_size, 256, 256, output_pic_dim)
			decode_layer_8_deconv = gen_decode_conv2d(relu(decode_layer_7_concat), self.output_pic_dim, name='decode_layer_8_deconv') 
			generator_output = tf.nn.tanh(decode_layer_8_deconv)

			return generator_output

	def B2A_generator(self, input_image, reuse=False):
		with tf.variable_scope("B2A_gen", reuse=reuse):

			# print("\n------  generator layers shape  ------\n")
			# print("input_image shape: {}".format(input_image.shape))


			num_filter = 64

			            # ---------- encoder part ----------
			# gen_encode_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 256, 256, input_pic_dim]

			# gen_encode_layer_1_output = (batch_size, 128, 128, num_filter)
			gen_encode_layer_1_conv = gen_encode_conv2d(input_image, num_filter, name='gen_encode_layer_1_conv') 

			# gen_encode_layer_2_output = (batch_size, 64, 64, num_filter*2)
			gen_encode_layer_2_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_1_conv), num_filter*2, name='gen_encode_layer_2_conv') 
			gen_encode_layer_2_batchnorm = self.B2A_gen_encode_layer2_batchnorm(gen_encode_layer_2_conv)

			# gen_encode_layer_3_output = (batch_size, 32, 32, num_filter*4)
			gen_encode_layer_3_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_2_batchnorm), num_filter*4, name='gen_encode_layer_3_conv')
			gen_encode_layer_3_batchnorm = self.B2A_gen_encode_layer3_batchnorm(gen_encode_layer_3_conv)

			# gen_encode_layer_4_output = (batch_size, 16, 16, num_filter*8)
			gen_encode_layer_4_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_3_batchnorm), num_filter*8, name='gen_encode_layer_4_conv') 
			gen_encode_layer_4_batchnorm = self.B2A_gen_encode_layer4_batchnorm(gen_encode_layer_4_conv)

			# gen_encode_layer_5_output = (batch_size, 8, 8, num_filter*8)
			gen_encode_layer_5_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_4_batchnorm), num_filter*8, name='gen_encode_layer_5_conv') 
			gen_encode_layer_5_batchnorm = self.B2A_gen_encode_layer5_batchnorm(gen_encode_layer_5_conv)

			# gen_encode_layer_6_output = (batch_size, 4, 4, num_filter*8)
			gen_encode_layer_6_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_5_batchnorm), num_filter*8, name='gen_encode_layer_6_conv') 
			gen_encode_layer_6_batchnorm = self.B2A_gen_encode_layer6_batchnorm(gen_encode_layer_6_conv)

			# gen_encode_layer_7_output = (batch_size, 2, 2, num_filter*8)
			gen_encode_layer_7_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_6_batchnorm), num_filter*8, name='gen_encode_layer_7_conv') 
			gen_encode_layer_7_batchnorm = self.B2A_gen_encode_layer7_batchnorm(gen_encode_layer_7_conv)

			# gen_encode_layer_8_output = (batch_size, 1, 1, num_filter*8)
			gen_encode_layer_8_conv = gen_encode_conv2d(leaky_relu(gen_encode_layer_7_batchnorm), num_filter*8, name='gen_encode_layer_8_conv') 
			gen_encode_layer_8_batchnorm = self.B2A_gen_encode_layer8_batchnorm(gen_encode_layer_8_conv)

			# ---------- decoder part ----------
			# gen_decode_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 1, 1, num_filter*8]

			# decode_layer_1_output = (batch_size, 2, 2, num_filter*8*2)
			decode_layer_1_deconv = gen_decode_conv2d(relu(gen_encode_layer_8_batchnorm), num_filter*8, name='decode_layer_1_deconv') 
			decode_layer_1_batchnorm = self.B2A_gen_decode_layer1_batchnorm(decode_layer_1_deconv)
			decode_layer_1_dropout = tf.nn.dropout(decode_layer_1_batchnorm, rate=0.5)
			decode_layer_1_concat = tf.concat([decode_layer_1_dropout, gen_encode_layer_7_batchnorm], 3)

			# decode_layer_2_output = (batch_size, 4, 4, num_filter*8*2)
			decode_layer_2_deconv = gen_decode_conv2d(relu(decode_layer_1_concat), num_filter*8, name='decode_layer_2_deconv') 
			decode_layer_2_batchnorm = self.B2A_gen_decode_layer2_batchnorm(decode_layer_2_deconv)
			decode_layer_2_dropout = tf.nn.dropout(decode_layer_2_batchnorm, rate=0.5)
			decode_layer_2_concat = tf.concat([decode_layer_2_dropout, gen_encode_layer_6_batchnorm], 3)

			# decode_layer_3_output = (batch_size, 8, 8, num_filter*8*2)
			decode_layer_3_deconv = gen_decode_conv2d(relu(decode_layer_2_concat), num_filter*8, name='decode_layer_3_deconv') 
			decode_layer_3_batchnorm = self.B2A_gen_decode_layer3_batchnorm(decode_layer_3_deconv)
			decode_layer_3_dropout = tf.nn.dropout(decode_layer_3_batchnorm, rate=0.5)
			decode_layer_3_concat = tf.concat([decode_layer_3_dropout, gen_encode_layer_5_batchnorm], 3)

			# decode_layer_4_output = (batch_size, 16, 16, num_filter*8*2)
			decode_layer_4_deconv = gen_decode_conv2d(relu(decode_layer_3_concat), num_filter*8, name='decode_layer_4_deconv') 
			decode_layer_4_batchnorm = self.B2A_gen_decode_layer4_batchnorm(decode_layer_4_deconv)
			decode_layer_4_dropout = tf.nn.dropout(decode_layer_4_batchnorm, rate=0.5)
			decode_layer_4_concat = tf.concat([decode_layer_4_dropout, gen_encode_layer_4_batchnorm], 3)

			# decode_layer_5_output = (batch_size, 32, 32, num_filter*4*2)
			decode_layer_5_deconv = gen_decode_conv2d(relu(decode_layer_4_concat), num_filter*4, name='decode_layer_5_deconv') 
			decode_layer_5_batchnorm = self.B2A_gen_decode_layer5_batchnorm(decode_layer_5_deconv)
			decode_layer_5_dropout = tf.nn.dropout(decode_layer_5_batchnorm, rate=0.5)
			decode_layer_5_concat = tf.concat([decode_layer_5_dropout, gen_encode_layer_3_batchnorm], 3)

			# decode_layer_6_output = (batch_size, 64, 64, num_filter*2*2)
			decode_layer_6_deconv = gen_decode_conv2d(relu(decode_layer_5_concat), num_filter*2, name='decode_layer_6_deconv') 
			decode_layer_6_batchnorm = self.B2A_gen_decode_layer6_batchnorm(decode_layer_6_deconv)
			decode_layer_6_dropout = tf.nn.dropout(decode_layer_6_batchnorm, rate=0.5)
			decode_layer_6_concat = tf.concat([decode_layer_6_dropout, gen_encode_layer_2_batchnorm], 3)

			# decode_layer_7_output = (batch_size, 128, 128, num_filter*1*2)
			decode_layer_7_deconv = gen_decode_conv2d(relu(decode_layer_6_concat), num_filter, name='decode_layer_7_deconv') 
			decode_layer_7_batchnorm = self.B2A_gen_decode_layer7_batchnorm(decode_layer_7_deconv)
			decode_layer_7_dropout = tf.nn.dropout(decode_layer_7_batchnorm, rate=0.5)
			decode_layer_7_concat = tf.concat([decode_layer_7_dropout, gen_encode_layer_1_conv], 3)


			# decode_layer_8_output = (batch_size, 256, 256, output_pic_dim)
			decode_layer_8_deconv = gen_decode_conv2d(relu(decode_layer_7_concat), self.output_pic_dim, name='decode_layer_8_deconv') 
			generator_output = tf.nn.tanh(decode_layer_8_deconv)

			return generator_output

	def A2B_descriptor(self, input_image, reuse=False):
		with tf.variable_scope('A2B_des', reuse=reuse):

			# print("\n------  descriptor layers shape  ------\n")
			# print("input_image shape: {}".format(input_image.shape))

			num_filter = 64

			# ---------- descriptor part ----------
			# descriptor_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 256, 256, input_pic_dim]

			# layer_0_conv = (batch_size, 128, 128, num_filter)
			layer_0_conv = des_conv2d(input_image, num_filter, name='layer_0_conv')

			# layer_1_conv = (batch_size, 64, 64, num_filter*2)
			layer_1_conv = des_conv2d(leaky_relu(layer_0_conv), num_filter*2, name='layer_1_conv')
			layer_1_batchnorm = self.A2B_des_layer_1_batchnorm(layer_1_conv)

			# layer_2_conv = (batch_size, 32, 32, num_filter*4)
			layer_2_conv = des_conv2d(leaky_relu(layer_1_batchnorm), num_filter*4, name='layer_2_conv')
			layer_2_batchnorm = self.A2B_des_layer_2_batchnorm(layer_2_conv)

			# layer_3_conv = (batch_size, 16, 16, num_filter*8)
			layer_3_conv = des_conv2d(leaky_relu(layer_2_batchnorm), num_filter*8, name='layer_3_conv')
			layer_3_batchnorm = self.A2B_des_layer_3_batchnorm(layer_3_conv)

			# linearization the descriptor result
			# # print(layer_3_batchnorm.shape) # (1, 16, 16, 512)
			# # print(layer_3_reshape.shape) # (1, 131072)

			layer_4_fully_connected = des_fully_connected(leaky_relu(layer_3_batchnorm), 100, name="layer_4_fully_connected")

			return layer_4_fully_connected 

	def B2A_descriptor(self, input_image, reuse=False):
		with tf.variable_scope('B2A_des', reuse=reuse):

			# print("\n------  descriptor layers shape  ------\n")
			# print("input_image shape: {}".format(input_image.shape))

			num_filter = 64

			# ---------- descriptor part ----------
			# descriptor_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 256, 256, input_pic_dim]

			# layer_0_conv = (batch_size, 128, 128, num_filter)
			layer_0_conv = des_conv2d(input_image, num_filter, name='layer_0_conv')

			# layer_1_conv = (batch_size, 64, 64, num_filter*2)
			layer_1_conv = des_conv2d(leaky_relu(layer_0_conv), num_filter*2, name='layer_1_conv')
			layer_1_batchnorm = self.B2A_des_layer_1_batchnorm(layer_1_conv)

			# layer_2_conv = (batch_size, 32, 32, num_filter*4)
			layer_2_conv = des_conv2d(leaky_relu(layer_1_batchnorm), num_filter*4, name='layer_2_conv')
			layer_2_batchnorm = self.B2A_des_layer_2_batchnorm(layer_2_conv)

			# layer_3_conv = (batch_size, 16, 16, num_filter*8)
			layer_3_conv = des_conv2d(leaky_relu(layer_2_batchnorm), num_filter*8, name='layer_3_conv')
			layer_3_batchnorm = self.B2A_des_layer_3_batchnorm(layer_3_conv)

			# linearization the descriptor result
			# # print(layer_3_batchnorm.shape) # (1, 16, 16, 512)
			# # print(layer_3_reshape.shape) # (1, 131072)

			layer_4_fully_connected = des_fully_connected(leaky_relu(layer_3_batchnorm), 100, name="layer_4_fully_connected")

			return layer_4_fully_connected 

	def A2B_des_langevin_revision(self, input_image_arg):
		# print("input_image_arg.shape: ",input_image_arg.shape)
		# self.pic_list = []
		def cond(i, input_image):
			return tf.less(i, self.langevin_revision_steps)

		def body(i, input_image):
			# print("input_image.shape: ",input_image.shape)
			# save_images(input_image, [self.batch_size, 1],
			# 	'./{}/test.png'.format(self.output_dir))
			noise = tf.random_normal(shape=[1, self.image_size, self.image_size, 3], name='noise')
			descripted_input_image = self.A2B_descriptor(input_image, reuse=True)

			grad = tf.gradients(descripted_input_image, input_image, name='grad_des')[0]
			input_image = input_image - 0.5 * self.langevin_step_size * self.langevin_step_size * (input_image / self.sigma1 / self.sigma1 - grad) + self.langevin_step_size * noise
			# print("input_image.shape: ",input_image.shape)
			return tf.add(i, 1), input_image

		with tf.name_scope("A2B_des_langevin_revision"):
			i = tf.constant(0)
			i, input_image = tf.while_loop(cond, body, [i, input_image_arg])
			return input_image

	def B2A_des_langevin_revision(self, input_image_arg):
		# print("input_image_arg.shape: ",input_image_arg.shape)
		# self.pic_list = []
		def cond(i, input_image):
			return tf.less(i, self.langevin_revision_steps)

		def body(i, input_image):
			# print("input_image.shape: ",input_image.shape)
			# save_images(input_image, [self.batch_size, 1],
			# 	'./{}/test.png'.format(self.output_dir))
			noise = tf.random_normal(shape=[1, self.image_size, self.image_size, 3], name='noise')
			descripted_input_image = self.B2A_descriptor(input_image, reuse=True)

			grad = tf.gradients(descripted_input_image, input_image, name='grad_des')[0]
			input_image = input_image - 0.5 * self.langevin_step_size * self.langevin_step_size * (input_image / self.sigma1 / self.sigma1 - grad) + self.langevin_step_size * noise
			# print("input_image.shape: ",input_image.shape)
			return tf.add(i, 1), input_image

		with tf.name_scope("B2A_des_langevin_revision"):
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

	# def lang_1(self, input_image_arg):
	# 	def cond(i, input_image):
	# 		return tf.less(i, 1)

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

	# def lang_10(self, input_image_arg):
	# 	def cond(i, input_image):
	# 		return tf.less(i, 10)

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

	# def lang_30(self, input_image_arg):
	# 	def cond(i, input_image):
	# 		return tf.less(i, 30)

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

	# def lang_50(self, input_image_arg):
	# 	def cond(i, input_image):
	# 		return tf.less(i, 50)

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

	# def lang_100(self, input_image_arg):
	# 	def cond(i, input_image):
	# 		return tf.less(i, 100)

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


	# def lang_200(self, input_image_arg):
	# 	def cond(i, input_image):
	# 		return tf.less(i, 200)

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


