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
				discriminator_learning_rate = 0.01,
				generator_learning_rate = 0.0001,
				recover_learning_rate = 0.0001,
				dataset_name='edges2handbags', dataset_dir ='./test_datasets', 
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
		self.discriminator_learning_rate = discriminator_learning_rate 
		self.generator_learning_rate  = generator_learning_rate 
		self.recover_learning_rate  = recover_learning_rate 
		# print(1e-5) # 0.00001
		

		self.dataset_dir = dataset_dir
		self.dataset_name = dataset_name

		self.output_dir = output_dir
		self.checkpoint_dir = checkpoint_dir
		self.log_dir = log_dir
		self.epoch_startpoint = 0

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

		self.rec_encode_layer2_batchnorm = batch_norm(name='rec_encode_layer_2_batchnorm')
		self.rec_encode_layer3_batchnorm = batch_norm(name='rec_encode_layer_3_batchnorm')
		self.rec_encode_layer4_batchnorm = batch_norm(name='rec_encode_layer_4_batchnorm')
		self.rec_encode_layer5_batchnorm = batch_norm(name='rec_encode_layer_5_batchnorm')
		self.rec_encode_layer6_batchnorm = batch_norm(name='rec_encode_layer_6_batchnorm')
		self.rec_encode_layer7_batchnorm = batch_norm(name='rec_encode_layer_7_batchnorm')
		self.rec_encode_layer8_batchnorm = batch_norm(name='rec_encode_layer_8_batchnorm')

		self.rec_decode_layer1_batchnorm = batch_norm(name='rec_decode_layer_1_batchnorm')
		self.rec_decode_layer2_batchnorm = batch_norm(name='rec_decode_layer_2_batchnorm')
		self.rec_decode_layer3_batchnorm = batch_norm(name='rec_decode_layer_3_batchnorm')
		self.rec_decode_layer4_batchnorm = batch_norm(name='rec_decode_layer_4_batchnorm')
		self.rec_decode_layer5_batchnorm = batch_norm(name='rec_decode_layer_5_batchnorm')
		self.rec_decode_layer6_batchnorm = batch_norm(name='rec_decode_layer_6_batchnorm')
		self.rec_decode_layer7_batchnorm = batch_norm(name='rec_decode_layer_7_batchnorm')

		self.des_layer_1_batchnorm = batch_norm(name='des_layer_1_batchnorm')
		self.des_layer_2_batchnorm = batch_norm(name='des_layer_2_batchnorm')
		self.des_layer_3_batchnorm = batch_norm(name='des_layer_3_batchnorm')

		# batch normalization : deals with poor initialization helps gradient flow
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
		self.d_bn3 = batch_norm(name='d_bn3')

		self.sigma1 = 0.016
		self.sigma2 = 0.3
		self.beta1 = 0.5
		self.L1_lambda = 100

		# self.input_revised_B = tf.placeholder(tf.float32,
		# 		[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
		# 		name='input_revised_B')
		self.input_generated_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_generated_B')
		self.input_real_data_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_real_data_B')
		self.input_real_data_A = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_real_data_A')
		self.input_encoder_mu = tf.placeholder(tf.float32,
				[1, 1, 1, self.input_pic_dim],
				name='input_encoder_mu')




		# self.input_recovered_A = tf.placeholder(tf.float32,
		# 		[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
		# 		name='input_recovered_A')

	def build_model(self):

		# generator 
		self.encoder_output_origin = self.generator_encoder(self.input_real_data_A, reuse = False)
		self.generated_B_origin = self.generator_decoder(self.encoder_output_origin, reuse = False)

		self.generated_B_mu = self.generator_decoder(self.input_encoder_mu, reuse = True)

		# self.encoder_output_case_A = self.add_noise(self.encoder_output_origin, 1)
		# self.encoder_output_case_B = self.add_noise(self.encoder_output_origin, 10)
		# self.encoder_output_case_C = self.add_noise(self.encoder_output_origin, 30)
		# self.encoder_output_case_D = self.add_noise(self.encoder_output_origin, 50)
		# self.encoder_output_case_E = self.add_noise(self.encoder_output_origin, 100)
		# self.encoder_output_case_F = self.add_noise(self.encoder_output_origin, 200)

		# self.generated_B_case_A = self.generator_decoder(self.encoder_output_case_A, reuse = True)
		# self.generated_B_case_B = self.generator_decoder(self.encoder_output_case_B, reuse = True)
		# self.generated_B_case_C = self.generator_decoder(self.encoder_output_case_C, reuse = True)
		# self.generated_B_case_D = self.generator_decoder(self.encoder_output_case_D, reuse = True)
		# self.generated_B_case_E = self.generator_decoder(self.encoder_output_case_E, reuse = True)
		# self.generated_B_case_F = self.generator_decoder(self.encoder_output_case_F, reuse = True)
		# self.generated_B_case_G = self.generator_decoder(self.encoder_output_case_G, reuse = True)

		# descriptor
		# described_real_data_B = self.descriptor(self.input_real_data_B, reuse=False)
		# described_revised_B = self.descriptor(self.input_revised_B, reuse=True)
		# described_generated_B = self.descriptor(self.input_generated_B, reuse=True)

		# discriminator
		self.real_AB = tf.concat([self.input_real_data_A, self.input_real_data_B], 3)
		self.fake_AB = tf.concat([self.input_real_data_A, self.generated_B_origin], 3)
		self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
		self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

		# recover
		self.recovered_A = self.recover(self.input_generated_B, reuse = False)

		# symbolic langevins
		# self.des_langevin_revision_output = self.des_langevin_revision(self.input_generated_B)
		# self.lang_1_output = self.lang_1(self.input_revised_B)
		# self.lang_10_output = self.lang_10(self.input_revised_B)
		# self.lang_30_output = self.lang_30(self.input_revised_B)
		# self.lang_50_output = self.lang_50(self.input_revised_B)
		# self.lang_100_output = self.lang_100(self.input_revised_B)
		# self.lang_200_output = self.lang_200(self.input_revised_B)

		t_vars = tf.trainable_variables()
		self.dis_vars = [var for var in t_vars if var.name.startswith('discriminator')]
		# self.des_vars = [var for var in t_vars if var.name.startswith('des')]
		self.gen_vars = [var for var in t_vars if var.name.startswith('gen')]
		self.rec_vars = [var for var in t_vars if var.name.startswith('rec')]

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
		# self.des_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(described_revised_B, axis=0), tf.reduce_mean(described_real_data_B, axis=0)))

		# self.des_optim = tf.train.AdamOptimizer(self.descriptor_learning_rate, beta1=self.beta1).minimize(self.des_loss, var_list=self.des_vars)


		# discriminator loss functions
		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

		# self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
		# self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

		self.dis_loss = self.d_loss_real + self.d_loss_fake

		# self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

		# discriminator loss functions
		# self.des_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(described_revised_B, axis=0), tf.reduce_mean(described_real_data_B, axis=0)))

		self.dis_optim = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=self.beta1).minimize(self.dis_loss, var_list=self.dis_vars)

		# Compute Mean square error(MSE) for generated data and real data
		# self.mse_loss = tf.reduce_mean(
  #           tf.pow(tf.subtract(tf.reduce_mean(self.input_generated_B, axis=0), tf.reduce_mean(self.input_revised_B, axis=0)), 2))


		# generator loss functions
		# self.gen_loss = tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.input_revised_B - self.generated_B_origin), axis=0))
		
		
		self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.input_real_data_B - self.generated_B_origin))

		self.gen_optim = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=self.beta1).minimize(self.gen_loss, var_list=self.gen_vars)


		# recover loss functions
		self.rec_loss = tf.reduce_mean((self.recovered_A - self.input_real_data_A)**2)

		# tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.recovered_A - self.input_real_data_A), axis=0))

		self.rec_optim = tf.train.AdamOptimizer(self.recover_learning_rate, beta1=self.beta1).minimize(self.rec_loss, var_list=self.rec_vars)

		self.saver = tf.train.Saver(max_to_keep=10)


	def train(self,sess):

		# build model
		self.build_model()

		# prepare training data
		training_data = glob('{}/{}/train/*.jpg'.format(self.dataset_dir, self.dataset_name))

		# prepare color data
		img_list = []
		color_data = glob('./color/*.png')
		for i in xrange(10): # num_batch
			# find picture list index*self.batch_size to (index+1)*self.batch_size (one batch)
			# if batch_size = 2, get one batch = batch[0], batch[1]
			color_files = color_data[i:(i+1)] 

			color_batch = [load_one_data(color_file) for color_file in color_files]
			color_batch_images = np.array(color_batch).astype(np.float32)
			# print(color_batch_images.shape)

			# for i in xrange(color_batch_images.shape[0]):
			img = color_batch_images[:, :, :, : 3] 
			img_list.append(img)
			# print(img.shape)
			# save_images(img, [self.batch_size, 1], './{}/test_color_{}.png'.format(self.output_dir, i))

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
				# print(batch_images.shape)

				# data domain A and data domain B
				data_A = batch_images[:, :, :, : self.input_pic_dim] 
				data_B = batch_images[:, :, :, self.input_pic_dim:self.input_pic_dim+self.output_pic_dim] 
				# print(data_A.shape)

				# step G1: try to generate B domain(target domain) picture
				generated_B_origin = sess.run(self.generated_B_origin , feed_dict={self.input_real_data_A: data_A})
				generated_B_case_A = sess.run(self.generated_B_origin , feed_dict={self.input_real_data_A: data_A}) + img_list[0]
				generated_B_case_B = sess.run(self.generated_B_origin , feed_dict={self.input_real_data_A: data_A}) + img_list[1] 
				generated_B_case_C = sess.run(self.generated_B_origin , feed_dict={self.input_real_data_A: data_A}) + img_list[2]
				generated_B_case_D = sess.run(self.generated_B_origin , feed_dict={self.input_real_data_A: data_A}) + img_list[3]
				generated_B_case_E = sess.run(self.generated_B_origin , feed_dict={self.input_real_data_A: data_A}) + img_list[4]
				generated_B_case_F = sess.run(self.generated_B_origin , feed_dict={self.input_real_data_A: data_A}) + img_list[5]
				generated_B_case_G = sess.run(self.generated_B_origin , feed_dict={self.input_real_data_A: data_A}) + img_list[6]
				generated_B_case_H = sess.run(self.generated_B_origin , feed_dict={self.input_real_data_A: data_A}) + img_list[7]
				generated_B_case_I = sess.run(self.generated_B_origin , feed_dict={self.input_real_data_A: data_A}) + img_list[8]
				generated_B_case_J = sess.run(self.generated_B_origin , feed_dict={self.input_real_data_A: data_A}) + img_list[9]
				# generated_B_case_A = sess.run(self.generated_B_case_A , feed_dict={self.input_real_data_A: data_A}) + img_list[0]
				# generated_B_case_B = sess.run(self.generated_B_case_B , feed_dict={self.input_real_data_A: data_A}) + img_list[1] 
				# generated_B_case_C = sess.run(self.generated_B_case_C , feed_dict={self.input_real_data_A: data_A}) + img_list[2]
				# generated_B_case_D = sess.run(self.generated_B_case_D , feed_dict={self.input_real_data_A: data_A}) + img_list[3]
				# generated_B_case_E = sess.run(self.generated_B_case_E , feed_dict={self.input_real_data_A: data_A}) + img_list[4]
				# generated_B_case_F = sess.run(self.generated_B_case_F , feed_dict={self.input_real_data_A: data_A}) + img_list[5]
				# generated_B_case_G = sess.run(self.generated_B_case_G , feed_dict={self.input_real_data_A: data_A}) + img_list[6]


				# # step D1: descriptor try to revised image:"generated_B"
				# revised_B_origin = sess.run(self.des_langevin_revision_output, feed_dict={self.input_generated_B: generated_B_origin})
				# revised_B_mu = sess.run(self.des_langevin_revision_output, feed_dict={self.input_generated_B: generated_B_mu})

				# step R1: recover origin picture
				recovered_A_origin = sess.run(self.recovered_A, feed_dict={self.input_generated_B: generated_B_origin})
				recovered_A_case_A = sess.run(self.recovered_A, feed_dict={self.input_generated_B: generated_B_case_A})
				recovered_A_case_B = sess.run(self.recovered_A, feed_dict={self.input_generated_B: generated_B_case_B})
				recovered_A_case_C = sess.run(self.recovered_A, feed_dict={self.input_generated_B: generated_B_case_C})
				recovered_A_case_D = sess.run(self.recovered_A, feed_dict={self.input_generated_B: generated_B_case_D})
				recovered_A_case_E = sess.run(self.recovered_A, feed_dict={self.input_generated_B: generated_B_case_E})
				recovered_A_case_F = sess.run(self.recovered_A, feed_dict={self.input_generated_B: generated_B_case_F})
				recovered_A_case_G = sess.run(self.recovered_A, feed_dict={self.input_generated_B: generated_B_case_G})
				recovered_A_case_H = sess.run(self.recovered_A, feed_dict={self.input_generated_B: generated_B_case_H})
				recovered_A_case_I = sess.run(self.recovered_A, feed_dict={self.input_generated_B: generated_B_case_I})
				recovered_A_case_J = sess.run(self.recovered_A, feed_dict={self.input_generated_B: generated_B_case_J})


				# print(list(generated_B_origin.shape))
				# print(type(generated_B_origin))
				# test_color = sess.run(tf.zeros(list(generated_B_origin.shape) , tf.int32))
				# test_color = np.full(generated_B_origin.shape, 0)

				# test_color_2 = np.full(generated_B_origin.shape, -1)
				# test_color_3 = np.full(generated_B_origin.shape, 1)
				# test_color_4 = np.full(generated_B_origin.shape, 0.5)
				# test_color_5 = np.full(generated_B_origin.shape, -0.5)
				
				# print(test_color)
				# print(test_color_2)
				# print(test_color_3)
				# print(test_color_4)
				# print(generated_B_origin)



				# print(type(test_color))
				

				



				# step D2: update discriminator net
				discriminator_loss , _ = sess.run([self.dis_loss, self.dis_optim],
                                  		feed_dict={self.input_real_data_A: data_A, self.input_real_data_B: data_B})

				# step G2: update generator net
				generator_loss , _ = sess.run([self.gen_loss, self.gen_optim],
                                  		feed_dict={self.input_real_data_A: data_A, self.input_real_data_B: data_B}) # self.input_revised_B: revised_B,

				# self.input_generated_B: generated_B,

				# print(descriptor_loss)

				# step R2: update recover net
				rec_list = []

				recover_loss , _ = sess.run([self.rec_loss, self.rec_optim], feed_dict={self.input_generated_B: generated_B_origin, self.input_real_data_A: data_A})
				rec_list.append(recover_loss)
				recover_loss , _ = sess.run([self.rec_loss, self.rec_optim], feed_dict={self.input_generated_B: generated_B_case_A, self.input_real_data_A: data_A})
				rec_list.append(recover_loss)
				recover_loss , _ = sess.run([self.rec_loss, self.rec_optim], feed_dict={self.input_generated_B: generated_B_case_B, self.input_real_data_A: data_A})
				rec_list.append(recover_loss)
				recover_loss , _ = sess.run([self.rec_loss, self.rec_optim], feed_dict={self.input_generated_B: generated_B_case_C, self.input_real_data_A: data_A})
				rec_list.append(recover_loss)
				recover_loss , _ = sess.run([self.rec_loss, self.rec_optim], feed_dict={self.input_generated_B: generated_B_case_D, self.input_real_data_A: data_A})
				rec_list.append(recover_loss)
				recover_loss , _ = sess.run([self.rec_loss, self.rec_optim], feed_dict={self.input_generated_B: generated_B_case_E, self.input_real_data_A: data_A})
				rec_list.append(recover_loss)
				recover_loss , _ = sess.run([self.rec_loss, self.rec_optim], feed_dict={self.input_generated_B: generated_B_case_F, self.input_real_data_A: data_A})
				rec_list.append(recover_loss)
				recover_loss , _ = sess.run([self.rec_loss, self.rec_optim], feed_dict={self.input_generated_B: generated_B_case_G, self.input_real_data_A: data_A})
				rec_list.append(recover_loss)
				recover_loss , _ = sess.run([self.rec_loss, self.rec_optim], feed_dict={self.input_generated_B: generated_B_case_H, self.input_real_data_A: data_A})
				rec_list.append(recover_loss)
				recover_loss , _ = sess.run([self.rec_loss, self.rec_optim], feed_dict={self.input_generated_B: generated_B_case_I, self.input_real_data_A: data_A})
				rec_list.append(recover_loss)
				recover_loss , _ = sess.run([self.rec_loss, self.rec_optim], feed_dict={self.input_generated_B: generated_B_case_J, self.input_real_data_A: data_A})
				rec_list.append(recover_loss)


				recover_loss_avg = sum(rec_list)/len(rec_list)
				# print(sum(rec_list), len(rec_list))

				# recover_loss , _ = sess.run([self.rec_loss, self.rec_optim],
    #                               		feed_dict={self.input_generated_B: generated_B_origin, self.input_real_data_A: data_A})


				# Compute Mean square error(MSE) for generated data and revised data
				# mse_loss = sess.run(self.mse_loss, feed_dict={self.input_revised_B: revised_B, self.input_generated_B: generated_B})


				# put picture in sample picture
				# sample_results[index : (index + 1)] = revised_B

				print("Epoch: [{:4d}] [{:4d}/{:4d}] time: {}, eta: {}, d_loss: {:.4f}, g_loss: {:.4f}, r_loss_avg: {:.4f}"
					.format(epoch, index, self.num_batch, 
						str(datetime.timedelta(seconds=int(time.time()-start_time))),
							str(datetime.timedelta(seconds=int((time.time()-start_time)*(counter_end-(self.epoch_startpoint*self.num_batch)-counter)/counter))),
								 discriminator_loss, generator_loss, recover_loss_avg))

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
					# save_images(test_color, [self.batch_size, 1], './{}/test_color.png'.format(self.output_dir))
					# save_images(test_color_2, [self.batch_size, 1], './{}/test_color_2.png'.format(self.output_dir))
					# save_images(test_color_3, [self.batch_size, 1], './{}/test_color_3.png'.format(self.output_dir))
					# save_images(test_color_4, [self.batch_size, 1], './{}/test_color_4.png'.format(self.output_dir))
					# save_images(test_color_5, [self.batch_size, 1], './{}/test_color_5.png'.format(self.output_dir))
					# save_images(test_color_6, [self.batch_size, 1], './{}/test_color_6.png'.format(self.output_dir))
					# save_images(test_color_7, [self.batch_size, 1], './{}/test_color_7.png'.format(self.output_dir))

					# color_data = glob('./color/*.png')
					# # print(color_data)
					
					# # print(color_files)
					# # load data : list format, amount = one batch


					# for i in xrange(7): # num_batch
					# 	# find picture list index*self.batch_size to (index+1)*self.batch_size (one batch)
					# 	# if batch_size = 2, get one batch = batch[0], batch[1]
					# 	color_files = color_data[i:(i+1)] 

					# 	color_batch = [load_one_data(color_file) for color_file in color_files]
					# 	color_batch_images = np.array(color_batch).astype(np.float32)
					# 	print(color_batch_images.shape)

					# 	# for i in xrange(color_batch_images.shape[0]):
					# 	img = color_batch_images[:, :, :, : 3] 
					# 	print(img.shape)
					# 	save_images(img, [self.batch_size, 1], './{}/test_color_{}.png'.format(self.output_dir, i))



					save_images(data_A, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_01_input_data_A.png'.format(self.output_dir, epoch, index))
					save_images(data_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_02_input_data_B.png'.format(self.output_dir, epoch, index))
					save_images(generated_B_origin, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_030_generated_B_origin.png'.format(self.output_dir, epoch, index))
					# save_images(revised_B_origin, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_04_revised_B_origin.png'.format(self.output_dir, epoch, index))
					save_images(recovered_A_origin, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_031_recovered_A_origin.png'.format(self.output_dir, epoch, index))

					save_images(generated_B_case_A, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_040_generated_B_case_A.png'.format(self.output_dir, epoch, index))
					# save_images(revised_B_mu, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_07_revised_B_mu.png'.format(self.output_dir, epoch, index))
					save_images(recovered_A_case_A, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_041_recovered_A_case_A.png'.format(self.output_dir, epoch, index))

					save_images(generated_B_case_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_050_generated_B_case_B.png'.format(self.output_dir, epoch, index))
					save_images(recovered_A_case_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_051_recovered_A_case_B.png'.format(self.output_dir, epoch, index))

					save_images(generated_B_case_C, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_060_generated_B_case_C.png'.format(self.output_dir, epoch, index))
					save_images(recovered_A_case_C, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_061_recovered_A_case_C.png'.format(self.output_dir, epoch, index))

					save_images(generated_B_case_D, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_070_generated_B_case_D.png'.format(self.output_dir, epoch, index))
					save_images(recovered_A_case_D, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_071_recovered_A_case_D.png'.format(self.output_dir, epoch, index))

					save_images(generated_B_case_E, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_080_generated_B_case_E.png'.format(self.output_dir, epoch, index))
					save_images(recovered_A_case_E, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_081_recovered_A_case_E.png'.format(self.output_dir, epoch, index))

					save_images(generated_B_case_F, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_090_generated_B_case_F.png'.format(self.output_dir, epoch, index))
					save_images(recovered_A_case_F, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_091_recovered_A_case_F.png'.format(self.output_dir, epoch, index))

					save_images(generated_B_case_G, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_100_generated_B_case_G.png'.format(self.output_dir, epoch, index))
					save_images(recovered_A_case_G, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_101_recovered_A_case_G.png'.format(self.output_dir, epoch, index))

					save_images(generated_B_case_H, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_110_generated_B_case_E.png'.format(self.output_dir, epoch, index))
					save_images(recovered_A_case_H, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_111_recovered_A_case_E.png'.format(self.output_dir, epoch, index))

					save_images(generated_B_case_I, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_120_generated_B_case_F.png'.format(self.output_dir, epoch, index))
					save_images(recovered_A_case_I, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_121_recovered_A_case_F.png'.format(self.output_dir, epoch, index))

					save_images(generated_B_case_J, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_130_generated_B_case_G.png'.format(self.output_dir, epoch, index))
					save_images(recovered_A_case_J, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_131_recovered_A_case_G.png'.format(self.output_dir, epoch, index))



					# save_images(lang_1_output, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_099_lang_001.png'.format(self.output_dir, epoch, index))
					# save_images(lang_10_output, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_099_lang_010.png'.format(self.output_dir, epoch, index))
					# save_images(lang_30_output, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_099_lang_030.png'.format(self.output_dir, epoch, index))
					# save_images(lang_50_output, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_099_lang_050.png'.format(self.output_dir, epoch, index))
					# save_images(lang_100_output, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_099_lang_100.png'.format(self.output_dir, epoch, index))
					# save_images(lang_200_output, [self.batch_size, 1],
					# 	'./{}/ep{:02d}_{:04d}_099_lang_200.png'.format(self.output_dir, epoch, index))
					

				counter += 1

			self.save(self.checkpoint_dir, epoch)

			
			# # print("time: {:.4f} , Epoch: {} ".format(time.time() - start_time, epoch))
			# print('Epoch #{:d}, avg.descriptor loss: {:.4f}, avg.generator loss: {:.4f}, avg.L2 distance: {:4.4f}, '
			# 	'time: {:.2f}s'.format(epoch, np.mean(des_loss_avg), np.mean(gen_loss_avg), np.mean(mse_avg), time.time() - start_time))

	def generator_decoder(self, input_image, reuse=False):
		with tf.variable_scope("gen", reuse=reuse):

			num_filter = 64
			# ---------- decoder part ----------
			# gen_decode_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 1, 1, num_filter*8]

			# gen_decode_layer_1_output = (batch_size, 2, 2, num_filter*8*2)
			gen_decode_layer_1_deconv = gen_decode_conv2d(self.gen_encode_output, num_filter*8, name='gen_decode_layer_1_deconv') 
			gen_decode_layer_1_batchnorm = self.gen_decode_layer1_batchnorm(gen_decode_layer_1_deconv)
			gen_decode_layer_1_dropout = tf.nn.dropout(gen_decode_layer_1_batchnorm, rate=0.5)
			gen_decode_layer_1_concat = tf.concat([gen_decode_layer_1_dropout, self.gen_encode_layer_7_batchnorm], 3)

			# gen_decode_layer_2_output = (batch_size, 4, 4, num_filter*8*2)
			gen_decode_layer_2_deconv = gen_decode_conv2d(relu(gen_decode_layer_1_concat), num_filter*8, name='gen_decode_layer_2_deconv') 
			gen_decode_layer_2_batchnorm = self.gen_decode_layer2_batchnorm(gen_decode_layer_2_deconv)
			gen_decode_layer_2_dropout = tf.nn.dropout(gen_decode_layer_2_batchnorm, rate=0.5)
			gen_decode_layer_2_concat = tf.concat([gen_decode_layer_2_dropout, self.gen_encode_layer_6_batchnorm], 3)

			# gen_decode_layer_3_output = (batch_size, 8, 8, num_filter*8*2)
			gen_decode_layer_3_deconv = gen_decode_conv2d(relu(gen_decode_layer_2_concat), num_filter*8, name='gen_decode_layer_3_deconv') 
			gen_decode_layer_3_batchnorm = self.gen_decode_layer3_batchnorm(gen_decode_layer_3_deconv)
			gen_decode_layer_3_dropout = tf.nn.dropout(gen_decode_layer_3_batchnorm, rate=0.5)
			gen_decode_layer_3_concat = tf.concat([gen_decode_layer_3_dropout, self.gen_encode_layer_5_batchnorm], 3)

			# gen_decode_layer_4_output = (batch_size, 16, 16, num_filter*8*2)
			gen_decode_layer_4_deconv = gen_decode_conv2d(relu(gen_decode_layer_3_concat), num_filter*8, name='gen_decode_layer_4_deconv') 
			gen_decode_layer_4_batchnorm = self.gen_decode_layer4_batchnorm(gen_decode_layer_4_deconv)
			gen_decode_layer_4_dropout = tf.nn.dropout(gen_decode_layer_4_batchnorm, rate=0.5)
			gen_decode_layer_4_concat = tf.concat([gen_decode_layer_4_dropout, self.gen_encode_layer_4_batchnorm], 3)

			# gen_decode_layer_5_output = (batch_size, 32, 32, num_filter*4*2)
			gen_decode_layer_5_deconv = gen_decode_conv2d(relu(gen_decode_layer_4_concat), num_filter*4, name='gen_decode_layer_5_deconv') 
			gen_decode_layer_5_batchnorm = self.gen_decode_layer5_batchnorm(gen_decode_layer_5_deconv)
			gen_decode_layer_5_dropout = tf.nn.dropout(gen_decode_layer_5_batchnorm, rate=0.5)
			gen_decode_layer_5_concat = tf.concat([gen_decode_layer_5_dropout, self.gen_encode_layer_3_batchnorm], 3)

			# gen_decode_layer_6_output = (batch_size, 64, 64, num_filter*2*2)
			gen_decode_layer_6_deconv = gen_decode_conv2d(relu(gen_decode_layer_5_concat), num_filter*2, name='gen_decode_layer_6_deconv') 
			gen_decode_layer_6_batchnorm = self.gen_decode_layer6_batchnorm(gen_decode_layer_6_deconv)
			gen_decode_layer_6_dropout = tf.nn.dropout(gen_decode_layer_6_batchnorm, rate=0.5)
			gen_decode_layer_6_concat = tf.concat([gen_decode_layer_6_dropout, self.gen_encode_layer_2_batchnorm], 3)

			# gen_decode_layer_7_output = (batch_size, 128, 128, num_filter*1*2)
			gen_decode_layer_7_deconv = gen_decode_conv2d(relu(gen_decode_layer_6_concat), num_filter, name='gen_decode_layer_7_deconv') 
			gen_decode_layer_7_batchnorm = self.gen_decode_layer7_batchnorm(gen_decode_layer_7_deconv)
			gen_decode_layer_7_dropout = tf.nn.dropout(gen_decode_layer_7_batchnorm, rate=0.5)
			gen_decode_layer_7_concat = tf.concat([gen_decode_layer_7_dropout, self.gen_encode_layer_1_conv], 3)


			# gen_decode_layer_8_output = (batch_size, 256, 256, output_pic_dim)
			gen_decode_layer_8_deconv = gen_decode_conv2d(relu(gen_decode_layer_7_concat), self.output_pic_dim, name='gen_decode_layer_8_deconv') 
			generator_output = tf.nn.tanh(gen_decode_layer_8_deconv)

			return generator_output


	def generator_encoder(self, input_image, reuse=False):
		with tf.variable_scope("gen", reuse=reuse):

			# print("\n------  generator layers shape  ------\n")
			# print("input_image shape: {}".format(input_image.shape))


			num_filter = 64

			# ---------- encoder part ----------
			# gen_encode_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 256, 256, input_pic_dim]

			# gen_encode_layer_1_output = (batch_size, 128, 128, num_filter)
			self.gen_encode_layer_1_conv = gen_encode_conv2d(input_image, num_filter, name='gen_encode_layer_1_conv') 

			# gen_encode_layer_2_output = (batch_size, 64, 64, num_filter*2)
			gen_encode_layer_2_conv = gen_encode_conv2d(leaky_relu(self.gen_encode_layer_1_conv), num_filter*2, name='gen_encode_layer_2_conv') 
			self.gen_encode_layer_2_batchnorm = self.gen_encode_layer2_batchnorm(gen_encode_layer_2_conv)
			
			# gen_encode_layer_3_output = (batch_size, 32, 32, num_filter*4)
			gen_encode_layer_3_conv = gen_encode_conv2d(leaky_relu(self.gen_encode_layer_2_batchnorm), num_filter*4, name='gen_encode_layer_3_conv')
			self.gen_encode_layer_3_batchnorm = self.gen_encode_layer3_batchnorm(gen_encode_layer_3_conv)

			# gen_encode_layer_4_output = (batch_size, 16, 16, num_filter*8)
			gen_encode_layer_4_conv = gen_encode_conv2d(leaky_relu(self.gen_encode_layer_3_batchnorm), num_filter*8, name='gen_encode_layer_4_conv') 
			self.gen_encode_layer_4_batchnorm = self.gen_encode_layer4_batchnorm(gen_encode_layer_4_conv)

			# gen_encode_layer_5_output = (batch_size, 8, 8, num_filter*8)
			gen_encode_layer_5_conv = gen_encode_conv2d(leaky_relu(self.gen_encode_layer_4_batchnorm), num_filter*8, name='gen_encode_layer_5_conv') 
			self.gen_encode_layer_5_batchnorm = self.gen_encode_layer5_batchnorm(gen_encode_layer_5_conv)

			# gen_encode_layer_6_output = (batch_size, 4, 4, num_filter*8)
			gen_encode_layer_6_conv = gen_encode_conv2d(leaky_relu(self.gen_encode_layer_5_batchnorm), num_filter*8, name='gen_encode_layer_6_conv') 
			self.gen_encode_layer_6_batchnorm = self.gen_encode_layer6_batchnorm(gen_encode_layer_6_conv)

			# gen_encode_layer_7_output = (batch_size, 2, 2, num_filter*8)
			gen_encode_layer_7_conv = gen_encode_conv2d(leaky_relu(self.gen_encode_layer_6_batchnorm), num_filter*8, name='gen_encode_layer_7_conv') 
			self.gen_encode_layer_7_batchnorm = self.gen_encode_layer7_batchnorm(gen_encode_layer_7_conv)

			# gen_encode_layer_8_output = (batch_size, 1, 1, num_filter*8)
			gen_encode_layer_8_conv = gen_encode_conv2d(leaky_relu(self.gen_encode_layer_7_batchnorm), num_filter*8, name='gen_encode_layer_8_conv') 
			gen_encode_layer_8_batchnorm = self.gen_encode_layer8_batchnorm(gen_encode_layer_8_conv)
			self.gen_encode_output = relu(gen_encode_layer_8_batchnorm)

			return self.gen_encode_output

	def add_noise(self, gen_encode_output, stddev):
		# self.gen_encode_output_origin = relu(gen_encode_layer_8_batchnorm)
		# self.gen_encode_sigma = relu(gen_encode_layer_8_batchnorm)

		eps = tf.random_normal(
			shape=tf.shape(gen_encode_output),
			mean=0, stddev=stddev, dtype=tf.float32)
		gen_encode_output_mu = gen_encode_output + tf.sqrt(tf.exp(gen_encode_output)) * eps

		# print(tf.sqrt(tf.exp(gen_encode_output)) * eps)

		# if stddev == 200:
		# 	# print(gen_encode_output_mu)
		# 	gen_encode_output_mu = tf.add(gen_encode_output_mu, 100)
		# 	# print(gen_encode_output_mu)

		return gen_encode_output_mu







	def recover(self, input_image, reuse=False):
		with tf.variable_scope("rec", reuse=reuse):

			# print("\n------  recover layers shape  ------\n")
			# print("input_image shape: {}".format(input_image.shape))


			num_filter = 64

			# ---------- encoder part ----------
			# rec_encode_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 256, 256, input_pic_dim]

			# rec_encode_layer_1_output = (batch_size, 128, 128, num_filter)
			rec_encode_layer_1_conv = rec_encode_conv2d(input_image, num_filter, name='rec_encode_layer_1_conv') 

			# rec_encode_layer_2_output = (batch_size, 64, 64, num_filter*2)
			rec_encode_layer_2_conv = rec_encode_conv2d(leaky_relu(rec_encode_layer_1_conv), num_filter*2, name='rec_encode_layer_2_conv') 
			rec_encode_layer_2_batchnorm = self.rec_encode_layer2_batchnorm(rec_encode_layer_2_conv)
			
			# rec_encode_layer_3_output = (batch_size, 32, 32, num_filter*4)
			rec_encode_layer_3_conv = rec_encode_conv2d(leaky_relu(rec_encode_layer_2_batchnorm), num_filter*4, name='rec_encode_layer_3_conv')
			rec_encode_layer_3_batchnorm = self.rec_encode_layer3_batchnorm(rec_encode_layer_3_conv)

			# rec_encode_layer_4_output = (batch_size, 16, 16, num_filter*8)
			rec_encode_layer_4_conv = rec_encode_conv2d(leaky_relu(rec_encode_layer_3_batchnorm), num_filter*8, name='rec_encode_layer_4_conv') 
			rec_encode_layer_4_batchnorm = self.rec_encode_layer4_batchnorm(rec_encode_layer_4_conv)

			# rec_encode_layer_5_output = (batch_size, 8, 8, num_filter*8)
			rec_encode_layer_5_conv = rec_encode_conv2d(leaky_relu(rec_encode_layer_4_batchnorm), num_filter*8, name='rec_encode_layer_5_conv') 
			rec_encode_layer_5_batchnorm = self.rec_encode_layer5_batchnorm(rec_encode_layer_5_conv)

			# rec_encode_layer_6_output = (batch_size, 4, 4, num_filter*8)
			rec_encode_layer_6_conv = rec_encode_conv2d(leaky_relu(rec_encode_layer_5_batchnorm), num_filter*8, name='rec_encode_layer_6_conv') 
			rec_encode_layer_6_batchnorm = self.rec_encode_layer6_batchnorm(rec_encode_layer_6_conv)

			# rec_encode_layer_7_output = (batch_size, 2, 2, num_filter*8)
			rec_encode_layer_7_conv = rec_encode_conv2d(leaky_relu(rec_encode_layer_6_batchnorm), num_filter*8, name='rec_encode_layer_7_conv') 
			rec_encode_layer_7_batchnorm = self.rec_encode_layer7_batchnorm(rec_encode_layer_7_conv)

			# rec_encode_layer_8_output = (batch_size, 1, 1, num_filter*8)
			rec_encode_layer_8_conv = rec_encode_conv2d(leaky_relu(rec_encode_layer_7_batchnorm), num_filter*8, name='rec_encode_layer_8_conv') 
			rec_encode_layer_8_batchnorm = self.rec_encode_layer8_batchnorm(rec_encode_layer_8_conv)

			# ---------- decoder part ----------
			# rec_decode_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 1, 1, num_filter*8]

			# rec_decode_layer_1_output = (batch_size, 2, 2, num_filter*8*2)
			rec_decode_layer_1_deconv = rec_decode_conv2d(relu(rec_encode_layer_8_batchnorm), num_filter*8, name='rec_decode_layer_1_deconv') 
			rec_decode_layer_1_batchnorm = self.rec_decode_layer1_batchnorm(rec_decode_layer_1_deconv)
			rec_decode_layer_1_dropout = tf.nn.dropout(rec_decode_layer_1_batchnorm, rate=0.5)
			rec_decode_layer_1_concat = tf.concat([rec_decode_layer_1_dropout, rec_encode_layer_7_batchnorm], 3)

			# rec_decode_layer_2_output = (batch_size, 4, 4, num_filter*8*2)
			rec_decode_layer_2_deconv = rec_decode_conv2d(relu(rec_decode_layer_1_concat), num_filter*8, name='rec_decode_layer_2_deconv') 
			rec_decode_layer_2_batchnorm = self.rec_decode_layer2_batchnorm(rec_decode_layer_2_deconv)
			rec_decode_layer_2_dropout = tf.nn.dropout(rec_decode_layer_2_batchnorm, rate=0.5)
			rec_decode_layer_2_concat = tf.concat([rec_decode_layer_2_dropout, rec_encode_layer_6_batchnorm], 3)

			# rec_decode_layer_3_output = (batch_size, 8, 8, num_filter*8*2)
			rec_decode_layer_3_deconv = rec_decode_conv2d(relu(rec_decode_layer_2_concat), num_filter*8, name='rec_decode_layer_3_deconv') 
			rec_decode_layer_3_batchnorm = self.rec_decode_layer3_batchnorm(rec_decode_layer_3_deconv)
			rec_decode_layer_3_dropout = tf.nn.dropout(rec_decode_layer_3_batchnorm, rate=0.5)
			rec_decode_layer_3_concat = tf.concat([rec_decode_layer_3_dropout, rec_encode_layer_5_batchnorm], 3)

			# rec_decode_layer_4_output = (batch_size, 16, 16, num_filter*8*2)
			rec_decode_layer_4_deconv = rec_decode_conv2d(relu(rec_decode_layer_3_concat), num_filter*8, name='rec_decode_layer_4_deconv') 
			rec_decode_layer_4_batchnorm = self.rec_decode_layer4_batchnorm(rec_decode_layer_4_deconv)
			rec_decode_layer_4_dropout = tf.nn.dropout(rec_decode_layer_4_batchnorm, rate=0.5)
			rec_decode_layer_4_concat = tf.concat([rec_decode_layer_4_dropout, rec_encode_layer_4_batchnorm], 3)

			# rec_decode_layer_5_output = (batch_size, 32, 32, num_filter*4*2)
			rec_decode_layer_5_deconv = rec_decode_conv2d(relu(rec_decode_layer_4_concat), num_filter*4, name='rec_decode_layer_5_deconv') 
			rec_decode_layer_5_batchnorm = self.rec_decode_layer5_batchnorm(rec_decode_layer_5_deconv)
			rec_decode_layer_5_dropout = tf.nn.dropout(rec_decode_layer_5_batchnorm, rate=0.5)
			rec_decode_layer_5_concat = tf.concat([rec_decode_layer_5_dropout, rec_encode_layer_3_batchnorm], 3)

			# rec_decode_layer_6_output = (batch_size, 64, 64, num_filter*2*2)
			rec_decode_layer_6_deconv = rec_decode_conv2d(relu(rec_decode_layer_5_concat), num_filter*2, name='rec_decode_layer_6_deconv') 
			rec_decode_layer_6_batchnorm = self.rec_decode_layer6_batchnorm(rec_decode_layer_6_deconv)
			rec_decode_layer_6_dropout = tf.nn.dropout(rec_decode_layer_6_batchnorm, rate=0.5)
			rec_decode_layer_6_concat = tf.concat([rec_decode_layer_6_dropout, rec_encode_layer_2_batchnorm], 3)

			# rec_decode_layer_7_output = (batch_size, 128, 128, num_filter*1*2)
			rec_decode_layer_7_deconv = rec_decode_conv2d(relu(rec_decode_layer_6_concat), num_filter, name='rec_decode_layer_7_deconv') 
			rec_decode_layer_7_batchnorm = self.rec_decode_layer7_batchnorm(rec_decode_layer_7_deconv)
			rec_decode_layer_7_dropout = tf.nn.dropout(rec_decode_layer_7_batchnorm, rate=0.5)
			rec_decode_layer_7_concat = tf.concat([rec_decode_layer_7_dropout, rec_encode_layer_1_conv], 3)


			# rec_decode_layer_8_output = (batch_size, 256, 256, output_pic_dim)
			rec_decode_layer_8_deconv = rec_decode_conv2d(relu(rec_decode_layer_7_concat), self.output_pic_dim, name='rec_decode_layer_8_deconv') 
			recover_output = tf.nn.tanh(rec_decode_layer_8_deconv)

			return recover_output

	def discriminator(self, input_image, reuse=False):

		with tf.variable_scope("discriminator") as scope:

			num_filter = 64

			# image is 256 x 256 x (input_c_dim + output_c_dim)
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse == False

			h0 = leaky_relu(conv2d(input_image, num_filter, name='d_h0_conv'))
			# h0 is (128 x 128 x self.df_dim)
			h1 = leaky_relu(self.d_bn1(conv2d(h0, num_filter*2, name='d_h1_conv')))
			# h1 is (64 x 64 x self.df_dim*2)
			h2 = leaky_relu(self.d_bn2(conv2d(h1, num_filter*4, name='d_h2_conv')))
			# h2 is (32x 32 x self.df_dim*4)
			h3 = leaky_relu(self.d_bn3(conv2d(h2, num_filter*8, d_h=1, d_w=1, name='d_h3_conv')))
			# h3 is (16 x 16 x self.df_dim*8)
			h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

			return tf.nn.sigmoid(h4), h4


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




