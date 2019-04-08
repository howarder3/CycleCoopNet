import tensorflow as tf
import numpy as np
import time

from glob import glob
from six.moves import xrange

# --------- self define function ---------
# ops: layers structure
from ops import *
# utils: for loading data
from utils import *

class Coop_pix2pix(object):
	def __init__(self, sess, 
				epoch=1, 
				batch_size=10,
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


	def build_model(self):
		self.input_data_A = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_images_A')
		self.input_data_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_images_B')

		# self.input_data = tf.placeholder(tf.float32,
		# 		[self.batch_size, self.image_size, self.image_size, self.input_pic_dim+self.output_pic_dim],
		# 		name='input_A_and_B_images')

		# # data domain A and data domain B
		# self.data_A = self.input_data[:, :, :, :self.input_pic_dim]
		# self.data_B = self.input_data[:, :, :, self.input_pic_dim:self.input_pic_dim+self.output_pic_dim]

		# print("data_A shape = {}".format(self.data_A.shape)) # data_A shape = (1, 256, 256, 3)
		# print("data_B shape = {}".format(self.data_B.shape)) # data_B shape = (1, 256, 256, 3)


		self.generated_B = self.generator(self.input_data_A, reuse=False)
		self.descripted_B = self.descriptor(self.input_data_B, reuse=False)






	def train(self,sess):
		# build model
		self.build_model()

		# initialize training
		sess.run(tf.global_variables_initializer())

		# start training
		start_time = time.time()	
		print("time: {:.4f} , Start training model......".format(0))
		

		for epoch in xrange(self.epoch): # how many epochs to train
			print("time: {:.4f} , Epoch: {} ".format(time.time() - start_time, epoch))
			# prepare training data
			training_data = glob('{}/{}/train/*.jpg'.format(self.dataset_dir, self.dataset_name))
			
			# iteration(num_batch) = picture_amount/batch_size
			num_batch = min(len(training_data), self.picture_amount) // self.batch_size

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
				# print("time: {:.4f} , Loading data finished! ".format(time.time() - start_time))


				# step G1: try to generate B domain(target domain) picture
				generated_B = sess.run(self.generated_B, feed_dict={self.input_data_A: data_A})
				# print(generated_B.shape) # (1, 256, 256, 3)

				# step D1: descriptor try to revised image:"generated_B"
				revised_B = sess.run(self.descripted_B, feed_dict={self.input_data_B: generated_B})
				



	def generator(self, input_image, reuse=False):
		with tf.variable_scope("generator", reuse=reuse):

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


	def descriptor(self, input_image, reuse=False):
		with tf.variable_scope('descriptor', reuse=reuse):

			num_filter = 64

			# ---------- descriptor part ----------
			# descriptor_conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 256, 256, input_pic_dim]

			# des_layer_0_conv = (batch_size, 128, 128, num_filter)
			des_layer_0_conv = descriptor_conv2d(input_image, num_filter, name='des_layer_0_conv')

			# des_layer_1_conv = (batch_size, 64, 64, num_filter*2)
			des_layer_1_conv = descriptor_conv2d(leaky_relu(des_layer_0_conv), num_filter*2, name='des_layer_1_conv')
			des_layer_1_batchnorm = self.descriptor_batchnorm_layer1(des_layer_1_conv)

			# des_layer_2_conv = (batch_size, 32, 32, num_filter*4)
			des_layer_2_conv = descriptor_conv2d(leaky_relu(des_layer_1_batchnorm), num_filter*4, name='des_layer_2_conv')
			des_layer_2_batchnorm = self.descriptor_batchnorm_layer2(des_layer_2_conv)
			
			# des_layer_3_conv = (batch_size, 16, 16, num_filter*8)
			des_layer_3_conv = descriptor_conv2d(leaky_relu(des_layer_2_batchnorm), num_filter*8, name='des_layer_3_conv')
			des_layer_3_batchnorm = self.descriptor_batchnorm_layer3(des_layer_3_conv)

			# linearization the descriptor result
			des_layer_3_reshape = tf.reshape(leaky_relu(des_layer_3_batchnorm), [self.batch_size, -1])
			# des_layer_3_linearization = linearization(des_layer_3_reshape, 1, 'des_layer_3_linearization')
			# print(des_layer_3_batchnorm.shape) # (1, 16, 16, 512)
			print(des_layer_3_reshape.shape) # (1, 131072)

			return des_layer_3_reshape








