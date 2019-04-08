import tensorflow as tf
import numpy as np
import time

from glob import glob
from six.moves import xrange

# self define function 
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

	def build_model(self):
		self.input_data = tf.placeholder(tf.float32,
										[self.batch_size, self.image_size, self.image_size, self.input_pic_dim+self.output_pic_dim],
										name='input_A_and_B_images')

		# data domain A and data domain B
		self.data_A = self.input_data[:, :, :, :self.input_pic_dim]
		self.data_B = self.input_data[:, :, :, self.input_pic_dim:self.input_pic_dim+self.output_pic_dim]

		print("data_A shape = {}".format(self.data_A.shape)) # data_A shape = (1, 256, 256, 3)
		print("data_B shape = {}".format(self.data_B.shape)) # data_B shape = (1, 256, 256, 3)

		self.generated_B = self.generator(self.data_A, reuse=False)






	def train(self,sess):
		self.build_model()
		start_time = time.time()	
		print("time: {:.4f} , Start training model......".format(0))
		

		for epoch in xrange(self.epoch): # how many epochs to train
			print("time: {:.4f} , Epoch: {} ".format(time.time() - start_time, epoch))
			# prepare training data
			training_data = glob('{}/{}/train/*.jpg'.format(self.dataset_dir, self.dataset_name))
			
			# iteration(num_batch) = picture_amount/batch_size
			num_batch = min(len(training_data), self.picture_amount) // self.batch_size

			for index in xrange(1): # num_batch
				# find picture list index*self.batch_size to (index+1)*self.batch_size (one batch)
				# if batch_size = 2, get one batch = batch[0], batch[1]
				batch_files = training_data[index*self.batch_size:(index+1)*self.batch_size] 

				# load data : list format, amount = one batch
				batch = [load_data(batch_file) for batch_file in batch_files]
				batch_images = np.array(batch).astype(np.float32)
				print("time: {:.4f} , Loading data finished! ".format(time.time() - start_time))



	def generator(self, input_image, reuse=False):
		with tf.variable_scope("generator", reuse=reuse):

			# output_size = self.output_size
			# output_size_2 = output_size/2
			# output_size_4 = output_size/4
			# output_size_8 = output_size/8
			# output_size_16 = output_size/16
			# output_size_32 = output_size/32
			# output_size_64 = output_size/64
			# output_size_128 = output_size/128

			num_encoder_filter = 64

			# ---------- encoder part ----------
			# conv2d(input_image, output_dimension (by how many filters), scope_name)
			# input image = [batch_size, 256, 256, input_pic_dim]
			gen_encode_layer_1_output = encode_conv2d(input_image, num_encoder_filter, name='gen_encode_layer_1_conv') 
			# gen_encode_layer_1_output = (batch_size, 128, 128, num_encoder_filter)
			gen_encode_layer_2_output = encode_conv2d(lrelu(gen_encode_layer_1_output), num_encoder_filter*2, name='gen_encode_layer_2_conv') 
			# gen_encode_layer_2_output = (batch_size, 64, 64, num_encoder_filter*2)
			gen_encode_layer_3_output = encode_conv2d(lrelu(gen_encode_layer_2_output), num_encoder_filter*4, name='gen_encode_layer_3_conv')
			# gen_encode_layer_3_output = (batch_size, 32, 32, num_encoder_filter*4)
			gen_encode_layer_4_output = encode_conv2d(lrelu(gen_encode_layer_3_output), num_encoder_filter*8, name='gen_encode_layer_4_conv')
			# gen_encode_layer_4_output = (batch_size, 16, 16, num_encoder_filter*8)
			gen_encode_layer_5_output = encode_conv2d(lrelu(gen_encode_layer_4_output), num_encoder_filter*8, name='gen_encode_layer_5_conv')
			# gen_encode_layer_5_output = (batch_size, 8, 8, num_encoder_filter*8)
			gen_encode_layer_6_output = encode_conv2d(lrelu(gen_encode_layer_5_output), num_encoder_filter*8, name='gen_encode_layer_6_conv')
			# gen_encode_layer_6_output = (batch_size, 4, 4, num_encoder_filter*8)
			gen_encode_layer_7_output = encode_conv2d(lrelu(gen_encode_layer_6_output), num_encoder_filter*8, name='gen_encode_layer_7_conv')
			# gen_encode_layer_7_output = (batch_size, 2, 2, num_encoder_filter*8)
			gen_encode_layer_8_output = encode_conv2d(lrelu(gen_encode_layer_7_output), num_encoder_filter*8, name='gen_encode_layer_8_conv')
			# gen_encode_layer_8_output = (batch_size, 1, 1, num_encoder_filter*8)


			


			return 0
'''


            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)
'''
