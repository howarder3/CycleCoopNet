import tensorflow as tf
import numpy as np
import time

from glob import glob

class Coop_pix2pix(object):
	def __init__(self, sess, epoch = 200, dataset_name = 'facades',
				output_dir='./output_dir', checkpoint_dir='./checkpoint_dir', log_dir='./log_dir'):
		"""
		args:
			sess: tensorflow session

		"""


		self.sess = sess
		self.epoch = epoch


		self.dataset_name = dataset_name
		self.output_dir = output_dir
		self.checkpoint_dir = checkpoint_dir
		self.log_dir = log_dir

	def build_model(self):
		pass



	def train(self,sess):
		self.build_model()
		start_time = time.time()	
		print("time: {:.4f} , Start training model......".format(0))
			

		

		# prepare training data
		training_data = glob('./datasets/{}/train/*.jpg'.format(self.dataset_name))
		print("time: {:.4f} , Loading data finished! ".format(time.time() - start_time))




