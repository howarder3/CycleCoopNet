import tensorflow as tf
import numpy as np
import time

from glob import glob
from six.moves import xrange

from utils import *

class Coop_pix2pix(object):
	def __init__(self, sess, 
				epoch=200, 
				batch_size=1,
				picture_amount=99999,
				dataset_name='facades', dataset_dir ='./datasets', 
				output_dir='./output_dir', checkpoint_dir='./checkpoint_dir', log_dir='./log_dir'):
		"""
		args:
			sess: tensorflow session
			batch_size: how many pic in one group(batch), iteration(num_batch) = picture_amount/batch_size

		"""


		self.sess = sess
		self.epoch = epoch		
		self.batch_size = batch_size
		self.picture_amount = picture_amount

		self.dataset_dir = dataset_dir
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
		training_data = glob('{}/{}/train/*.jpg'.format(self.dataset_dir, self.dataset_name))
		


		# iteration(num_batch) = picture_amount/batch_size
		num_batch = min(len(training_data), self.picture_amount) // self.batch_size

		for index in xrange(0, 1): #num_batch
			# find picture list index*self.batch_size to (index+1)*self.batch_size (one batch)
			# if batch_size = 2, get one batch = batch[0], batch[1]
			batch_files = training_data[index*self.batch_size:(index+1)*self.batch_size] 
			print(batch_files)

			# load data 
			# list format, amount = one batch
			batch = [load_data(batch_file) for batch_file in batch_files]
			batch_images = np.array(batch).astype(np.float32)

			print(batch_images)
			
			print("time: {:.4f} , Loading data finished! ".format(time.time() - start_time))



