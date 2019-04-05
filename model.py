import tensorflow as tf
import numpy as np


class Cooppix2pix(object):
	def __init__(self, sess, epoch=200, checkpoint_dir="./checkpoint_dir", sample_dir="./sample_dir",test_dir="./test_dir"):
		"""
		args:
			sess: tensorflow session

		"""

		
		self.sess = sess
		self.epoch = epoch


		self.checkpoint_dir = checkpoint_dir
		self.sample_dir = sample_dir
		self.test_dir = test_dir
