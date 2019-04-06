import tensorflow as tf
import numpy as np


class Coop_pix2pix(object):
	def __init__(self, sess, epoch=200, output_dir="./output_dir", checkpoint_dir="./checkpoint_dir", log_dir="./log_dir"):
		"""
		args:
			sess: tensorflow session

		"""


		self.sess = sess
		self.epoch = epoch

		self.output_dir = output_dir
		self.checkpoint_dir = checkpoint_dir
		self.log_dir = log_dir

	def train(self,sess):
		print("Now training model......")


