'''
D:
cd D:\test\mycode
python main.py
'''

import tensorflow as tf
import os

from model import Cooppix2pix

# parameters setting
tf.app.flags.DEFINE_integer('epoch',200,'how many epochs to train')

# folder position
tf.app.flags.DEFINE_string('output_dir', './output_dir', 'output directory')




FLAGS = tf.app.flags.FLAGS





def main(_):
	output_dir = FLAGS.output_dir

	with tf.Session() as sess:
		if tf.gfile.Exists(output_dir):
			user_input = input('Warning! Output folder exists! Enter \'y\' to delete folder! ')
			if user_input == 'y':
				tf.gfile.DeleteRecursively(output_dir)
				tf.gfile.MakeDirs(output_dir)
		else:
			tf.gfile.MakeDirs(output_dir)

	# if not os.path.exists(args.checkpoint_dir):
	# 	os.makedirs(args.checkpoint_dir)
	# 	if not os.path.exists(args.sample_dir):
	# 		os.makedirs(args.sample_dir)
	# 		if not os.path.exists(args.test_dir):
	# 			os.makedirs(args.test_dir)
	# 			print("test")


# program start here
if __name__ == '__main__':
	tf.app.run()







