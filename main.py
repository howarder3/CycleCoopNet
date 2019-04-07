'''
# windows
D:
cd D:\test\mycode
conda activate mlenv
python main.py

# mac
cd /Users/howarder3/Documents/GitHub/Coop_pix2pix 
python main.py

# pip packages 
pip freeze > requirements.txt
pip -r requirements.txt

'''

import tensorflow as tf
import os

from model import Coop_pix2pix

# parameters setting
tf.app.flags.DEFINE_integer('epoch',1,'how many epochs to train')
tf.app.flags.DEFINE_integer('batch_size',10,'how many pic in one group(batch), iteration = picture_amount/batch_size')
tf.app.flags.DEFINE_integer('picture_amount',99999,'how many pictures to train')
tf.app.flags.DEFINE_integer('image_size',256,'input image size')
tf.app.flags.DEFINE_integer('output_size',256,'output image size')
tf.app.flags.DEFINE_integer('input_pic_dim',3,'input picture dimension : colorful = 3, grayscale = 1')
tf.app.flags.DEFINE_integer('output_pic_dim',3,'output picture dimension : colorful = 3, grayscale = 1')

# dataset floder name
tf.app.flags.DEFINE_string('dataset_dir', './test_datasets', 'dataset directory')
tf.app.flags.DEFINE_string('dataset_name', 'facades', 'dataset name')

# folder position
tf.app.flags.DEFINE_string('output_dir', './output_dir', 'output directory')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint_dir', 'checkpoint directory')
tf.app.flags.DEFINE_string('log_dir', './log_dir', 'log directory')



FLAGS = tf.app.flags.FLAGS





def main(_):
	dataset_dir = FLAGS.dataset_dir+"/"+FLAGS.dataset_name
	output_dir = FLAGS.output_dir
	checkpoint_dir = FLAGS.checkpoint_dir
	log_dir = FLAGS.log_dir

	with tf.Session() as sess:
		if tf.gfile.Exists(dataset_dir):
			pass
		else:
			print("\nError! Dataset not found!\n")
			return
			
		if tf.gfile.Exists(output_dir):
			# user_input = input('\nWarning! Output directory exists! Enter \'y\' to delete folder!\n')
			# if user_input == 'y':
			tf.gfile.DeleteRecursively(output_dir)
			tf.gfile.MakeDirs(output_dir)
		else:
			tf.gfile.MakeDirs(output_dir)

		if tf.gfile.Exists(checkpoint_dir):
			# user_input = input('\nWarning! Checkpoint directory exists! Enter \'y\' to delete folder!\n')
			# if user_input == 'y':
			tf.gfile.DeleteRecursively(checkpoint_dir)
			tf.gfile.MakeDirs(checkpoint_dir)
		else:
			tf.gfile.MakeDirs(checkpoint_dir)

		if tf.gfile.Exists(log_dir):
			# user_input = input('\nWarning! Log directory exists! Enter \'y\' to delete folder!\n')
			# if user_input == 'y':
			tf.gfile.DeleteRecursively(log_dir)
			tf.gfile.MakeDirs(log_dir)
		else:
			tf.gfile.MakeDirs(log_dir)


		model = Coop_pix2pix(sess, 
				epoch=FLAGS.epoch, 
				batch_size=FLAGS.batch_size,
				picture_amount=FLAGS.picture_amount,
				image_size=FLAGS.image_size,
				output_size = FLAGS.output_size,
				input_pic_dim = FLAGS.input_pic_dim, 
				output_pic_dim = FLAGS.output_pic_dim,
				dataset_name=FLAGS.dataset_name, dataset_dir =FLAGS.dataset_dir, 
				output_dir=FLAGS.output_dir, checkpoint_dir=FLAGS.checkpoint_dir, log_dir=FLAGS.log_dir)


		model.train(sess)



# program start here
if __name__ == '__main__':
	tf.app.run()







