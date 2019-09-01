import tensorflow as tf
import os

from model import Cycle_CoopNet


# parameters setting
tf.app.flags.DEFINE_integer('epoch',300,'how many epochs to train')
tf.app.flags.DEFINE_integer('batch_size',1,'how many pic in one group(batch), iteration = picture_amount/batch_size')
tf.app.flags.DEFINE_integer('picture_amount',400,'how many pictures to train')
tf.app.flags.DEFINE_integer('image_size',256,'input image size')
tf.app.flags.DEFINE_integer('output_size',256,'output image size')
tf.app.flags.DEFINE_integer('input_pic_dim',3,'input picture dimension : colorful = 3, grayscale = 1')
tf.app.flags.DEFINE_integer('output_pic_dim',3,'output picture dimension : colorful = 3, grayscale = 1')
tf.app.flags.DEFINE_integer('output_descriptor_dim',100,'output descriptor dimension')


# learning rate
tf.app.flags.DEFINE_float('descriptor_learning_rate',0.01,'descriptor learning rate') # 0.01 # 0.007 # 1e-6 # 0.01 # 0.001 # 1e-6 # 0.01 # 0.007
tf.app.flags.DEFINE_float('generator_learning_rate',0.0001,'generator learning rate') # 0.0001 # 1e-5 # 0.0001 # 1e-4 # 0.0001 # 0.0001
tf.app.flags.DEFINE_float('cycle_learning_rate',0.0001,'cycle_learning_rate') 
tf.app.flags.DEFINE_integer('langevin_revision_steps',30,'langevin revision steps') #100 # 30 # 10
tf.app.flags.DEFINE_float('langevin_step_size',0.002,'langevin step size') # 0.002
tf.app.flags.DEFINE_float('L1_lambda',100.0,'L1_lambda') # 0.002


# dataset floder name
tf.app.flags.DEFINE_string('dataset_dir', './test_datasets', 'dataset directory')
tf.app.flags.DEFINE_string('dataset_name', 'edges2handbags', 'dataset name') 
# cityscapes # edges2handbags # edges2shoes # facades # maps	
# vangogh2photo

# folder position
tf.app.flags.DEFINE_string('output_dir', './output', 'output directory')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', 'checkpoint directory')
tf.app.flags.DEFINE_string('sample_dir', './sample', 'sample directory')
tf.app.flags.DEFINE_string('log_dir', './log', 'log directory')

FLAGS = tf.app.flags.FLAGS

def main(_):
	dataset_dir = FLAGS.dataset_dir+"/"+FLAGS.dataset_name
	output_dir = FLAGS.output_dir # +"_p"+str(FLAGS.picture_amount)
	checkpoint_dir = FLAGS.checkpoint_dir
	sample_dir = FLAGS.sample_dir
	log_dir = FLAGS.log_dir

	with tf.Session() as sess:
		if tf.gfile.Exists(dataset_dir):
			pass
		else:
			print("\nError! Dataset not found!\n")
			return

		# user_input = input('\n [!] Warning! Checkpoint exists! Continue training? (y/n) ')
		user_input = 'y'

		if tf.gfile.Exists(checkpoint_dir):
			if user_input == 'n':
				tf.gfile.DeleteRecursively(checkpoint_dir)
				tf.gfile.MakeDirs(checkpoint_dir)
		else:
			tf.gfile.MakeDirs(checkpoint_dir)


		if tf.gfile.Exists(output_dir):
			if user_input == 'n':
				tf.gfile.DeleteRecursively(output_dir)
				tf.gfile.MakeDirs(output_dir)
		else:
			tf.gfile.MakeDirs(output_dir)


		if tf.gfile.Exists(sample_dir):
			if user_input == 'n':
				tf.gfile.DeleteRecursively(sample_dir)
				tf.gfile.MakeDirs(sample_dir)
		else:
			tf.gfile.MakeDirs(sample_dir)

		if tf.gfile.Exists(log_dir):
			if user_input == 'n':
				tf.gfile.DeleteRecursively(log_dir)
				tf.gfile.MakeDirs(log_dir)
		else:
			tf.gfile.MakeDirs(log_dir)



		model = Cycle_CoopNet(sess, 
				epoch=FLAGS.epoch, 
				batch_size=FLAGS.batch_size,
				picture_amount=FLAGS.picture_amount,
				image_size=FLAGS.image_size, output_size = FLAGS.output_size,
				input_pic_dim = FLAGS.input_pic_dim, output_pic_dim = FLAGS.output_pic_dim,
				output_descriptor_dim = FLAGS.output_descriptor_dim,
				langevin_revision_steps = FLAGS.langevin_revision_steps,
				langevin_step_size = FLAGS.langevin_step_size,
				descriptor_learning_rate = FLAGS.descriptor_learning_rate,
				generator_learning_rate = FLAGS.generator_learning_rate,
				cycle_learning_rate = FLAGS.cycle_learning_rate,
				L1_lambda = FLAGS.L1_lambda,
				dataset_name=FLAGS.dataset_name, dataset_dir =FLAGS.dataset_dir, log_dir=FLAGS.log_dir,
				output_dir=FLAGS.output_dir, checkpoint_dir=FLAGS.checkpoint_dir, sample_dir=FLAGS.sample_dir)


		model.train(sess)



# program start here
if __name__ == '__main__':
	tf.app.run()







