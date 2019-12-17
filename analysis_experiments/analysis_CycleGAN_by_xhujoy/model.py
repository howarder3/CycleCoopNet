from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *


# 20191114 -> add vis_util: visualization (borrow from CoopNet)
from vis_util import *


class cyclegan(object):
	def __init__(self, sess, args):
		self.sess = sess
		self.batch_size = args.batch_size
		self.image_size = args.fine_size
		self.input_c_dim = args.input_nc
		self.output_c_dim = args.output_nc
		self.L1_lambda = args.L1_lambda
		self.dataset_dir = args.dataset_dir
		self.output_dir = args.output_dir
		self.sample_dir = args.sample_dir

		# 20191114 -> add log picutres
		self.log_dir = args.log_dir



		self.discriminator = discriminator
		if args.use_resnet:
		    self.generator = generator_resnet
		else:
		    self.generator = generator_unet
		if args.use_lsgan:
		    self.criterionGAN = mae_criterion
		else:
		    self.criterionGAN = sce_criterion

		OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
		                      gf_dim df_dim output_c_dim is_training')
		self.options = OPTIONS._make((args.batch_size, args.fine_size,
		                              args.ngf, args.ndf, args.output_nc,
		                              args.phase == 'train'))

		self._build_model()
		self.saver = tf.train.Saver()
		self.pool = ImagePool(args.max_size)

	def _build_model(self):
		self.real_data = tf.placeholder(tf.float32,
		                                [None, self.image_size, self.image_size,
		                                 self.input_c_dim + self.output_c_dim],
		                                name='real_A_and_B_images')

		self.real_A = self.real_data[:, :, :, :self.input_c_dim]
		self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

		self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
		self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")

		self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
		self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

		self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
		self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
		# a2b loss (fake and judge fake(ones_like)) 
		self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
		    + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
		    + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)




		# b2a loss (fake and judge fake(ones_like)) 
		self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
		    + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
		    + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

		# total loss
		self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
		    + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
		    + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
		    + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

		self.fake_A_sample = tf.placeholder(tf.float32,
		                                    [None, self.image_size, self.image_size,
		                                     self.input_c_dim], name='fake_A_sample')
		self.fake_B_sample = tf.placeholder(tf.float32,
		                                    [None, self.image_size, self.image_size,
		                                     self.output_c_dim], name='fake_B_sample')
		self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
		self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
		self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
		self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

		self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real)) # b loss (real and judge_fake(ones_like)) 
		self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample)) # b loss (fake and judge_real(zeros_like)) 
		self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
		self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real)) # a loss (fake and judge_real(zeros_like))
		self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample)) # a loss (fake and judge_real(zeros_like)) 
		self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
		self.d_loss = self.da_loss + self.db_loss



		# g summary
		self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b) # a2b loss (real and judge real) 
		self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
		self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
		self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])

		# d summary
		self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real) # b loss (real and judge real) 
		self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake) # b loss (fake and judge fake) 
		self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss) # total b loss

		self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real) # a loss (real and judge real) 
		self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake) # a loss (fake and judge fake) 
		self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss) # total a loss

		self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss) # total loss
		self.d_sum = tf.summary.merge(
		    [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
		     self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
		     self.d_loss_sum]
		)

		self.test_A = tf.placeholder(tf.float32,
		                             [None, self.image_size, self.image_size,
		                              self.input_c_dim], name='test_A')
		self.test_B = tf.placeholder(tf.float32,
		                             [None, self.image_size, self.image_size,
		                              self.output_c_dim], name='test_B')
		self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
		self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
		self.g_vars = [var for var in t_vars if 'generator' in var.name]
		for var in t_vars: print(var.name)


		# add cycle loss 20191114
		self.cycle_loss_a2b2a = self.L1_lambda * abs_criterion(self.real_A, self.fake_A_)
		self.cycle_loss_b2a2b = self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

	def train(self, args):




		# 20191114 -> add picture Visualizer



		epoch_avg_A2B_des_loss_vis = Visualizer(title='epoch_avg_A2B_des_loss_vis', xlabel='training epoch_avgs', ylabel='epoch_avg_A2B_des_loss_vis',
		                          save_figpath=self.log_dir + '/epoch_avg_A2B_des_loss_vis.png', avg_period = self.batch_size)
		epoch_avg_A2B_gen_loss_vis = Visualizer(title='epoch_avg_A2B_gen_loss_vis', xlabel='training epoch_avgs', ylabel='epoch_avg_A2B_gen_loss_vis',
		                          save_figpath=self.log_dir + '/epoch_avg_A2B_gen_loss_vis.png', avg_period = self.batch_size)
		epoch_avg_A2B_cycle_loss_vis = Visualizer(title='epoch_avg_A2B_cycle_loss_vis', xlabel='training epoch_avgs', ylabel='epoch_avg_A2B_cycle_loss_vis', 
		                  		  save_figpath=self.log_dir + '/epoch_avg_A2B_cycle_loss_vis.png', avg_period = self.batch_size)
		epoch_avg_B2A_des_loss_vis = Visualizer(title='epoch_avg_B2A_des_loss_vis', xlabel='training epoch_avgs', ylabel='epoch_avg_B2A_des_loss_vis', 
		                          save_figpath=self.log_dir + '/epoch_avg_B2A_des_loss_vis.png', avg_period = self.batch_size)
		epoch_avg_B2A_gen_loss_vis = Visualizer(title='epoch_avg_B2A_gen_loss_vis', xlabel='training epoch_avgs', ylabel='epoch_avg_B2A_gen_loss_vis',
		                          save_figpath=self.log_dir + '/epoch_avg_B2A_gen_loss_vis.png', avg_period = self.batch_size)
		epoch_avg_B2A_cycle_loss_vis = Visualizer(title='epoch_avg_B2A_cycle_loss_vis', xlabel='training epoch_avgs', ylabel='epoch_avg_B2A_cycle_loss_vis', 
		                  		  save_figpath=self.log_dir + '/epoch_avg_B2A_cycle_loss_vis.png', avg_period = self.batch_size)




		# """Train cyclegan"""
		self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
		self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
		    .minimize(self.d_loss, var_list=self.d_vars)
		self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
		    .minimize(self.g_loss, var_list=self.g_vars)

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

		counter = 1
		start_time = time.time()

		if args.continue_train:
			if self.load(args.checkpoint_dir):
			    print(" [*] Load SUCCESS")
			else:
			    print(" [!] Load failed...")

		for epoch in range(args.epoch):


			A2B_des_loss_avg, A2B_gen_loss_avg, A2B_cycle_loss_avg = [], [], [] 
			B2A_des_loss_avg, B2A_gen_loss_avg, B2A_cycle_loss_avg = [], [], [] 


			dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
			dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
			np.random.shuffle(dataA)
			np.random.shuffle(dataB)
			batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
			lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

			test_dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
			test_dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))

			for idx in range(0, batch_idxs):
				batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
				                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
				batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
				batch_images = np.array(batch_images).astype(np.float32)

				# real_A -> fake_B -> fake_A_
				# real_B -> fake_A -> fake_B_

				# Update G network and record fake outputs
				# 20191114 -> add get loss g_loss_a2b, g_loss_b2a
				real_A, real_B, fake_A, fake_B, fake_A_, fake_B_, _, summary_str, g_loss_a2b, g_loss_b2a = self.sess.run(
				    [self.real_A, self.real_B, self.fake_A, self.fake_B, self.fake_A_, self.fake_B_, self.g_optim, self.g_sum, self.g_loss_a2b, self.g_loss_b2a],
				    feed_dict={self.real_data: batch_images, self.lr: lr})
				self.writer.add_summary(summary_str, counter)
				[fake_A, fake_B] = self.pool([fake_A, fake_B])

				# 20191114 -> add get loss d_loss_a2b, d_loss_b2a, cycle_loss_a2b2a, cycle_loss_b2a2b	
				_, summary_str, d_loss_a2b, d_loss_b2a, cycle_loss_a2b2a, cycle_loss_b2a2b = self.sess.run(
				    [self.d_optim, self.d_sum, self.da_loss, self.db_loss, self.cycle_loss_a2b2a, self.cycle_loss_b2a2b ],
				    feed_dict={self.real_data: batch_images,
				               self.fake_A_sample: fake_A,
				               self.fake_B_sample: fake_B,
				               self.lr: lr})
				self.writer.add_summary(summary_str, counter)


				A2B_des_loss_avg.append(d_loss_a2b)
				A2B_gen_loss_avg.append(g_loss_a2b)
				A2B_cycle_loss_avg.append(cycle_loss_a2b2a)

				B2A_des_loss_avg.append(d_loss_b2a)
				B2A_gen_loss_avg.append(g_loss_b2a)
				B2A_cycle_loss_avg.append(cycle_loss_b2a2b)

				counter += 1
				print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
				    epoch, idx, batch_idxs, time.time() - start_time)))

				# if np.mod(counter, 1) == 0:

				# 	# real_A -> fake_B -> fake_A_
				# 	# real_B -> fake_A -> fake_B_
				# 	save_images(real_A, [self.batch_size, 1],
				# 		'./{}/ep{:02d}_{:04d}_01_A2B_real_A.png'.format(self.output_dir, epoch, idx))
				# 	save_images(fake_B, [self.batch_size, 1],
				# 		'./{}/ep{:02d}_{:04d}_03_B2A_gen_A.png'.format(self.output_dir, epoch, idx))
				# 	save_images(fake_A_, [self.batch_size, 1],
				# 		'./{}/ep{:02d}_{:04d}_04_A2B_recover_A.png'.format(self.output_dir, epoch, idx))

					
				# 	save_images(real_B, [self.batch_size, 1],
				# 		'./{}/ep{:02d}_{:04d}_11_B2A_real_B.png'.format(self.output_dir, epoch, idx))
				# 	save_images(fake_A, [self.batch_size, 1],
				# 		'./{}/ep{:02d}_{:04d}_13_B2A_gen_B.png'.format(self.output_dir, epoch, idx))
				# 	save_images(fake_B_, [self.batch_size, 1],
				# 		'./{}/ep{:02d}_{:04d}_14_B2A_recover_B.png'.format(self.output_dir, epoch, idx))


				# 	# compare part
				# 	save_images(real_B, [self.batch_size, 1],
				# 		'./{}/ep{:02d}_{:04d}_02_B2A_real_B.png'.format(self.output_dir, epoch, idx))
				# 	save_images(real_A, [self.batch_size, 1],
				# 		'./{}/ep{:02d}_{:04d}_12_A2B_real_A.png'.format(self.output_dir, epoch, idx))


			for sample_index in range(min(len(test_dataA), len(test_dataB))):

				test_batch_files = list(zip(test_dataA[sample_index * self.batch_size:(sample_index + 1) * self.batch_size],
				                       		test_dataB[sample_index * self.batch_size:(sample_index + 1) * self.batch_size]))
				test_batch_images = [load_train_data(test_batch_file, 286, 256, is_testing=True) for test_batch_file in test_batch_files]
				test_batch_images = np.array(test_batch_images).astype(np.float32)


				# test_batch_files = list(zip(test_dataA[:self.batch_size], test_dataB[:self.batch_size]))
				# test_sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in test_batch_files]
				# test_sample_images = np.array(test_sample_images).astype(np.float32)

				real_A, real_B, fake_A, fake_B, fake_A_, fake_B_,= self.sess.run(
				    [self.real_A, self.real_B, self.fake_A, self.fake_B, self.fake_A_, self.fake_B_],
				    feed_dict={self.real_data: test_batch_images})

	            # real_A -> fake_B -> fake_A_
	            # real_B -> fake_A -> fake_B_

				save_images(real_A, [self.batch_size, 1],
							'./{}/testA_{:02d}_ep{:02d}_01_real_A.png'.format(self.sample_dir, sample_index, epoch))
				save_images(real_B, [self.batch_size, 1],
							'./{}/testA_{:02d}_ep{:02d}_02_real_B.png'.format(self.sample_dir, sample_index, epoch))
				save_images(fake_B, [self.batch_size, 1],
							'./{}/testA_{:02d}_ep{:02d}_03_gen_A.png'.format(self.sample_dir, sample_index, epoch))
				save_images(fake_A_, [self.batch_size, 1],
							'./{}/testA_{:02d}_ep{:02d}_05_recovered_A.png'.format(self.sample_dir, sample_index, epoch))


				save_images(real_B, [self.batch_size, 1],
							'./{}/testB_{:02d}_ep{:02d}_11_real_B.png'.format(self.sample_dir, sample_index, epoch))
				save_images(real_A, [self.batch_size, 1],
							'./{}/testB_{:02d}_ep{:02d}_12_real_A.png'.format(self.sample_dir, sample_index, epoch))
				save_images(fake_A, [self.batch_size, 1],
							'./{}/testB_{:02d}_ep{:02d}_13_gen_B.png'.format(self.sample_dir, sample_index, epoch))
				save_images(fake_B_, [self.batch_size, 1],
							'./{}/testB_{:02d}_ep{:02d}_15_recovered_B.png'.format(self.sample_dir, sample_index, epoch))



			epoch_avg_A2B_des_loss_vis.add_loss_val(epoch, np.mean(A2B_des_loss_avg))
			epoch_avg_A2B_gen_loss_vis.add_loss_val(epoch, np.mean(A2B_gen_loss_avg))
			epoch_avg_A2B_cycle_loss_vis.add_loss_val(epoch, np.mean(A2B_cycle_loss_avg))
			epoch_avg_B2A_des_loss_vis.add_loss_val(epoch, np.mean(B2A_des_loss_avg))
			epoch_avg_B2A_gen_loss_vis.add_loss_val(epoch, np.mean(B2A_gen_loss_avg))
			epoch_avg_B2A_cycle_loss_vis.add_loss_val(epoch, np.mean(B2A_cycle_loss_avg))

			epoch_avg_A2B_des_loss_vis.draw_figure()
			epoch_avg_A2B_gen_loss_vis.draw_figure()
			epoch_avg_A2B_cycle_loss_vis.draw_figure()
			epoch_avg_B2A_des_loss_vis.draw_figure()
			epoch_avg_B2A_gen_loss_vis.draw_figure()
			epoch_avg_B2A_cycle_loss_vis.draw_figure()

			self.save(args.checkpoint_dir, counter)
					


	def save(self, checkpoint_dir, step):
		model_name = "cyclegan.model"
		model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
		    os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoint...")

		model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
		    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		    self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
		    return True
		else:
		    return False


	def test(self, args):
		"""Test cyclegan"""
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		if args.which_direction == 'AtoB':
		    sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
		elif args.which_direction == 'BtoA':
		    sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
		else:
		    raise Exception('--which_direction must be AtoB or BtoA')

		if self.load(args.checkpoint_dir):
		    print(" [*] Load SUCCESS")
		else:
		    print(" [!] Load failed...")

		# write html for visual comparison
		index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
		index = open(index_path, "w")
		index.write("<html><body><table><tr>")
		index.write("<th>name</th><th>input</th><th>output</th></tr>")

		out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
		    self.testA, self.test_B)

		for sample_file in sample_files:
		    print('Processing image: ' + sample_file)
		    sample_image = [load_test_data(sample_file, args.fine_size)]
		    sample_image = np.array(sample_image).astype(np.float32)
		    image_path = os.path.join(args.test_dir,
		                              '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
		    fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
		    save_images(fake_img, [1, 1], image_path)
		    index.write("<td>%s</td>" % os.path.basename(image_path))
		    index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
		        '..' + os.path.sep + sample_file)))
		    index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
		        '..' + os.path.sep + image_path)))
		    index.write("</tr>")
		index.close()
