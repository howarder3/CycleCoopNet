import math 
import numpy as np 
import scipy

def clip_by_value(input_, min=0, max=1):
    return np.minimum(max, np.maximum(min, input_))

def img2cell(images, row_num=10, col_num=10, margin_syn=2):
    [num_images, image_size] = images.shape[0:2]
    num_cells = int(math.ceil(num_images / (col_num * row_num)))
    cell_image = np.zeros((num_cells, row_num * image_size + (row_num-1)*margin_syn,
                           col_num * image_size + (col_num-1)*margin_syn, 3))
    for i in range(num_images):
        cell_id = int(math.floor(i / (col_num * row_num)))
        idx = i % (col_num * row_num)
        ir = int(math.floor(idx / col_num))
        ic = idx % col_num
        temp = clip_by_value(np.squeeze(images[i]), -1, 1)
        temp = (temp + 1) / 2 * 255
        temp = clip_by_value(np.round(temp), min=0, max=255)
        gLow = np.min(temp, axis=(0, 1, 2))
        gHigh = np.max(temp, axis=(0, 1, 2))
        temp = (temp - gLow) / (gHigh - gLow)
        cell_image[cell_id, (image_size+margin_syn)*ir:image_size + (image_size+margin_syn)*ir,
                    (image_size+margin_syn)*ic:image_size + (image_size+margin_syn)*ic,:] = temp
    return cell_image

def saveSampleResults(sample_results, filename, col_num=10, margin_syn=2):
    cell_image = img2cell(sample_results, col_num, col_num, margin_syn)
    scipy.misc.imsave(filename, np.squeeze(cell_image))





def sample_model(self, sample_dir, epoch, idx):
    sample_images = self.load_random_samples()
    samples, d_loss, g_loss = self.sess.run(
        [self.fake_B_sample, self.d_loss, self.g_loss],
        feed_dict={self.real_data: sample_images}
    )
    save_images(samples, [self.batch_size, 1],
                './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def save(self, checkpoint_dir, step):
    model_name = "pix2pix.model"
    model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

def inverse_transform(images):
    return (images+1.)/2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img