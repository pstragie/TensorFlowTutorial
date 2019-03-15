import ast
import numpy as np
import math
import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array as img_to_array
from tensorflow.keras.preprocessing.image import load_img as load_img


def load_image(image_path, size):
    # data augmentation logic such as random rotations can be added here
    return img_to_array(load_img(image_path, target_size=(size, size))) / 255.


class KagglePlanetSequence(tf.keras.utils.Sequence):
    """
    Custom Sequence object to train a model on out-of-memory datasets.
    """

    def __init__(self, df_path, data_path, im_size, batch_size, mode='train'):
        """
        df_path: path to a .csv file that contains columns with image names and labels
        data_path: path that contains the training images
        im_size: image size
        mode: when in training mode, data will be shuffled between epochs
        """
        self.df = pd.read_csv(df_path)
        self.im_size = im_size
        self.batch_size = batch_size
        self.mode = mode

        # Take labels and a list of image locations in memory
        self.wlabels = self.df['weather_labels'].apply(lambda x: ast.literal_eval(x)).tolist()
        self.glabels = self.df['ground_labels'].apply(lambda x: ast.literal_eval(x)).tolist()
        self.image_list = self.df['image_name'].apply(lambda x: os.path.join(data_path, x + '.jpg')).tolist()

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.batch_size)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch
        self.indexes = range(len(self.image_list))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return [self.wlabels[idx * self.batch_size: (idx + 1) * self.batch_size],
                self.glabels[idx * self.batch_size: (idx + 1) * self.batch_size]]

    def get_batch_features(self, idx):
        # Fetch a batch of images
        batch_images = self.image_list[idx * self.batch_size: (1 + idx) * self.batch_size]
        return np.array([load_image(im, self.im_size) for im in batch_images])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y