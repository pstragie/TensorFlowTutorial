from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pathlib
import os
import random

tf.enable_eager_execution()
tf.VERSION

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/trainjpg/"
TRAIN_LABELS = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/train_labels.csv"
data_root = TRAIN_DIR
data_root = pathlib.Path(data_root)
print(data_root)

print(len([name for name in os.listdir(data_root)]))

class HistopathologyDataset:
    """Create a dictionary from labels file"""
    def __init__(self):
        super(HistopathologyDataset, self).__init__()

    """Load and format the images."""
    def load_labels(self):
        bestand = open(TRAIN_LABELS, "r")
        header = bestand.readline()
        print("header: ", header)
        labels_dict = {}
        for line in bestand:
            # print("line: " + line)
            name, label = line.split(",")
            labels_dict[name] = int(label.strip("\n"))
        bestand.close()
        print("lengte: ", len(labels_dict.keys()))
        return labels_dict

    def preprocess_image(self, image, label):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_images(image, [96, 96])
        image /= 255 # normalize to [0, 1] range
        return image, label

    def load_and_preprocess_image(self, path, label):
        image = tf.read_file(path)

        return self.preprocess_image(image, label)

    def load_and_preprocess_from_path_label(self, path, label):
        return self.load_and_preprocess_image(path, label)

    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        return image, label

    def dataset(self):
        labels_dict = self.load_labels()
        print("len: ", len(labels_dict))
        # step 1
        filenames = tf.constant([TRAIN_LABELS+name+".jpg" for name in labels_dict.keys()][0:10])
        labels = tf.constant([label for label in labels_dict.values()][0:10])

        # step 2: create a dataset returning slices of 'filenames'
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        # step 3: parse every image in the dataset using map
        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(2)

        # step 4: create iterator and final input tensor
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        all_image_paths = [name for name in labels_dict][0:10]  # Limited for test purposes
        ds = tf.data.Dataset.from_tensor_slices((all_image_paths, [label for label in labels_dict.values()][0:10]))
        print('shape: ', repr(ds.output_shapes))
        print('type: ', ds.output_types)
        print()
        print(ds)
        image_label_ds = ds.map(self.load_and_preprocess_from_path_label)
        return image_label_ds

if __name__ == '__main__':
    CS = HistopathologyDataset()
    CS.dataset()