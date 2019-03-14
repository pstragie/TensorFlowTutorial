from __future__ import absolute_import, division, print_function
import os, argparse, sys
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import pandas as pd
### This guide uses tf.keras, a high-level API to build and train models in TensorFlow.
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
HEIGHT = 96
WIDTH = 96
CHANNELS = 1
FLAGS = None

class Histopathology:
    """Histopathology. Recognize a tumor."""
    print(tf.__version__)
    print(tf.keras.__version__)

    def __init__(self):
        super(Histopathology, self).__init__()
        self.class_category = ['True', 'False']
        self.filePathTrain = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/trainjpg/"
        self.filePathTest = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/testjpg/"
        self.fileLabelsTrain = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/train_labels.csv"
        self.train_tensor = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/data/train_images_0.pickle"
        self.test_tensor = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/data/test_images_0.pickle"
        self.label_tensor = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/data/train_labels_0.pickle"

        #self.labels_dict = self.load_labels()
        #self.id_labels = {Image:Id for Image, Id in zip(self.names, self.labels)}

    def load_labels(self):
        bestand = open(self.fileLabelsTrain, "r")
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

    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        return image, label

    def dataset(self):
        labels_dict = self.load_labels()
        print("len: ", len(labels_dict))
        # step 1
        #filenames = tf.constant([self.filePathTrain+name+".jpg" for name in labels_dict.keys()][0:10])
        #labels = tf.constant([label for label in labels_dict.values()][0:10])

        # step 2: create a dataset returning slices of 'filenames'
        d = tf.data.Dataset.from_tensors(self.train_tensor)
        print(d.output_shapes)
        print(d.output_types)
        dataset = tf.data.Dataset.from_tensor_slices((self.train_tensor, self.label_tensor))
        print(dataset.output_types)
        print(dataset.output_shapes)
        # step 3: parse every image in the dataset using map
        dataset = dataset.map(self._parse_function)
        #dataset = dataset.batch(10)

        iterator = dataset.make_one_shot_iterator()
        next_image, next_label = iterator.get_next()

        with tf.Session() as sess:
            sess.run(next_image)
            self.showPlotImages(next_image, next_label)




    def importTrainSet(self, filePathTrain, fileLabelsTrain):
        """Import the training set."""
        # get data

        train_images = ""
        train_labels = ""
        # subselection
        train_images = train_images[:100]
        train_labels = train_labels[:100]
        return (train_images, train_labels)

    def importCSVLabels(self, fileLabels):
        label = pd.read_csv(self.fileLabelsTrain)
        labels_dict = {}
        for line in label:
            # print("line: " + line)
            name, label = line.split(",")
            labels_dict[name] = int(label.strip("\n"))
            print(len(labels_dict.keys()))
            return labels_dict

    def importTestSet(self, filePathTest):
        """Import the test set."""
        test_images = ""
        test_labels = ""
        # subselection
        test_images = test_images[:100]
        test_labels = test_labels[:100]
        return (test_images, test_labels)

    def exploreData(self, train_images, train_labels):
        """Explore the data."""


        train_images.shape

        #(test_images, test_labels) = self.importTestSet(self.filePathTest)
        #print("Number of test labels: " + len(test_labels))

        self.showPlotHisto(train_images)

        #test_images = test_images / 255.0
        self.showPlotImages(train_images, train_labels)

    def buildModel(self):
        model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation=tf.keras.activations.softmax)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])

        return model

    def showPlotHisto(self, train_images):
        """Plot one image."""
        plt.figure()
        plt.imshow(train_images[0])
        plt.colorbar()
        plt.grid(False)
        plt.show()

    def showPlotImages(self, train_images, train_labels):
        """Plot an array of images with labels."""
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[train_labels[i]])
        plt.show()




if __name__ == "__main__":
    H = Histopathology()
    H.dataset()


