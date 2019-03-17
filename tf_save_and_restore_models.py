from __future__ import absolute_import, division, print_function
import os
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
### This guide uses tf.keras, a high-level API to build and train models in TensorFlow.
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from imagesToTFRecord import imagesToTfRecord
""" TensorFlow Noobie Tutorial """


TRAIN_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/trainjpg/"
TEST_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/testjpg/"
TRAIN_LABELS = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/train_labels.csv"
OUT_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/Records/"



class TensorFlow:
    """
    Explore, Build, Learn

    This guide trains a neural network model to classify images of clothing,
    like sneakers and shirts. It's okay if you don't understand all the details,
    this is a fast-paced overview of a complete TensorFlow program with the details
    explained as we go.

    This guide uses tf.keras, a high-level API to build and train models in TensorFlow.
    """
    print(tf.__version__)
    print(tf.keras.__version__)
    # Each image is mapped to a single label. Since the class names are not included with the dataset,
    # store them here to use later when plotting the images:
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def mainDef(self):
        """Import the Fashion MNIST dataset"""
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        BEGIN = 0
        EINDE = 5000
        #S = startML()
        tfrecord_filename = OUT_DIR + "histo_" + str(BEGIN) + "_" + str(EINDE) + "_TF_record.tfrecords"
        T2 = imagesToTfRecord(tfrecord_filename, TRAIN_DIR, TRAIN_LABELS, 96, 96, 3)
        parsed_image_dataset = T2.readRecord(tfrecord_filename)
        print(parsed_image_dataset)
        # for parsed_record in parsed_image_dataset.take(5):
        # print("parsed_record: ", repr(parsed_record))
        # labels = parsed_record[1][0]
        parsed_image_dataset.batch(32)
        iterator = parsed_image_dataset.make_one_shot_iterator()
        image_ds, label_ds = iterator.get_next()

        test_images, test_labels = image_ds, label_ds
        train_images, train_labels = image_ds, label_ds
        """Explore the data"""
        # Loading the dataset returns four NumPy arrays: train_images and train_labels
        # The model is tested against the test set, the test_images and test_labels arrays
        #train_images.shape

        #train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0  # Divide by 255.0 to scale values to a range of 0 to 1
        #test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

        """Preprocess the data"""
        # Preprocessing example
        #self.showPlotShoe(train_images)

        #test_images = test_images / 255.0
        # Display the first 25 images from the training set and display the class name below each image
        #self.showPlotImages(train_images, train_labels)

        """Define a model"""
        model = self.create_model()
        model.fit(train_images, train_labels, epochs=5, steps_per_epoch=100)
        model.summary()

        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        model = self.create_model()

        model.fit(train_images, train_labels,  epochs=5,
                  steps_per_epoch=100,
                  validation_data=(test_images, test_labels),
                  callbacks=[cp_callback])  # pass callback to training

        # This may generate warnings related to saving the state of the optimizer.
        # These warnings (and similar warnings throughout this notebook)
        # are in place to discourage outdated usage, and can be ignored.

        """Create a new, untrained model."""
        model = self.create_model()
        model.fit(test_images, test_labels, epochs=5,
                  steps_per_epoch=100)
        loss, acc = model.evaluate(test_images,
                                     y=test_labels,
                                     batch_size=32,
                                     verbose=1,
                                     sample_weight=None,
                                     steps=10,
                                     max_queue_size=10,
                                     workers=1,
                                     use_multiprocessing=False
                                     )
        print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

        """Then load the weights from the checkpoint, and re-evaluate."""
        model.load_weights(checkpoint_path)
        loss, acc = model.evaluate(test_images, test_labels)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

        """Checkpoint callback options."""
        # include the epoch in the file name. (uses `str.format`)
        checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_weights_only=True,
            # Save weights, every 5-epochs.
            period=5)

        model = self.create_model()
        model.save_weights(checkpoint_path.format(epoch=0))
        model.fit(train_images, train_labels,
                  epochs=10, callbacks=[cp_callback],
                  steps_per_epoch=100,
                  validation_data=(test_images, test_labels),
                  verbose=0)

        latest = tf.train.latest_checkpoint(checkpoint_dir)
        latest

        """To test, reset the model and load the latest checkpoint."""
        model = self.create_model()
        model.fit(test_images, test_labels, epochs=5,
                  steps_per_epoch=100)
        model.load_weights(latest)
        loss, acc = model.evaluate(test_images, test_labels)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

        """Manually save weights."""
        # Save the weights
        model.save_weights('./checkpoints/my_checkpoint')

        # Restore the weights
        model = self.create_model()
        model.fit(test_images, test_labels, epochs=5, steps_per_epoch=100)
        model.load_weights('./checkpoints/my_checkpoint')

        loss, acc = model.evaluate(test_images, test_labels)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

        """Save the entire moel: as an hdf5 file."""
        model = self.create_model()

        model.fit(train_images, train_labels, epochs=5, steps_per_epoch=100)

        # Save entire model to a HDF5 file
        model.save('my_model.h5')

        """ Now recreate the model from that file."""
        # Recreate the exact same model, including weights and optimizer.
        new_model = keras.models.load_model('my_model.h5')
        new_model.summary()

        """Check its accuracy."""
        loss, acc = new_model.evaluate(test_images, test_labels)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


        """Build a fresh model:"""
        model = self.create_model()

        model.fit(train_images, train_labels, epochs=5, steps_per_epoch=100)

        """Create a saved_model"""
        saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")

        """Reload a fresh keras model from the saved model."""
        new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
        new_model.summary()

        """Run the restored model."""
        # The model has to be compiled before evaluating.
        # This step is not required if the saved model is only being deployed.

        new_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

        # Evaluate the restored model.
        loss, acc = new_model.evaluate(test_images, test_labels)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))



    def predictArrayOfImages(self, predictions, test_labels, test_images, row=5, col=3):
        """ Predict an array of images.
        Args:
            predictions
            test_labels
            test_images
            row (int): The number of rows
            col (int): The number columns

        Returns:
            plt: A Plot of images (col x row)
        """
        num_rows = row
        num_cols = col
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_image(i, predictions, test_labels, test_images)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_value_array(i, predictions, test_labels)
        plt.show()

    def predictSingleImage(self, i, predictions, test_labels, test_images):
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        self.plot_image(i, predictions, test_labels, test_images)
        plt.subplot(1, 2, 2)
        self.plot_value_array(i, predictions, test_labels)
        plt.show()

    def showPlotShoe(self, train_images):
        plt.figure()
        plt.imshow(train_images[0])
        plt.colorbar()
        plt.grid(False)
        plt.show()

    def showPlotImages(self, train_images, train_labels):
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[train_labels[i]])
        plt.show()

    def plot_image(self, i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             self.class_names[true_label]),
                   color=color)

    def plot_value_array(self, i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(96, 96, 3)),  # This layer has no parameters to learn
            keras.layers.Dense(256, activation=tf.nn.relu),  # 128 nodes
            keras.layers.Dense(1, activation=tf.nn.sigmoid)  # array of 2 probability scores
        ])
        '''
        model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation=tf.keras.activations.softmax)
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        '''
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

if __name__ == "__main__":
    TF = TensorFlow()
    print(TensorFlow.__doc__)
    TF.mainDef()
