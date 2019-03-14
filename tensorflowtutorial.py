from __future__ import absolute_import, division, print_function
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
### This guide uses tf.keras, a high-level API to build and train models in TensorFlow.
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

""" TensorFlow Noobie Tutorial """

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
    def __init__(self):
        super(TensorFlow, self).__init__()
        # Each image is mapped to a single label. Since the class names are not included with the dataset,
        # store them here to use later when plotting the images:
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


    def mainDef(self):
        """Import the Fashion MNIST dataset"""
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



        """Explore the data"""
        # Loading the dataset returns four NumPy arrays: train_images and train_labels
        # The model is tested against the test set, the test_images and test_labels arrays
        train_images.shape
        print(len(train_labels))
        train_labels
        print(len(test_labels))

        train_images = train_images / 255.0     # Divide by 255.0 to scale values to a range of 0 to 1

        """Preprocess the data"""
        # Preprocessing example
        self.showPlotShoe(train_images)

        test_images = test_images / 255.0
        # Display the first 25 images from the training set and display the class name below each image
        self.showPlotImages(train_images, train_labels)

        """Build the model"""
        """ Building the neural network requires configuring the layers of the model, then compiling the model. 
        The basic building block of a neural network is the layer. Layers extract representations from the data 
        fed into them. And, hopefully, these representations are more meaningful for the problem at hand. Most 
        of deep learning consists of chaining together simple layers. Most layers, 
        like tf.keras.layers.Dense, have parameters that are learned during training."""
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        """ Compile the model
        Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:

        Loss function —This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
        Optimizer —This is how the model is updated based on the data it sees and its loss function.
        Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified."""
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        """ Train the model
        Training the neural network model requires the following steps:
        
        Feed the training data to the model—in this example, the train_images and train_labels arrays.
        The model learns to associate images and labels.
        We ask the model to make predictions about a test set—in this example, the test_images array. We verify that the predictions match the labels from the test_labels array.
        To start training, call the model.fit method—the model is 
        "fit" to the training data:"""
        model.fit(train_images, train_labels, epochs=5)

        """Evaluate accuracy"""
        # Compare how the model performs on the test dataset:
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)

        """Make predictions"""
        # With the model trained, we can use it to make predictions about some images
        predictions = model.predict(test_images)
        print(predictions[0])
        print(np.argmax(predictions[0]))
        print(test_labels[0])

        self.predictSingleImage(12, predictions, test_labels, test_images)

        # Plot the first X test images, their predicted label, and the true label
        # Color correct predictions in blue, incorrect predictions in red
        self.predictArrayOfImages(predictions, test_labels, test_images, row=5, col=3)


        img = test_images[0]
        print(img.shape)
        img = (np.expand_dims(img, 0))
        predictions_single = model.predict(img)
        print(predictions_single)

        self.plot_value_array(0, predictions_single, test_labels)
        _ = plt.xticks(range(10), self.class_names, rotation=45)
        print(np.argmax(predictions_single[0]))

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


if __name__ == "__main__":
    TF = TensorFlow()
    print(TensorFlow.__doc__)
    TF.mainDef()
