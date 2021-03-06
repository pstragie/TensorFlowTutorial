from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import load_model
import time
import pathlib
import random
import matplotlib.pyplot as plt
tf.enable_eager_execution()
tf.VERSION


AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/trainjpg/"
TEST_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/testjpg/"
TRAIN_LABELS = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/train_labels.csv"
OUT_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/"
TFRECORD = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/"
class_names = {0: "Negative", 1: "Positive"}
data_root = pathlib.Path(TRAIN_DIR)
BEGIN = 0
EINDE = 50000
BATCH_SIZE = 32
all_image_paths = list(data_root.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]
image_count = len(all_image_paths)
#print(image_count)

label_root = pathlib.Path(TRAIN_LABELS)
#print(all_image_paths[:10])

def load_labels():
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

label_to_index = load_labels()
all_image_paths = [TRAIN_DIR+naam+".jpg" for naam in label_to_index.keys()]
all_image_labels = [label for label in label_to_index.keys()]

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [96, 96])
  image /= 255.0  # normalize to [0,1] range
  return image
"""TFRecord File"""
def writeToTFRecord(all_image_paths):
    image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.read_file)
    tfrec = tf.data.experimental.TFRecordWriter(OUT_DIR+'images.tfrec')
    tfrec.write(image_ds)
def readFromTFRecord(label_ds, filename="images.tfrec"):
    print("Reading from TFRecord.........")
    image_ds = tf.data.TFRecordDataset(OUT_DIR+filename).map(preprocess_image)
    # zip with the labels to get (image, label) pairs
    ds = tf.data.Dataset.zip((image_ds, label_ds))
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    print(ds)
    return ds

all_image_labels = [label for label in label_to_index.values()][BEGIN:EINDE]
#all_image_labels = tf.expand_dims(all_image_labels, axis=-1)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
dataset = readFromTFRecord(label_ds, 'images.tfrec')
train_images, train_labels = next(iter(dataset))
test_images, test_labels = next(iter(dataset))

train_images = train_images[0:1000]
train_labels = train_labels[0:1000]
test_images = test_images[1000:2000]
test_labels = test_labels[1000:2000]


def predictSingleImage(i, predictions, test_labels, test_images):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions, test_labels)
    plt.show()

def showPlotShoe(train_images):
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

def showPlotImages(train_images, train_labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])
    plt.show()
"""Preprocess the data"""
# Preprocessing example
#showPlotShoe(train_images)


"""Build the model"""
""" Building the neural network requires configuring the layers of the model, then compiling the model. 
The basic building block of a neural network is the layer. Layers extract representations from the data 
fed into them. And, hopefully, these representations are more meaningful for the problem at hand. Most 
of deep learning consists of chaining together simple layers. Most layers, 
like tf.keras.layers.Dense, have parameters that are learned during training."""


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(96, 96, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])


""" Compile the model
Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:

Loss function —This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
Optimizer —This is how the model is updated based on the data it sees and its loss function.
Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified."""

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

""" Train the model
Training the neural network model requires the following steps:

Feed the training data to the model—in this example, the train_images and train_labels arrays.
The model learns to associate images and labels.
We ask the model to make predictions about a test set—in this example, the test_images array. We verify that the predictions match the labels from the test_labels array.
To start training, call the model.fit method—the model is 
"fit" to the training data:"""

model.fit(x=dataset,
          y=None,
          batch_size=None,
          epochs=3,
          verbose=1,
          callbacks=None,
          validation_split=0.0,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=313,
          validation_steps=None,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False)
model.summary()

"""model.save('my_model.h5', overwrite=True, include_optimizer=True)
model.save_weights('model_weights.h5', overwrite=True, save_format=None)
del model
model = load_model('my_model.h5')
"""

"""Evaluate accuracy"""
# Compare how the model performs on the test dataset:
test_loss, test_acc = model.evaluate(dataset,
                                     y=None,
                                     batch_size=32,
                                     verbose=1,
                                     sample_weight=None,
                                     steps=32,
                                     max_queue_size=10,
                                     workers=1,
                                     use_multiprocessing=False
                                     )
print('Test accuracy:', test_acc)

"""Make predictions"""
# With the model trained, we can use it to make predictions about some images
predictions = model.predict(dataset,
                            batch_size=None,
                            verbose=1,
                            steps=32,
                            max_queue_size=10,
                            workers=1,
                            use_multiprocessing=False)
print(predictions[0:10])
print(np.argmax(predictions[0]))


def predictArrayOfImages(predictions, test_labels, test_images, row=5, col=3):
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
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()




def plot_image(i, predictions_array, true_label, img):
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

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100 * np.max(predictions_array),
                                         true_label),
                                         color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


#predictSingleImage(12, predictions, test_labels, test_images)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
predictArrayOfImages(predictions, test_labels, test_images, row=5, col=3)


img = test_images[0]
print("img_shape: ", img.shape)
img = (np.expand_dims(img, 0))
predictions_single = model.predict(img)
print("predictions_single: ", predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
print("np argmax: ", np.argmax(predictions_single[0]))