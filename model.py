import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import imagesToTFRecord as imagesToTfRecord
from imagesToTFRecord import imagesToTfRecord

tf.enable_eager_execution()
IM_SIZE = 96

def plotImage(images):
    plt.figure()
    plt.imshow(images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

def plotGridOfImages(train_images, label_ds, row=5, col=5):
    plt.figure(figsize=(row, row))
    for i in range(row*col):
        plt.subplot(row, col, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(label_ds[label_ds[i]])
    plt.show()

def runModel(image_ds, label_ds, test_images, test_labels, epochs=5, steps_per_epoch=100):

    train_images = image_ds
    print("train images shape = ", train_images.shape)
    #plotImage(train_images)
    #plotGridOfImages(train_images, label_ds, row=5, col=5)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(96, 96, 3)),  # This layer has no parameters to learn
        keras.layers.Dense(256, activation=tf.nn.relu),  # 128 nodes
        keras.layers.Dense(1, activation=tf.nn.sigmoid)  # array of 2 probability scores
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit( x=image_ds,
               y=label_ds,
               batch_size=None,
               epochs=epochs,
               verbose=1,
               callbacks=None,
               validation_split=0.0,
               validation_data=None,
               shuffle=False,
               class_weight=None,
               sample_weight=None,
               initial_epoch=0,
               steps_per_epoch=steps_per_epoch,
               validation_steps=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False)

    model.summary()


    """Evaluate accuracy"""
    # Compare how the model performs on the test dataset:
    test_loss, test_acc = model.evaluate( test_images,
                                          y=test_labels,
                                          batch_size=32,
                                          verbose=1,
                                          sample_weight=None,
                                          steps=10,
                                          max_queue_size=10,
                                          workers=1,
                                          use_multiprocessing=False
                                          )
    print('Test accuracy:', test_acc)


    """Make predictions"""
    # With the model trained, we can use it to make predictions about some images
    predictions = model.predict( test_images,
                                 batch_size=32,
                                 verbose=1,
                                 steps=10,
                                 max_queue_size=10,
                                 workers=1,
                                 use_multiprocessing=False)
    print(predictions[0:10])
    print(np.argmax(predictions[0]))



