import tensorflow as tf
from tensorflow import keras
import numpy as np
import imagesToTFRecord as imagesToTfRecord
from imagesToTFRecord import imagesToTfRecord

tf.enable_eager_execution()
IM_SIZE = 96
def runModel(image_ds, label_ds):

    train_images = image_ds
    print("train images shape = ", train_images.shape)
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(96, 96, 3)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=image_ds,
              y=label_ds,
              batch_size=None,
              epochs=5,
              verbose=1,
              callbacks=None,
              validation_split=0.0,
              validation_data=None,
              shuffle=False,
              class_weight=None,
              sample_weight=None,
              initial_epoch=0,
              steps_per_epoch=100,
              validation_steps=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False)

    model.summary()

    """Evaluate accuracy"""
    # Compare how the model performs on the test dataset:
    test_loss, test_acc = model.evaluate(image_ds,
                                         y=label_ds,
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
    predictions = model.predict(image_ds,
                                batch_size=None,
                                verbose=1,
                                steps=32,
                                max_queue_size=10,
                                workers=1,
                                use_multiprocessing=False)
    print(predictions[0:10])
    print(np.argmax(predictions[0]))



