from model import runModel
import os
from imagesToTFRecord import imagesToTfRecord
import tensorflow as tf

class startML:

    def __init__(self, height=96, width=96, channels=3):
        super(startML, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels

    def create_tfrecord(self, tfrecord_filename, begin=0, einde=-1):
        """ Write images and labels to TFRecord file """
        # Call imagesToTfRecord class to build dataset and store in TFRecord
        T2.writeRecord(tfrecord_filename, TRAIN_DIR, begin=begin, einde=einde)

    def train(self, tfrecord_filename, begin=0, einde=1000):
        print("Start training with data")

        print("Image Height: %d, Width: %d, Channels: %d" % (self.height, self.width, self.channels))

        T2 = imagesToTfRecord(tfrecord_filename, TRAIN_DIR, TRAIN_LABELS, self.width, self.height, self.channels)
        if os.path.exists(tfrecord_filename):
            """ Read images and labels (tensors) from TFRecord file """
            parsed_image_dataset = T2.readRecord(tfrecord_filename)
            print(parsed_image_dataset)
            #for parsed_record in parsed_image_dataset.take(5):
                #print("parsed_record: ", repr(parsed_record))
                #labels = parsed_record[1][0]
            parsed_image_dataset.batch(32)
            iterator = parsed_image_dataset.make_one_shot_iterator()
            image_ds, label_ds = iterator.get_next()
            test_images, test_labels = image_ds, label_ds
            # print("iterator: ", iterator)

            runModel(image_ds, label_ds, test_images, test_labels)
        else:
            print("File does not exist: ", tfrecord_filename)


    def test(self):
        print("Test on unknown data")


if __name__ == "__main__":

    TRAIN_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/trainjpg/"
    TEST_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/testjpg/"
    TRAIN_LABELS = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/train_labels.csv"
    OUT_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/Records/"


    BEGIN = 0
    EINDE = 5000
    S = startML()
    tfrecord_filename =  OUT_DIR + "histo_" + str(BEGIN) + "_" + str(EINDE) + "_TF_record.tfrecords"
    T2 = imagesToTfRecord(tfrecord_filename, TRAIN_DIR, TRAIN_LABELS, 96, 96, 3)

    """ Write TFRecord """
    #S.create_tfrecord(tfrecord_filename, begin=BEGIN, einde=EINDE)

    with tf.Session() as sess:
        sess.run(tf.initialize_local_variables())
        tf.train.start_queue_runners()
        try:
            while True:
                data_batch = sess.run(S.train(tfrecord_filename, begin=0, einde=1000))

                # process data
        except tf.errors.OutOfRangeError:
            pass