import tensorflow as tf
import os

class imagesToTfRecord:


    def __init__(self, record_filepath, image_folder, label_file, batch_size=32, autotune="AUTOTUNE", x_px=96, y_px=96, preprocessing_channels=3):
        super(imagesToTfRecord, self).__init__()
        self.record_filepath = record_filepath
        self.batch_size = batch_size
        self.autotune = "tf.data.experimental.%s", autotune
        self.preprocessing_channels = preprocessing_channels
        self.image_folder = image_folder
        self.label_file = label_file
        self.x_px = x_px
        self.y_px = y_px

    def load_labels(self, label_file, begin=0, einde=-1):
        """ Labels laden in dictionary
            @Return Dictionary, keys: image_id, values: label
        """
        bestand = open(label_file, "r")
        header = bestand.readline()
        print("header: ", header)
        labels_dict = {}
        for line in bestand:
            for _ in range(begin, einde):
                name, label = line.split(",")
                labels_dict[name] = str(label.strip("\n"))
        bestand.close()
        print("lengte: ", len(labels_dict.keys()))
        return labels_dict

    def buildDataset(self, label_file, begin=0, einde=-1):
        """Build a tf.data.Dataset"""
        label_to_index = self.load_labels(label_file, begin, einde)
        all_image_paths = [self.image_folder + naam + ".jpg" for naam in label_to_index.keys()]
        all_image_labels = [label for label in label_to_index.keys()]

        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_ds = path_ds.map(self.load_and_preprocess_image, num_parallel_calls=self.autotune)

        all_labels = [label for label in label_to_index.values()]
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_labels, tf.int64))

        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        print('image shape: ', image_label_ds.output_shapes[0])
        print('label shape: ', image_label_ds.output_shapes[1])
        print('types: ', image_label_ds.output_types)
        print()
        print('image_label_ds: ', image_label_ds)

        ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
        return ds

    def image_count(self, dataset):
        """ Aantal afbeeldingen in de dataset
            @Return Integer
        """
        image_count = len(dataset)
        return image_count

    def preprocess_image(self, image, x_px=96, y_px=96):
        """ Decode, resize and normalize image
            @Return image
        """
        image = tf.image.decode_jpeg(image, channels=self.preprocessing_channels)
        image = tf.image.resize_images(image, [self.x_px, self.y_px])
        image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(self, path):
        """ Read and preprocess image
            @Return preprocessed image
        """
        print("path: ", path)
        image = tf.read_file(path)
        return self.preprocess_image(image)


    def _write_to_tf_record(self, dataset):
        """ Create TFRecord file """
        #image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.read_file)
        tfrecord = tf.data.experimental.TFRecordWriter(self.record_filepath)
        tfrecord.write(dataset)

    def _read_from_tf_record(self, record_filepath):
        """ Read a TFRecord file
            NOT Shuffled are randomized!
        """
        print("Reading from TFRecord.........")
        ds = tf.data.TFRecordDataset(record_filepath).map(self.preprocess_image)
        # zip with the labels to get (image, label) pairs
        #ds = tf.data.Dataset.zip((image_ds, label_ds))
        #ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self.image_count(ds)))
        ds = ds.batch(self.batch_size).prefetch(self.autotune)
        print(ds)
        return ds

    def verifyRecord(self, filename):
        for example in tf.python_io.tf_record_iterator(filename):
            result = tf.train.Example.FromString(example)
        return result

    def writeRecord(self, begin=0, einde=-1):
        """ Write images and labels to TFRecord
            @begin: slice array begin
            @einde: slice array einde
        """
        dataset = self.buildDataset(self.label_file, begin=begin, einde=einde)
        self._write_to_tf_record(dataset)
        example = self.verifyRecord(self.record_filepath)
        print(example)

    def readRecord(self, record_filepath):
        """ Read a TFRecord file as dataset
            @Return tf.dataset
        """
        dataset = self._read_from_tf_record(record_filepath)
        return dataset

class imageResolution:

    def __init__(self):
        super(imageResolution, self).__init__()

    def getResolution(self, image):
        return image.shape()

if __name__ == '__main__':
    TRAIN_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/trainjpg/"
    TEST_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/testjpg/"
    TRAIN_LABELS = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/train_labels.csv"
    OUT_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/"


    y_px, x_px, preprocessing_channels = 96, 96, 3
    print("Image Height: %d, Width: %d, Channels: %d" % (y_px, x_px, preprocessing_channels))
    # Call imagesToTfRecord class to build dataset and store in TFRecord
    T2 = imagesToTfRecord(OUT_DIR+"histo_1000_TF_record.h5", TRAIN_DIR, TRAIN_LABELS, x_px, y_px, preprocessing_channels)
    T2.writeRecord(begin=0, einde=1000)
