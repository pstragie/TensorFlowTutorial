import tensorflow as tf
import os, sys
import numpy as np

import IPython.display as display
from PIL import Image, ImageDraw
import glob

print(tf.VERSION)

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

    def _int64_feature(self, value):
        """ Converting the values into features
        _int_64 is used for numeric values
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        """ _bytes is used for string/char values """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def load_labels(self, label_file, begin=0, einde=-1):
        """ Labels laden in dictionary
            @Return Dictionary, keys: image_id, values: label
        """
        bestand = open(label_file, "r")
        header = bestand.readline()
        print("header: ", header)
        labels_dict = {}
        for line in bestand:
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
        print(path_ds)
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
        image = tf.image.resize_images(image, [x_px, y_px])
        #image /= 255  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(self, path):
        """ Read and preprocess image
            @Return preprocessed image
        """
        image = tf.read_file(path)
        return self.preprocess_image(image)


    def _write_to_tf_record(self, tfrecord_filename, image_path, begin=0, einde=-1):
        """ Create TFRecord file """
        print("Write to TFRecord...........")
        writer = tf.python_io.TFRecordWriter(tfrecord_filename)

        # List of image paths, np array of labels
        images = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
        label_dict = self.load_labels(self.label_file)
        #print("images: ", images)
        print("Aantal afbeeldingen gekozen: %d" % (len(images[begin:einde])))

        # Loop over images and labels, wrap in TF Examples, write away to TFRecord file
        for index, image in enumerate(images[begin:einde]):
            img_id = os.path.basename(image).strip(".jpg")
            if not index % 1000:
                print("Train data: {}/{}".format(index, len(images[begin:einde])))
                sys.stdout.flush()
            #img = self.load_and_preprocess_image(image_path+image)
            img = Image.open(image_path+image)
            img = np.array(img.resize((96, 96)))
            rows = img.shape[0]
            cols = img.shape[1]
            depth = img.shape[2]
            image_string = open(image_path+image, 'rb').read()

            label = int(label_dict[img_id])
            image_name = str.encode(image)
            feature = {'image_name': self._bytes_feature(image_name),
                       'height': self._int64_feature(rows),
                       'width': self._int64_feature(cols),
                       'depth': self._int64_feature(depth),
                       'label': self._int64_feature(label),
                       'image_raw': self._bytes_feature(image_string)}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            #print("ex: ", repr(example))
            # Write the serialized example
            writer.write(example.SerializeToString())
        writer.close()
        print("Writing finished.")
        sys.stdout.flush()


    def _read_from_tf_record(self, filepath, x_px=96, y_px=96):
        """ Read a TFRecord file
            NOT Shuffled or randomized!
        """
        filenames = [filepath]


        #Create a dictionary describing the features.
        image_feature_description = {
            'image_name': tf.FixedLenFeature([], dtype=tf.string),
            'height': tf.FixedLenFeature([], dtype=tf.int64),
            'width': tf.FixedLenFeature([], dtype=tf.int64),
            'depth': tf.FixedLenFeature([], dtype=tf.int64),
            'label': tf.FixedLenFeature([], dtype=tf.int64),
            'image_raw': tf.FixedLenFeature([], dtype=tf.string)
        }


        def _parse_record(example_proto, clip=False):
            # Parse a single record into image, label
            example = tf.parse_single_example(example_proto, image_feature_description)

            height = tf.cast(example['height'], tf.int64)
            width = tf.cast(example['width'], tf.int64)
            depth = tf.cast(example['depth'], tf.int64)
            im_shape = tf.stack([height, width, depth])

            im = tf.image.decode_jpeg(example['image_raw'], channels=3)
            #im = tf.divide(tf.cast(im, tf.float32), tf.constant(255.0, dtype=tf.float32))
            im = tf.reshape(im, im_shape)

            diagnosis = tf.cast(example['label'], tf.int64)

            return im, diagnosis

        # Construct a TFRecordDataset
        ds_train = tf.data.TFRecordDataset(filepath)
        ds_train = ds_train.map(_parse_record)
        ds_train = ds_train.shuffle(100).repeat(5).batch(32)

        return ds_train
        """
        def _parse_image_function(example_proto, clip=False):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.parse_single_example(example_proto, image_feature_description)

        raw_image_dataset = tf.data.TFRecordDataset(filenames)
        parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
        parsed_image_dataset = parsed_image_dataset.shuffle(1000).batch(32)
        
        return parsed_image_dataset
        """


    def writeRecord(self, tfrecord_filename, TRAIN_DIR, begin=0, einde=-1):
        """ Write images and labels to TFRecord
            @begin: slice array begin
            @einde: slice array einde
        """
        self._write_to_tf_record(tfrecord_filename, TRAIN_DIR, begin=begin, einde=einde)

    def readRecord(self, record_filepath):
        """ Read a TFRecord file as dataset
            @Return tf.dataset
        """
        print("record filepath = ", record_filepath)
        #self._convert_tf_record_to_dataset(filepath=record_filepath)
        dataset = self._read_from_tf_record(record_filepath)
        return dataset

class imageResolution:

    def __init__(self):
        super(imageResolution, self).__init__()

    def getResolution(self, image):
        return image.shape()

