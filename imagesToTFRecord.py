import tensorflow as tf
import os, sys
import glob

tf.enable_eager_execution()
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
        print('resizing values: %d en %d' % (x_px, y_px))
        image = tf.image.decode_jpeg(image, channels=self.preprocessing_channels)
        image = tf.image.resize_images(image, [x_px, y_px])
        image /= 255  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(self, path):
        """ Read and preprocess image
            @Return preprocessed image
        """
        image = tf.read_file(path)
        return self.preprocess_image(image)


    def _write_to_tf_record(self, tfrecord_filename, image_path, begin=0, einde=-1, x_px=96, y_px=96):
        """ Create TFRecord file """
        print("Write to TFRecord...........")
        writer = tf.python_io.TFRecordWriter(tfrecord_filename)
        images = [f for f in os.listdir(image_path) if os.path.isfile( os.path.join(image_path, f) )]
        label_dict = self.load_labels(self.label_file)
        #print("images: ", images)
        for index, image in enumerate(images[begin:einde]):
            if not index % 1000:
                print("Train data: {}/{}".format(index, len(images)))
                sys.stdout.flush()
            img = self.load_and_preprocess_image(image_path+image)
            print("image = ", img)
            print(img.shape)
            rows = img.shape[0]
            cols = img.shape[1]
            depth = img.shape[2]

            img_id = os.path.basename(image).strip(".jpg")
            label = label_dict[img_id]
            feature = {'label': self._int64_feature(int(label)),
                       'height': self._int64_feature(rows),
                       'width': self._int64_feature(cols),
                       'depth': self._int64_feature(depth),
                       'image': self._bytes_feature(tf.compat.as_bytes(img.__str__()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Write the serialized example
            print("example: ", example)
            writer.write(example.SerializeToString())
        writer.close()
        sys.stdout.flush()


    def _read_from_tf_record(self, folder):
        """ Read a TFRecord file
            NOT Shuffled are randomized!
        """
        reader = tf.TFRecordReader()
        filenames = glob.glob(folder+'*.tfrecords')
        filename_queue = tf.train.string_input_producer(
            filenames)
        _, serialized_example = reader.read(filename_queue)
        feature_set = { 'image': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.int64) }
        features = tf.parse_single_example( serialized_example, features = feature_set )
        label = features['label']
        image = features['image']
        with tf.Session() as sess:
            print(sess.run([image, label]))

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
    OUT_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/Records"
    BEGIN = 0
    EINDE = 5

    y_px, x_px, preprocessing_channels = 96, 96, 3
    tfrecord_filename = OUT_DIR+"histo_"+str(BEGIN) + "_" + str(EINDE) + "_TF_record.h5"

    print("Image Height: %d, Width: %d, Channels: %d" % (y_px, x_px, preprocessing_channels))
    # Call imagesToTfRecord class to build dataset and store in TFRecord
    T2 = imagesToTfRecord(tfrecord_filename, TRAIN_DIR, TRAIN_LABELS, x_px, y_px, preprocessing_channels)
    T2.writeRecord(tfrecord_filename, TRAIN_DIR, begin=BEGIN, einde=EINDE)
