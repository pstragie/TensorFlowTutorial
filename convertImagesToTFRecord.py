import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

TRAIN_DIR = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/trainjpg/"
TRAIN_LABELS = "/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/train_labels.csv"
#construct list for image paths and their corresponding masks in the same order
def load_labels():
    bestand = open(TRAIN_LABELS, "r")
    header = bestand.readline()
    print("header: ", header)
    id_label = []
    for line in bestand:
        # print("line: " + line)
        name, label = line.split(",")
        id_label.append((name, label.strip("\n")))
    bestand.close()
    print("lengte: ", len(id_label))
    return id_label

label_pairs = load_labels()
#image_path = "../data/train/*.jpg"
image_path = []
mask_path = []
for (name, label) in label_pairs:
    image_path.append(TRAIN_DIR+name+".jpg")
    mask_path.append(label)

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# path to where the tfrecords will be stored (change to your customized path).
# I can't run this here in Kaggle because of permission.
# Any comment on how to write temporary files in Kaggle will be appreciated.
tfrecords_filename = '/media/pieter/bd7f5343-172e-43f6-8e0f-417aa96d3113/Downloads/ML/Histopathology/training_data.tfrecords'
#get a writer for the tfrecord file.
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
#write data/masks into tfrecords
for i in range(len(image_path)):
    img = np.array(mpimg.imread(image_path[i]))
    mask = np.array(mpimg.imread(mask_path[i]))

    height = img.shape[0]
    width = img.shape[1]
    img_raw = img.tostring()
    label_string = mask.tostring()
    #save the heights and widths as well so, which
    #are needed when decoding from tfrecords back to images
    example = tf.train.Example(features=tf.train.Features(feature={
                                                          'height': _int64_feature(height),
                                                          'width': _int64_feature(width),
                                                          'image_raw': _bytes_feature(img_raw),
                                                          'label': label_string}
                                                          ))
    writer.write(example.SerializeToString())
writer.close()



#run the following to verify the created tfrecord file.
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    height = int(example.features.feature['height'].int64_list.value[0])
    width = int(example.features.feature['width'].int64_list.value[0])
    img_string = (example.features.feature['image_raw'].bytes_list.value[0])
    mask_string = (example.features.feature['label'])
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    #mask_1d = np.fromstring(mask_string, dtype=np.uint8)
    #reshape back to their original shape from a 1D array read from tfrecords
    img = img_1d.reshape((height, width, -1))
    #mask = mask_1d.reshape((height, width))
    plt.imshow(img)
    plt.show()
