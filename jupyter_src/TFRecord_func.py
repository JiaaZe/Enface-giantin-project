import os
import tensorflow as tf
from tensorflow.keras.utils import Progbar


def image_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.tostring()])
    )


def mask_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.tostring()])
    )


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def tfrecord_example(image_id, image, mask):
    img_shape = image.shape
    feature = {
        'id': bytes_feature(image_id),
        'num': int64_feature(img_shape[0]),
        'height': int64_feature(img_shape[1]),
        'width': int64_feature(img_shape[2]),
        'image': image_feature(image),
        'mask': mask_feature(mask),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    image_feature_description = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'num': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, image_feature_description)
    # example["image"] = tf.io.decode_png(example["image"], dtype=tf.uint16, channels=1)
    example["image"] = tf.reshape(tf.io.decode_raw(example['image'], tf.uint16),
                                  shape=(example["num"], example["height"], example["width"]))
    example["mask"] = tf.reshape(tf.io.decode_raw(example['mask'], tf.bool),
                                 shape=(example["num"], example["height"], example["width"]))
    # example['id'] =  example['id'].decode("utf-8")
    return example


def data_to_TFRecord(id_list, giantin_patches_list, mask_patches_list, TFRecord_path):
    """

    :param id_list: image id list
    :param giantin_patches_list: shape: (m,n,256,256): m images, n rois
    :param mask_patches_list:
    :param TFRecord_path:
    :return:
    """
    if not os.path.exists(TFRecord_path):
        os.makedirs(TFRecord_path)
    lens = len(id_list)
    p = Progbar(lens)
    for i in range(lens):
        p.update(i)
        image_id = id_list[i]
        image_patches = giantin_patches_list[i]
        mask_patches = mask_patches_list[i]
        record_file = os.path.join(TFRecord_path, '{}.tfrecords'.format(image_id))
        with tf.io.TFRecordWriter(record_file) as writer:
            tf_example = tfrecord_example(image_id, image_patches, mask_patches)
            writer.write(tf_example.SerializeToString())
        print(" " + image_id + " write to TFRecord sucessfully!")
    print("data to TFRecord DONE!")


def TFRecord_to_data(TFRecord_path):
    TFRecord_files = os.listdir(TFRecord_path)
    d = Progbar(len(TFRecord_files))
    images = []
    masks = []
    ids = []
    for n, file in enumerate(TFRecord_files):
        raw_image_dataset = tf.data.TFRecordDataset(TFRecord_path + "/" + file)
        parsed_dataset = raw_image_dataset.map(parse_tfrecord_fn)
        for features in parsed_dataset.take(1):
            image = features["image"].numpy()
            mask = features["mask"].numpy()
            image_id = features["id"].numpy().decode()
            images.append(image)
            masks.append(mask)
            ids.append(image_id)
        d.update(n)
    return images, masks, ids
