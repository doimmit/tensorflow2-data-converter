
from absl import flags
import tensorflow as tf
import cv2, os, io
import numpy as np
import csv
from PIL import Image
import re
from pathlib import Path
from datetime import datetime

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util


flags.DEFINE_string('data_dir', './data', '')
flags.DEFINE_string('train_dir', './train', '')
flags.DEFINE_string('pretrained_dir', './pretrained/efficientdet_d0_coco17_tpu-32', '')
flags.DEFINE_integer('batch_size', 16, 'Number of batch_size.')
flags.DEFINE_integer('num_steps', 100000, 'Number of num_steps.')
flags.DEFINE_integer('dimension', 0, 'Number of dimensions.')

FLAGS = flags.FLAGS

def convert_dataset():
    
    print(f'Start to create customizing TFRecord and config file')
    print(f'data_dir: {FLAGS.data_dir}')
    print(f'train_dir: {FLAGS.train_dir}')
    print(f'pretrained_dir: {FLAGS.pretrained_dir}')
    print(f'batch_size: {str(FLAGS.batch_size)}')
    print(f'num_steps: {str(FLAGS.num_steps)}')
    print(f'dimension: {str(FLAGS.dimension)}')

    Path(os.path.join(FLAGS.train_dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(FLAGS.train_dir, 'config')).mkdir(parents=True, exist_ok=True)
    CONFIG_DIR = os.path.join(FLAGS.train_dir, 'config')
    num_shard = {}

    classes = get_classes()
    label_map_dict = create_label_map(classes, CONFIG_DIR)

    for d in ["train", "test"]:

        all_image = get_all_image_files(os.path.join(FLAGS.data_dir, d))
        output_filebase = os.path.join(CONFIG_DIR, f'petopia_{d}.tfrecord')
        num_shard[d] = int(len(all_image) / 1000) + 1

        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack, output_filebase, num_shard[d])

            # writer = tf.io.TFRecordWriter(output_filebase)

            print(f'### Start convert csv annotation file to tf record : {d}, {num_shard[d]}')
            for id, file_path in enumerate(all_image):
                tf_image = create_tf_image(id, file_path, classes, label_map_dict)
                if tf_image is not None:
                    # writer.write(tf_image.SerializeToString())
                    output_shard_index = id % num_shard[d]
                    output_tfrecords[output_shard_index].write(tf_image.SerializeToString())

                if id % 1000 == 0:
                    print('ing...')

            # writer.close()
            print(f'### Complete convert csv to tf record : {d}')

    create_config(classes, CONFIG_DIR, num_shard)
    
def get_all_image_files(path):
    all_files = []
    exts = ['.png', '.jpeg', '.jpg']
    for (path, directory, files) in os.walk(path):
        for filename in files:
            file_abs_path = os.path.join(path, filename)
            ext = str(os.path.splitext(file_abs_path)[1]).lower()
            if ext in exts:
                all_files.append(str(file_abs_path))

    print(f'### Done find image file : {str(len(all_files))}')
    return all_files


def create_tf_image(id, file_path, classes, label_map_dict):
    tf_example = None
    name, image_format = os.path.splitext(file_path)
    image_format = image_format.split('.')[1]
    file_name = os.path.basename(file_path)
    csv_file = f'{name}-o.csv'

    encoded_jpg = None
    with tf.io.gfile.GFile(file_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    # width = 0
    # height = 0
    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    class_names = []  # List of string class name of bounding box (1 per box)
    class_ids = []  # List of integer class id of bounding box (1 per box)

    with open(csv_file, "r", encoding='utf-8') as f:
        rdr = csv.reader(f, delimiter=',')
        lines = list(rdr)

        for idx, line in enumerate(lines):

            if idx != 0:
                if target == 'all' or line[1] == target:
                    if line[2] in classes:
                        class_names.append(line[2].encode('utf8'))
                        class_ids.append(label_map_dict[line[2]])
                        xmins.append(float(line[3]) / float(width))
                        ymins.append(float(line[4]) / float(height))
                        xmaxs.append((float(line[3]) + float(line[5])) / float(width))
                        ymaxs.append((float(line[4]) + float(line[6])) / float(height))

    print(f'{str(height)}, {str(width)}, {file_name.encode("utf8")}, \
            {image_format.encode("utf8")}, {str(class_names)}, {str(class_ids)}')

    if len(class_names) != 0:
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(file_name.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(file_name.encode('utf8')),
            # 'image/source_id': dataset_util.bytes_feature(str(id).encode('utf8')),
            # 'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(class_names),
            'image/object/class/label': dataset_util.int64_list_feature(class_ids),
        }))

    # print(f'Done : {str(class_names)}, {str(class_ids)}, {str(file_name)} ')
        return tf_example
    else:
        return None


def get_classes():
    classes_file = os.path.join(FLAGS.data_dir, 'label.txt')
    classes = []
    with open(classes_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            classes.append(line.split('\n')[0])

    print(f'### Done load label list : {str(classes)}')
    return classes


def create_label_map(classes, CONFIG_DIR):
    labe_map_file = os.path.join(CONFIG_DIR, 'label_map.pbtxt')
    label_map_dict = {}
    with open(labe_map_file, "w", encoding='utf-8') as f:
        for i, n in enumerate(classes):
            f.write('item {\n')
            f.write(f'  id: {i + 1}\n')
            f.write(f'  name: "{n}"\n')
            f.write('}\n\n')
            label_map_dict[n] = i + 1

    print(f'### Done create label map : {str(label_map_dict)}')
    return label_map_dict


def create_config(classes, CONFIG_DIR, num_shard):
    config_file = os.path.join(FLAGS.train_dir, 'pipeline.config')
    new_config_file = os.path.join(CONFIG_DIR, 'pipeline.config')
    with open(config_file, "r", encoding='utf-8') as f:
        s = f.read()

    with open(new_config_file, "w", encoding='utf-8') as f:

        # dimension change.
        if dimension != 0:
            s = re.sub('min_dimension: [0-9]+',
                       f'min_dimension: {str(dimension)}', s)
            s = re.sub('max_dimension: [0-9]+',
                       f'max_dimension: {str(dimension)}', s)
            s = re.sub('output_size: [0-9]+',
                       f'output_size: {str(dimension)}', s)

        # fine_tune_checkpoint
        s = re.sub('fine_tune_checkpoint: ".*?"',
                   f'fine_tune_checkpoint: "{FLAGS.train_dir}/checkpoint/ckpt-0"', s)

        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")',
            f'input_path: "{CONFIG_DIR}/petopia_train.tfrecord-?????-of-0000{num_shard["train"]}"', s)
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")',
            f'input_path: "{CONFIG_DIR}/petopia_test.tfrecord-?????-of-0000{num_shard["test"]}"', s)

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"',
            f'label_map_path: "{CONFIG_DIR}/label_map.pbtxt"', s)

        # Set training steps, num_steps
        s = re.sub('total_steps: [0-9]+',
                   f'total_steps: {str(num_steps)}', s)


        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                   f'batch_size: {str(batch_size)}', s)

        # Init eval batch_size.
        s = re.sub('batch_size: [0-9]+;',
                   f'batch_size: 1;', s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                   f'num_steps: {str(num_steps)}', s)

        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                   f'num_classes: {str(len(classes))}', s)

        # fine-tune checkpoint type
        s = re.sub('fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "detection"', s)

        # metrics_set
        s = re.sub('metrics_set: ".*?"', 'metrics_set: "pascal_voc_detection_metrics"', s)

        f.write(s)


if __name__ == '__main__':

