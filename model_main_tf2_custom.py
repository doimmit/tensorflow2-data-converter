# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Creates and runs TF2 object detection models.

For local training/evaluation run:
PIPELINE_CONFIG_PATH=path/to/pipeline.config
MODEL_DIR=/tmp/model_outputs
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main_tf2.py -- \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr
"""
from absl import flags
# from object_detection import model_lib_v2
from model_lib_v2_petopia import eval_continuously, train_loop
import cv2
import csv
import shutil
import numpy as np
from six import BytesIO
from PIL import Image
from pathlib import Path
from datetime import datetime

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(1)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9500)])
tf.config.experimental.set_memory_growth(gpus[1], True)
tf.config.experimental.set_virtual_device_configuration(gpus[1], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9500)])

flags.DEFINE_string('train_path', None, 'Path to config '
                                        'file.')
flags.DEFINE_string('test_data_dir', None, '')
flags.DEFINE_string('best_ckpt', None, '')
flags.DEFINE_bool('eval_only', False, 'eval_only')
flags.DEFINE_bool('inference_only', False, 'inference_only')
flags.DEFINE_bool('inference_video', False, 'inference_only')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                                                  'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_bool('eval_on_train_data', False, 'Enable evaluating on train '
                                               'data (only supported in distributed training).')
flags.DEFINE_integer('sample_1_of_n_eval_examples', None, 'Will sample one of '
                                                          'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                                                                'one of every n train input examples for evaluation, '
                                                                'where n is provided. This is only used if '
                                                                '`eval_training_data` is True.')
flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
                       'where event and checkpoint files will be written.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
                            '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
                            'writing resulting metrics to `model_dir`.')

flags.DEFINE_integer('eval_timeout', 3600, 'Number of seconds to wait for an'
                                           'evaluation checkpoint before exiting.')

flags.DEFINE_bool('use_tpu', False, 'Whether the job is executing on a TPU.')
flags.DEFINE_string(
    'tpu_name',
    default=None,
    help='Name of the Cloud TPU for Cluster Resolvers.')
flags.DEFINE_integer(
    'num_workers', 1, 'When num_workers > 1, training uses '
                      'MultiWorkerMirroredStrategy. When num_workers = 1 it uses '
                      'MirroredStrategy.')
flags.DEFINE_integer(
    'checkpoint_every_n', 1000, 'Integer defining how often we checkpoint.')
flags.DEFINE_boolean('record_summaries', True,
                     ('Whether or not to record summaries during'
                      ' training.'))

FLAGS = flags.FLAGS


def main(unused_argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    tf.config.set_soft_device_placement(True)

    pipeline_config_path = os.path.join(FLAGS.train_path, 'config', 'pipeline.config')
    model_dir = os.path.join(FLAGS.train_path, 'model')
    label_map_path = os.path.join(FLAGS.train_path, 'config', 'label_map.pbtxt')

    if FLAGS.inference_only:
        best_ckpt, checkpoint_list = get_best_ckpt(model_dir, FLAGS.best_ckpt)
        if FLAGS.test_data_dir is None:
            print('Cannot inference images. Please input test_data_dir value.')
        else:
            inference_image(pipeline_config_path, os.path.join(model_dir, best_ckpt), label_map_path,
                            FLAGS.test_data_dir,
                            os.path.join(FLAGS.train_path, 'model', f'inference_image_{best_ckpt}'))
    elif FLAGS.inference_video:
        best_ckpt, checkpoint_list = get_best_ckpt(model_dir, FLAGS.best_ckpt)
        if FLAGS.test_data_dir is None:
            print('Cannot inference video. Please input test_data_dir value.')
        else:
            inference_video(pipeline_config_path, os.path.join(model_dir, best_ckpt), label_map_path,
                            FLAGS.test_data_dir,
                            os.path.join(FLAGS.train_path, 'model', f'inference_video_{best_ckpt}'))
    elif FLAGS.eval_only:
        best_ckpt, checkpoint_list = get_best_ckpt(model_dir, FLAGS.best_ckpt)
        eval_petopia(pipeline_config_path, model_dir, checkpoint_list)
    else:
        if FLAGS.use_tpu:
            # TPU is automatically inferred if tpu_name is None and
            # we are running under cloud ai-platform.
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
        elif FLAGS.num_workers > 1:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.compat.v2.distribute.MirroredStrategy()

        with strategy.scope():
            train_loop(
                pipeline_config_path=pipeline_config_path,
                model_dir=model_dir,
                train_steps=FLAGS.num_train_steps,
                use_tpu=FLAGS.use_tpu,
                checkpoint_every_n=FLAGS.checkpoint_every_n,
                record_summaries=FLAGS.record_summaries,
                checkpoint_max_to_keep=400)

        best_ckpt, checkpoint_list = get_best_ckpt(model_dir, FLAGS.best_ckpt)
        eval_petopia(pipeline_config_path, model_dir, checkpoint_list)
    print("!!!!!!!! Done !!!!!!!!")


def get_best_ckpt(model_dir, best_ckpt=None):
    lines = open(os.path.join(model_dir, 'checkpoint'), "r", encoding='utf-8').readlines()
    checkpoint_list = []
    if best_ckpt is None:
        best_ckpt = lines[0].split("\"")[1].split("\"")[0]

        for line in lines:
            if 'all_model_checkpoint_paths' in line:
                ckpt = line.split('\"')[1].split('\"')[0]
                if os.path.isfile(os.path.join(model_dir, f'{ckpt}.index')):
                    checkpoint_list.append(os.path.join(model_dir, ckpt))

        if len(checkpoint_list) > 200:
            checkpoint_list = checkpoint_list[-200:]
    else:
        checkpoint_list = [os.path.join(model_dir, best_ckpt)]

    return best_ckpt, checkpoint_list


def eval_petopia(pipeline_config_path, model_dir, checkpoint_list):
    eval_metrics_dir = os.path.join(os.path.dirname(checkpoint_list[0]),
                                    f'eval_metrics_{datetime.now().strftime("%m%d_%H%M")}')
    Path(eval_metrics_dir).mkdir(parents=True, exist_ok=True)
    for checkpoint_dir in checkpoint_list:
        eval_continuously(
            pipeline_config_path=pipeline_config_path,
            model_dir=model_dir,
            train_steps=FLAGS.num_train_steps,
            sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
            sample_1_of_n_eval_on_train_examples=(
                FLAGS.sample_1_of_n_eval_on_train_examples),
            checkpoint_dir=checkpoint_dir,
            eval_metrics_dir=eval_metrics_dir,
            wait_interval=300, timeout=FLAGS.eval_timeout,
            custom_list={'peeing_female': ['peeing_female', 'peeing_female_etc'],
                         'peeing_female_etc': ['peeing_female', 'peeing_female_etc']},
            custom_map_list=['fourlegs', 'sit', 'kneel', 'headdown', 'peeing_male', 'peeing_female', 'pooping',
                             'lay_belly',
                             'lay_curled', 'lay_side', 'lay_sprawled', 'mouth_open', 'mouth_close', 'eating_mouth',
                             'eating_bowl',
                             'eating_hide', 'licking', 'yawning', 'sleeping', 'bowl', 'feed', 'pad', 'toilet'])


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


def get_all_video_files(path):
    all_files = []
    exts = ['.mp4', '.avi']
    for (path, directory, files) in os.walk(path):
        for filename in files:
            file_abs_path = os.path.join(path, filename)
            ext = str(os.path.splitext(file_abs_path)[1]).lower()
            if ext in exts:
                all_files.append(str(file_abs_path))

    print(f'### Done find video file : {str(len(all_files))}')
    return all_files


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    if image.getdata().mode == "RGBA":
        image = image.convert('RGB')
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_capture_image_into_numpy_array(image):
    VID_WIDTH = image.shape[1]
    VID_HEIGHT = image.shape[0]
    return np.array(image).reshape(
        (VID_HEIGHT, VID_WIDTH, 3)).astype(np.uint8)


def detect_fn(model, image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


def inference_video(pipeline_config_path, model_dir, label_map_path, test_data_dir, inference_dir):
    # all_video = get_all_video_files(test_data_dir)
    #
    # for video_path in all_video:

    video_path = "/home/petopia-02/mnt/inference_video/0924/MD_20200421132714_peeing.mp4"
    # MD_20200330174329_pooping.mp4
    # MD_20200330123235_scratching.mp4
    # hard_pooping.mp4
    # yawning_2.mp4
    # peeing_male_1.mp4
    # eating_bowl_1.mp4
    # 9221031-eating_bowl.mp4
    output_path = os.path.join(inference_dir, f'{os.path.splitext(os.path.basename(video_path))[0]}_out_.avi')
    Path(inference_dir).mkdir(parents=True, exist_ok=True)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']

    detection_model = model_builder.build(
        model_config=model_config, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)
    ckpt.restore(model_dir)

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, float(cap.get(5)), (int(cap.get(3)), int(cap.get(4))))

    try:
        frame_count = 0
        print(f'video : {str(video_path)}')
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame_count += 1
                image_np = load_capture_image_into_numpy_array(frame)
                input_tensor = tf.convert_to_tensor(
                    np.expand_dims(image_np, 0), dtype=tf.float32)
                detections, predictions_dict, shapes = detect_fn(detection_model, input_tensor)

                boxes = detections['detection_boxes'][0].numpy()
                classes = (detections['detection_classes'][0].numpy() + 1).astype(int)
                scores = detections['detection_scores'][0].numpy()
                display_str = f'-- frame : {str(frame_count)}'
                for i in range(boxes.shape[0]):
                    score = round(100 * scores[i])
                    if score >= 50:
                        display_str = f'{display_str} / {category_index[classes[i]]["name"]}: {str(round(100 * scores[i]))}% '

                # print(display_str)

                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    detections['detection_boxes'][0].numpy(),
                    (detections['detection_classes'][0].numpy() + 1).astype(int),
                    detections['detection_scores'][0].numpy(),
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=200,
                    min_score_thresh=.5,
                    agnostic_mode=False,
                    line_thickness=10
                )

                out.write(image_np)
                # save_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                # Path(os.path.join(inference_dir, os.path.splitext(os.path.basename(video_path))[0])).mkdir(parents=True, exist_ok=True)
                # cv2.imwrite(os.path.join(inference_dir, os.path.splitext(os.path.basename(video_path))[0], f'{str(frame_count)}.jpg'), save_img)

                if frame_count > 3000:
                    break
                # cv2.imshow('frame',image_np)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

        # Release everything if job is finished
        print('releasing objects...')
        cap.release()
        out.release()
        print('Done!!!!!!!')
        # cv2.destroyAllWindows()
    except Exception as e:
        print(f'### Exception : {str(e)}')


def inference_image(pipeline_config_path, model_dir, label_map_path, test_data_dir, inference_dir):
    Path(inference_dir).mkdir(parents=True, exist_ok=True)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)
    ckpt.restore(model_dir)

    # detect_fn = get_model_detection_function(detection_model)
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    all_image = get_all_image_files(test_data_dir)

    for image_path in all_image:
        file_name = os.path.basename(image_path)
        name, image_format = os.path.splitext(image_path)
        try:
            image_np = load_image_into_numpy_array(image_path)

            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32)
            detections, predictions_dict, shapes = detect_fn(detection_model, input_tensor)
            width = shapes.numpy()[1]
            height = shapes.numpy()[0]
            boxes = detections['detection_boxes'][0].numpy()
            classes = (detections['detection_classes'][0].numpy() + 1).astype(int)
            scores = detections['detection_scores'][0].numpy()
            display_str = f'image : {file_name}'
            for i in range(boxes.shape[0]):
                score = round(100 * scores[i])
                if score >= 25:
                    # print(boxes[i])
                    display_str = f'{display_str} / {category_index[classes[i]]["name"]}: {str(round(100 * scores[i]))}% ' \
                                  f'({str(boxes[i])})'
                    ## xmin, ymin, xmax, ymax

            print(display_str)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.25,
                agnostic_mode=False,
            )

            save_img = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)

            class_dir = os.path.dirname(image_path).split('/')[-1]
            Path(os.path.join(inference_dir, class_dir)).mkdir(parents=True, exist_ok=True)
            shutil.copy(f'{name}-o.csv', os.path.join(inference_dir, class_dir, f'{os.path.basename(name)}-o.csv'))
            cv2.imwrite(os.path.join(inference_dir, class_dir, file_name), save_img)
        except Exception as e:
            print(f'### Exception : {file_name} - {str(e)}')


if __name__ == '__main__':
    tf.compat.v1.app.run()
