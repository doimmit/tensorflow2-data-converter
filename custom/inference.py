import cv2
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import logging
# tf.get_logger().setLevel(logging.INFO)
tf.get_logger().setLevel(logging.ERROR)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])
tf.config.experimental.set_memory_growth(gpus[1], True)
tf.config.experimental.set_virtual_device_configuration(gpus[1], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])


pipeline_config = os.path.join('/home/petopia-02/git/models/research/object_detection/petopia/train/d0_36/config/pipeline.config')
label_map_path = '/home/petopia-02/git/models/research/object_detection/petopia/train/d0_36/config/label_map.pbtxt'

model_dir = '/home/petopia-02/git/models/research/object_detection/petopia/train/d0_36/model/ckpt-201'
EVAL_DIR = '/home/petopia-02/git/models/research/object_detection/petopia/train/d0_36/model/eval-limmi'
TEST_DATA_DIR = '/home/petopia-02/git/models/research/object_detection/petopia/data/36all/test'
# TEST_DATA_DIR = '/petopia/mnt/testlimmi'


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



def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn




if __name__ == '__main__':
    Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)
    ckpt.restore(model_dir)


    detect_fn = get_model_detection_function(detection_model)
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    all_image = get_all_image_files(TEST_DATA_DIR)

    for image_path in all_image:
        file_name = os.path.basename(image_path)
        print(f'### Try to inference image : {file_name}')
        try:
            image_np = load_image_into_numpy_array(image_path)

            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32)
            detections, predictions_dict, shapes = detect_fn(input_tensor)

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
                min_score_thresh=.5,
                agnostic_mode=False,
            )

            save_img = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(EVAL_DIR, file_name), save_img)
        except Exception as e:
            print(f'### Exception : {file_name} - {str(e)}')


