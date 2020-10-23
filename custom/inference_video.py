
import numpy as np
import sys
import tensorflow as tf
import os

import argparse
import cv2

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])
# tf.config.experimental.set_memory_growth(gpus[1], True)
# tf.config.experimental.set_virtual_device_configuration(gpus[1], [
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])


parser = argparse.ArgumentParser(description='Simple object detection inference for *.avi videos.')
parser.add_argument('-v','--video_path', help='The video file path.')
parser.add_argument('-m','--train_path', default='', help='')
parser.add_argument('-c','--ckpt', default='ckpt-1', help='')
parser.add_argument('-f','--fps', default=24.0, help='Video file frame rate.')
args = parser.parse_args()

TRAINNING_DIR = args.train_path
CKPT = args.ckpt

lines = open(os.path.join(TRAINNING_DIR, 'model', 'checkpoint'), "r", encoding='utf-8').readlines()
checkpoint_list = []
if CKPT is None:
    CKPT = lines[0].split("\"")[1].split("\"")[0]

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = os.path.join(TRAINNING_DIR, 'model', f'{CKPT}.data-00000-of-00001')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(TRAINNING_DIR, 'config', 'label_map.pbtxt')


FRAME_RATE = args.fps

VIDEO_PATH = args.video_path
VIDEO_OUT_PATH = os.path.join(TRAINNING_DIR, 'model', f'{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_out.mpt')

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# # model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# # model_dir =  "../../bigdata/models/" + model_name + "/saved_model"
# detection_model = tf.saved_model.load(str(PATH_TO_FROZEN_GRAPH))
# detection_model = detection_model.signatures['serving_default']

# Load a label map that maps label indexes to category names
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Helper code
def load_image_into_numpy_array(image):
    VID_WIDTH = image.shape[1]
    VID_HEIGHT = image.shape[0]
    return np.array(image).reshape(
        (VID_HEIGHT, VID_WIDTH, 3)).astype(np.uint8)

def run_inference_for_single_image_one(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]

    # output_dict is a dict  with keys detection_classes , num_detections , detection_boxes(4 coordinates of each box) , detection_scores for 100 boxes
    output_dict = model(input_tensor)
    # print(1,output_dict)

    # num_detections gives number of objects in current frame
    num_detections = int(output_dict.pop('num_detections'))
    # print(2,num_detections)

    # output_dict is a dict  with keys detection_classes , detection_boxes(4 coordinates of each box) , detection_scores for num_detections boxes
    output_dict = {key:value[0, :num_detections].numpy()
                   for key,value in output_dict.items()}

    # adding num_detections that was earlier popped out
    output_dict['num_detections'] = num_detections

    # converting all values in detection_classes as ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    print(5,output_dict)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        print(5,detection_masks_reframed)
    return output_dict


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

# Process video
cap = cv2.VideoCapture(VIDEO_PATH)
# Define the codec and create VideoWriter object
# Modify the next line if you want to process different codecs
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(VIDEO_OUT_PATH,fourcc, float(FRAME_RATE), (int(cap.get(3)),int(cap.get(4))))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        image_np = load_image_into_numpy_array(frame)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # output_dict = run_inference_for_single_image_one(detection_model, image_np)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=2)
        #output_image = Image.fromarray(image_np)
        out.write(image_np)

        cv2.imshow('frame',image_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #else:
        #break

# Release everything if job is finished
print('releasing objects...')
cap.release()
out.release()
cv2.destroyAllWindows()
print('done')