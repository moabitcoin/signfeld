import os
import sys
import tqdm
import random
import operator
import tarfile
import zipfile
import logging
import argparse
import fontconfig
import numpy as np
from pathlib import Path
import six.moves.urllib as urllib

import skvideo.io
import skvideo.datasets
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from fonts.otf import font_files
from IPython.display import display, Image

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from utils import label_map_util

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
system_fonts = fontconfig.query(family='ubuntu', lang='en')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_graph(model_path):

  try:

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(model_path.as_posix(), 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    logging.info('Done loading frozen graph from {}'.format(model_path))

    return detection_graph

  except Exception as err:
    logging.error('Error loading frozen graph from {}'.format(model_path))
    return None


def build_model(default_graph, session):

  tensor_list = ['num_detections', 'detection_boxes', 'detection_scores',
                 'detection_classes', 'detection_masks']
  tensor_dict = {}

  with default_graph.as_default():
    with session.as_default():
      # Get handles to input and output tensors
      ops = default_graph.get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      for key in tensor_list:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = default_graph.get_tensor_by_name(tensor_name)

      image_tensor = default_graph.get_tensor_by_name('image_tensor:0')

    return tensor_dict, image_tensor


def load_label_map(label_map_file):

  map_loader = label_map_util.create_category_index_from_labelmap
  return map_loader(label_map_file.as_posix(), use_display_name=True)


def fetch_images(source_images, ext='.jpg'):

  if source_images.is_file():
    with source_images.open() as pfile:
      test_images = pfile.readlines()
      test_images = [t.strip() for t in test_images]

  elif source_images.is_dir():

    test_images = [img for img in source_images.iterdir()
                   if img.suffix == ext]
  else:
    logger.error('Neither an image list'
                 ' or a directory {}'.format(source_images))
    return None

  return test_images


def fetch_frames(vid_path):

  try:
    vidgen = skvideo.io.vreader(vid_path.as_posix())
    return vidgen

  except Exception as err:
    logging.error('Error parsing video {}, {}'.format(vid_path, err))
    return None


def run_inference(sess, output_dict, image_tensor, image):

  with sess.as_default():
    output_dict = sess.run(output_dict,
                           feed_dict={image_tensor: np.expand_dims(image,
                                                                   0)})

    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
      output_dict['detection_masks'] = output_dict['detection_masks'][0]

  return output_dict


def draw_detections(source_img, bboxes, scores, labels,
                    label_map, threshold=0.5):

  source_img = source_img.convert("RGBA")
  draw = ImageDraw.Draw(source_img)

  width, height = source_img.size

  for bbox, score, label in zip(bboxes, scores, labels):

      if score < threshold:
          continue

      ymin = int(bbox[0] * height)
      ymax = int(bbox[2] * height)

      xmin = int(bbox[1] * width)
      xmax = int(bbox[3] * width)

      rect_width = int(min(32, 0.1 * (xmax - xmin)))
      font_size = int(min(32, 0.5 * (xmax - xmin)))

      draw.rectangle(((xmin, ymin), (xmax, ymax)),
                     fill=None, outline='red', width=rect_width)
      object_string = '{} : {:.2f} %'.format(label_map[label]['name'], score)
      draw.text((xmin, ymax), object_string,
                font=ImageFont.truetype(system_fonts[0].file, font_size))

  source_img = source_img.convert("RGB")

  return source_img


def run_detection(video_path, images_path, model_path,
                  labels_file, destination, im_size, threshold):

  is_video = video_path is not None
  is_image = images_path is not None

  assert operator.xor(is_video, is_image), \
      "Señor! Either provide images or video but not both"

  if video_path:
    writer = skvideo.io.FFmpegWriter(destination.as_posix())
    image_gen = fetch_frames(video_path)
  if images_path:
    image_gen = fetch_images(images_path)

  graph = load_graph(model_path)

  session = tf.Session(graph=graph)
  output_tensors, input_tensor = build_model(graph, session)
  labels_map = load_label_map(labels_file)

  for image in tqdm.tqdm(image_gen):

    if is_image:
      pil_image = PILImage.open(image)
      pil_image.thumbnail(im_size, PILImage.ANTIALIAS)
      np_image = np.array(pil_image).astype(np.uint8)
    if is_video:
      pil_image = PILImage.fromarray(image)
      pil_image.thumbnail(im_size, PILImage.ANTIALIAS)
      np_image = np.array(pip_image).astype(np.uint8)

    preds = run_inference(session, output_tensors,
                          input_tensor, np_image)

    pil_image = draw_detections(pil_image, preds['detection_boxes'],
                                preds['detection_scores'],
                                preds['detection_classes'],
                                labels_map, threshold=threshold)
    if is_video:
      writer.writeFrame(np.array(pil_image))

    if is_image:
      detection_image_path = destination.joinpath(os.path.basename(image))
      pil_image.save(detection_image_path.as_posix())

  if is_video:
    writer.close()


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Running traffic sign detection model on '
                                   'Images (list or directory) / Video')
  parser.add_argument('-v', '-video', dest='video', default=None, type=Path,
                      help='Path to input video')
  parser.add_argument('-i', '-images', dest='images', default=None, type=Path,
                      help='Path directory or file'
                      ' list of images')
  parser.add_argument('-m', '-model', dest='model', type=Path, required=True,
                      help='Frozen detection graph')
  parser.add_argument('-l', '-labels', dest='labels', type=Path, required=True,
                      help='Labels prototxt file')
  parser.add_argument('-t', '-threshold', dest='threshold', type=float,
                      default=0.5, help='Detection threshold')
  parser.add_argument('-d', '-destinaton', dest='destinaton', type=Path,
                      required=True, help='Directory for images, video path'
                      ' otherwise')
  parser.add_argument('-s', '-size', dest='size', type=int, nargs='+',
                      required=True, help='Aspect ratio preserving resizing'
                      ' to be applied before detections (W H)')

  args = parser.parse_args()

  run_detection(args.video, args.images, args.model, args.labels,
                args.destinaton, args.size, args.threshold)
