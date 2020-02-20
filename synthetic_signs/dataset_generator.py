import argparse
import functools
import io
import logging
import multiprocessing
import numpy as np
import operator
import os
import pathlib
import random
import shutil
import sys
import traceback

import cv2
from tqdm import tqdm
import yaml

from synthetic_signs import blending
from synthetic_signs import image_composition
from synthetic_signs import utils

_RANDOM_DISTRACTOR_CATEGORY = "distractor"


def read_backgrounds(background_path, max_num_images=None):
  """Read background images from directory or list file.

  Args:
    background_path (pathlib.Path): Path to image directory or path to file
      which contains image filenames (one file per line).
    max_num_images (int, optional): Maximum number of background images to
      read. A random selection is made if the background_path contains more
      images.

  Returns:
    output (List[str]): List of filenames.
  """
  if background_path.is_dir():
    filenames = [str(i) for i in background_path.iterdir()]
  elif background_path.is_file():
    with background_path.open() as file_handle:
      filenames = file_handle.readlines()
      filenames = [i.strip() for i in filenames]
  else:
    logging.error(
        "Neither a directory or a file list {}".format(background_path))
    sys.exit()

  logging.info("Found {} images ".format(len(filenames)))
  if max_num_images is not None:
    filenames = np.random.choice(filenames, size=max_num_images)
  return filenames


def read_templates(templates_path, max_template_size):
  """Read image templates from directory.

  Args:
    templates_path (pathlib.Path): Path to template image directory.
    max_template_size (int): Template images are resized such that the larger
      side has size size.

  Returns:
    output (List[Template]): List of templates.
  """
  if templates_path.is_dir():
    template_names = [str(t) for t in templates_path.iterdir()]
  elif templates_path.is_file():
    with templates_path.open() as pfile:
      template_names = [t.strip() for t in pfile.readlines()]
  else:
    logging.error(
        "Neither a directory or a file list {}".format(templates_path))

  templates = []
  progress_bar = tqdm(template_names)
  for template_path in progress_bar:
    progress_bar.set_description("Reading {}".format(template_path))
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template.ndim != 3 or template.shape[-1] != 4:
      logging.error("Not an RGBA image {}".format(template_path))
      os.exit_stack()

    image, mask = template[:, :, :3], template[:, :, 3]
    image, mask = utils.remove_padding(image, mask)

    h, w = template.shape[:2]
    new_height, new_width = ((None, max_template_size) if h < w else
                             (max_template_size, None))
    image = utils.resize_image(image, new_width, new_height)
    mask = utils.resize_image(mask, new_width, new_height)

    category = os.path.splitext(os.path.basename(template_path))[0]
    templates.append(
        image_composition.Template(image=image, mask=mask, category=category))

  return templates


def generate_random_distractors(templates, background_filenames,
                                num_distractors_per_template):
  """ Generate random distractors by cropping background images.

  Args:
    templates (List[Template]): List of templates. The generated distractors
      have the same shape/mask as the templates in this list.
    background_filenames (List[str]): List of background filenames. A random
      selection of images from this list is used to generate the distractor
      textures.
    num_distractors_per_template (int): For every template, generate this many
      random distractors.

  Returns:
    output (List[Template]): A list of length num_distractors_per_template *
      templates which contains the generated distractors.
  """
  distractors = []

  progress_bar = tqdm(range(num_distractors_per_template))
  progress_bar.set_description("Generating distractors")
  for i in progress_bar:
    filename = random.choice(background_filenames)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    image_h, image_w = image.shape[:2]

    for template in templates:
      template_h, template_w = template.mask.shape[:2]

      if image_h < template_h or image_w < template_w:
        logging.warning("Skipping distractor generation due to incompatible "
                        "image size.")
        continue

      y0 = np.random.randint(0, image_h - template_h)
      x0 = np.random.randint(0, image_w - template_w)

      y1 = y0 + template_h
      x1 = x0 + template_w

      crop = image[y0:y1, x0:x1, :]
      assert crop.shape[0] == template_h and crop.shape[1] == template_w

      distractors.append(
          image_composition.Template(image=crop,
                                     mask=template.mask,
                                     category=_RANDOM_DISTRACTOR_CATEGORY))

  return distractors


def generate_chunks(contents, n_chunks):
  """Generate equally sized chunks.

  Args:
    contents (list): A list to be split into chunks.
    n_chunks (int): Number of chunks; must be smaller or equal to the length
      of contents.

  Returns:
    output: Generator object of n_chunks elements.
  """
  size = len(contents)
  chunk_size = size // n_chunks

  for i in range(0, size, chunk_size):
    yield contents[i:i + chunk_size]


def generate_synthetic_images(thread_idx,
                              templates,
                              image_configs,
                              result_dict,
                              background_size=None,
                              min_template_size=None):
  """Generate synthetic images.

  Args:
    thread_idx (int): Unique thread identifier; used to display progress bar
      status updates.
    templates (List[Template]): List of templates to place on background
      images.
    image_configs (List[ImageCompositionConfig]): List of image configurations.
    result_dict (multiprocessing.managers.DictProxy): Generated groundtruth
      annotations are stored in this dictionary under entry thread_idx.
    background_size (Tuple[int, int], optional): If not None, crop and resize
      background to this size (height, width).
    min_template_size (int, optional): If not None, pasted templates are
      at least this wide/high.
  """
  progress_bar = tqdm(image_configs,
                      desc='worker-{}'.format(thread_idx),
                      position=thread_idx,
                      unit='sample')

  binary_groundtruth = []
  multiclass_groundtruth = []

  for idx, image_config in enumerate(progress_bar):
    try:
      binary, multiclass = image_composition.compose_image(
          image_config, templates, background_size, min_template_size)
      binary_groundtruth.append(binary)
      multiclass_groundtruth.append(multiclass)
    except Exception as err:
      exc_buffer = io.StringIO()
      traceback.print_exc(file=exc_buffer)
      logging.error("Error running {}, {}\n{}".format(
          image_config.background_filename, err, exc_buffer.getvalue()))

  result_dict[thread_idx] = [binary_groundtruth, multiclass_groundtruth]


def execute_tasks(templates,
                  task_chunks,
                  background_size=None,
                  min_template_size=None):
  """Execute dataset generation tasks in parallel.

  Args:
    templates (List[Template]): List of templates to place on top of background
      images.
    task_chunks (Sequence[List[ImageCompositionConfig]]): Every element in the
      sequence contains a list of configurations to be executed by a separate
      process.
    background_size (Tuple[int, int], optional): If not None, crop and resize
      background image to this size (height, width).
    min_template_size (int, optional): If not None, pasted templates are
      at least this wide/high.

  Returns:
    output (Tuple[str, str]): Returns a tuple with the following entries: (1)
      string of binary classification ground truth file, (2) string of
      multiclass annotation file.
  """
  # Execute tasks in parallel.
  manager = multiprocessing.Manager()
  return_dict = manager.dict()

  jobs = []
  for chunk_id, task_list in enumerate(task_chunks):
    process = multiprocessing.Process(target=generate_synthetic_images,
                                      args=(chunk_id, templates, task_list,
                                            return_dict, background_size,
                                            min_template_size))
    jobs.append(process)
    process.start()
  for job in jobs:
    job.join()

  # Concatenate annotation results.
  binary_annotations = []
  multiclass_annotations = []

  for binary_annotation, multiclass_annotation in return_dict.values():
    binary_annotations.extend(binary_annotation)
    multiclass_annotations.extend(multiclass_annotation)

  binary_annotations = functools.reduce(operator.iconcat, binary_annotations,
                                        [])
  multiclass_annotations = functools.reduce(operator.iconcat,
                                            multiclass_annotations, [])
  return binary_annotations, multiclass_annotations


def generate_task_args(
    background_filenames, templates, distractors, output_dir,
    brightness_bias_lo, brightness_bias_hi, brightness_scale_lo,
    brightness_scale_hi, lighting_gain_lo, lighting_gain_hi, saturation_gain_lo,
    saturation_gain_hi, perspective_rate_lo, perspective_rate_hi,
    rotation_angle_deg_lo, rotation_angle_deg_hi, scale_factor_lo,
    scale_factor_hi, noise_magnitude_lo, noise_magntidue_hi, noise_probability,
    blur_stddev_lo, blur_stddev_hi, blur_probability, num_signs_in_image_lo,
    num_signs_in_image_hi, num_distractors_in_image_lo,
    num_distractors_in_image_hi, pixelate_factor_min, pixelate_factor_max,
    pixelate_probability, cutout_size_lo, cutout_size_hi, cutout_probability,
    blend_mode_exclude, template_exclude_rotation, template_exclude_blur,
    template_exclude_perspective, template_exclude_scaling):
  """Generate list of image composition tasks, each describing one synthetic
  image.

  Args:
    background_filenames (List[str]): List of paths to background images.
    templates (List[Template]): List of templates to place on top of
      background.
    distractors (List[Template]): List of distractor templates to place on top
      of background. Distractors are synthetic objects which are not part of
      the ground truth.
    output_dir (str): Output directory where the synthetic images will be
      stored.
    brightness_bias_lo (float): Lower bound of a uniform distribution that
      specifies an additive brightness augmentation applied to the
      background image. See synthetic_signs.augmentation.adjust_brightness
      for more information.
    brightness_bias_hi (float): Upper bound of a uniform distribution that
      specifies an additive brightness augmentation applied to the
      background image. See synthetic_signs.augmentation.adjust_brightness
      for more information.
    brightness_scale_lo (float): Lower bound of a uniform distribution that
      specifies a brightness scale augmentation applied to background and
      template images. See synthetic_signs.augmentation.adjust_brightness
      for more information.
    brightness_scale_hi (float): Upper bound of a uniform distribution that
      specifies a brightness scale augmentation applied to background and
      template images. See synthetic_signs.augmentation.adjust_brightness
      for more information.
    lighting_gain_lo (float): Lower bound of a uniform distribution that
      specifies the lighting gain applied to template images.
    lighting_gain_hi (float): Upper bound of a uniform distribution that
      specifies the lighting gain applied to template images.
    saturation_gain_lo (float): Lower bound of a uniform distribution that
      specifies the saturation gain applied to template images.
    saturation_gain_hi (float): Upper bound of a uniform distribution that
      specifies the saturation gain applied to template images.
    perspective_rate_lo (float): Lower bound of a uniform distribution that
      specifies a perspective distortion to template images. Reasonable
      values are in [0.0, 0.2]. Can be controled on a per-category level
      in category_to_augmentations. See
      synthetic_signs.augmentations.distort_perspective for more information.
    perspective_rate_hi (float): Upper bound of a uniform distribution that
      specifies a perspective distortion to template images. Reasonable
      values are in [0.0, 0.2]. Can be controled on a per-category level
      in category_to_augmentations. See
      synthetic_signs.augmentations.distort_perspective for more information.
    rotation_angle_deg_lo (float): Lower bound of a uniform distribution that
      specifies a rotation augmentation applied to template images. Can be
      controled on a per-category level in category_to_augmentations. See
      synthetic_signs.augmentations.rotate for more information.
    rotation_angle_deg_hi (float): Upper bound of a uniform distribution that
      specifies a rotation augmentation applied to template images. Can be
      controled on a per-category level in category_to_augmentations. See
      synthetic_signs.augmentations.rotate for more information.
    scale_factor_lo (float): Lower bound of a uniform distribution that
      specifies a scale augmentation applied to templatae images. Can be
      controled on a per-category level in category_to_augmentations. See
      synthetic_signs.augmentation.scale for more information.
    scale_factor_hi (float): Upper bound of a uniform distribution that
      specifies a scale augmentation applied to templates images. Can be
      controled on a per-category level in category_to_augmentations. See
      synthetic_signs.augmentation.scale for more information.
    noise_magnitude_lo (float): Lower bound of a uniform distribution that
      specifies an additive per-pixel noise augmentation. See
      synthetic_signs.augmentations.add_uniform_noise for more information.
    noise_magnitude_hi (float): Upper bound of a uniform distribution that
      specifies an additive per-pixel noise augmentation. See
      synthetic_signs.augmentations.add_uniform_noise for more information.
    noise_probability (float): Apply additive per-pixel noise augmentation
      to templates with this probability.
    blur_stddev_lo (float): Lower bound of a uniform distribution that
      specifies a Gaussian blur augmentation that is applied to templates.
      Can be controled on a per-category level in category_to_augmentations.
      See synthetic_signs.augmentations.blur for more information.
    blur_stddev_hi (float): Upper bound of a uniform distribution that
      specifies a Gaussian blur augmentation that is applied to templates.
      Can be controled on a per-category level in category_to_augmentations.
      See synthetic_signs.augmentations.blur for more information.
    blur_probability (bool): Apply Gaussian blur to templates with this
      probability.
    num_signs_in_image_lo (int): Place at least this number of templates on
      top of background images.
    num_signs_in_image_hi (int): Place at most this number of templates on
      top of background images.
    num_distractors_in_image_lo (int): Place at least this number of
      distractors in images. This value is ignored if the list of distractors
      is empty.
    num_distractors_in_image_hi (int): Place at most this number of
      distractors in images. This value is ignored if the list of distractors
      is empty.
    pixelate_factor_min (float): Lower bound of a uniform distribution that
      specifies an image down and upscale factor that is used to pixelate the
      image. Must be in ]0, 1]. See augmentations.pixelate for more information.
    pixelate_factor_max (float): Upper bound of a uniform distribution that
      specifies an image down and upscale factor that is used to pixelate the
      image. Must be in ]0, 1]. See augmentations.pixelate for more information.
    pixelate_probability (float): Apply image pixelation with this probability.
    coutout_size_lo (float): Lower bound of a uniform distribution that
      specifies the minimum width/height of a rectangular region that is cut
      out of templates randomly. Must be a number in ]0, 1] (specified in
      homogeneous template coordinates).
    cutout_size_hi (float): Upper bound of a uniform distribution that
      specifies the maximum width/height of rectangular region that is cut out
      of templates randomly. Must be a number in ]0, 1] (specified in
      homogeneous template coordinates).
    cutout_probability (float): A number in [0, 1] that specifies the
      probability that a rectangular region is cut out of the templatae
      randomly.
    blend_mode_exclude (List[str]): A list of blending modes that should not
      be applied. See synthetic_signs.blending.BLEND_NAME_TO_FUNCTION for an
      overview of available blending operations.
    template_exclude_rotation (List[str]): List of template categories which
      should not be rotated.
    template_exclude_blur (List[str]): List of template categories which should
      not be blurred.
    template_exclude_perspective (List[str]): List of template categories which
      should not be distorted by a perspective transformation.
    template_exclude_scaling (List[str]): List of template categories which
      should not be scaled.

  Returns:
    tasks (List[ImageCompositionConfig]): List of image composition tasks. For
      each random composition (template positions and augmentations), the task
      list contains multiple versions with different image blending modes. See
      synthetic_signs.blending.BLEND_NAME_TO_FUNCTION for an overview of
      blending operations.
    templates_and_distractors (List[Template]): Merged list of templates and
      distractors.
  """

  def generate_template_config(template_idx, rotation_flag, perspective_flag,
                               blur_flag, scale_flag, noise_flag, cutout_flag):

    rotation_angle_deg = (np.random.uniform(rotation_angle_deg_lo,
                                            rotation_angle_deg_hi)
                          if rotation_flag else None)

    perspective_rate = (np.random.uniform(
        perspective_rate_lo, perspective_rate_hi) if perspective_flag else None)

    scale_factor = (np.random.uniform(scale_factor_lo, scale_factor_hi)
                    if scale_flag else None)

    blur_stddev = (np.random.uniform(blur_stddev_lo, blur_stddev_hi)
                   if blur_flag else None)

    noise_magnitude = (np.random.randint(noise_magnitude_lo, noise_magntidue_hi)
                       if noise_flag else None)

    lighting_gain = np.random.randint(lighting_gain_lo, lighting_gain_hi)
    saturation_gain = np.random.randint(saturation_gain_lo, saturation_gain_hi)

    x, y = np.random.uniform(0.0, 1.0, (2, ))

    cutout_boxes = []
    if cutout_flag:
      y0, x0 = np.random.rand(2)
      h, w = np.random.uniform(cutout_size_lo, cutout_size_hi, (2, ))
      y1, x1 = y0 + h, x0 + w
      cutout_boxes.append(image_composition.Box(x0=x0, y0=y0, x1=x1, y1=y1))

    template_config = image_composition.TemplateAugmentationConfig(
        additive_noise_range=noise_magnitude,
        blur_stddev=blur_stddev,
        brightness_scale=None,
        lighting_gain=lighting_gain,
        saturation_gain=saturation_gain,
        perspective_rate=perspective_rate,
        rotation_angle_deg=rotation_angle_deg,
        scale_factor=scale_factor,
        position=image_composition.Point(x=x, y=y),
        template_idx=template_idx,
        cutout_boxes=cutout_boxes)
    return template_config

  progress_bar = tqdm(background_filenames)
  progress_bar.set_description("Generating tasks")

  image_configs = []
  for background_filename in progress_bar:
    brightness_bias = float(
        np.random.uniform(brightness_bias_lo, brightness_bias_hi))
    brightness_scale = float(
        np.random.uniform(brightness_scale_lo, brightness_scale_hi))

    if np.random.rand() <= pixelate_probability:
      pixelate_factor = float(
          np.random.uniform(pixelate_factor_min, pixelate_factor_max))
    else:
      pixelate_factor = None

    # Generate template augmentation configuration.
    num_signs = np.random.randint(num_signs_in_image_lo, num_signs_in_image_hi)
    template_indices = np.random.permutation(len(templates))[:num_signs]

    templates_in_image = []
    for template_idx in template_indices:
      category = templates[template_idx].category

      rotation_flag = category not in template_exclude_rotation
      perspective_flag = category not in template_exclude_perspective
      scale_flag = category not in template_exclude_scaling
      blur_flag = (category not in template_exclude_blur
                   and np.random.rand() <= blur_probability)
      noise_flag = np.random.rand() <= noise_probability
      cutout_flag = np.random.rand() <= cutout_probability

      template_config = generate_template_config(
          template_idx,
          rotation_flag=rotation_flag,
          perspective_flag=perspective_flag,
          blur_flag=blur_flag,
          scale_flag=scale_flag,
          noise_flag=noise_flag,
          cutout_flag=cutout_flag)
      templates_in_image.append(template_config)

    # Generate distractor augmentation configuration.
    max_num_distractors = min(len(distractors), num_distractors_in_image_hi)
    min_num_distractors = min(max_num_distractors - 1,
                              num_distractors_in_image_lo)
    num_distractors = (np.random.randint(min_num_distractors,
                                         max_num_distractors)
                       if max_num_distractors > 0 else 0)
    distractor_indices = np.random.permutation(
        len(distractors))[:num_distractors]

    distractors_in_image = []
    for distractor_idx in distractor_indices:
      template_config = generate_template_config(
          len(templates) + distractor_idx,
          rotation_flag=True,
          perspective_flag=True,
          blur_flag=np.random.rand() <= blur_probability,
          scale_flag=True,
          noise_flag=np.random.rand() <= noise_probability,
          cutout_flag=np.random.rand() <= cutout_probability)
      distractors_in_image.append(template_config)

    # Iterate over blending modes.
    all_blend_modes = set(blending.BLEND_NAME_TO_FUNCTION.keys())
    selected_blend_modes = all_blend_modes - set(blend_mode_exclude)

    for blending_mode in selected_blend_modes:
      background_basename = os.path.basename(background_filename)
      output_filename = os.path.join(
          output_dir, "{:05d}_{}".format(len(image_configs),
                                         background_basename))

      image_configs.append(
          image_composition.ImageCompositionConfig(
              background_filename=background_filename,
              blending_mode=blending_mode,
              brightness_bias=brightness_bias,
              brightness_scale=brightness_scale,
              pixelate_factor=pixelate_factor,
              output_filename=output_filename,
              template_configs=templates_in_image,
              distractor_configs=distractors_in_image,
          ))

  return image_configs, templates + distractors


def main(args):
  """Main program entry point.

  Args:
    args (argparse.Namespace): Command line arguments. See parse_args() for
      more information.
  """
  # Read input data.
  backgrounds = read_backgrounds(args.backgrounds, args.max_images)

  templates = read_templates(args.templates_path, args.max_template_size)
  logging.info("Found {} templates".format(len(templates)))

  if args.distractors_path is not None:
    distractors = read_templates(args.distractors_path, args.max_template_size)
  else:
    distractors = []
  logging.info("Found {} distractors".format(len(distractors)))

  if args.random_distractors > 0:
    random_distractors = generate_random_distractors(templates, backgrounds,
                                                     args.random_distractors)
    num_generated_distractors = len(random_distractors)

    distractors = distractors + random_distractors
    del random_distractors
  else:
    num_generated_distractors = 0

  logging.info("Generated additional {} random distractors".format(
      num_generated_distractors))

  with args.augmentations.open() as file_handle:
    augmentations = yaml.safe_load(file_handle)

  # Prepare output directory.
  image_output_directory = os.path.join(args.out_path, "imgs")
  os.makedirs(image_output_directory, exist_ok=True)
  shutil.copyfile(args.augmentations, args.out_path.joinpath("augmentations.yaml"))

  # Generate tasks and split into chunks according to number of jobs.
  task_args, templates_and_distractors = generate_task_args(
      backgrounds, templates, distractors, image_output_directory,
      **augmentations)
  task_args = task_args[:args.max_images]
  task_chunks = generate_chunks(task_args, args.jobs)

  # Generate synthetic dataset and write output files.
  if args.background_size is not None and len(args.background_size) > 0:
    background_size = [int(x.strip()) for x in args.background_size.split(',')]
    if len(background_size) != 2:
      raise ValueError("Failed to parse background size")
  else:
    background_size = None

  if args.min_template_size is not None and args.min_template_size != 0:
    min_template_size = args.min_template_size
  else:
    min_template_size = None

  binary_annotations, multiclass_annotations = execute_tasks(
      templates_and_distractors, task_chunks, background_size,
      min_template_size)

  logging.info("Generating annotations...")
  with args.out_path.joinpath("multiclass.csv").open(mode="w") as file_handle:
    file_handle.write("\n".join(multiclass_annotations) + "\n")

  with args.out_path.joinpath("binary.csv").open(mode="w") as file_handle:
    file_handle.write("\n".join(binary_annotations) + "\n")
  logging.info("Done.")


def parse_args():
  """Parse command line arguments.

  Returns:
    output (argparse.Namespace): Parsed command line arguments. See executable
      help page for more information.
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--backgrounds",
                      type=pathlib.Path,
                      required=True,
                      help="Path to the directory containing "
                      "background images to be used (or file list).")
  parser.add_argument("--templates-path",
                      required=True,
                      type=pathlib.Path,
                      help="Path (or file list) of templates.")
  parser.add_argument("--augmentations",
                      required=True,
                      type=pathlib.Path,
                      help="Path to augmentation configuration file.")
  parser.add_argument("--distractors-path",
                      type=pathlib.Path,
                      help="Path (or file list) of distractors.")
  parser.add_argument(
      "--random-distractors",
      type=int,
      default=1000,
      help="Generate this many random distractors for each template.")
  parser.add_argument(
      "--out-path",
      type=pathlib.Path,
      required=True,
      help="Path to the directory to save the generated images to.")
  parser.add_argument("--max-images",
                      type=int,
                      required=True,
                      help="Number of images to be generated.")
  parser.add_argument("--n",
                      dest="jobs",
                      type=int,
                      default=1,
                      help="Maximum number of parallel processes.")
  parser.add_argument("--max-template-size",
                      type=int,
                      default=196,
                      help="Maximum template size.")
  parser.add_argument("--min-template-size",
                      type=int,
                      default=16,
                      help="Minimum template size.")
  parser.add_argument(
      "--background-size",
      default="800,1360",
      help="If not None (or empty string), image shape 'height,width'")
  return parser.parse_args()


if __name__ == "__main__":
  main(parse_args())
