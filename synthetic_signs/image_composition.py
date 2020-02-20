import collections
import logging

import cv2
import numpy as np

from synthetic_signs import blending
from synthetic_signs import augmentation
from synthetic_signs import utils

Point = collections.namedtuple("Point", ["x", "y"])
Point.__doc__ = ("""
A point in two dimensional (image) space.

Attributes
  x (int | float): Position on x-axis.
  y (int | float): Position on y-axis.
""")

Box = collections.namedtuple("Box", ["x0", "y0", "x1", "y1"])
Box.__doc__ = ("""
This structure holds the coordinates of a bounding box.

Attributes:
  x0 (int): Top-left corner location on x-axis.
  y0 (int): Top-left corner location on y-axis.
  x1 (int): One behind bottom-right corner location on x-axis.
  y1 (int): One behind bottom-right corner location on y-axis.
""")

Template = collections.namedtuple("Template", [
    "image",
    "mask",
    "category",
])
Template.__doc__ = ("""
This structure holds a template that can be pasted on top of background images.

Attributes:
  image (np.ndarray): Array of shape (H, W, 3) which contains a BGR color image
    of dtype np.uint8.
  mask (np.ndarray): Array of shape (H, W) which contains a foreground mask of
    dtype np.uint8, such that the mask evaluates to 0 if the pixel is
    background and 255 if the pixel is foreground with 100 percent certainty.
  category (str): A category string which characterizes this template; used as
    class name for classification.
""")

TemplateAugmentationConfig = collections.namedtuple(
    "TemplateAugmentationConfig", [
        "additive_noise_range",
        "blur_stddev",
        "brightness_scale",
        "cutout_boxes",
        "lighting_gain",
        "saturation_gain",
        "perspective_rate",
        "rotation_angle_deg",
        "scale_factor",
        "template_idx",
        "position",
    ])
TemplateAugmentationConfig.__doc__ = ("""
This structure describes how a single template should be placed and augmented
on top of the background image. Augmentation parameters can take None values
to indicate that the particular augmentation should not be performed.

Attributes:
  additive_noise_range (int, optional): Add uuniform pixel noise in range
    [-x, x] to the template, where x is this value.
  blur_stddev (float, optional): Apply Gaussian blur of given standard
    deviation.
  brightness_scale (float, optional): Scale template brightness by this
    factor. A value of 0 maps to black and 1 impairs no change.
  cutout_boxes (List[Box]): A list of homogeneous box coordinates in [0, 1]
    which describe regions of the template where the mask should be set to
    transparent.
  lighting_gain (float, optional): Add this value to the lighting channel
    of the template's HSV image representation.
  saturation_gain (float, optional): Add this value to the saturation channel
    of the template's HSV image representation.
  perspective_rate (float, optional): Apply perspective distortion to the
    tempalte. A rate of 0 corresponds to no distortion, a rate of 1 collapses
    one side of the template into a single point. Reasonable values are in
    [0.0, 0.2].
  rotation_angle_deg (float, optional): Rotate template by rotation_angle_deg
    degrees.
  scale_factor (float, optional): Scale template by this factor.
  template_idx (int): Index of the template to be placed on background,
    referring to a global list of templates.
  position (Point): Position of the top-left corner of the pasted template in
    homogeneous background image coordinates [0, 1]. NOTE that the position is
    relative to a valid image region where the template can be pasted such that
    it is fully contained in the image.
""")

ImageCompositionConfig = collections.namedtuple("ImageCompositionConfig", [
    "background_filename",
    "blending_mode",
    "brightness_bias",
    "brightness_scale",
    "pixelate_factor",
    "output_filename",
    "template_configs",
    "distractor_configs",
])
ImageCompositionConfig.__doc__ = ("""
This structure describes the composition of a single synthetic image.

Attributes:
  background_filename (str): Path to background image file.
  blending_mode (str): Describes the blending mode used to paste the
    template on top of the background image. See
    synthetic_signs.blending.BLEND_NAME_TO_FUNCTION for a list of
    choices.
  brightness_bias (float): Background image additive brightness bias.
  brightness_scale (float): Composed image brightness scale factor in [0, 1].
  pixelate_factor (float, optional): Composed image pixelation factor in [0, 1].
  output_filename (str): Filename of the generated image.
  template_configs (List[TemplateAugmentationConfig]): List of template
    configurations. Each describes the placement and augmentation of a single
    template.
  distractor_configs (List[TemplateAugmentationConfig]): List of distractor
    configurations. Each describes the placement and augmentation of a single
    distractor.
""")

_BRIGHTNESS_NORMALIZATION_TARGET_VALUE = 170
_BINARY_CLASSNAME = "traffic_sign"


def draw_template(foreground_image, foreground_mask, background, template,
                  template_config, boxes, min_template_size):
  """Draw template on top of background image.

  Args:
    foreground_image (np.ndarray): Array of shape (H, W, 3) and dtype np.uint8
      which contains the foreground (pasted templates) in BGR color space. This
      is where the template is drawn into.
    foreground_mask (np.ndarray): Array of shape (H, W) and dtype np.uint8
      which contains the foreground mask (255 where an object has been pasted
      and 0 otherwise). This is where the template mask is drawn into.
    background (np.ndarray): Array of shape (H, W, 3) and dtype np.uint8 which
      contains the background image in BGR color space. This array is not
      manipulated.
    template (Template): The template to draw on top of the background.
    template_config (TemplateAugmentationConfig): Configuration that describes
      augmentations and blending mode.
    boxes (List[Box]): List of boxes that should not be overwritten.
    min_template_size (int, optional): If not None, pasted templates are
      at least this wide/high.

  Returns:
    output (Box | NoneType): Returns the bounding box of the drawn template
      or None if drawing failed due to overlap with an existing box in
      boxes.
  """
  assert foreground_image.shape[:2] == foreground_mask.shape[:2]
  assert foreground_image.shape[:2] == background.shape[:2]

  # Change template appearance according to template_config.
  template_image, template_mask = template.image, template.mask

  for box in template_config.cutout_boxes:
    h = int(box.y1 - box.y0) * template_mask.shape[0]
    w = int(box.x1 - box.x0) * template_mask.shape[1]

    placement_region_h = max(0, template_mask.shape[0] - h)
    placement_region_w = max(0, template_mask.shape[1] - w)

    y0 = int(box.y0 * placement_region_h)
    y1 = int(box.y1 * placement_region_h)
    x0 = int(box.x0 * placement_region_w)
    x1 = int(box.x1 * placement_region_w)

    template_mask = template_mask.copy()
    template_mask[y0:y1, x0:x1, ...] = 0

  lighting_gain = (template_config.lighting_gain
                   if template_config.lighting_gain is not None else 0)
  saturation_gain = (template_config.saturation_gain
                     if template_config.saturation_gain is not None else 0)
  if lighting_gain > 0 or saturation_gain > 0:
    template_image = augmentation.add_saturation_lighting_noise(
        template_image, saturation_gain, lighting_gain)

  if template_config.blur_stddev is not None:
    template_image = augmentation.blur_gaussian(template_image,
                                                template_config.blur_stddev,
                                                mask=template_mask)

  if template_config.additive_noise_range is not None:
    template_image = augmentation.add_uniform_noise(
        template_image,
        lo=-template_config.additive_noise_range,
        hi=template_config.additive_noise_range)

  if template_config.brightness_scale is not None:
    template_image = augmentation.adjust_brightness(
        template_image,
        multiply_value=template_config.brightness_scale,
        add_value=0)

  if template_config.perspective_rate is not None:
    direction = "left" if template_config.position.x > 0.5 else "right"
    template_image, template_mask = augmentation.distort_perspective(
        template_image, template_mask, template_config.perspective_rate,
        direction)

  if template_config.rotation_angle_deg is not None:
    template_image, template_mask = augmentation.rotate(
        template_image, template_mask, template_config.rotation_angle_deg)

  if template_config.scale_factor is not None:
    if min_template_size is not None:
      minsize = min(*template_image.shape[:2])
      scaled_minsize = template_config.scale_factor * minsize
      scale_factor = (min_template_size /
                      float(minsize) if scaled_minsize < min_template_size else
                      template_config.scale_factor)
    else:
      scale_factor = template_config.scale_factor

    template_image, template_mask = augmentation.scale(template_image,
                                                       template_mask,
                                                       scale_factor)

  # Geometric distortions may introduce padding which we need to remove for an
  # accurate annotation box.
  template_image, template_mask = utils.remove_padding(template_image,
                                                       template_mask)

  # Determine spatial location, making sure we don't generate partially visible
  # templates.
  template_h, template_w = template_image.shape[:2]
  target_h, target_w = background.shape[:2]
  placement_region_h = target_h - template_h
  placement_region_w = target_w - template_w
  if placement_region_h <= 0 or placement_region_h <= 0:
    # Template is too large to fit the image. This should rarely happen,
    # but can if the user sets a very large scale factor or the image is
    # very small.
    return None

  x0 = int(template_config.position.x * placement_region_w)
  y0 = int(template_config.position.y * placement_region_h)
  assert x0 >= 0 and y0 >= 0, "Template placement logic failed"

  x1 = x0 + template_w
  y1 = y0 + template_h
  assert x1 <= target_w and y1 <= target_h, "Template placement logic failed"

  # Adapt brightness to background.
  template_image = augmentation.normalize_brightness(
      template_image,
      template_mask,
      background[y0:y1, x0:x1],
      target_value=_BRIGHTNESS_NORMALIZATION_TARGET_VALUE)

  # Finally, draw template onto background if the location is not occupied
  # by a template already.
  box = Box(x0=x0, y0=y0, x1=x1, y1=y1)
  if utils.has_box_overlap(box, boxes):
    return None

  foreground_image[:, :, :] = blending.paste_to(template_image, template_mask,
                                                foreground_image, (y0, x0))
  foreground_mask[:, :] = blending.paste_to(
      np.full_like(template_mask, fill_value=255), template_mask,
      foreground_mask, (y0, x0))

  return box


def compose_image(image_config,
                  templates,
                  background_size=None,
                  min_template_size=None):
  """Generate a synthetic image for training an object detector.

  Args:
    image_config (ImageCompositionConfig): The configuration which describes
      how templates are pasted onto background.
    templates (List[Template]): A list of templates. The index in the
      TemplateAugmentationConfig of image_config.template_configs should refer
      to this list.
    background_size (Tuple[int, int], optional): If not None, crop and resize background
      image to this (height, width).
    min_template_size (int, optional): If not None, pasted templates are
      at least this wide/high.
  """
  # Load image.
  background = cv2.imread(image_config.background_filename, cv2.IMREAD_COLOR)
  if background is None:
    logging.info("Error loading '{}'.".format(image_config.background_filename))
    return [], []

  if background_size is not None:
    fx = background_size[1] / float(background.shape[1])
    fy = background_size[0] / float(background.shape[0])
    scale_factor = max(fx, fy)

    background = cv2.resize(background, (0, 0),
                            fx=scale_factor,
                            fy=scale_factor)

    assert background.shape[1] >= background_size[1]
    assert background.shape[0] >= background_size[0]

    background = background[:background_size[0], :background_size[1]]

  # Augment background image.
  background = augmentation.adjust_brightness(
      background,
      multiply_value=image_config.brightness_scale,
      add_value=image_config.brightness_bias)

  # Draw templates on foreground image.
  foreground_image = background.copy()
  foreground_mask = np.zeros(foreground_image.shape[:2], dtype=np.uint8)

  boxes, categories = [], []
  for template_config in image_config.template_configs:
    template = templates[template_config.template_idx]
    box = draw_template(foreground_image, foreground_mask, background, template,
                        template_config, boxes, min_template_size)

    if box is None:
      # Template placement failed, e.g. due to overlap with other boxes.
      continue

    boxes.append(box)
    categories.append(template.category)

  # Draw distractors on foreground image.
  for template_config in image_config.distractor_configs:
    template = templates[template_config.template_idx]
    box = draw_template(foreground_image, foreground_mask, background, template,
                        template_config, boxes, min_template_size)

    if box is None:
      # Template placement failed, e.g. due to overlap with other box.
      continue

    boxes.append(box)
    categories.append(None)

  # Blend foreground and background.
  blending_fn = blending.BLEND_NAME_TO_FUNCTION.get(image_config.blending_mode,
                                                    None)
  if blending_fn is None:
    raise ValueError("Invalid blending mode {}".format(blending_mode))
  background = blending_fn(foreground_image, foreground_mask, background)

  # Pixelate composed image.
  perform_pixelation = (image_config.pixelate_factor is not None
                        and image_config.pixelate_factor > 0)
  if perform_pixelation:
    background = augmentation.pixelate(background, image_config.pixelate_factor)

  # Generate annotation lines and write synthetic image to file.
  binary_annotation_lines = []
  multiclass_annotation_lines = []

  for box, category in zip(boxes, categories):
    if category is None:
      continue  # This is a distractor.

    binary_line = ','.join([
        image_config.output_filename,
        str(box.x0),
        str(box.y0),
        str(box.x1),
        str(box.y1),
        _BINARY_CLASSNAME,
    ])
    binary_annotation_lines.append(binary_line)

    multiclass_line = ','.join([
        image_config.output_filename,
        str(box.x0),
        str(box.y0),
        str(box.x1),
        str(box.y1),
        category,
    ])
    multiclass_annotation_lines.append(multiclass_line)

  cv2.imwrite(image_config.output_filename, background)
  return binary_annotation_lines, multiclass_annotation_lines
