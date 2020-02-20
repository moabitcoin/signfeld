import cv2
import numpy as np


def clip_image_at_border(source_image, target_shape, offset):
  """Clip source image at border of image with given target shape.

    Args:
      source_image (np.ndarray): Array of shape (H, W, ...) which contains
        the source image.
      target_shape (Tuple[int, int]): Height and width of the target image.
      offset (Tuple[int, int]): Location (y, x) of the top-left corner of
        the source image in the target image.

    Returns:
      output (Tuple[np.ndarray, Tuple[int, int]]): A tuple with the following
        entries:

        * Array of shape (H', W', ...) which contains the part of the
          source_image which overlaps with the target image.
        * Location (y, x) of the top-left corner of the clipped source image.
    """
  source_shape = np.asarray(source_image.shape[:2])
  target_shape = np.asarray(target_shape)
  offset = np.asarray(offset)

  # Compute region of interest in target image.
  box_min, box_max = offset, offset + source_shape - 1
  box_min = np.clip(box_min, a_min=0, a_max=None)
  box_max = np.clip(box_max, a_min=None, a_max=target_shape - 1)

  # Catch invalid configuration and default to empty box.
  if np.any(box_max < box_min):
    clipped_image = np.zeros([0] * source_image.ndim, source_image.dtype)
    return clipped_image, (0, 0)

  box_max = np.clip(box_max, a_min=box_min, a_max=None)
  box_min = np.clip(box_min, a_max=box_max, a_min=None)

  # Compute corrected offset and source_image.
  start = box_min - offset
  size = box_max - box_min + 1
  end = start + size

  return source_image[start[0]:end[0], start[1]:end[1]], tuple(box_min)


def resize_image(image, new_width=None, new_height=None):
  """Resize image.

  Args:
    image (np.ndarray): Array of shape (H, W) or (H, W, C) which contains
      the input image.
    new_width (int, optional): If not None, the output image is resized to
      this width. If None, new_height must not be None.
    new_height (int, optional): If not None, the output image is resized to
      this height. If None, new_width must not be None.

  Returns:
    output (np.ndarray): The resized image of shape (new_height, new_width,
      ...). If one of new_height or new_width is not given, they are computed
      to fit the aspect ratio of the input image.
  """
  if new_width is None and new_height is None:
    raise ValueError("At least one of 'new_width', 'new_height' must be set")

  height, width = image.shape[:2]
  if new_width is not None and new_height is None:
    aspect_ratio = new_width / width
    new_height = int(height * aspect_ratio)
  elif new_width is None and new_height is not None:
    aspect_ratio = new_height / height
    new_width = int(width * aspect_ratio)
  new_image = cv2.resize(image, (new_width, new_height))
  return new_image


def remove_padding(image, mask):
  """Remove padding from mask.

  Args:
    image (np.ndarray): Array of shape (H, W, ...) which contains the input
      image.
    mask (np.ndarray): Array of shape (H, W) which contains the foreground
      mask with transparent/background pixels assigned 0 values.

  Returns:
    output (Tuple[np.ndarray, np.ndarray]): Returns the cropped image and
      mask where transparent borders have been removed (tightest possible
      fit).
  """
  foreground_pixels = np.argwhere(mask > 0)
  y0, x0 = foreground_pixels.min(axis=0)
  y1, x1 = foreground_pixels.max(axis=0) + 1
  return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


# TODO(niwojke): Make max_iou configurable through app.
def has_box_overlap(box, boxes, max_iou=0.3):
  """Check if bounding box overlaps with list of other boxes.

  Args:
    box (Box): Bounding box.
    boxes (List[Box]): List of other boxes.
    max_iou (float, optional): Return true if intersection over union
      between box and any box in boxes is larger than this value. Must
      be a value in [0, 1] (larger means more occlusion is allowed).

  Returns:
    output (bool): True if box overlaps with any box in boxes, False
      otherwise.
  """
  box_area = (box.x1 - box.x0) * (box.y1 - box.y0)
  for other_box in boxes:
    other_area = (other_box.x1 - other_box.x0) * (other_box.y1 - other_box.y0)

    intersection_x0 = max(box.x0, other_box.x0)
    intersection_y0 = max(box.y0, other_box.y0)

    intersection_x1 = min(box.x1, other_box.x1)
    intersection_y1 = min(box.y1, other_box.y1)

    intersection_area = ((intersection_x1 - intersection_x0) *
                         (intersection_y1 - intersection_y0))
    intersection_area = max(0, intersection_area)

    denominator = float(box_area + other_area - intersection_area)
    iou = intersection_area / denominator if denominator > 0.0 else 1.0
    if iou > max_iou:
      return True
  return False
