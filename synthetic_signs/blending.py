import functools
import random

import cv2
import numpy as np

from synthetic_signs.external import pb


def _check_location(source, target, offset):
  """Check that the source is fully contained in target."""
  source_shape = np.asarray(source.shape[:2])
  target_shape = np.asarray(target.shape[:2])
  offset = np.asarray(offset)

  start, end = offset, offset + source_shape
  if np.any(start < 0) or np.any(end > target_shape):
    raise ValueError("Source is not fully contained in target")
  return start, end


def paste_to(source, source_mask, target, offset):
  """Paste source into target image using alpha blending.

  Args:
    source (np.ndarray): Array of shape (H, W) or (H, W, C) which contains
      the source image (dtype np.uint8).
    source_mask (np.ndarray): Array of shape (H, W) which contains the
      source foreground mask (dtype np.uint8). Background pixels should be
      assigned 0 and foreground 255. Values inbetween are used to interpolate
      between source and target (i.e., alpha blending).
    target (np.ndarray): Array of shape (H', W') or (H', W', C) which contains
      the target image.
    offset (Tuple[int, int]): Location (y, x) of the top-left corner of the
      pasted source image in the target image.

  Returns:
    output (np.ndarray): Array of the same shape as target containing the
      blended image.
  """
  start, end = _check_location(source, target, offset)

  if source_mask.ndim < target.ndim:
    source_mask = source_mask[:, :, np.newaxis]

  target_crop = target[start[0]:end[0], start[1]:end[1], ...]
  weight = source_mask.astype(np.float32) / 255.0

  blended_crop = (target_crop.astype(np.float32) * (1.0 - weight) +
                  source.astype(np.float32) * weight)

  blended = target.copy()
  blended[start[0]:end[0], start[1]:end[1], ...] = np.clip(blended_crop,
                                                           a_min=0,
                                                           a_max=255)
  return blended


def poisson_paste_to(source, source_mask, target, offset):
  """Paste source into target using Poisson image editing.

  Args:
    source (np.ndarray): Array of shape (H, W, C) which contains the
      source image (dtype np.uint8).
    source_mask (np.ndarray): Array of shape (H, W) which contains the
      source foreground mask (dtype np.uint8). Only pixels which are
      assigned non-zero values in the mask are copied.
    target (np.ndarray): Array of shape (H', W', C) which contains the target
      image.
    offset (Tuple[int, int]): Location (y, x) of the top-left corner of the
      pasted source image in the target image.

  Returns:
    output (np.ndarray): Array of the same shape as target containing the
      blended image.
  """
  # Increase the size of the source and mask by one pixel. This prevents
  # boundary conditions (foreground next to image border) which would cause
  # the Poisson solver to erode the mask.
  source = cv2.copyMakeBorder(source,
                              1,
                              1,
                              1,
                              1,
                              cv2.BORDER_CONSTANT,
                              value=(0, 0, 0))

  _, source_mask = cv2.threshold(source_mask, 0, 255, cv2.THRESH_BINARY)
  source_mask = cv2.copyMakeBorder(source_mask,
                                   1,
                                   1,
                                   1,
                                   1,
                                   cv2.BORDER_CONSTANT,
                                   value=0)

  offset = offset[0] - 1, offset[1] - 1

  # Execute Poisson solver.
  source = source.astype(np.float64)
  source_mask = source_mask.astype(np.float64) / 255.0
  target = target.astype(np.uint8)

  image_mask, image_source, offset_adjust = pb.create_mask(
      source_mask, target, source, offset)

  blended = pb.poisson_blend(image_mask,
                             image_source,
                             target,
                             method='normal',
                             offset_adj=offset_adjust)
  return blended


def blend_normal(source, source_mask, target):
  """Blend source on top of target image using weighted alpha blending.

  Args:
    source (np.ndarray): Array of shape (H, W, C) which contains the source
      image (dtype np.uint8).
    source_mask (np.ndarray): Array of shape (H, W) which contains the
      source foreground mask (dtype np.uint8). Background pixels should be
      assigned 0 and foreground 255. Values inbetween are used to interpolate
      between source and target.
    target (np.ndarray): Array of shape (H, W, C) which contains the target
      image.

  Returns:
    output (np.ndarray): Array of the same shape as target containing the
      blended image.
  """
  return paste_to(source, source_mask, target, (0, 0))


def blend_box_blur(source, source_mask, target, kernel_size):
  """Blend source on top of target image using weighted alpha blending.

  This function uses a Box filter to generate smooth transition between
  source and target image.

  Args:
    source (np.ndarray): Array of shape (H, W, C) which contains the source
      image (dtype np.uint8).
    source_mask (np.ndarray): Array of shape (H, W) which contains the
      source foreground mask (dtype np.uint8). Background pixels should be
      assigned 0 and foreground 255. Values inbetween are used to interpolate
      between source and target.
    target (np.ndarray): Array of shape (H, W, C) which contains the target
      image.
    kernel_size (int): Size of the box blur mask. Larger mask size corresponds
      to more blur.

  Returns:
    output (np.ndarray): Array of the same shape as target containing the
      blended image.
  """
  blended = blend_normal(source, source_mask, target)
  blended = cv2.boxFilter(blended, -1, (kernel_size, kernel_size))
  return blend_normal(source, source_mask, target)


def blend_gaussian_blur(source, source_mask, target, kernel_size):
  """Blend source on top of target image using weighted alpha blending.

  This function uses a Gaussian filter to generate a smooth transition between
  source and target image.

  Args:
    source (np.ndarray): Array of shape (H, W, C) which contains the source
      image (dtype np.uint8).
    source_mask (np.ndarray): Array of shape (H, W) which contains the
      source foreground mask (dtype np.uint8). Background pixels should be
      assigned 0 and foreground 255. Values inbetween are used to interpolate
      between source and target.
    target (np.ndarray): Array of shape (H, W, C) which contains the target
      image.
    kernel_size (int): Size of the Gaussian blur mask; controls the standard
      deviation. Larger mask size corresponds to more blur.

  Returns:
    output (np.ndarray): Array of the same shape as target containing the
      blended image.
  """
  blended = blend_normal(source, source_mask, target)
  blended = cv2.GaussianBlur(blended, (kernel_size, kernel_size), 0)
  return blended


def blend_motion_blur(source, source_mask, target, kernel_size, angle):
  """Blend source on top of target image using weighted alpha blending.

  This function uses motion blur to generate a smooth transition between
  source and target image.

  Args:
    source (np.ndarray): Array of shape (H, W, C) which contains the source
      image (dtype np.uint8).
    source_mask (np.ndarray): Array of shape (H, W) which contains the
      source foreground mask (dtype np.uint8). Background pixels should be
      assigned 0 and foreground 255. Values inbetween are used to interpolate
      between source and target.
    target (np.ndarray): Array of shape (H, W, C) which contains the target
      image.
    kernel_size (int): Size of the motion blur mask. Larger mask size
      corresponds to more blur.
    angle (float): Direction of blur in degrees.

  Returns:
    output (np.ndarray): Array of the same shape as target containing the
      blended image.
  """
  blended = blend_normal(source, source_mask, target)

  kernel_shape = kernel_size, kernel_size
  blur_kernel = np.zeros(kernel_shape, np.float32)
  blur_kernel[(kernel_size - 1) // 2, :] = 1.0 / kernel_size

  rotation_matrix = cv2.getRotationMatrix2D(
      (kernel_size // 2, kernel_size // 2), angle, scale=1)
  blur_kernel = cv2.warpAffine(blur_kernel,
                               rotation_matrix,
                               kernel_shape,
                               flags=cv2.INTER_LINEAR)

  blended = cv2.filter2D(blended, -1, blur_kernel)
  return blended


def blend_poisson(source, source_mask, target, blur_kernel_size=None):
  """Blend source on top of target image using Poisson blending.

  Args:
    source (np.ndarray): Array of shape (H, W, C) which contains the source
      image (dtype np.uint8).
    source_mask (np.ndarray): Array of shape (H, W) which contains the
      source foreground mask (dtype np.uint8). Background pixels should be
      assigned 0 and foreground 255. Values inbetween 0 and 255 are assigned
      to background or foreground based on a thresholding operation at
      intensity 127.
    target (np.ndarray): Array of shape (H, W, C) which contains the target
      image.
    blur_kernel_size (int, optional): If given, adds Gaussian blur with
      given kernel size to the blended image (blur standard deviation is
      computed from kernel size by OpenCV).

  Returns:
    output (np.ndarray): Array of the same shape as target containing the
      blended image.
  """
  # Poisson image editing only works with binary masks. Use thresholding to
  # generate hard foreground/background assignment.
  _, source_mask = cv2.threshold(source_mask, 127, 255, cv2.THRESH_BINARY)

  # Computational complexity of the Poisson image editing implementation grows
  # quickly with image size (every pixel is a variable in the optimization
  # problem). To speed things up, we find connected components in the mask
  # and paste components individually. This way less pixels that cannot change
  # values end up in the optimization problem.
  contours, _ = cv2.findContours(source_mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
  contours = [c[:, 0, :] for c in contours]

  blended = target.copy()
  for contour in contours:
    # Pad up to two pixels around the object for a smooth blending output.
    x0, y0 = np.maximum(contour.min(axis=0) - 2, 0)
    x1, y1 = np.minimum(
        contour.max(axis=0) + 3, [source.shape[1], source.shape[0]])

    source_crop = source[y0:y1, x0:x1, :]
    source_mask_crop = source_mask[y0:y1, x0:x1]
    target_crop = target[y0:y1, x0:x1, :]

    blended_crop = poisson_paste_to(source_crop, source_mask_crop, target_crop,
                                    (0, 0))
    blended[y0:y1, x0:x1] = blended_crop

  # Apply Gaussian blur.
  if blur_kernel_size is not None:
    blended = cv2.GaussianBlur(blended,
                               ksize=(blur_kernel_size, blur_kernel_size),
                               sigmaX=0)

  return blended


def _blend_random_box_blur(source, source_mask, target):
  kernel_size = random.choice([3, 5, 7, 9, 11, 13])
  return blend_box_blur(source, source_mask, target, kernel_size)


def _blend_random_gaussian_blur(source, source_mask, target):
  kernel_size = random.choice([3, 5, 7, 9, 11, 13])
  return blend_gaussian_blur(source, source_mask, target, kernel_size)


def _blend_random_motion_blur(source, source_mask, target):
  kernel_size = random.choice([3, 5, 7, 9, 11, 13])
  angle = np.random.uniform(0.0, 180.0)
  return blend_motion_blur(source, source_mask, target, kernel_size, angle)


def _blend_random_poisson_blur(source, source_mask, target):
  blur_kernel_size = random.choice([None, 3, 5])
  return blend_poisson(source, source_mask, target, blur_kernel_size)


"""
Entries in this dictionary are used for image blending by the
dataset_generator.
"""
BLEND_NAME_TO_FUNCTION = {
    "normal": blend_normal,
    "box": _blend_random_box_blur,
    "gaussian": _blend_random_gaussian_blur,
    "motion": _blend_random_motion_blur,
    "poisson": _blend_random_poisson_blur,
}
