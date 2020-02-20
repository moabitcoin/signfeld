import cv2
import numpy as np

from synthetic_signs import utils


def adjust_brightness(image, multiply_value, add_value):
  """Adjust image brightness.

  Args:
    image (np.ndarray): Array of shape (H, W, ...) and dtype np.uint8 which
      contains the image.
    multiply_value (float | np.ndarray): Scalar scale factor or array of the
      same shape as (or broadcastable to) image.
    add_value (float | np.ndarray): Scalar additive intensity adjustment
      value or array of the same shape as (or broadcastable to) image.

  Returns:
    output (np.ndarray): Adjusted image multiply_value * (image + add_value).
  """
  adjusted_image = multiply_value * (image.astype(np.float32) + add_value)
  return np.clip(adjusted_image, a_min=0, a_max=255).astype(np.uint8)


def add_uniform_noise(image, lo, hi):
  """Add uniform noise in U(lo, hi) to every color channel and pixel in a given
  image.

  Args:
    image (np.ndarray): Array of shape (H, W, ...) and dtype np.uint8 which
      contains the image.
    lo (int): Lower bound of uniform noise distribution (inclusive).
    hi (int): Upper bound of uniform noise distribution (inclusive).

  Returns:
    output (np.ndarray): Noisy image [image]_ij + delta_ij where
      delta_ij ~ U(lo, hi).
  """
  if hi <= lo:
    return image  # Nothing to do.

  noise = np.random.randint(lo, hi, size=image.shape)
  return np.clip(image + noise, a_min=0, a_max=255).astype(np.uint8)


def add_saturation_lighting_noise(image,
                                  saturation_gain,
                                  lighting_gain,
                                  saturation_threshold=5):
  """Change saturation and lighting.

  Args:
    image (np.ndarray): Array of shape (H, W, C) and dtype np.uint8 which
      contains the BGR color image.
    saturation_gain (int): Add this value to the saturation channel of the
      HSV image representation.
    lighting_gain (int): Add this value to the lighting channel of the HSV
      image representation.
    saturation_threshold (int, optional): Do not apply changes to pixels with
      saturation lower than this value. Used to mask out black and white pixels
      to avoid unwanted color jitter. Default: 5.

  Returns:
    output (np.ndarray): Array of the same shape and dtype as the input image
      with modified saturation and lighting.
  """
  image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)

  valid_mask = image_hsv[:, :, 1] > saturation_threshold
  image_hsv[valid_mask, 1] += saturation_gain
  image_hsv[valid_mask, 2] += lighting_gain
  image_hsv = np.clip(image_hsv, a_min=0, a_max=255).astype(np.uint8)

  image_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
  return image_bgr


def normalize_brightness(image, mask, reference_patch, target_value):
  """Adjust brightness of image to match brightness of a reference patch.

  This function compares the average brightness of a given reference patch
  to a target value and increases/decreases the brightness of the image by
  this difference.

  Args:
    image (np.ndarray): Array of shape (H, W, 3) and dtype np.uint8 which
      contains a BGR color image.
    mask (np.ndarray): Array of shape (H, W) and type np.uint8 which contains
      a foreground mask associated with the image. Foreground pixels should
      be assigned non-zero values.
    reference_patch (np.ndarray): Array of shape (H', W', 3) which contains
      the reference image patch in BGR color space (dtype np.uint8).
    target_value (int): The target value which the average brightness of the
      reference_patch is compared against.

  Returns:
    output (np.ndarray): Returns the brightness adjusted version of the input
      image. Roughly image + (reference_patch.mean() - target_value), but
      taking care of masked pixels.
  """
  reference_patch = cv2.cvtColor(reference_patch, cv2.COLOR_BGR2GRAY)
  reference_mean = reference_patch.mean()
  difference = reference_mean.astype(np.int) - target_value

  image = image.astype(np.int)
  image[mask > 0, ...] += difference
  image = np.clip(image, a_min=0, a_max=255).astype(np.uint8)
  return image


def blur_gaussian(image, blur_stddev, mask=None):
  """Apply Gaussian blur.

  Args:
    image (np.ndarray): Array of shape (H, W, 3) and dtype np.uint8 which
      contains a BGR color image.
    blur_stddev (float): Blur standard deviation.
    mask (np.ndarray, optional): If given, array of shape (H, W) which
      contains a foreground mask where invalid pixels are assigned 0 and
      valid pixels 255. Only valid pixels are taken into consideration during
      blurring.

  Returns:
    output (np.ndarray): Blurred image of the same shape and dtype.
  """
  if mask is None:
    # No invalid pixels to take care of, therefore simple blur.
    return cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=blur_stddev)

  # Make sure we have a binary mask by thresholding at 50%.
  _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

  blurred_image = image.copy()
  blurred_image[binary_mask == 0] = 0
  blurred_image = cv2.GaussianBlur(blurred_image,
                                   ksize=(0, 0),
                                   sigmaX=blur_stddev)

  blurred_mask = cv2.GaussianBlur(binary_mask, ksize=(0, 0), sigmaX=blur_stddev)

  invalid_mask = np.logical_or(binary_mask == 0, blurred_mask == 0)
  valid_mask = np.logical_not(invalid_mask)

  blurred_image[invalid_mask] = image[invalid_mask]
  blurred_image[valid_mask] = np.clip(
      255.0 * blurred_image[valid_mask].astype(np.float32) /
      blurred_mask[valid_mask, np.newaxis],
      a_min=0,
      a_max=255).astype(np.uint8)
  return blurred_image


def distort_perspective(template, template_mask, rate, direction):
  """Generate perspective effect.

  Args:
    template (np.ndarray): Array of shape (H, W, C) and dtype np.uint8 which
      contains the input template image.
    template_mask (np.ndarray): Array of shape (H, W, C) and dtype np.uint8
      which contains the input template foreground mask.
    rate (float): Distortation rate in range [0, 1] where 0 corresponds to
      no distortion and 1 collapses one side of the template image into
      a single point. Reasonable values are in [0, 0.2].
    direction (str): Distoration direction that specifies the side of the
      template image to distort; either 'left' or 'right'.

  Returns:
    output(Tuple[np.ndarray, np.ndarray]): Returns the transformed template
    image and mask. The shape remains the same.
  """
  if rate < 0 or rate > 1.0:
    raise ValueError("rate must be in [0, 1]")
  min_size = min(template.shape[0], template.shape[1])

  h, w, o = template.shape[0] - 1, template.shape[1] - 1, rate * min_size

  points1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
  if direction == "left":
    points2 = np.float32([[o, o], [w, 0], [o, h - o], [w, h]])
  elif direction == "right":
    points2 = np.float32([[0, 0], [w - o, o], [0, h], [w - o, h - o]])
  else:
    raise ValueError("Direction must be either 'left' or 'right'")

  transform = cv2.getPerspectiveTransform(points1, points2)
  template = cv2.warpPerspective(template,
                                 transform,
                                 (template.shape[1], template.shape[0]),
                                 flags=cv2.INTER_LINEAR)
  template_mask = cv2.warpPerspective(template_mask,
                                      transform,
                                      (template.shape[1], template.shape[0]),
                                      flags=cv2.INTER_LINEAR)
  return template, template_mask


def rotate(template, template_mask, angle_deg):
  """Rotate template.

  Args:
    template (np.ndarray): Array of shape (H, W, C) and dtype np.uint8 which
      contains the input template image.
    template_mask (np.ndarray): Array of shape (H, W, C) and dtype np.uint8
      which contains the input template foreground mask.
    angle_deg (float): Rotation angle in degrees.

  Returns:
    output (Tuple[np.ndarray, np.ndarray]): Returns the transformed template
    image and mask. The shape remains the same.
  """
  max_size = max(template.shape[:2])
  padding = max_size // 2 + 1
  template = cv2.copyMakeBorder(template, padding, padding, padding, padding,
                                cv2.BORDER_CONSTANT)
  template_mask = cv2.copyMakeBorder(template_mask, padding, padding, padding,
                                     padding, cv2.BORDER_CONSTANT)

  h, w = template.shape[:2]
  transform = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, scale=1)
  template = cv2.warpAffine(template, transform, (w, h), flags=cv2.INTER_LINEAR)
  template_mask = cv2.warpAffine(template_mask,
                                 transform, (w, h),
                                 flags=cv2.INTER_LINEAR)

  template, template_mask = utils.remove_padding(template, template_mask)

  return template, template_mask


def scale(template, template_mask, scale_factor):
  """Scale template and mask.

  Args:
    template (np.ndarray): Array of shape (H, W, C) and dtype np.uint8 which
      contains the input template image.
    template_mask (np.ndarray): Array of shape (H, W, C) and dtype np.uint8
      which contains the input template foreground mask.
    scale_factor (float): Scale image and foreground mask by given factor.

  Returns:
    output (Tuple[np.ndarray, np.ndarray]): Returns the transformed template
    image and mask.
  """
  template = cv2.resize(template, (0, 0),
                        fx=scale_factor,
                        fy=scale_factor,
                        interpolation=cv2.INTER_LINEAR)
  template_mask = cv2.resize(template_mask, (0, 0),
                             fx=scale_factor,
                             fy=scale_factor,
                             interpolation=cv2.INTER_LINEAR)
  return template, template_mask


def pixelate(image, scale_factor):
  """Create low-resolution effect by image down and up scaling.

  Args:
    image (np.ndarray): Array of shape (H, W, C) and dtype np.uint8 which
      contains the input image.
    scale_factor (float): Image scale factor in ]0, 1]. The image is first
      downscaled by this factor and then resized up to its original size.
  """
  h, w = image.shape[:2]
  image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
  image = cv2.resize(image, (w, h))
  return image
