import unittest

import numpy as np

from synthetic_signs import utils


class TestUtils(unittest.TestCase):

  def setUp(self):
    self.source_image = np.zeros((5, 5), dtype=np.uint8)
    self.target_image = np.ones((11, 11), dtype=np.uint8)

  def test_clip_image_at_border_contained(self):
    clipped_image, clipped_offset = utils.clip_image_at_border(
        self.source_image, self.target_image.shape[:2], offset=(2, 2))

    self.assertTupleEqual(clipped_image.shape, self.source_image.shape)
    self.assertTupleEqual(clipped_offset, (2, 2))

  def test_clip_image_at_border_outside(self):
    clipped_image, clipped_offset = utils.clip_image_at_border(
        self.source_image, self.target_image.shape[:2], offset=(-5, 11))

    actual_shape = np.asarray(clipped_image.shape)
    expected_shape = np.zeros((2, ), dtype=np.int64)
    self.assertTupleEqual(clipped_image.shape, (0, 0))
    self.assertTupleEqual(clipped_offset, (0, 0))

  def test_clip_image_at_border_clip_lo(self):
    clipped_image, clipped_offset = utils.clip_image_at_border(
        self.source_image, self.target_image.shape[:2], offset=(-2, 1))

    self.assertTupleEqual(clipped_image.shape, (3, 5))
    self.assertTupleEqual(clipped_offset, (0, 1))

  def test_clip_image_at_border_clip_hi(self):
    clipped_image, clipped_offset = utils.clip_image_at_border(
        self.source_image, self.target_image.shape[:2], offset=(1, 7))

    self.assertTupleEqual(clipped_image.shape, (5, 4))
    self.assertTupleEqual(clipped_offset, (1, 7))
