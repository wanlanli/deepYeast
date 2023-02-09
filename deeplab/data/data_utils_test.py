# coding=utf-8
# Copyright 2022 The Deeplab2 Authors.
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

"""Tests for data_utils."""

import io
import numpy as np
from skimage.io import imsave
import tensorflow as tf

from data import data_utils


def _encode_png_image(image):
    """Helper method to encode input image in PNG format."""
    imsave("./test.png", image)
    with tf.io.gfile.GFile("./test.png", 'rb') as f:
        image_data = f.read()
    # Image.fromarray(image).save(buffer, format='png')
    # return buffer.getvalue()
    return image_data


class DataUtilsTest(tf.test.TestCase):

  def _create_test_image(self, height, width):
    # rng = np.random.RandomState(319281498)
    data = np.random.randint(0,10000, size=(height, width, 1), dtype=np.uint16)
    return data# rng.randn(0, 1, size=(height, width, 1), dtype=np.float32)

  def test_encode_and_decode(self):
    """Checks decode created tf.Example for semantic segmentation."""
    test_image_height = 20
    test_image_width = 15
    filename = 'dummy'

    image = self._create_test_image(test_image_height, test_image_width)
    # Take the last channel as dummy label.
    # label = image[..., 0]

    example = data_utils.create_tfexample(
        image_data=_encode_png_image(image),
        image_format='png', filename=filename,
        label_data=None, label_format=None)

    # Parse created example, expect getting identical results.
    parser = data_utils.SegmentationDecoder(is_panoptic_dataset=False)
    parsed_tensors = parser(example.SerializeToString())

    self.assertIn('image', parsed_tensors)
    self.assertIn('image_name', parsed_tensors)
    # self.assertIn('label', parsed_tensors)
    self.assertEqual(filename, parsed_tensors['image_name'])
    np.testing.assert_array_equal(image, parsed_tensors['image'].numpy())
    # Decoded label is a 3-D array with last dimension of 1.
    # decoded_label = parsed_tensors['label'].numpy()
    # np.testing.assert_array_equal(label, decoded_label[..., 0])

  def test_encode_and_decode_panoptic(self):
    test_image_height = 31
    test_image_width = 17
    filename = 'dummy'

    image = self._create_test_image(test_image_height, test_image_width)
    # Create dummy panoptic label in np.int32 dtype.
    label = np.dot(image.astype(np.int32), [1]).astype(np.int32)
    example = data_utils.create_tfexample(
        image_data=_encode_png_image(image),
        image_format='png', filename=filename,
        label_data=label.tostring(), label_format='raw')

    parser = data_utils.SegmentationDecoder(is_panoptic_dataset=True)
    parsed_tensors = parser(example.SerializeToString())

    self.assertIn('image', parsed_tensors)
    self.assertIn('image_name', parsed_tensors)
    # self.assertIn('label', parsed_tensors)
    self.assertEqual(filename, parsed_tensors['image_name'])
    return image, parsed_tensors
    np.testing.assert_array_equal(image, parsed_tensors['image'].numpy())
    # Decoded label is a 3-D array with last dimension of 1.
    # decoded_label = parsed_tensors['label'].numpy()
    # np.testing.assert_array_equal(label, decoded_label[..., 0])
    

if __name__ == '__main__':
  tf.test.main()
