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

"""Contains common utility functions and classes for building dataset."""

import collections
import io

import numpy as np
# from PIL import Image
# from PIL import ImageOps
import tensorflow as tf
from skimage.io import imread

import common


def read_image(image_data):
    """Decodes image from in-memory data.

    Args:
      image_data: Bytes data representing encoded image.

    Returns:
      Decoded PIL.Image object.
    """
    # image = Image.open(io.BytesIO(image_data))
    try: ##读数据有问题需要重新做
        image = imread(image_data, plugin='imageio')
        # norm image to 0 to 1
        # print(np.max(image))
        # image = image.astype(np.float32)
        # image = (image - np.min(image))/(np.max(image) - np.min(image))
    #    image_data = io.BytesIO()
    #   image.save(image_data, format=image_format)
    #   image_data = image_data.getvalue()
    except TypeError:
        # capture and ignore this bug:
        # https://github.com/python-pillow/Pillow/issues/3973
        pass
    return image


# def get_image_dims(image_data):
#     """Decodes image and return its height and width.

#     Args:
#         image_data: Bytes data representing encoded image.
#         check_is_rgb: Whether to check encoded image is RGB.

#     Returns:
#         Decoded image size as a tuple of (height, width)

#     Raises:
#         ValueError: If check_is_rgb is set and input image has other format.
#     """
#     image = read_image(image_data)
#     width, height = image.shape
#     return height, width


def _int64_list_feature(values):
    """Returns a TF-Feature of int64_list.

    Args:
      values: A scalar or an iterable of integer values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
      A TF-Feature.
    """
    if isinstance(values, str):
        values = values.encode()

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def create_features(image_data,
                    image_format,
                    filename,
                    label_data=None,
                    label_format=None,):
    """Creates image/segmentation features.

    Args:
      image_data: String or byte stream of encoded image data.
      image_format: String, image data format, should be either 'jpeg', 'jpg', or
        'png'.
      filename: String, image filename.
      label_data: String or byte stream of (potentially) encoded label data. If
        None, we skip to write it to tf.train.Example.
      label_format: String, label data format, should be either 'png' or 'raw'. If
        None, we skip to write it to tf.train.Example.

    Returns:
      A dictionary of feature name to tf.train.Feature maaping.
    """
    if image_format not in ('jpeg', 'png', 'jpg'):
        raise ValueError('Unsupported image format: %s' % image_format)

    # Check color mode, and convert grey image to rgb image.
    image = read_image(image_data)
    # print("max:", np.max(image))
    # if image.mode != 'RGB':
    #   # image = image.convert('RGB')
    #   a = np.array(image)
    #   a = (a-a.min())/(a.max()-a.min())
    #   a = np.uint8(a*255)
    #   image = Image.fromarray(a).convert("RGB")
    #   image_data = io.BytesIO()
    #   image.save(image_data, format=image_format)
    #   image_data = image_data.getvalue()
    width, height = image.shape
    # height, width = get_image_dims(image_data, check_is_rgb=False, image_path=image_path)

    feature_dict = {
        common.KEY_ENCODED_IMAGE: _bytes_list_feature(image_data),
        common.KEY_IMAGE_FILENAME: _bytes_list_feature(filename),
        common.KEY_IMAGE_FORMAT: _bytes_list_feature(image_format),
        common.KEY_IMAGE_HEIGHT: _int64_list_feature(height),
        common.KEY_IMAGE_WIDTH: _int64_list_feature(width),
        common.KEY_IMAGE_CHANNELS: _int64_list_feature(1),
    }

    if label_data is None:
        return feature_dict

    if label_format == 'png':
        label_height, label_width = np.fromstring(label_data, dtype=np.int32).shape # get_image_dims(label_data, image_path=image_path)
        if (label_height, label_width) != (height, width):
            raise ValueError('Image (%s) and label (%s) shape mismatch' %
                        ((height, width), (label_height, label_width)))
    elif label_format == 'raw':
      # Raw label encodes int32 array.
        expected_label_size = height * width * np.dtype(np.int32).itemsize
        if len(label_data) != expected_label_size:
            raise ValueError('Expects raw label data length %d, gets %d' % 
                              (expected_label_size, len(label_data)))
    else:
        raise ValueError('Unsupported label format: %s' % label_format)

    feature_dict.update({
        common.KEY_ENCODED_LABEL: _bytes_list_feature(label_data),
        common.KEY_LABEL_FORMAT: _bytes_list_feature(label_format)
    })

    return feature_dict


def create_tfexample(image_data,
                     image_format,
                     filename,
                     label_data=None,
                     label_format=None):
    """Converts one image/segmentation pair to TF example.

    Args:
      image_data: String or byte stream of encoded image data.
      image_format: String, image data format, should be either 'jpeg' or 'png'.
      filename: String, image filename.
      label_data: String or byte stream of (potentially) encoded label data. If
        None, we skip to write it to tf.train.Example.
      label_format: String, label data format, should be either 'png' or 'raw'. If
        None, we skip to write it to tf.train.Example.

    Returns:
      TF example proto.
    """
    feature_dict = create_features(image_data, image_format, filename, label_data,
                                  label_format)
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


class SegmentationDecoder(object):
    """Basic parser to decode serialized tf.Example."""

    def __init__(self,
                is_panoptic_dataset=True,
                decode_groundtruth_label=True):
        self._is_panoptic_dataset = is_panoptic_dataset
        self._decode_groundtruth_label = decode_groundtruth_label
        string_feature = tf.io.FixedLenFeature((), tf.string)
        int_feature = tf.io.FixedLenFeature((), tf.int64)
        self._keys_to_features = {
            common.KEY_ENCODED_IMAGE: string_feature,
            common.KEY_IMAGE_FILENAME: string_feature,
            common.KEY_IMAGE_FORMAT: string_feature,
            common.KEY_IMAGE_HEIGHT: int_feature,
            common.KEY_IMAGE_WIDTH: int_feature,
            common.KEY_IMAGE_CHANNELS: int_feature,
        }
        if decode_groundtruth_label:
            self._keys_to_features[common.KEY_ENCODED_LABEL] = string_feature

    def _decode_image(self, parsed_tensors, key):
        """Decodes image udner key from parsed tensors."""
        image = tf.io.decode_image(
            parsed_tensors[key],
            channels=1,
            dtype=tf.dtypes.uint16,
            expand_animations=False)
        image.set_shape([None, None, 1])
        return image

    def _decode_label(self, parsed_tensors, label_key):
        """Decodes segmentation label under label_key from parsed tensors."""
        if self._is_panoptic_dataset:
            flattened_label = tf.io.decode_raw(
                parsed_tensors[label_key], out_type=tf.int32)
            label_shape = tf.stack([
                parsed_tensors[common.KEY_IMAGE_HEIGHT],
                parsed_tensors[common.KEY_IMAGE_WIDTH],
                1
            ])
            label = tf.reshape(flattened_label, label_shape)
            return label
        label = tf.io.decode_image(parsed_tensors[label_key], channels=1)
        label.set_shape([None, None, 1])
        return label

    def __call__(self, serialized_example):
        parsed_tensors = tf.io.parse_single_example(
            serialized_example, features=self._keys_to_features)
        return_dict = {
            'image':
                self._decode_image(parsed_tensors, common.KEY_ENCODED_IMAGE),
            'image_name':
                parsed_tensors[common.KEY_IMAGE_FILENAME],
            'height':
                tf.cast(parsed_tensors[common.KEY_IMAGE_HEIGHT], dtype=tf.int32),
            'width':
                tf.cast(parsed_tensors[common.KEY_IMAGE_WIDTH], dtype=tf.int32),
        }
        return_dict['label'] = None
        if self._decode_groundtruth_label:
            return_dict['label'] = self._decode_label(parsed_tensors,
                                                    common.KEY_ENCODED_LABEL)
        return return_dict
