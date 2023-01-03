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

"""This file contains code to create run an experiment."""
from typing import Optional, Sequence

import tensorflow as tf

import config_yml
from data import dataset
from model import deeplab


# Use 1 channel as input (uint16).
_SINGLE_FRAME_INPUT_CHANNELS = 1


def create_deeplab_model(
    config: config_yml.ExperimentOptions,
    dataset_descriptor: dataset.DatasetDescriptor) -> tf.keras.Model:
  """Creates DeepLab model based on config."""
  return deeplab.DeepLab(config, dataset_descriptor)


def build_deeplab_model(deeplab_model: tf.keras.Model,
                        crop_size: Sequence[int],
                        batch_size: Optional[int] = None):
  """Builds DeepLab model with input crop size."""
  input_shape = list(crop_size) + [_SINGLE_FRAME_INPUT_CHANNELS]
  deeplab_model(
        tf.keras.Input(input_shape, batch_size=batch_size), training=False)
  return input_shape
