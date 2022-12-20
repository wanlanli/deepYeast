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

"""This file contains helper functions to run training in a distributed way."""
import tensorflow as tf


def create_strategy(num_gpus: int = 0) -> tf.distribute.Strategy:
  """Creates a strategy based on the given parameters.

  The strategies are created based on the following criteria and order:
  1. If A tpu_address is not None, a TPUStrategy is used.
  2. If num_gpus > 1, a MirrorStrategy is used which replicates the model on
    each GPU.
  3. If num_gpus == 1, a OneDevice strategy is used on the GPU.
  4. If num_gpus == 0, a OneDevice strategy is used on the CPU.

  Args:
    tpu_address: The optional name or address of the TPU to connect to or None.
    num_gpus: A non-negative integer specifying the number of GPUs.

  Returns:
    A tf.distribute.Strategy.

  Raises:
    ValueError: If `num_gpus` is negative and tpu_address is None.
  """
  if num_gpus < 0:
    raise ValueError('`num_gpus` must not be negative.')
  elif num_gpus == 0:
    devices = ['device:CPU:0']
  else:
    devices = ['device:GPU:%d' % i for i in range(num_gpus)]
  if len(devices) == 1:
    return tf.distribute.OneDeviceStrategy(devices[0])
  return tf.distribute.MirroredStrategy(devices)
