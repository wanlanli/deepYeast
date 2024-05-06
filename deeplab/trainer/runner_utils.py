# # coding=utf-8
# # Copyright 2022 The Deeplab2 Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """Utility functions for the trainer and evaluator runner."""
from typing import Any
from typing import Mapping
from typing import Union

import tensorflow as tf

import config_yml as config_yml


def maybe_load_checkpoint(initial_checkpoint: Union[str, None],
                          load_dict: Mapping[Any, Any]) -> None:
    """Maybe load a checkpoint.

    Args:
      initial_checkpoint: A string or None, specifying a path to a checkpoint.
      load_dict: A dictionary that defines what to load from the checkpoint.

    Raises:
      ValueError: If load_dict does not contain the 'encoder'.
    """
    if not initial_checkpoint:
        return

    if 'encoder' not in load_dict:
        raise ValueError('Load_dict should contain the encoder, but it is missing.')

    if tf.io.gfile.isdir(initial_checkpoint):
        initial_checkpoint = tf.train.latest_checkpoint(initial_checkpoint)

    # if _load_tf_model_garden_vision_checkpoint(initial_checkpoint):
    #   checkpoint = tf.train.Checkpoint(
    #       backbone=tf.train.Checkpoint(
    #           _encoder=load_dict['encoder']))
    # else:
    checkpoint = tf.train.Checkpoint(**load_dict)
    status = checkpoint.read(initial_checkpoint)
    # Motion-DeepLab models require nontrivial_match, as the input channels for
    # the first convolution change.
    status.expect_partial().assert_nontrivial_match()
