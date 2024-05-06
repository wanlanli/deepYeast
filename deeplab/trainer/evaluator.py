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

"""This file contains code to create an evaluator runner.

Note that the evaluator is not well-optimized for inference speed. There are
some redundant outputs, e.g., visualization results, evaluation loss, and so
on. We still compute them in this implementation with the goal to provide more
detailed information for research development. One should remove those
redundant outputs for a faster inference speed.
"""

# import os
import orbit
import tensorflow as tf

from model import utils


class Evaluator(orbit.StandardEvaluator):
  """Implements an evaluator for DeepLab models."""

  def __init__(self, config, model, global_step):
    """Initializes the Evaluator.

    Args:
      config: A config_pb2.ExperimentOptions configuration.
      model: A tf.keras.Model.
      loss: A tf.keras.losses.Loss.
      global_step: A tf.Variable that records the global training step.
      model_dir: A path to store all experimental artifacts.
    """
    self._strategy = tf.distribute.get_strategy()

    self._supported_tasks = utils.get_supported_tasks(config)
    self._config = config
    self._model = model
    self._global_step = global_step
    self._sample_counter = 0

  def _reset(self):
    pass

  def eval_begin(self):
    pass

  def eval_step(self, iterator):
    def step_fn(inputs):
      step_outputs = self._eval_step(inputs)
      return step_outputs

    distributed_outputs = self._strategy.run(step_fn, args=(next(iterator),))
    return tf.nest.map_structure(self._strategy.experimental_local_results,
                                 distributed_outputs)

  def _eval_step(self, inputs):
    return {}
