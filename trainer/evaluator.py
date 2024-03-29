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

import os
import orbit
import tensorflow as tf

import common
from data import dataset
from evaluation import panoptic_quality
from model import utils
from trainer import runner_utils
from trainer import vis

_PANOPTIC_METRIC_OFFSET = 256 * 256 * 256
_PREDICTIONS_KEY = 'unique_key_for_storing_predictions'
_LABELS_KEY = 'unique_key_for_storing_labels'


class Evaluator(orbit.StandardEvaluator):
  """Implements an evaluator for DeepLab models."""

  def __init__(self, config, model, loss, global_step, model_dir):
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
    eval_dataset = runner_utils.create_dataset(
        config.eval_dataset_options,
        is_training=False,
        only_semantic_annotations=(common.TASK_PANOPTIC_SEGMENTATION
                                   not in self._supported_tasks))
    eval_dataset = orbit.utils.make_distributed_dataset(self._strategy,
                                                        eval_dataset)
    evaluator_options_override = orbit.StandardEvaluatorOptions(
        config.evaluator_options.use_tf_function)
    super(Evaluator, self).__init__(eval_dataset, evaluator_options_override)
    self._config = config
    self._model = model
    self._loss = loss
    self._global_step = global_step
    self._sample_counter = 0
    self._enable_visualization = config.evaluator_options.save_predictions
    self._num_vis_samples = 3 # config.evaluator_options.num_vis_samples
    self._save_raw_predictions = config.evaluator_options.save_raw_predictions
    self._decode_groundtruth_label = (
        config.eval_dataset_options.decode_groundtruth_label)
    if hasattr(config.evaluator_options, 'override_save_dir'):
      self._vis_dir = config.evaluator_options.override_save_dir
    else:
      self._vis_dir = os.path.join(model_dir, 'vis')

    self._dataset_info = dataset.MAP_NAME_TO_DATASET_INFO[
        config.eval_dataset_options.dataset]

    # Create eval loss metrics.
    self._eval_loss_metric_dict = runner_utils.create_loss_metric_dict(
        loss.get_loss_names(), prefix='eval_')
    # Create metrics (PQ, IoU).
    self._ignore_label = self._dataset_info.ignore_label
    self._eval_iou_metric = tf.keras.metrics.MeanIoU(
        self._dataset_info.num_classes, 'IoU')

    if common.TASK_PANOPTIC_SEGMENTATION in self._supported_tasks:
      self._eval_pq_metric = panoptic_quality.PanopticQuality(
          self._dataset_info.num_classes,
          self._dataset_info.ignore_label,
          self._dataset_info.panoptic_label_divisor,
          offset=_PANOPTIC_METRIC_OFFSET)
    ##重写此处的测量
    # if common.TASK_INSTANCE_SEGMENTATION in self._supported_tasks:
    #   self._eval_ap_metric = instance_ap.PanopticInstanceAveragePrecision(
    #       self._dataset_info.num_classes,
    #       self._dataset_info.class_has_instances_list,
    #       self._dataset_info.panoptic_label_divisor,
    #       self._dataset_info.ignore_label)
    # if common.TASK_VIDEO_PANOPTIC_SEGMENTATION in self._supported_tasks:
    #   self._eval_tracking_metric = stq.STQuality(
    #       self._dataset_info.num_classes,
    #       self._dataset_info.class_has_instances_list,
    #       self._dataset_info.ignore_label,
    #       self._dataset_info.panoptic_label_divisor,
    #       offset=_PANOPTIC_METRIC_OFFSET)
    # if (common.TASK_DEPTH_AWARE_VIDEO_PANOPTIC_SEGMENTATION
    #     in self._supported_tasks):
    #   # We compute two-frame video panoptic quality as an additional metric
    #   # for the task of depth-aware video panoptic segmentation.
    #   self._eval_vpq_metric = vpq.VideoPanopticQuality(
    #       self._dataset_info.num_classes,
    #       self._dataset_info.ignore_label,
    #       self._dataset_info.panoptic_label_divisor,
    #       offset=_PANOPTIC_METRIC_OFFSET)
    #   self._eval_depth_metric = depth_metrics.DepthMetrics()

  def _reset(self):
    for metric in self._eval_loss_metric_dict.values():
      metric.reset_states()
    self._eval_iou_metric.reset_states()
    if common.TASK_PANOPTIC_SEGMENTATION in self._supported_tasks:
      self._eval_pq_metric.reset_states()
    self._sample_counter = 0

  def eval_begin(self):
    """Called once at the beginning of the evaluation.

    This method is called before dataset iterators creation.
    """
    self._reset()
    tf.io.gfile.makedirs(self._vis_dir)
    if self._save_raw_predictions:
      tf.io.gfile.makedirs(os.path.join(self._vis_dir, 'raw_semantic'))
      if common.TASK_PANOPTIC_SEGMENTATION in self._supported_tasks:
        tf.io.gfile.makedirs(os.path.join(self._vis_dir, 'raw_panoptic'))
      # if (common.TASK_DEPTH_AWARE_VIDEO_PANOPTIC_SEGMENTATION
      #     in self._supported_tasks):
      #   tf.io.gfile.makedirs(os.path.join(self._vis_dir, 'raw_depth'))

  def eval_step(self, iterator):
    """Implements one step of evaluation.

    Runs one step of evaluation with respect to the chosen strategy. In case of
    a distributed strategy, the replica results are gathered and returned.

    Note that all operations within `_eval_step` are tf.function compatible, as
    they will be traced with tf.function. Any other/numpy operations are put in
    `eval_begin`, `eval_end` or `eval_reduce` functions.

    Args:
      iterator: A tf.nest-compatible structure of tf.data Iterator or
        DistributedIterator.

    Returns:
      An output which is passed as `step_outputs` argument into `eval_reduce`
      function.
    """

    def step_fn(inputs):
      step_outputs = self._eval_step(inputs)
      return step_outputs

    distributed_outputs = self._strategy.run(step_fn, args=(next(iterator),))
    return tf.nest.map_structure(self._strategy.experimental_local_results,
                                 distributed_outputs)

  def _eval_step(self, inputs):
    tf.assert_equal(
        tf.shape(inputs[common.IMAGE])[0], 1, 'Currently only a '
        'batchsize of 1 is supported in evaluation due to resizing.')
    outputs = self._model(inputs[common.IMAGE], training=False)
    raw_size = [
        inputs[common.GT_SIZE_RAW][0, 0], inputs[common.GT_SIZE_RAW][0, 1]
    ]
    resized_size = [
        tf.shape(inputs[common.RESIZED_IMAGE])[1],
        tf.shape(inputs[common.RESIZED_IMAGE])[2],
    ]

    step_outputs = {}
    if self._decode_groundtruth_label:

      loss_dict = self._loss(inputs, outputs)
      # Average over the batch.
      average_loss_dict = {
          key: tf.reduce_mean(value) for key, value in loss_dict.items()
      }

      for name, value in average_loss_dict.items():
        self._eval_loss_metric_dict[name].update_state(value)

      # We only undo-preprocess for those defined in tuples in model/utils.py.
      outputs = utils.undo_preprocessing(outputs, resized_size,
                                         raw_size)

      self._eval_iou_metric.update_state(
          tf.where(
              tf.equal(inputs[common.GT_SEMANTIC_RAW], self._ignore_label), 0,
              inputs[common.GT_SEMANTIC_RAW]),
          outputs[common.PRED_SEMANTIC_KEY],
          tf.where(
              tf.equal(inputs[common.GT_SEMANTIC_RAW], self._ignore_label), 0.0,
              1.0))
      if common.TASK_PANOPTIC_SEGMENTATION in self._supported_tasks:
        step_outputs[self._eval_pq_metric.name] = (
            inputs[common.GT_PANOPTIC_RAW], outputs[common.PRED_PANOPTIC_KEY])
    else:
      # We only undo-preprocess for those defined in tuples in model/utils.py.
      outputs = utils.undo_preprocessing(outputs, resized_size,
                                         raw_size)
    # We only undo-preprocess for those defined in tuples in model/utils.py.
    inputs = utils.undo_preprocessing(inputs, resized_size,
                                      raw_size)
    if common.SEQUENCE_ID in inputs:
      step_outputs[common.SEQUENCE_ID] = inputs[common.SEQUENCE_ID]
    if self._enable_visualization or self._save_raw_predictions:
      step_outputs[_PREDICTIONS_KEY] = outputs
      step_outputs[_LABELS_KEY] = inputs
    return step_outputs

  def eval_end(self, state=None):
    """Called at the end of the evaluation.

    Args:
      state: The outputs from `eval_reduce` after the last eval step.

    Returns:
      A dictionary of `Tensors`, which will be written to logs and as
      TensorBoard summaries.
    """
    if not self._decode_groundtruth_label:
      return {}

    eval_logs = {}
    for loss_metric in self._eval_loss_metric_dict.values():
      eval_logs['losses/' + loss_metric.name] = loss_metric.result()
    eval_logs['evaluation/iou/' + self._eval_iou_metric.name] = (
        self._eval_iou_metric.result())
    if common.TASK_PANOPTIC_SEGMENTATION in self._supported_tasks:
      pq_results = self._eval_pq_metric.result()
      eval_logs['evaluation/pq/PQ'] = pq_results[0]
      eval_logs['evaluation/pq/SQ'] = pq_results[1]
      eval_logs['evaluation/pq/RQ'] = pq_results[2]
      eval_logs['evaluation/pq/TP'] = pq_results[3]
      eval_logs['evaluation/pq/FN'] = pq_results[4]
      eval_logs['evaluation/pq/FP'] = pq_results[5]

    return eval_logs

  def eval_reduce(self, state=None, step_outputs=None):
    """A function to do the reduction on the evaluation outputs per step.

    Args:
      state: A maintained state throughout the evaluation.
      step_outputs: Outputs from the current evaluation step.

    Returns:
      An output which is passed as `state` argument into `eval_reduce` function
      for the next step. After evaluation is finished, the output from last step
      will be passed into `eval_end` function.
    """
    if self._save_raw_predictions:
      sequence = None
      if self._dataset_info.is_video_dataset:
        sequence = step_outputs[_LABELS_KEY][common.SEQUENCE_ID][0][0]
      vis.store_raw_predictions(
          step_outputs[_PREDICTIONS_KEY],
          step_outputs[_LABELS_KEY][common.IMAGE_NAME][0][0],
          self._dataset_info,
          self._vis_dir,
          sequence,
          raw_panoptic_format=(
              self._config.evaluator_options.raw_panoptic_format),
          # convert_to_eval=self._config.evaluator_options.convert_raw_to_eval_ids
      )
    if not self._decode_groundtruth_label:
      # The followed operations will all require decoding groundtruth label, and
      # thus we will simply return if decode_groundtruth_label is False.
      return state

    if (self._enable_visualization and
        (self._sample_counter < self._num_vis_samples)):
      print("!!!!!!",self._sample_counter, self._num_vis_samples)
      predictions = step_outputs[_PREDICTIONS_KEY]
      inputs = step_outputs[_LABELS_KEY]
      if self._dataset_info.is_video_dataset:
        inputs[common.IMAGE] = tf.expand_dims(
            inputs[common.IMAGE][0][..., :3], axis=0)
      vis.store_predictions(predictions, inputs, self._sample_counter,
                            self._dataset_info, self._vis_dir)
      self._sample_counter += 1

    # Accumulates PQ, AP_Mask and STQ.
    if common.TASK_PANOPTIC_SEGMENTATION in self._supported_tasks:
      for gt_panoptic, pred_panoptic in zip(
          step_outputs[self._eval_pq_metric.name][0],
          step_outputs[self._eval_pq_metric.name][1]):
        batch_size = tf.shape(gt_panoptic)[0]
        for i in range(batch_size):
          self._eval_pq_metric.update_state(gt_panoptic[i], pred_panoptic[i])
    return state
