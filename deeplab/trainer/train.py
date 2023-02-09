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

"""This file contains code to run a model."""
import os
import functools
from typing import MutableMapping, Text, Any

import tensorflow as tf
import orbit

from data import dataset
from trainer import distribution_utils
from model.deeplab import DeepLab
from model.loss.loss_builder import DeepLabFamilyLoss
from model import utils
from trainer import trainer as trainer_lib
from trainer import runner_utils
from data.preprocessing import input_preprocessing


class DeepCellModule(tf.Module):
    def __init__(self, mode, config, num_gpus: int, model_dir: str) -> None:
        global_step = orbit.utils.create_global_step()
        strategy = distribution_utils.create_strategy(num_gpus)
        dataset_name = config.train_dataset_options.dataset
        datasets = dataset.MAP_NAME_TO_DATASET_INFO[dataset_name]

        with strategy.scope():
            model = DeepLab(config, datasets)

            losses = DeepLabFamilyLoss(
                        loss_options=config.trainer_options.loss_options,
                        num_classes=datasets.num_classes,
                        ignore_label=datasets.ignore_label,
                        ignore_depth=datasets.ignore_depth,
                        thing_class_ids=datasets.class_has_instances_list)
            trainer = None
            evaluator = None
            if "train" in mode:
                trainer = trainer_lib.Trainer(config, model, losses, global_step)
            if "evl" in mode:
                from trainer import evaluator as evaluator_lib
                evaluator = evaluator_lib.Evaluator(config, model, losses, global_step, model_dir)
            if (trainer is None) and (evaluator is None):
                from trainer import evaluator as evaluator_lib
                evaluator = evaluator_lib.Evaluator(config, model, losses, global_step, model_dir)

            checkpoint_dict = dict(global_step=global_step)

        checkpoint_dict.update(model.checkpoint_items)
        if trainer is not None:
            checkpoint_dict['optimizer'] = trainer.optimizer
            if trainer.backbone_optimizer is not None:
                checkpoint_dict['backbone_optimizer'] = trainer.backbone_optimizer
        checkpoint = tf.train.Checkpoint(**checkpoint_dict)
        init_dict = model.checkpoint_items
        if model_dir is None:
            model_dir = config.model_options.initial_checkpoint
        print(model_dir)
        init_fn = functools.partial(runner_utils.maybe_load_checkpoint,
                                    config.model_options.initial_checkpoint,
                                    init_dict)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=model_dir,
            max_to_keep=config.trainer_options.num_checkpoints_to_keep,
            step_counter=global_step,
            checkpoint_interval=config.trainer_options.save_checkpoints_steps,
            init_fn=init_fn)

        self.controller = orbit.Controller(
            strategy=strategy,
            trainer=trainer,
            evaluator=evaluator,
            global_step=global_step,
            steps_per_loop=config.trainer_options.steps_per_loop,
            checkpoint_manager=checkpoint_manager,
            summary_interval=config.trainer_options.save_summaries_steps,
            summary_dir=os.path.join(model_dir, 'train'),
            eval_summary_dir=os.path.join(model_dir, 'eval')
        )
        self.model = model
        self.config = config

        dataset_options = config.eval_dataset_options
        crop_height, crop_width = dataset_options.crop_size
        self._preprocess_fn = functools.partial(
            input_preprocessing.preprocess_image_and_label,
            label=None,
            crop_height=crop_height,
            crop_width=crop_width,
            prev_label=None,
            min_resize_value=dataset_options.min_resize_value,
            max_resize_value=dataset_options.max_resize_value,
            resize_factor=dataset_options.resize_factor,
            is_training=False)

    def train(self):
        self.controller.train(steps=self.config.trainer_options.solver_options.training_number_of_steps)

    def evaluta(self, steps=-1):
        self.controller.evaluate(steps=steps)

    def get_input_spec(self):
        """Returns TensorSpec of input tensor needed for inference."""
        # We expect a single 3D, uint16 tensor with shape [height, width, channels].
        self._input_depth = 1
        return tf.TensorSpec(shape=[None, None, self._input_depth], dtype=tf.float16)

    def predict(self, image):
        """Performs a forward pass.

        Args:
        input_tensor: An uint16 input tensor of type tf.Tensor with shape [height,
            width, channels].

        Returns:
        A dictionary containing the results of the specified DeepLab architecture.
        The results are bilinearly upsampled to input size before returning.
        """
        if len(image.shape) == 2:
            input_tensor = tf.cast(image.reshape(image.shape[0], image.shape[1], 1), dtype=tf.float32)
        elif len(image.shape) == 3:
            input_tensor = tf.cast(image.reshape(image.shape[0], image.shape[1], image.shape[2], 1), dtype=tf.float32)
        # output = self.model.predict(image)
        # return output

        input_size = [tf.shape(input_tensor)[0], tf.shape(input_tensor)[1]]
        (resized_image, processed_image, _, _, _, _) = self._preprocess_fn(
            image=input_tensor)

        resized_size = tf.shape(resized_image)[0:2]
        # Making input tensor to 4D to fit model input requirements.
        outputs = self.model(tf.expand_dims(processed_image, 0), training=False)
        # We only undo-preprocess for those defined in tuples in model/utils.py.
        return utils.undo_preprocessing(outputs, resized_size,
                                        input_size)

    @tf.function
    def __call__(self, input_tensor: tf.Tensor) -> MutableMapping[Text, Any]:
        """Performs a forward pass.

        Args:
        input_tensor: An uint16 input tensor of type tf.Tensor with shape [height,
            width, channels].

        Returns:
        A dictionary containing the results of the specified DeepLab architecture.
        The results are bilinearly upsampled to input size before returning.
        """
        input_size = [tf.shape(input_tensor)[0], tf.shape(input_tensor)[1]]
        (resized_image, processed_image, _, _, _, _) = self._preprocess_fn(
            image=input_tensor)

        resized_size = tf.shape(resized_image)[0:2]
        # Making input tensor to 4D to fit model input requirements.
        outputs = self.model(tf.expand_dims(processed_image, 0), training=False)
        # We only undo-preprocess for those defined in tuples in model/utils.py.
        output = utils.undo_preprocessing(outputs, resized_size,
                                        input_size)
        return output

    def save_weight(self, save_path):
        self.model.save_weights(save_path)

    def save_model(self, save_path):
        signatures = self.__call__.get_concrete_function(self.get_input_spec())
        tf.saved_model.save(self, save_path, signatures=signatures)
