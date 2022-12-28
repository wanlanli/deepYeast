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
import sys
sys.path.append("/work/FAC/FBM/DMF/smartin/cellfusion/wanlan/project/deepyeast/")

import os
from absl import app
from absl import flags
import functools

import tensorflow as tf
import orbit
import yaml

from config_yml import ExperimentOptions
from data import dataset
from  trainer import distribution_utils
from project.deepyeast.model.deeplab import DeepLab
from model.loss.loss_builder import DeepLabFamilyLoss
from trainer import trainer as trainer_lib
from trainer import evaluator as evaluator_lib
from trainer import runner_utils


flags.DEFINE_enum(
    'mode',
    default=None,
    enum_values=['train', 'eval', 'train_and_eval'],
    help='Mode to run: `train`, `eval`, `train_and_eval`.')

flags.DEFINE_string(
    'model_dir',
    default=None,
    help='The base directory where the model and training/evaluation summaries'
    'are stored. The path will be combined with the `experiment_name` defined '
    'in the config file to create a folder under which all files are stored.')

flags.DEFINE_string(
    'config_file',
    default=None,
    help='Proto file which specifies the experiment configuration. The proto '
    'definition of ExperimentOptions is specified in config.proto.')

flags.DEFINE_integer(
    'num_gpus',
    default=0,
    help='The number of GPUs to use for. If `master` flag is not set, this'
    'parameter specifies whether GPUs should be used and how many of them '
    '(default: 0).')

FLAGS = flags.FLAGS


def main(_):
    with open("./configs/config_wl.yaml", 'r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    config = ExperimentOptions(config_data)

    controller = run_experiment(FLAGS.model_dir, FLAGS.num_gpus, config)
    if FLAGS.mode == 'train':
        controller.train(steps=config.trainer_options.solver_options.training_number_of_steps)
    elif FLAGS.mode == 'evl':
        controller.evaluate(steps=-1)


def run_experiment(model_dir, num_gpus, config):
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

        trainer = trainer_lib.Trainer(config, model, losses, global_step)

        evaluator = evaluator_lib.Evaluator(config, model, losses, global_step, model_dir)

    checkpoint_dict = dict(global_step=global_step)
    checkpoint_dict.update(model.checkpoint_items)
    if trainer is not None:
        checkpoint_dict['optimizer'] = trainer.optimizer
        if trainer.backbone_optimizer is not None:
            checkpoint_dict['backbone_optimizer'] = trainer.backbone_optimizer
    checkpoint = tf.train.Checkpoint(**checkpoint_dict)
    init_dict = model.checkpoint_items
    init_fn = functools.partial(runner_utils.maybe_load_checkpoint,
                                config.model_options.initial_checkpoint,
                                init_dict)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=model_dir,
        max_to_keep=1,  # configs.trainer_options.num_checkpoints_to_keep,
        step_counter=global_step,
        checkpoint_interval=config.trainer_options.save_checkpoints_steps,
        init_fn=init_fn)

    controller = orbit.Controller(
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
    return controller


if __name__ == '__main__':
  app.run(main)
