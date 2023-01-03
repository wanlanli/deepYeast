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
sys.path.append("../../../deepyeast/")
import os
from absl import app
from absl import flags

from utils import batch_processing

flags.DEFINE_string(
    'root_dir',
    default=None,
    help='The base directory where the images are stored')

flags.DEFINE_string(
    'model_dir',
    default=None,
    help='The base directory where the model are stored')

flags.DEFINE_string(
    'config_file',
    default=None,
    help='Proto file which specifies the experiment configuration. The proto '
    'definition of ExperimentOptions is specified in config.proto.')

flags.DEFINE_string(
    'save_path',
    default=None,
    help='The base directory where the result will be stored')

FLAGS = flags.FLAGS

def main(_):
    batch_processing(FLAGS.root_dir,
                     FLAGS.model_dir,
                     FLAGS.config_file,
                     FLAGS.save_path,)

if __name__ == '__main__':
  app.run(main)

