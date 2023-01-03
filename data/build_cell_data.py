"""Converts Yeast data to sharded TFRecord file format with Example protos.

"""
import math
import os

import tensorflow as tf
from absl import flags
from absl import app
from skimage.io import imread
import numpy as np

from data import data_utils


_NUM_SHARDS = 1

FLAGS = flags.FLAGS

flags.DEFINE_string('image_root', None, 'Image dataset root folder.')

flags.DEFINE_string('output_dir', None,
                    'Path to save converted TFRecord of TensorFlow examples.')

flags.DEFINE_boolean('create_panoptic_data', True,
                     'Whether to create semantic or panoptic dataset.')

flags.DEFINE_boolean('treat_crowd_as_ignore', False,
                     'Whether to apply ignore labels to crowd pixels in '
                     'panoptic label.')

_FOLDERS_MAP = {
    'image': 'images',
    'label': 'instance_maps',
}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': '',
    'label': '',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}

_PANOPTIC_LABEL_FORMAT = 'raw'


def _get_images(image_root, dataset_split):
    """Gets files for the specified data type and dataset split.

    Args:
        image_root: String, path to Cityscapes dataset root folder.
        dataset_split: String, dataset split ('train', 'val', 'test')

    Returns:
        A list of sorted file names or None when getting label for
        test set.
    """
    pattern = '*%s.%s' % (_POSTFIX_MAP['image'], _DATA_FORMAT_MAP['image'])
    search_files = os.path.join(
        image_root, _FOLDERS_MAP['image'], dataset_split, '*', pattern)
    filenames = tf.io.gfile.glob(search_files)
    return sorted(filenames)


def _split_image_path(image_path):
    """Helper method to extract split paths from input image path.

    Args:
        image_path: String, path to the image file.

    Returns:
        A tuple of (cityscape root, dataset split, cityname and shared filename
        prefix).
    """
    image_path = os.path.normpath(image_path)
    path_list = image_path.split(os.sep)
    image_folder, dataset_split, city_name, file_name = path_list[-4:]
    if image_folder != _FOLDERS_MAP['image']:
        raise ValueError('Expects image path %s containing image folder.'
                        % image_path)

    # pattern = '%s.%s' % (_POSTFIX_MAP['image'], _DATA_FORMAT_MAP['image'])
    # if not file_name.endswith(pattern):
    #     raise ValueError('Image file name %s should end with %s' %
    #                     (file_name, pattern))
    # file_prefix = file_name[:-len(pattern)]

    return os.sep.join(path_list[:-4]), dataset_split, city_name, file_name


def _generate_panoptic_label(data):
    panoptic_label = imread(data, plugin='imageio')
    return panoptic_label.astype(np.int32)


def _create_panoptic_label(cityscapes_root, dataset_split, city_name, annotation_file_name):
    """Creates labels for panoptic segmentation."""
    panoptic_annotation_file = os.path.join(cityscapes_root, _FOLDERS_MAP['label'], dataset_split, city_name, annotation_file_name)
    with tf.io.gfile.GFile(panoptic_annotation_file, 'rb') as f:
        panoptic_data = f.read()
    panoptic_label = _generate_panoptic_label(panoptic_data)
    print(np.max(panoptic_label))
    return panoptic_label.tostring(), _PANOPTIC_LABEL_FORMAT


def _convert_dataset(image_root, dataset_split, output_dir, create_panoptic_data):
    """Converts the specified dataset split to TFRecord format.

    Args:
        image_root: String, path to Cityscapes dataset root folder.
        dataset_split: String, the dataset split (one of `train`, `val` and `test`).
        output_dir: String, directory to write output TFRecords to.

    Raises:
        RuntimeError: If loaded image and label have different shape, or if the
        image file with specified postfix could not be found.
  """
    image_files = _get_images(image_root, dataset_split)

    num_images = len(image_files)
    # print(num_images)
    # expected_dataset_size = _SPLITS_TO_SIZES[_convert_split_name(dataset_split)]
    # if num_images != expected_dataset_size:
    #   raise ValueError('Expects %d images, gets %d' %
    #                    (expected_dataset_size, num_images))

    # segments_dict = None
    # if FLAGS.create_panoptic_data:
    #     segments_dict = _read_segments(FLAGS.image_root, dataset_split)

    num_per_shard = int(math.ceil(len(image_files) / _NUM_SHARDS))

    for shard_id in range(_NUM_SHARDS):
        shard_filename = '%s-%05d-of-%05d.tfrecord' % (
            dataset_split, shard_id, _NUM_SHARDS)
        output_filename = os.path.join(output_dir, shard_filename)
        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                # Read the image.
                print(image_files[i])
                with tf.io.gfile.GFile(image_files[i], 'rb') as f:
                    image_data = f.read()
                _, _, city_name, file_prefix = _split_image_path(image_files[i])
                if dataset_split == 'test':
                    label_data, label_format = None, None
                elif create_panoptic_data:
                    label_data, label_format = _create_panoptic_label(image_root, dataset_split, city_name, file_prefix)
                else:
                    label_data, label_format = None, None
                    # label_data, label_format = _create_semantic_label(image_files[i])

                # Convert to tf example.
                # _, _, _, file_prefix = _split_image_path(image_files[i])
                example = data_utils.create_tfexample(image_data,
                                                    _DATA_FORMAT_MAP['image'],
                                                    file_prefix,
                                                    label_data,
                                                    label_format)
                tfrecord_writer.write(example.SerializeToString())


def main():
    tf.io.gfile.makedirs(FLAGS.output_dir)
    for dataset_split in ('train', 'val', 'test'):
        # logging.info('Starts to processing dataset split %s.', dataset_split)
        _convert_dataset(FLAGS.image_root, dataset_split, FLAGS.output_dir, FLAGS.create_panoptic_data)


if __name__ == "__main__":
    flags.mark_flags_as_required(['image_root', 'output_dir'])
    app.run(main)
