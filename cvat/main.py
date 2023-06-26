from absl import app
from absl import flags
from image_to_xml import main


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input',
    default="",
    help='The base directory where the model and training/evaluation summaries'
    'are stored. The path will be combined with the `experiment_name` defined '
    'in the config file to create a folder under which all files are stored.')


flags.DEFINE_string(
    'output',
    default="",
    help='Proto file which specifies the experiment configuration. The proto '
    'definition of ExperimentOptions is specified in config.proto.')


def run(_):
    main(input_path=FLAGS.input,
         out_file=FLAGS.output)


if __name__ == '__main__':
    # python upload_mask_to_cvat.py --input="" --output=""
    app.run(run)
