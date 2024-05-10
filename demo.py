import os
import sys
sys.path.append(os.path.abspath("./deeplab/"))
import yaml


from deeplab.config_yml import ExperimentOptions
from deeplab.trainer.train import DeepCellModule
from deeplab.postprocess.post_process_utils import post_process_panoptic


def load_segment_model(model_dir: str = os.path.abspath("./deepYeast//models/v1.0.0/checkpoint/"),
                       num_gpus: int = 0,
                       config_path: str = os.path.abspath("./deepYeast/deeplab/configs/config_wd.yaml")):
    """
    Loads a segmentation model from a specified directory, configuring it based on a given YAML configuration file.
    This function is specifically tailored for deep learning models, potentially supporting GPU acceleration if
    available and specified. The model and its configuration are intended for use in segmenting images, with a focus
    on biological data such as yeast cells.

    Parameters:
    ----------
    model_dir : str, optional
        The directory path where the model's checkpoint files are stored. Defaults to a relative path pointing to
        a versioned model directory.

    num_gpus : int, optional
        The number of GPUs to be used for the model. If 0, the model will run on CPU. Defaults to 0.

    config_path : str, optional
        The file path to the YAML configuration file that contains model parameters and settings. Defaults to a
        relative path pointing to a specific configuration file.

    Returns:
    ----------
    model : object
        The loaded model object, ready for performing segmentation tasks. The exact type of this object depends on
        the deep learning framework used (e.g., TensorFlow, PyTorch) and the specific model architecture.

    Raises:
    ----------
    FileNotFoundError
        If the `model_dir` does not exist or the `config_path` points to a non-existent configuration file.

    Note:
    -----
    This function assumes that the necessary deep learning libraries (e.g., TensorFlow, PyTorch) and any required
    custom modules are already installed and available in the Python environment. It also assumes that the YAML
    configuration file adheres to a structure compatible with the model being loaded.
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    configs = ExperimentOptions(config)
    model = DeepCellModule(configs, num_gpus, model_dir=model_dir)
    return model


def main() -> None:
    import numpy as np
    from skimage.io import imread
    image = imread("./example.tif")
    model = load_segment_model()
    output = model.predict(image)
    output = output["panoptic_pred"][0].numpy()
    output = post_process_panoptic(output)

    for label in np.unique(output)[1:]:
        print(label)


if __name__ == "__main__":
    main()  # pragma: no cover
