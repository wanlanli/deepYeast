import os
import sys
sys.path.append(os.path.abspath("./deeplab/"))
import yaml
import numpy as np

from deeplab.config_yml import ExperimentOptions
from deeplab.trainer.train import DeepCellModule
from deeplab.postprocess.post_process_utils import post_process_panoptic
from skimage.measure import find_contours, approximate_polygon
from postprocess.post_process_utils import post_process_panoptic


def load_segment_model(model_dir: str = os.path.abspath("./deepYeast/models/v_1.0.0/checkpoint/"),
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


def to_contours(output):
    masks = output["panoptic_pred"][0].numpy()
    masks = post_process_panoptic(masks)
    labels = np.unique(masks)
    labels = labels[labels!=0]
    result = []
    for label in labels:
        mask = masks == label
        # contours = find_contours(mask)[0]
        # length = len(contours)
        # number = 20
        # if length != 0:
        #     x = np.arange(0, length)
        #     z = np.linspace(0, length, number)
        #     cont_x = np.around(np.interp(z, x, contours[:, 0]), decimals=2)
        #     cont_y = np.around(np.interp(z, x, contours[:, 1]), decimals=2)
        # contours = (np.array([cont_y, cont_x]).T).flatten().tolist()
        polygon = to_cvat_polygon(mask)
        if polygon is not None:
            cvat_mask = to_cvat_mask(mask)
            result.append({
                "confidence": "1",
                "label": "cell",
                "points": polygon,
                "mask": cvat_mask,
                "type": "mask",
                })
        else:
            continue
    return result


def to_cvat_mask(mask):
    x, y = np.where(mask)
    # bbox = [y.min(), x.min(), y.max(), x.max()]
    xtl, ytl, xbr, ybr = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
    flattened = ((mask*1).astype(np.int8))[xtl:xbr + 1, ytl:ybr + 1].T.flat[:].tolist()
    flattened.extend([ytl, xtl, ybr, xbr])
    return flattened


def to_cvat_polygon(mask):
    contour = find_contours(mask)[0]
    contour = np.flip(contour, axis=1)
    contour = approximate_polygon(contour, tolerance=2.5)

    if len(contour) < 3:
        return None
    else:
        return contour.ravel().tolist()


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
