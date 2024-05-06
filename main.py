import os
import yaml


from deeplab.config_yml import ExperimentOptions
from deeplab.trainer.train import DeepCellModule


def load_segment_model(model_dir: str = os.path.abspath("./model/checkpoint/"),
                       num_gpus: int = 0,
                       config_path: str = os.path.abspath("./deeplab/configs/config_wd.yaml")):
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


# def __single_image_segment(model, image):
#     """
#     Performs segmentation on a single image using a preloaded model, processes the segmentation output,
#     and returns the segmented image as an array.

#     Parameters:
#     ----------
#     model : object
#         A segmentation model loaded and ready to predict.

#     image : np.ndarray
#         The 2d image to be segmented.

#     Returns:
#     ----------
#     np.ndarray
#         The segmented image, post-processed and converted to a uint16 array. The exact shape and content
#         of this array will depend on the model's segmentation and the subsequent post-processing.
#     """
#     output = model.predict(image)
#     post_ouput = output["panoptic_pred"][0].numpy()
#     post_ouput = post_process_panoptic(post_ouput)
#     seg = post_ouput.astype("uint16")
#     return seg


# def segment(model, image, channel=0, start: int = 0, end: int = None):
#     """
#     Segments cells from a single image or a sequence of images (movie) using a provided segmentation model.
#     The function is capable of processing either 2D single images or 3D image stacks (e.g., time-lapse movies),
#     applying the model frame by frame to segment cells. The segmentation can be applied over a specified range
#     of frames.

#     Parameters:
#     ----------
#     model : object
#         A segmentation model object.

#     image : np.ndarray
#         A 2D (Width, Height) or 3D (Frames, Width, Height) numpy array representing the image(s) to be segmented.
#         The function applies segmentation to each frame individually, iterating over the frame dimension
#         if the image is 3D.

#     channel : int, optional
#         The channel index specifying which index to iterate the segmentation process, if `image` is 3D.
#         Defaults to 0, the first channel.

#     start : int, optional
#         The starting frame index for segmentation if `image` is 3D, inclusive. Defaults to 0.

#     end : int, optional
#         The ending frame index for segmentation if `image` is 3D, exclusive. Defaults to the number of frames
#         in `image`. If `None`, segmentation is applied up to the last frame.

#     Returns:
#     ----------
#     np.ndarray
#         A 2D or 3D numpy array of the same shape as the input `image`, containing the segmentation results.
#         Each cell is typically represented by a unique label in the output array.

#     Note:
#     -----
#     The segmentation model's exact method of applying segmentation to each frame is not detailed here and
#     will depend on its implementation. Users should ensure that the `model` parameter is correctly configured
#     to work with the input `image` format and dimensions.
#     """
#     if image.ndim == 2:
#         segmentation = __single_image_segment(model, image)
#     elif image.ndim == 3:
#         if channel != 0:
#             image = np.moveaxis(image, channel, 0)
#         segmentation = np.zeros(image.shape, dtype="uint16")
#         if end is None:
#             end = image.shape[0]
#         for f in trange(start, end):
#             segmentation[f] = __single_image_segment(model, image[f])
#         if channel != 0:
#             segmentation = np.movaxis(segmentation, 0, channel)
#         return segmentation
#     else:
#         return None
