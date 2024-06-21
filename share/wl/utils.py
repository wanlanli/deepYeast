import os
import shutil
import re
from typing import List
# from skimage.io import imread, imsave


def file_traverse(file_path, file_regular=r'.*', **kwarg) -> List[str]:
    """
    Traverse a directory and return a list of file paths matching a regular expression.

    Parameters
    ----------
    file_path : str
        The root directory to traverse.
    file_regular : str
        The regular expression pattern to match files. Default is '.*', which matches all files.

    Returns
    -------
    List[str]
        A list of absolute file paths that match the given regular expression.
    """
    path = os.path.abspath(file_path)
    if (not os.path.isdir(path)):
        return [path]
    else:
        path_list = []
        for root, _, files in os.walk(path, topdown=False):
            for file in files:
                abs_path = os.path.join(root, file)
                if (not re.match(file_regular, abs_path) is None):
                    path_list.append(abs_path)
        path_list.sort()
        return path_list


def __rename(filename, flag: int = 1):
    """
    Rename a file based on the specified flag.

    Parameters:
    filename (str): The name of the file to be renamed.
    flag (int): Determines how the filename should be modified.
                1 for image, 2 for ground truth (gt). Default is 1.

    Returns:
    str: The renamed file with the same extension as the original.

    Raises:
    ValueError: If the flag is not 1 or 2.
    """

    if flag in (1, 2):
        # Split the filename by underscore and remove the last 'flag' number of parts
        extension = os.path.splitext(filename)[1]
        basename = ("_").join(filename.split("_")[:-flag])
        return basename + extension
    else:
        # Raise an error if the flag is not 1 or 2
        raise (ValueError(f"{flag}. Flag has to be 1 or 2!"))


def _get_matched_gt_image_list(gt_name_list, image_name_list):
    """
    Match ground truth (gt) images with their corresponding images based on their names.

    Parameters:
    gt_name_list (list): List of ground truth image names.
    image_name_list (list): List of image names.

    Returns:
    tuple: Two lists containing indices of matching ground truth images and images.
    """
    index_image = 0
    index_gt = 0
    new_gt_list = []
    new_image_list = []
    while ((index_gt < len(gt_name_list))
           & (index_image < len(image_name_list))):
        gt_name = gt_name_list[index_gt]
        image_name = image_name_list[index_image]
        if gt_name == image_name:
            new_gt_list.append(index_gt)
            new_image_list.append(index_image)
            index_image += 1
            index_gt += 1
            continue
        else:
            if gt_name not in image_name_list:
                index_gt += 1
            if image_name not in gt_name_list:
                index_image += 1
    return new_gt_list,  new_image_list


def _copyfile(gt_list, image_list, save_path,
              gt_folder="instance_maps", image_folder="images",
              train_flag="train", series_name="default"):
    """
    Copy ground truth (gt) and image files to a structured directory.

    Parameters:
    gt_list (list): List of paths to ground truth files.
    image_list (list): List of paths to image files.
    save_path (str): Base directory to save the copied files.
    gt_folder (str): Folder name for ground truth files within save_path.
    image_folder (str): Folder name for image files within save_path.
    train_flag (str): Subdirectory name within gt_folder and image_folder.
    series_name (str): Series name for further subdirectory structure.
    """
    save_gt_path = _mk_file_structural(save_path, gt_folder,
                                       train_flag, series_name)
    save_image_path = _mk_file_structural(save_path, image_folder,
                                          train_flag, series_name)
    for i in range(0, len(gt_list)):
        gt_old = gt_list[i]
        gt_new = __rename(os.path.basename(gt_old), 2)

        image_old = image_list[i]
        image_new = __rename(os.path.basename(image_old), 1)

        shutil.copyfile(gt_old, os.path.join(save_gt_path, gt_new))
        shutil.copyfile(image_old, os.path.join(save_image_path, image_new))


def _mk_file_structural(save_root: str, folder: str = "instance_maps", train_flag: str = "train", series_name: str = "default"):
    """
    Create and return a structured file path.

    Parameters:
    base_path (str): Base directory path.
    folder_name (str): Name of the subfolder.
    train_flag (str): Subdirectory name within the subfolder.
    series_name (str): Series name for further subdirectory structure.

    Returns:
    str: The created directory path.

    the file structural of training datasets:
    save_root
        --images
            --train
                --series1
                --series2
                --series3
            --val
                --series4
            --test
                --series5
        --instance_maps
            --train
                --series1
                --series2
                --series3
            --val
                --series4
            --test
                --series5
    """
    save_path = os.path.join(save_root, folder,
                             train_flag, series_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


def convert_cvat_to_trainingset(cvat_root, save_path, *args, **kwargs):
    """
    Convert CVAT annotations to a structured training set.

    Parameters
    ----------
    cvat_root : str
        Root directory of CVAT annotations.
    save_path : str
        Directory where the converted training set will be saved.
    """
    image_folder_name = "imgsFine"
    gt_folder_name = "gtFine"

    image_list = file_traverse(os.path.join(cvat_root, image_folder_name), file_regular=r'.*[(.png) | (.tif)]$')
    gt_list = file_traverse(os.path.join(cvat_root, gt_folder_name), file_regular=r'.*instanceIds.png$')

    gt_name_list = [__rename(os.path.basename(gt_name), 2) for gt_name in gt_list]
    image_name_list = [__rename(os.path.basename(image_name), 1) for image_name in image_list]

    matched_gt, matched_image = _get_matched_gt_image_list(gt_name_list,
                                                           image_name_list)

    if (len(matched_gt) > 0) & (len(matched_gt) == len(matched_image)):
        matched_gt_list = [gt_list[i] for i in matched_gt]
        matched_img_list = [image_list[i] for i in matched_image]
        _copyfile(matched_gt_list, matched_img_list, save_path, *args, **kwargs)


# def conver_to_uint16(file):
#     image = imread(file)
#     if image.dtype != "uint16":
#         image = image.astype("uint16")
#         imsave(file, image)
