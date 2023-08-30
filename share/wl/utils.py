import os
import shutil
from deepYeast.analyser.utils import file_traverse
from skimage.io import imread, imsave


def __rename(filename, flag: int = 1):
    """flag: 1:image, 2:gt
    """
    if (flag == 1) | (flag == 2):
        return ("_").join(filename.split("_")[:-flag])+".png"
    else:
        raise (ValueError(f"{flag}. Flag has to be 1 or 2!"))


def _get_matched_gt_image_list(gt_name_list, image_name_list):
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
            if gt_name in image_name_list:
                index_image += 1
            elif image_name in gt_name_list:
                index_gt += 1
            else:
                index_image += 1
                index_gt += 1
    return new_gt_list,  new_image_list


def _copyfile(gt_list, image_list, save_path,
              gt_folder="instance_maps", image_folder="images",
              train_flag="train", series_name="default"):
    save_gt_path = _mk_file_structral(save_path, gt_folder,
                                      train_flag, series_name)
    save_image_path = _mk_file_structral(save_path, image_folder,
                                         train_flag, series_name)
    for i in range(0, len(gt_list)):
        gt_old = gt_list[i]
        gt_new = __rename(os.path.basename(gt_old), 2)

        image_old = image_list[i]
        image_new = __rename(os.path.basename(image_old), 1)

        shutil.copyfile(gt_old, os.path.join(save_gt_path, gt_new))
        shutil.copyfile(image_old, os.path.join(save_image_path, image_new))


def _mk_file_structral(save_root: str, folder:
                       str = "instance_maps",
                       train_flag: str = "train",
                       series_name: str = "default"):
    """the file strutral of traingin datasets:
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


def conver_cvat_to_trainset(cvat_root, save_path, *args, **kwargs):
    image_list = file_traverse(os.path.join(cvat_root, "imgsFine"),
                               file_regular=r'.*.png$')
    gt_list = file_traverse(os.path.join(cvat_root, "gtFine"),
                            file_regular=r'.*instanceIds.png$')
    gt_name_list = [__rename(os.path.basename(gt_list[i]), 2)
                    for i in range(0, len(gt_list))]
    image_name_list = [__rename(os.path.basename(image_list[i]), 1)
                       for i in range(0, len(image_list))]
    matched_gt, matched_image = _get_matched_gt_image_list(gt_name_list,
                                                           image_name_list)
    if (len(matched_gt) > 0) & (len(matched_gt) == len(matched_image)):
        matched_gt_list = [gt_list[i] for i in matched_gt]
        matched_img_list = [image_list[i] for i in matched_image]
        _copyfile(matched_gt_list, matched_img_list, save_path, *args, **kwargs)


def conver_to_uint16(file):
    image = imread(file)
    if image.dtype != "uint16":
        image = image.astype("uint16")
        imsave(file, image)
