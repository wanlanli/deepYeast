"""This file contains code to create a post process.
"""
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes, disk, binary_erosion, dilation
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels, clear_border


def post_process_single_cell_region(data, area_threshold=200, erosion_disk=7):
    """Post processing the predicted mask.
    Remove small region -> fill holes -> erosion -> expand
    """
    # check the number of region, each label keep only one region.
    [h, n] = label(data, return_num=True, connectivity=1)
    # get the max area
    if n > 1:
        max_area = maxmum_area(h, area_threshold)
    else:
        max_area = area_threshold
    # open close operation to smooth outline

    img_rm_small = remove_small_objects(data, max_area, connectivity=1)
    img_fill_holes = remove_small_holes(img_rm_small, max_area, connectivity=2)
    footprint = disk(erosion_disk)
    img_erosion = binary_erosion(img_fill_holes, footprint)
    # img_expanded = expand_labels(img_erosion, distance=expand_disk)

    # # remove region touched border
    # if clean_border_region:
    #     img_expanded = clear_border(img_expanded)
    return img_erosion


def maxmum_area(data, vmin):
    max_area = vmin
    region = regionprops(data)
    for i in range(len(region)):
        max_area = max(region[i].area, max_area)
    return max_area


def post_process_panoptic(data, expand_disk=7, clean_border_region=False, **args):
    labels = np.unique(data)
    post_pred = np.zeros(data.shape, dtype=np.uint16)
    for label_i in labels:
        if label_i == 0:
            continue
        mask = data == label_i
        expanded = post_process_single_cell_region(mask, **args)
        post_pred[expanded] = label_i
    # img_erosion = erosion(post_pred, footprint)
    footprint = disk(expand_disk)
    img_expanded = dilation(post_pred, footprint)

    # remove region touched border
    if clean_border_region:
        img_expanded = clear_border(img_expanded)
    return img_expanded
