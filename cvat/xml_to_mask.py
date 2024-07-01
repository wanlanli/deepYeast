# Copyright 2024 wlli
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from skimage import draw
from skimage.io import imsave


def read_xml(path: str):
    """
    Read and parse an XML file.

    Parameters
    ----------
    path : str
        Path to the XML file.

    Returns
    -------
    Element
        The root element of the parsed XML tree.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    return root


def xml2mask(root: ET.Element):
    """
    Convert an XML root element to a mask dictionary.

    Parameters
    ----------
    root : Element
        The root element of the XML tree.

    Returns
    -------
    dict
        A dictionary representation of the mask.
    """
    mask_dict = {}
    for image in tqdm(root):
        if image.tag == "image":
            # print(image.tag, image.attrib)
            name = image.get("name")
            width = int(image.get("width"))
            height = int(image.get("height"))
            mask = np.zeros((width, height), dtype=np.uint16)
            order = 0
            for polygon in image:
                order += 1
                label = int(polygon.get("label"))
                points = polygon.get("points")
                point_list = np.array([point.split(",") for point in points.split(";")], dtype=np.float_)
                fill_row_coords, fill_col_coords = draw.polygon(point_list[:, 1], point_list[:, 0], mask.shape)
                mask[fill_row_coords, fill_col_coords] = label*1000+order
            mask_dict[name] = mask
    return mask_dict


def dump_dict2image(mask_dict: dict, save_path: str):
    """
    Save a mask dictionary as an image file.

    Parameters
    ----------
    mask_dict : dict
        The mask dictionary to be saved as an image.
    save_path : str
        Path to save the output image file.
    """
    for key, value in mask_dict.items():
        path = os.path.join(save_path, key)
        folder = os.path.split(path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        imsave(path, value, check_contrast=False)


def dump_xml2mask(load_path, save_path):
    """
    Read an XML file, convert it to a mask, and save the mask as an image.

    Parameters
    ----------
    load_path : str
        Path to the input XML file.
    save_path : str
        Path to save the output mask image.
    """
    root = read_xml(load_path)
    mask_dict = xml2mask(root)
    dump_dict2image(mask_dict, save_path)
