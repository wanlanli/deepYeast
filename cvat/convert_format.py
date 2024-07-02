import os
from pathlib import Path
from typing import List
import xml.dom.minidom as minidom

import re
import numpy as np

from skimage.io import imread
import zipfile
from skimage.measure import find_contours
# from .utils import file_traverse
# from absl import app
# from absl import flags
from tqdm import tqdm
import xml.etree.ElementTree as ET
from skimage import draw
import shutil
from skimage.io import imsave


class FORMAT:
    supported_formate = ["xml", "city", "training", "mask"]


IMAGE_FORMATE = [".png", ".tif", ".jpg"]
DIVISION = 1000
TYPEMAP = {
    1: '1',  # "cell",
    2: '2',  # "paired",
    3: '3',  # "zygotes",
    4: '4',  # "tetrads",
    5: '5',  # "lysis",
    6: '6',  # "single pair",
    7: '7',  # "fat",
    8: '8',  # "unknown",
}


class Convert_Format():
    def __init__(self) -> None:
        pass

    def xml2mask(self, input: str, output: str):
        xml_path_list = target_file_list(input, format="xml")
        i = 0
        for file in tqdm(xml_path_list, position=0):
            i += 1
            dump_xml2mask(file, output, series=str(i).zfill(3))

    def xml2training(self,  input: str, output: str):
        # self.xml2mask(input, output)
        # self.mask2training(output, output)
        pass

    def mask2xml(self, input: str, output: str, **kwargs):
        mask_path_list = target_file_list(input, format="image", **kwargs)
        dump_mask2xml(mask_path_list, output)

    def city2training(self, input: str, output: str):
        input_folder = target_file_list(input, format="folder")
        convert_cvat_to_trainingset(input_folder, output)

    def mask2training(self, input: str, output: str, folders=["train", "test", "val"]):
        # for folder in folders:
        #     gt_list = []
        #     mask_list = []
        #     # match_gt_mask
        #     # copy_to_structure
        pass




# FLAGS = flags.FLAGS
# flags.DEFINE_string(
#     'input',
#     default="",
#     help='The base directory where the model and training/evaluation summaries'
#     'are stored. The path will be combined with the `experiment_name` defined '
#     'in the config file to create a folder under which all files are stored.')


# flags.DEFINE_string(
#     'output',
#     default="",
#     help='Proto file which specifies the experiment configuration. The proto '
#     'definition of ExperimentOptions is specified in config.proto.')



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
    image_list = file_traverse(Path(cvat_root).joinpath(image_folder_name), file_regular=r'.*[(.png) | (.tif)]$')
    gt_list = file_traverse(Path(cvat_root).joinpath(gt_folder_name), file_regular=r'.*instanceIds.png$')

    gt_name_list = [__basename(os.path.basename(gt_name), 2) for gt_name in gt_list]
    image_name_list = [__basename(os.path.basename(image_name), 1) for image_name in image_list]

    matched_gt, matched_image = _get_matched_gt_image_list(gt_name_list,
                                                           image_name_list)

    if (len(matched_gt) > 0) & (len(matched_gt) == len(matched_image)):
        matched_gt_list = [gt_list[i] for i in matched_gt]
        matched_img_list = [image_list[i] for i in matched_image]
        _copyfile(matched_gt_list, matched_img_list, save_path, *args, **kwargs)


def __basename(filename, flag: int = 1):
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
        # extension = os.path.splitext(filename)[1]
        extension = Path(filename).suffix
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
        gt_new = __basename(os.path.basename(gt_old), 2)

        image_old = image_list[i]
        image_new = __basename(os.path.basename(image_old), 1)

        shutil.copyfile(gt_old, os.path.join(save_gt_path, gt_new))
        shutil.copyfile(image_old, os.path.join(save_image_path, image_new))


def make_training_folder_structural(save_root: str, folder: str = "instance_maps", train_flag: str = "train", series_name: str = "default"):
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
                    --image1.png
                    --image2.png
                    --image3.png
                --series2
                    --image1.png
                    --image2.png
                    --image3.png
                --series3
                    --image1.png
                    --image2.png
            --val
                --series4
                    --image1.png
                    --image2.png
            --test
                --series5
                    --image1.png
                    --image2.png
        --instance_maps
            --train
                --series1
                    --image1.png
                    --image2.png
                    --image3.png
                --series2
                    --image1.png
                    --image2.png
                    --image3.png
                --series3
                    --image1.png
                    --image2.png
            --val
                --series4
                    --image1.png
                    --image2.png
            --test
                --series5
                    --image1.png
                    --image2.png
    """
    save_path = Path(save_root).joinpath(folder, train_flag, series_name)
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


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
                    --image1.png
                    --image2.png
                    --image3.png
                --series2
                    --image1.png
                    --image2.png
                    --image3.png
                --series3
                    --image1.png
                    --image2.png
            --val
                --series4
                    --image1.png
                    --image2.png
            --test
                --series5
                    --image1.png
                    --image2.png
        --instance_maps
            --train
                --series1
                    --image1.png
                    --image2.png
                    --image3.png
                --series2
                    --image1.png
                    --image2.png
                    --image3.png
                --series3
                    --image1.png
                    --image2.png
            --val
                --series4
                    --image1.png
                    --image2.png
            --test
                --series5
                    --image1.png
                    --image2.png
    """
    save_path = os.path.join(save_root, folder,
                             train_flag, series_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


def is_image(path: str) -> str:
    extension = os.path.splitext(path)[1].lower()
    return extension in IMAGE_FORMATE


def is_xml(path: str) -> str:
    extension = os.path.splitext(path)[1].lower()
    return extension == ".xml"


def target_file_list(input: str, format="image", file_regular=None) -> list[str]:
    """
    Generate a list of file paths from a given input, which can be a zip file, directory, or single file, 
    based on the specified format.

    Parameters
    ----------
    input : str
        The path to the input file or directory.
    format : str, optional
        The format of the files to list ("image" or "xml"). Default is "image".
    file_regular : str, optional
        Regular expression to match the target files. Defaults to ".png$" for images and ".xml$" for XML files.

    Returns
    -------
    List[str]
        A list of file paths matching the specified format.
    """
    if format == "image":
        file_regular = r".*.png$" if file_regular is None else file_regular
    elif format == "xml":
        file_regular = r".*.xml$" if file_regular is None else file_regular
    if format == "folder":
        if os.path.isdir(input):
            return input
        elif input.endswith(".zip"):
            return unzip(input)
        else:
            return None
    else:
        if input.endswith(".zip"):
            folder_path = unzip(input)
            mask_path_list = file_traverse(folder_path, file_regular)
        elif os.path.isdir(input):
            folder_path = input
            mask_path_list = file_traverse(folder_path, file_regular)
        else:
            if format == "image":
                if is_image(input):
                    mask_path_list = [input]
                else:
                    mask_path_list = []
            elif format == "xml":
                if is_xml(input):
                    mask_path_list = [input]
                else:
                    mask_path_list = []
        return mask_path_list


def unzip(path: str) -> str:
    """
    Unzip a .zip file to a directory with the same name as the file (without the .zip extension).

    Parameters
    ----------
    path : str
        The path to the .zip file.

    Returns
    -------
    str
        The path to the directory where the files were extracted, or None if the file is not a .zip file or if an error occurs.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            extract_path = path[:-4]
            archive.extractall(extract_path)
        return extract_path
    except zipfile.BadZipFile as error:
        print(error)
        return None


def transform_coords_to_str(coords):
    """
    Transform an array of coordinates into a formatted string.

    Parameters
    ----------
    coords : array-like
        A 2D array where the first row contains y-coordinates and the second row contains x-coordinates.

    Returns
    -------
    str
        A string representing the coordinates in the format "x,y;x,y;...;x,y".
    """
    coords_str = ""
    for i in range(0, len(coords[0])):
        x = ('%.2f' % coords[1, i])
        y = ('%.2f' % coords[0, i])
        coords_str = coords_str + x + "," + y + ";"
    coords_str = coords_str + ('%.2f' % coords[1, 0])+","+('%.2f' % coords[0, 0])
    return coords_str


def create_polygons(mask: np.array, document: minidom.Document):
    """
    Create polygons from a mask and add them to an XML document.

    Parameters
    ----------
    mask : np.ndarray
        The mask array where each unique value represents a different label.
    document : Document
        The XML document to which the polygons will be added.

    Returns
    -------
    Document
        The XML document with added polygons.
    """
    labels = np.unique(mask)[1:]
    element = document.createElement('image')
    for i in range(0, len(labels)):
        label = labels[i]
        semantic = label//DIVISION
        new_polygon = document.createElement('polygon')
        new_polygon.setAttribute('label', TYPEMAP[semantic])
        new_polygon.setAttribute('occluded', "0")
        new_polygon.setAttribute('source', "manual")

        new_polygon.setAttribute('points', transform_coords_to_str(
            single_region_coordinate(mask == label)))
        new_polygon.setAttribute('z_order', "0")
        attribute = document.createElement('attribute')
        attribute.setAttribute('name', str(semantic))
        attribute.appendChild(document.createTextNode(str(semantic)))
        new_polygon.appendChild(attribute)
        element.appendChild(new_polygon)
    return element


def create_document_head():
    """
    Create the head of an XML document for annotations.

    Returns
    -------
    tuple
        A tuple containing the XML document and the root element.
    """
    dom = minidom.getDOMImplementation().createDocument(None, 'annotations', None)
    root = dom.documentElement
    version = dom.createElement("version")
    version.appendChild(dom.createTextNode('1.1'))
    root.appendChild(version)
    return dom, root


def dump_xml(path: str, document: minidom.Document):
    """
    Save an XML document to a specified file path.

    Parameters
    ----------
    path : str
        Path to save the XML document.
    document : Document
        The XML document to be saved.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        document.writexml(file, addindent='\t', newl='\n', encoding='utf-8')


def dump_mask2xml(mask_path_list: list[str], out_file):
    """
    Convert a list of mask images to a CVAT XML format and save it.

    Parameters
    ----------
    mask_path_list : List[str]
        List of file paths to the mask images.
    out_file : str
        Path to the output XML file.
    """
    dom, root = create_document_head()
    for f in range(0, len(mask_path_list)):
        mask = imread(mask_path_list[f])
        element = create_polygons(mask, dom)
        element.setAttribute('id', str(f))
        element.setAttribute('name', os.path.basename(mask_path_list[f]))
        element.setAttribute('width', str(mask.shape[0]))
        element.setAttribute('height', str(mask.shape[1]))
        root.appendChild(element)
    dump_xml(out_file, dom)


def single_region_coordinate(mask: np.array, number: int = 60):
    """
    Find iso-valued contours in a 2D array for a given level value (0.5).

    Parameters
    ----------
    mask : np.ndarray
        The 2D array (mask) from which to find contours.
    number : int
        The number of points to interpolate along the contour. Default is 60.

    Returns
    -------
    np.ndarray
        A 2D array where the first row contains interpolated x-coordinates and the second row contains interpolated y-coordinates.
    """
    if number <= 0:
        print("Border coordinate conversion failed. number %d <=0" % number)
    else:
        contour = find_contours(mask, level=0.5)[0]
        return resample(contour, number)
        # length = len(contour)
        # if length != 0:
        #     x = np.arange(0, length)
        #     z = np.linspace(0, length, number)
        #     cont_x = np.interp(z, x, contour[:, 0])
        #     cont_y = np.interp(z, x, contour[:, 1])
        #     return np.array([cont_x, cont_y])  # contour[index]
        # else:
        #     Warning("Border coordinate conversion failed. %d to %d" % (len(contour), number))
        #     return np.zeros(2, number)


def resample(contour, number: int) -> np.ndarray:
    """
    Resample a contour to a specified number of points.

    Parameters
    ----------
    contour : np.ndarray
        A 2D array where each row represents a point in the contour.
    number : int
        The number of points to resample to.

    Returns
    -------
    np.ndarray
        A 2D array where the first row contains the resampled x-coordinates and the second row contains the resampled y-coordinates.
    """
    length = len(contour)
    if length != 0:
        x = np.arange(0, length)
        z = np.linspace(0, length, number)
        cont_x = np.interp(z, x, contour[:, 0])
        cont_y = np.interp(z, x, contour[:, 1])
        return np.array([cont_x, cont_y])
    else:
        np.zeros(2, number)


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
    for image in tqdm(root, position=1, leave=False):
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
                mask[fill_row_coords, fill_col_coords] = label*DIVISION+order
            mask_dict[name] = mask
    return mask_dict


def dump_dict2image(mask_dict: dict, save_path: str, series: str = ""):
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
        path = Path(save_path).joinpath(series, key)
        # folder = os.path.split(path)[0]
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        imsave(path, value, check_contrast=False)


def dump_xml2mask(load_path, save_path, **kwarg):
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
    dump_dict2image(mask_dict, save_path, **kwarg)


def file_traverse(file_path, file_regular=r'.*') -> List[str]:
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
    path = Path(file_path).absolute()
    if (not path.is_dir()):
        return [path]
    else:
        path_list = []
        for path_object in path.rglob('*'):
            path_object = str(path_object)
            if (not re.match(file_regular, path_object) is None):
                path_list.append(path_object)
        path_list.sort()
        return path_list


# def main(_):
#     mask2xml(input_path=FLAGS.input, out_file=FLAGS.output)


# if __name__ == '__main__':
#     # python upload_mask_to_cvat.py --input="" --output=""
#     app.run(main)
