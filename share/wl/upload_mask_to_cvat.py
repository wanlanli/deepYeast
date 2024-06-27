import os
import xml.dom.minidom as minidom

import numpy as np

from skimage.io import imread
from skimage.measure import find_contours
from .utils import file_traverse
from absl import app
from absl import flags


class CVAT_Formate():
    def __init__(self) -> None:
        pass

    def xml2mask(self):
        pass

    def mask2xml(self, input: str, output):
        if input.endwith(".zip"):
            ??
            upzip
            newpath
        elif:
            is a folder??]
        else:
    
            mask_path_list = file_traverse(input)
            mask2xml(mask_path_list, output)

    def upzip(self):
        pass

    def mask2training(self):
        pass

    def xml2training(self):
        pass

    def checkzip(self):
        pass

    def tozip(self):
        pass


DIVISION = 1000


TYPEMAP = {
    1: '1',  # "cell",
    2: '2',  # "paired",
    3: '3',  # "zygotes",
    4: '4',  # "tetrads",
    5: '5',  # "lysis",
    6: '6',  # "single pair",
    7: '7',  # "fat",
    8: '8',  # "unoknow",
}


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
        print(label)
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
    with open(path, 'w', encoding='utf-8') as file:
        document.writexml(file, addindent='\t', newl='\n', encoding='utf-8')


def mask2xml(mask_path_list: list[str], out_file):
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


def main(_):
    mask2xml(input_path=FLAGS.input, out_file=FLAGS.output)


if __name__ == '__main__':
    # python upload_mask_to_cvat.py --input="" --output=""
    app.run(main)
