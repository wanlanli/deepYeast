import os
import xml.dom.minidom as minidom
from skimage.io import imread
from analyser.image_measure.meaure import MaskFeature
from analyser.utils import file_traverse
from absl import app
from absl import flags
from common import TYPEMAP


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





def transfor_coords_to_str(coords):
    coords_str = ""
    for i in range(0, len(coords[0])):
        x = ('%.2f' % coords[1, i])
        y = ('%.2f' % coords[0, i])
        coords_str = coords_str + x+","+y+";"
    coords_str = coords_str + ('%.2f' % coords[1, 0])+","+('%.2f' % coords[0, 0])
    return coords_str


def run(input_path, out_file):
    dom = minidom.getDOMImplementation().createDocument(None, 'annotations', None)
    root = dom.documentElement
    version = dom.createElement("version")
    version.appendChild(dom.createTextNode('1.1'))
    root.appendChild(version)
    file_path = file_traverse(input_path, file_regular=".*.png$")
    for f in range(0, len(file_path)):
        mask = imread(file_path[f])
        mf = MaskFeature(mask)
        mf.instance_properties = mf.init_instance_properties(number=100)
        #  add new image
        element = dom.createElement('image')
        element.setAttribute('id', str(f))
        element.setAttribute('name', os.path.basename(file_path[f]))
        element.setAttribute('width', str(mf.shape[0]))
        element.setAttribute('height', str(mf.shape[1]))
        for i in range(0, mf.instance_properties.shape[0]):
            #  add new polygon
            obj = mf.instance_properties.iloc[i]
            new_polygon = dom.createElement('polygon')
            new_polygon.setAttribute('label', TYPEMAP[obj.semantic])
            new_polygon.setAttribute('occluded', "0")
            new_polygon.setAttribute('source', "manual")
            new_polygon.setAttribute('points', transfor_coords_to_str(
                mf.instance_properties.iloc[i].coords))
            new_polygon.setAttribute('z_order', "0")
            attribute = dom.createElement('attribute')
            attribute.setAttribute('name', str(obj.semantic))
            attribute.appendChild(dom.createTextNode(str(obj.semantic)))
            new_polygon.appendChild(attribute)
            element.appendChild(new_polygon)
        root.appendChild(element)
    with open(out_file, 'w', encoding='utf-8') as f:
        dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')


def main(_):
    run(input_path=FLAGS.input,
        out_file=FLAGS.output)


if __name__ == '__main__':
    # python upload_mask_to_cvat.py --input="" --output=""
    app.run(main)
