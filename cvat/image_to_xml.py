import os
import xml.dom.minidom as minidom
from skimage.io import imread
from analyser.image_measure.meaure import ImageMeasure
from analyser.utils import file_traverse


def coordinates_to_string(coords):
    coords_str = ""
    for i in range(0, len(coords[0])):
        x = ('%.2f' % coords[1, i])
        y = ('%.2f' % coords[0, i])
        coords_str = coords_str + x+","+y+";"
    coords_str = coords_str + ('%.2f' % coords[1, 0])+","+('%.2f' % coords[0, 0])
    return coords_str


def create_element(mask, dom, index, filename):
    mf = ImageMeasure(mask)
    # mf.instance_properties = mf.init_instance_properties(number=100)
    #  add new image
    element = dom.createElement('image')
    element.setAttribute('id', str(index))
    element.setAttribute('name', os.path.basename(filename))
    element.setAttribute('width', str(mf.shape[0]))
    element.setAttribute('height', str(mf.shape[1]))
    for i in range(0, mf.instance_properties.shape[0]):
        #  add new polygon
        obj = mf.instance_properties.iloc[i]
        semantic_label = int(obj.label/1000)
        new_polygon = dom.createElement('polygon')
        new_polygon.setAttribute('label', str(semantic_label))
        new_polygon.setAttribute('occluded', "0")
        new_polygon.setAttribute('source', "manual")
        new_polygon.setAttribute('points', coordinates_to_string(
            mf.instance_properties.iloc[i].coords))
        new_polygon.setAttribute('z_order', "0")
        attribute = dom.createElement('attribute')
        attribute.setAttribute('name', str(semantic_label))
        attribute.appendChild(dom.createTextNode(str(semantic_label)))
        new_polygon.appendChild(attribute)
        element.appendChild(new_polygon)
    return element


def main(input_path, out_file):
    dom = minidom.getDOMImplementation().createDocument(None, 'annotations', None)
    root = dom.documentElement
    version = dom.createElement("version")
    version.appendChild(dom.createTextNode('1.1'))
    root.appendChild(version)
    file_path = file_traverse(input_path, file_regular=".*.png$")
    for f in range(0, len(file_path)):
        mask = imread(file_path[f])
        element = create_element(mask, dom, f, file_path[f])
        # mf = ImageMeasure(mask)
        # # mf.instance_properties = mf.init_instance_properties(number=100)
        # #  add new image
        # element = dom.createElement('image')
        # element.setAttribute('id', str(f))
        # element.setAttribute('name', os.path.basename(file_path[f]))
        # element.setAttribute('width', str(mf.shape[0]))
        # element.setAttribute('height', str(mf.shape[1]))
        # for i in range(0, mf.instance_properties.shape[0]):
        #     #  add new polygon
        #     obj = mf.instance_properties.iloc[i]
        #     semantic_label = int(obj.label/1000)
        #     new_polygon = dom.createElement('polygon')
        #     new_polygon.setAttribute('label', str(semantic_label))
        #     new_polygon.setAttribute('occluded', "0")
        #     new_polygon.setAttribute('source', "manual")
        #     new_polygon.setAttribute('points', coordinates_to_string(
        #         mf.instance_properties.iloc[i].coords))
        #     new_polygon.setAttribute('z_order', "0")
        #     attribute = dom.createElement('attribute')
        #     attribute.setAttribute('name', str(semantic_label))
        #     attribute.appendChild(dom.createTextNode(str(semantic_label)))
        #     new_polygon.appendChild(attribute)
        #     element.appendChild(new_polygon)
        root.appendChild(element)
    with open(out_file, 'w', encoding='utf-8') as f:
        dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')
