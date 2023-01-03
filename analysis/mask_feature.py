import math

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table, find_contours

from analysis.distance import find_nearnest_points
from analysis.config import DIVISION, CELL_IMAGE_PROPERTY


class MaskFeature(object):
    """Measure segemented regions' properties, such as area, center, boundingbox ect. al..

    Parameters
    ----------
    mask : 2D matrix,dtype:int
        mask is a int type 2d mask array. stored the labels of segementation.
    """
    def __init__(self, mask,) -> None:
        self.mask = mask
        self.instance_properties = self._init_region_props()
        self.labels = self.instance_properties.label.values
        self.cost = None

    def _init_region_props(self, **args):
        props = regionprops_table(self.mask, properties=CELL_IMAGE_PROPERTY)
        coords = []
        for label in props['label']:
            coord = self.__coordinates(label, **args)
            coords.append(coord)
        props['coords'] = coords

        data = pd.DataFrame(props)
        data["semantic"] = data["label"] // DIVISION
        data["instance"] = data["label"] % DIVISION
        data["out_of_border"] = self.__is_out_of_screen(data)
        data.columns = self.__rename_columns(data.columns)
        return data

    def __rename_columns(self, names):
        return [name.replace("-", "_") for name in names]

    def __coordinates(self, label, number=20):
        contour = find_contours(self.mask == label, 0.5)
        contour= contour[0]
        length = len(contour)
        index = np.arange(0, length, math.floor(length/number))
        index = np.append(index, [length-1])
        return contour[index]

    def __is_out_of_screen(self, data):
        bbox = data.loc[:, ['bbox-0','bbox-1','bbox-2','bbox-3']]
        shape = self.mask.shape
        min_row = bbox.iloc[:,0] == 0
        min_col = bbox.iloc[:,1] == 0
        max_row = bbox.iloc[:,2] == shape[0]
        max_col = bbox.iloc[:,3] == shape[1]
        out_of_border = min_row | min_col | max_row | max_col
        return out_of_border

    def get_instance_mask(self, label, crop_pad=-1):
        """Returen region mask by label. Add padding for better crop image.
        """
        mask = self.mask == label
        if crop_pad < 0 :
            return mask
        else:
            bbox = self.instance_properties.loc[self.instance_properties.label == label, ['bbox_0', 'bbox_1', 'bbox_2', 'bbox_3']].values[0]
            pad_mask = mask[bbox[0]-crop_pad:bbox[2]+crop_pad, bbox[1]-crop_pad:bbox[3]+crop_pad]
            return pad_mask

    def all_by_all_distance(self):
        if self.cost is None:
            columns = ['index_x', 'index_y', 'center_dist', 'nearnest_dis', 'nearnest_point_x_index', 'nearnest_point_y_index']
            data = pd.DataFrame(columns = columns)
            flag = 0
            for index_x in range(0,self.instance_properties.shape[0]):
                for index_y in range(index_x+1, self.instance_properties.shape[0]):
                    center_dist, nearnest_dis, ind_x, ind_y = self.two_regions_distance(index_x, index_y)
                    data.loc[flag, columns] = [index_x, index_y, center_dist, nearnest_dis, ind_x, ind_y]
                    flag+=1
            self.cost = data
        return self.cost

    def two_regions_distance(self, index_x, index_y):
        """Given two regions' label, return 2 types distance between 2 regions.
        """
        coods_x = self.instance_properties.loc[index_x, 'coords']
        coods_y = self.instance_properties.loc[index_y, 'coords']
        center_x = self.instance_properties.loc[index_x, ['centroid_0', 'centroid_1']]
        center_y = self.instance_properties.loc[index_y, ['centroid_0', 'centroid_1']]
        nearnest_dis, ind_x, ind_y = find_nearnest_points(coods_x, coods_y)
        center_dist = np.sqrt(np.sum(np.square(center_x - center_y)))
        return center_dist, nearnest_dis, ind_x, ind_y
