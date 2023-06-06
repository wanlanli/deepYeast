from typing import Optional, Sequence, Union, List
from collections import Iterable


import numpy as np
import pandas as pd
import math
from skimage.measure import regionprops_table, find_contours

from analyser.image_measure.distance import find_nearnest_points
from analyser.config import DIVISION, REGION_TABLE_VALUE, TRACE_IMAGE_PROPERTY
import analyser.common as common


class ImageMeasure(np.ndarray):
    """Extract segemented regions' information from mask, åsuch as area, center, boundingbox ect. al..
    Parameters
    ----------
    input_array : 2D matrix,dtype:int
        mask is a int type 2d mask array. stored the labels of segementation.
    """
    def __new__(cls, mask: np.ndarray):
        # Input array is an already formed ndarray instance first cast to be our class type
        obj = np.asarray(mask).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.instance_properties = self.init_instance_properties()
        # self.labels = self.instance_properties.label.values
        self.__cost = self.init_cost_matrix()

    def init_cost_matrix(self):
        length = self.instance_properties.shape[0]
        cost = np.zeros((length, length, 4))
        cost[:,:,:] = -1
        return cost

    def skregionprops_to_table(self, properties=REGION_TABLE_VALUE):
        """Return the values ​​of properties in the image as a table
        Parameters
        ----------
        properties : 2D matrix,dtype:int
        Notes
        ----------
        The following properties can be accessed as attributes or keys:
        areaint: Number of pixels of the region.
        area_bboxint: Number of pixels of bounding box.
        area_convexint: Number of pixels of convex hull image, which is the smallest convex polygon that encloses the region.
        area_filledint: Number of pixels of the region will all the holes filled in. Describes the area of the image_filled.
        axis_major_lengthfloat: The length of the major axis of the ellipse that has the same normalized second central moments as the region.
        axis_minor_lengthfloat: The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
        bboxtuple: Bounding box (min_row, min_col, max_row, max_col). Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col).
        centroidarray: Centroid coordinate tuple (row, col).
        centroid_localarray: Centroid coordinate tuple (row, col), relative to region bounding box.
        centroid_weightedarray: Centroid coordinate tuple (row, col) weighted with intensity image.
        centroid_weighted_localarray: Centroid coordinate tuple (row, col), relative to region bounding box, weighted with intensity image.
        coords(N, 2) ndarray: Coordinate list (row, col) of the region.
        eccentricityfloat: Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
        equivalent_diameter_areafloat: The diameter of a circle with the same area as the region.
        euler_numberint: Euler characteristic of the set of non-zero pixels. Computed as number of connected components subtracted by number of holes (input.ndim connectivity). In 3D, number of connected components plus number of holes subtracted by number of tunnels.
        extentfloat: Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
        feret_diameter_maxfloat
        Maximum Feret’s diameter computed as the longest distance between points around a region’s convex hull contour as determined by find_contours. [5]
        image(H, J) ndarray: Sliced binary region image which has the same size as bounding box.
        image_convex(H, J) ndarray: Binary convex hull image which has the same size as bounding box.
        image_filled(H, J) ndarray: Binary region image with filled holes which has the same size as bounding box.
        image_intensityndarray: Image inside region bounding box.
        inertia_tensorndarray: Inertia tensor of the region for the rotation around its mass.
        inertia_tensor_eigvalstuple: The eigenvalues of the inertia tensor in decreasing order.
        intensity_maxfloat: Value with the greatest intensity in the region.
        intensity_meanfloat: Value with the mean intensity in the region.
        intensity_minfloat: Value with the least intensity in the region.
        labelint: The label in the labeled input image.
        moments(3, 3) ndarray: Spatial moments up to 3rd order:
            m_ij = sum{ array(row, col) * row^i * col^j }
            Copy to clipboard
            where the sum is over the row, col coordinates of the region.
        moments_central(3, 3) ndarray: Central moments (translation invariant) up to 3rd order:
            mu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }
            Copy to clipboard
            where the sum is over the row, col coordinates of the region, and row_c and col_c are the coordinates of the region’s centroid.     
        moments_hutuple: Hu moments (translation, scale and rotation invariant).
        moments_normalized(3, 3) ndarray: Normalized moments (translation and scale invariant) up to 3rd order:
            nu_ij = mu_ij / m_00^[(i+j)/2 + 1]
            Copy to clipboard
            where m_00 is the zeroth spatial moment.
        moments_weighted(3, 3) ndarray: Spatial moments of intensity image up to 3rd order:
            wm_ij = sum{ array(row, col) * row^i * col^j }
            Copy to clipboard
            where the sum is over the row, col coordinates of the region.
        moments_weighted_central(3, 3) ndarray: Central moments (translation invariant) of intensity image up to 3rd order:
            wmu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }
            Copy to clipboard
            where the sum is over the row, col coordinates of the region, and row_c and col_c are the coordinates of the region’s weighted centroid.
        moments_weighted_hutuple: Hu moments (translation, scale and rotation invariant) of intensity image.
        moments_weighted_normalized(3, 3) ndarray: Normalized moments (translation and scale invariant) of intensity image up to 3rd order:
            wnu_ij = wmu_ij / wm_00^[(i+j)/2 + 1]
            Copy to clipboard
            where wm_00 is the zeroth spatial moment (intensity-weighted area).
        orientationfloat: Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.
        perimeterfloat: Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
        perimeter_croftonfloat: Perimeter of object approximated by the Crofton formula in 4 directions.
        slicetuple of slices: A slice to extract the object from the source image.
        solidityfloat: Ratio of pixels in the region to pixels of the convex hull image.
        """
        regionprops = regionprops_table(self.__array__(), properties=properties)
        regionprops = pd.DataFrame(regionprops)
        regionprops.columns = self.__rename_columns(regionprops.columns)
        return regionprops

    def __index(self,
                index: Union[int, Sequence] = None,
                label: Union[int, Sequence] = None):
        """Inner function that retrieve rows by index or label arbitrarily.
        Note: Two parameters can and can only specify one of them.
        """
        if index is not None:
            if label is None:
                return index
            else:
                Warning("`index` and `label` cannot be specified at the same time, the calculation is based on `index` only")
                return index
        else:
            if label is None:
                Warning("`index` and `label` cannot be None at the same time")
                return None
            else:
                return self.label2index(label)

    def __label(self,
                label: Union[int, Sequence] = None,
                index: Union[int, Sequence] = None,):
        """Inner function that retrieve rows by index or label arbitrarily.
        Note: Two parameters can and can only specify one of them.
        """
        if label is not None:
            if index is None:
                return label
            else:
                Warning("`index` and `label` cannot be specified at the same time, the calculation is based on `index` only")
                return label
        else:
            if index is None:
                Warning("`index` and `label` cannot be None at the same time")
                return None
            else:
                return self.index2label(index)

    def cal_coordinates(self, label: Union[int, Sequence], **args):
        """Returns the boundary of the region according to the specified label.
        Parameters
        ----------
        label_list:
        """
        coords = []
        if type(label) == int:
            return self.__single_region_coordinate(label, **args)
        elif isinstance(label, Iterable):
            for label_i in label:
                coord = self.__single_region_coordinate(label_i, **args)
                coords.append(coord)
            return coords
        else:
            return coords

    def __single_region_coordinate(self, label: int, number: int = common.IMAGE_CONTOURS_LENGTH):
        """Find iso-valued contours in a 2D array for a given level value(0.5).
        """
        if number <= 0:
            print("Border coordinate conversion failed. number %d <=0" % number)
        else:
            contour = find_contours(self.__array__() == label, level=0.5)[0]
            length = len(contour)
            if length != 0:
                x = np.arange(0, length)
                z = np.linspace(0, length, number)
                cont_x = np.interp(z, x, contour[:, 0])
                cont_y = np.interp(z, x, contour[:, 1])
                return np.array([cont_x, cont_y])  # contour[index]
            else:
                Warning("Border coordinate conversion failed. %d to %d" % (len(contour), number))
                return np.zeros(2, number)

    # def __is_out_of_screen(self, data):
    #     """
    #     """
    #     bbox = data.loc[:, common.IMAGE_BOUNDING_BOX_LIST]
    #     shape = self.shape
    #     min_row = bbox.iloc[:, 0] == 0
    #     min_col = bbox.iloc[:, 1] == 0
    #     max_row = bbox.iloc[:, 2] == shape[0]
    #     max_col = bbox.iloc[:, 3] == shape[1]
    #     out_of_border = min_row | min_col | max_row | max_col
    #    return out_of_border

    def label_list(self):
        if self.instance_properties is None:
            labellist = np.unique(self.__array__())
            return (labellist[labellist != 0]).astype(int)
        else:
            return self.instance_properties.label.astype(int)

    def __rename_columns(self, names):
        return [name.replace("-", "_") for name in names]

    def init_instance_properties(self, **args):
        """Calculate the attribute value of each instance of the generated mask.
        index: int, the order, from 0 to len(instances)
        label: int, the identify, equal with image values
        """
        props = self.skregionprops_to_table()
        # add properties for data
        if common.IMAGE_COORDINATE in TRACE_IMAGE_PROPERTY:
            coords = self.cal_coordinates(props[common.IMAGE_LABEL], **args)
            props[common.IMAGE_COORDINATE] = coords
        props.index = np.arange(0, props.shape[0])
        # props[common.OBJ_TABEL_ARG] = np.arange(0, props.shape[0])
        self.instance_properties = props
        return props

    def instance_mask(self, index=None, label=None, crop_pad=-1):
        """Returen region mask by label. Add padding for better crop image.
        """
        index_t = self.__index(index=index, label=label)
        print(index_t)
        if isinstance(index_t, Iterable):
            mask_all = np.zeros(self.shape)
            for i in index_t:
                mask = self.__array__() == self.instance_properties.loc[i].label
                if crop_pad < 0:
                    mask_all = mask_all + mask
                else:
                    bbox = self.instance_properties.loc[
                        i, common.IMAGE_BOUNDING_BOX_LIST].values[0]
                    pad_mask = mask[bbox[0]-crop_pad:bbox[2]+crop_pad,
                                    bbox[1]-crop_pad:bbox[3]+crop_pad]
                    mask_all = mask_all + pad_mask
            return mask_all
        else:
            mask = self.__array__() == self.instance_properties.loc[index_t].label
            if crop_pad < 0:
                return mask
            else:
                bbox = self.instance_properties.loc[
                    index_t, common.IMAGE_BOUNDING_BOX_LIST].values[0]
                pad_mask = mask[bbox[0]-crop_pad:bbox[2]+crop_pad,
                                bbox[1]-crop_pad:bbox[3]+crop_pad]
                return pad_mask

    # def all_by_all_distance(self, radius=200):
    #     """根据index索引
    #     """
    #     if self.__cost is None:
    #         columns = common.DISTANCE_COLUMNS
    #         data = pd.DataFrame(columns=columns)
    #         flag = 0
    #         for index_x in range(0, self.instance_properties.shape[0]):
    #             neibor_labels = self.nearnest_radius(self.index2label(index_x),
    #                                                  radius=radius)
    #             neibor_index = self.label2index(neibor_labels).values
    #             for index_y in range(index_x+1, self.instance_properties.shape[0]):
    #                 if index_y not in neibor_index:
    #                     center_dist, nearnest_dis, ind_x, ind_y = [-1, -1, -1, -1]
    #                     data.loc[flag, columns] = [index_x, index_y, center_dist, nearnest_dis, ind_x, ind_y]
    #                 else:
    #                     center_dist, nearnest_dis, ind_x, ind_y = self.two_regions_distance(index_x, index_y)
    #                     data.loc[flag, columns] = [index_x, index_y, center_dist, nearnest_dis, ind_x, ind_y]
    #                 flag += 1
    #         self.__cost = data
    #     return self.__cost

    def __distance_exist(self, x, y) -> bool:
        if self.__cost is not None:
            if self.__cost[x, y, 0] > 0:
                return True
            else:
                return False
        else:
            return False

    def two_regions_distance(self, target: int, source: int):
        """Given two regions' label, return 2 types distance between 2 regions.
        Parameters
        ----------
        target :int, index of target point
        source :int, index of source point
        Notes
        ----------
        """
        data = self.instance_properties.loc[[target, source]][
            common.IMAGE_CENTER_LIST+[common.IMAGE_COORDINATE]]
        nearnest_dis, ind_tgt, ind_src = find_nearnest_points(
            data.iloc[0, 2].T, data.iloc[1, 2].T)
        center_dist = np.sqrt(
            np.square(data.iloc[0, 0:2] - data.iloc[1, 0:2]).sum())
        return center_dist, nearnest_dis, ind_tgt, ind_src

    def ditance_matrix(self, sources: list, targets: list):
        """Given two regions' label, return 2 types distance between 2 regions.
        """
        if self.__cost is None:
            self.__cost = self.init_cost_matrix()
        for index_x in sources:
            for index_y in targets:
                if self.__distance_exist(index_x, index_y):
                    continue
                else:
                    center_dist, nearnest_dis, ind_x, ind_y = self.two_regions_distance(index_x, index_y)
                    self.__cost[index_x, index_y, :] = [center_dist, nearnest_dis, ind_x, ind_y]
                    self.__cost[index_y, index_x, :] = [center_dist, nearnest_dis, ind_y, ind_x]

        data = self.__cost[sources]
        data = data[:, targets]
        return data

    def nearneast_point(self, target, source):
        index = self.ditance_matrix([target], [source])[0, 0, 2:].astype(np.int_)
        points = list(self.instance_properties.iloc[[target, source]][common.IMAGE_COORDINATE])
        return points[0][:, index[0]], points[1][:, index[1]]

    def two_regions_angle(self, target, source):
        data = self.instance_properties.iloc[target][common.IMAGE_CENTER_LIST+[common.IMAGE_ORIENTATION]]
        p_t, p_s = self.nearneast_point(target, source)
        angel_x = self.angle_to_the_major_axis(data[0],
                                               data[1],
                                               p_t[0],
                                               p_t[1],
                                               data[2])

        data = self.instance_properties.iloc[source][common.IMAGE_CENTER_LIST+[common.IMAGE_ORIENTATION]]
        angel_y = self.angle_to_the_major_axis(data[0],
                                               data[1],
                                               p_s[0],
                                               p_s[1],
                                               data[2])
        return angel_x, angel_y

    def angle_to_the_major_axis(self, x1, y1, x2, y2, angle2):
        """
        x1: center x
        y1: center y
        x2: target x
        y2: target y
        angle2: orientation
        """
        angle1 = math.atan2(y2-y1, x2-x1)
        included_angle = angle1-angle2
        included_angle = included_angle - np.pi*2*math.floor(included_angle/(2 * np.pi))
        if abs(included_angle) > np.pi:
            included_angle = included_angle-np.pi*2
        return included_angle

    # def cost(self, source_x: Sequence = [], target_y: Sequence = [], **args):
    #     """
    #     source_x: index list
    #     target_y: index list
    #     """
    #     if self.__cost is None:
    #         self.__cost = self.all_by_all_distance(**args)
    #     cost = self.__cost.copy()
    #     a = cost.copy()
    #     a[['index_x', 'index_y', 'nearnest_point_x_index', 'nearnest_point_y_index']] = cost[['index_y', 'index_x', 'nearnest_point_y_index', 'nearnest_point_x_index']]
    #     cost = pd.concat([cost, a])
    #     if len(source_x):
    #         if not len(target_y):
    #             return cost.loc[cost.index_x.isin(source_x)]
    #         else:
    #             return cost.loc[(cost.index_x.isin(source_x)) & (cost.index_y.isin(target_y))]
    #     if len(target_y):
    #         if not len(source_x):
    #             return cost.loc[cost.index_y.isin(target_y)]
    #         else:
    #             return cost.loc[(cost.index_x.isin(source_x)) & (cost.index_y.isin(target_y))]
    #     return cost

    # def set_cost(self, cost):
    #     self.__cost = cost

    # def nearnestN(self, x_label: int, n: int = 1):
    #     x_index = self.label2index(x_label)
    #     x_cost = self.cost(source_x=[x_index])
    #     x_cost = x_cost.loc[x_cost[common.CENTER_DISTANCE] > 0]
    #     y_index, t_x, t_y = x_cost.iloc[
    #         np.argsort(x_cost["nearnest_dis"])[0:n]
    #         ][["index_y", 'nearnest_point_x_index', 'nearnest_point_y_index'
    #            ]].astype(int).values.T
    #     y_label = self.index2label(y_index)
    #     return y_label, t_x, t_y

    # def nearnest_radius(self, x_label, radius):
    #     center = self.get_centers([x_label]).values[0]
    #     mask = self.__generate_mask(center[0], center[1], radius=radius, w=self.shape[0], h=self.shape[1])
    #     neighbors = np.unique(mask*self.__array__())
    #     neighbors = neighbors[neighbors != 0]
    #     return neighbors

    # def __generate_mask(self, cx=50, cy=50, radius=10, w=100, h=100):
    #     x, y = np.ogrid[0: w, 0: h]
    #     mask = ((x-cx)**2 + (y-cy)**2) <= radius**2
    #     return mask

    # def get_cells(self, labels: Sequence = []):
    #     if len(labels) == 0:
    #         return self.__array__()
    #     else:
    #         mk = np.isin(self.__array__(), labels)
    #         return self.__array__()*mk

    def centers(self, index=None, labels=None):
        index = self.__index(index, labels)
        return self.instance_properties.iloc[index][common.IMAGE_CENTER_LIST]

    # def get_orientation(self, labels: Sequence = []):
    #     if not len(labels):
    #         return self.instance_properties[common.IMAGE_ORIENTATION]
    #     else:
    #         return self.instance_properties.loc[labels, common.IMAGE_ORIENTATION]

    # def get_major_length(self, labels: Sequence = []):
    #     if not len(labels):
    #         return self.instance_properties[common.IMAGE_MAJOR_AXIS]
    #     else:
    #         return self.instance_properties.loc[labels, common.IMAGE_MAJOR_AXIS]

    # def get_outline(self, labels: Sequence = []):
    #     if not len(labels):
    #         return self.instance_properties[common.IMAGE_COORDINATE]
    #     else:
    #         return self.instance_properties.loc[labels, common.IMAGE_COORDINATE]

    def index2label(self, index):
        return self.instance_properties.iloc[index].label

    def label2index(self, label: Union[int, Sequence]):
        """image label to arg
        """
        if type(label) == int:
            return self.instance_properties.loc[
                self.instance_properties.label == label].index[0]
        else:
            return list(self.instance_properties.loc[
                self.instance_properties.label.isin(label)].index)

    # def cost2matrix(self, source_x: Sequence = [], target_y: Sequence = []):
    #     cost = self.cost(source_x, target_y)
    #     if not len(source_x):
    #         source_x = np.unique(cost.index_x)
    #     if not len(target_y):
    #         target_y = np.unique(cost.index_y)
    #     mx = np.zeros((len(source_x), len(target_y), 2))
    #     source_x = list(source_x)
    #     target_y = list(target_y)
    #     for i in range(0, cost.shape[0]):
    #         x, y, d1, d2 = cost.iloc[i, 0:4]
    #         id_x = source_x.index(x)
    #         id_y = target_y.index(y)
    #         mx[id_x, id_y, :] = [d1, d2]
    #     return mx
    
    def instance_property(self, index=None, label=None):
        index_rt = self.__index(index=index, label=label)
        return self.instance_properties.loc[index_rt]
