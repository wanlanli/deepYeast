from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table, find_contours

from analyser.distance import find_nearnest_points
from analyser.config import DIVISION, REGION_TABLE_VALUE, TRACE_IMAGE_PROPERTY
import analyser.common as common


class MaskFeature(np.ndarray):
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
        # self.labels = None  # self.instance_properties.label.values
        self.__cost = None

    def get_skregionprops_table(self, properties=REGION_TABLE_VALUE):
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

    def cal_coordinates(self, label_list: Sequence, **args):
        """Returns the boundary of the region according to the specified label.
        Parameters
        ----------
        label_list:
        """
        coords = []
        for label in label_list:
            coord = self.__single_region_coordinate(label, **args)
            coords.append(coord)
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
                print("Border coordinate conversion failed. %d to %d" % (len(contour), number))

    def __is_out_of_screen(self, data):
        """
        """
        bbox = data.loc[:, common.IMAGE_BOUNDING_BOX_LIST]
        shape = self.shape
        min_row = bbox.iloc[:, 0] == 0
        min_col = bbox.iloc[:, 1] == 0
        max_row = bbox.iloc[:, 2] == shape[0]
        max_col = bbox.iloc[:, 3] == shape[1]
        out_of_border = min_row | min_col | max_row | max_col
        return out_of_border

    def get_region_label_list(self):
        if self.instance_properties is None:
            labellist = np.unique(self.__array__())
            return (labellist[labellist != 0]).astype(int)
        else:
            return self.instance_properties.label.astype(int)

    def __rename_columns(self, names):
        return [name.replace("-", "_") for name in names]

    def init_instance_properties(self, **args):
        """Calculate the attribute value of each instance of the generated mask.
        """
        props = self.get_skregionprops_table()
        # add properties for data
        if common.IMAGE_COORDINATE in TRACE_IMAGE_PROPERTY:
            coords = self.cal_coordinates(props[common.IMAGE_LABEL], **args)
            props[common.IMAGE_COORDINATE] = coords
        props[common.IMAGE_SEMANTIC_LABEL] = props[common.IMAGE_LABEL] // DIVISION
        props[common.IMAGE_INSTANCE_LABEL] = props[common.IMAGE_LABEL] % DIVISION
        props[common.IMAGE_IS_BORDER] = self.__is_out_of_screen(props)
        props.index = props[common.IMAGE_LABEL]
        self.instance_properties = props
        return props

    def get_instance_mask(self, label, crop_pad=-1):
        """Returen region mask by label. Add padding for better crop image.
        """
        mask = self.__array__() == label
        if crop_pad < 0:
            return mask
        else:
            bbox = self.instance_properties.loc[self.instance_properties.label == label, ['bbox_0', 'bbox_1', 'bbox_2', 'bbox_3']].values[0]
            pad_mask = mask[bbox[0]-crop_pad:bbox[2]+crop_pad, bbox[1]-crop_pad:bbox[3]+crop_pad]
            return pad_mask

    def all_by_all_distance(self):
        if self.__cost is None:
            columns = common.DISTANCE_COLUMNS
            data = pd.DataFrame(columns=columns)
            flag = 0
            for index_x in range(0, self.instance_properties.shape[0]):
                for index_y in range(index_x+1, self.instance_properties.shape[0]):
                    center_dist, nearnest_dis, ind_x, ind_y = self.two_regions_distance(index_x, index_y)
                    data.loc[flag, columns] = [index_x, index_y, center_dist, nearnest_dis, ind_x, ind_y]
                    flag += 1
            self.__cost = data
        return self.__cost

    def two_regions_distance(self, index_x, index_y):
        """Given two regions' label, return 2 types distance between 2 regions.
        """
        coods_x = self.instance_properties.iloc[index_x][common.IMAGE_COORDINATE]
        coods_y = self.instance_properties.iloc[index_y][common.IMAGE_COORDINATE]
        center_x = self.instance_properties.iloc[index_x][common.IMAGE_CENTER_LIST]
        center_y = self.instance_properties.iloc[index_y][common.IMAGE_CENTER_LIST]
        nearnest_dis, ind_x, ind_y = find_nearnest_points(coods_x.T, coods_y.T)
        center_dist = np.sqrt(np.sum(np.square(center_x - center_y)))
        return center_dist, nearnest_dis, ind_x, ind_y

    def cost(self, source_x: Sequence = [], target_y: Sequence = []):
        """
        source_x: index list
        target_y: index list
        """
        if self.__cost is None:
            self.__cost = self.all_by_all_distance()
        cost = self.__cost.copy()
        a = cost.copy()
        a[['index_x', 'index_y', 'nearnest_point_x_index', 'nearnest_point_y_index']] = cost[['index_y', 'index_x', 'nearnest_point_y_index', 'nearnest_point_x_index']]
        cost = pd.concat([cost, a])
        if len(source_x):
            if not len(target_y):
                return cost.loc[cost.index_x.isin(source_x)]
            else:
                return cost.loc[(cost.index_x.isin(source_x)) & (cost.index_y.isin(target_y))]
        if len(target_y):
            if not len(source_x):
                return cost.loc[cost.index_y.isin(target_y)]
            else:
                return cost.loc[(cost.index_x.isin(source_x)) & (cost.index_y.isin(target_y))]
        return cost

    def nearnestN(self, x_label: int, n: int = 1):
        x_index = self.label2index(x_label)
        x_cost = self.cost(source_x=[x_index])
        y_index, t_x, t_y = x_cost.iloc[np.argsort(x_cost["nearnest_dis"])[0:n]][["index_y", 'nearnest_point_x_index','nearnest_point_y_index']].astype(int).values.T
        y_label = self.index2label(y_index)
        return y_label, t_x, t_y

    def nearnest_radius(self, x_label, radius):
        center = self.get_centers([x_label]).values[0]
        mask = self.__generate_mask(center[0], center[1], radius=radius, w=self.shape[0], h=self.shape[1])
        neighbors = np.unique(mask*self.__array__())
        neighbors = neighbors[neighbors != 0]
        return neighbors

    def __generate_mask(self, cx=50, cy=50, radius=10, w=100, h=100):
        x, y = np.ogrid[0: w, 0: h]
        mask = ((x-cx)**2 + (y-cy)**2) <= radius**2
        return mask

    def get_cells(self, labels: Sequence = []):
        if len(labels) == 0:
            return self.__array__()
        else:
            mk = np.isin(self.__array__(), labels)
            return self.__array__()*mk

    def get_centers(self, labels: Sequence = []):
        if not len(labels):
            return self.instance_properties[common.IMAGE_CENTER_LIST]
        else:
            return self.instance_properties.loc[labels, common.IMAGE_CENTER_LIST]

    def get_outline(self, labels: Sequence = []):
        if not len(labels):
            return self.instance_properties[common.IMAGE_COORDINATE]
        else:
            return self.instance_properties.loc[labels, common.IMAGE_COORDINATE]

    def index2label(self, index):
        return self.instance_properties.iloc[index].label

    def label2index(self, label: Union[int, Sequence]):
        if type(label) == int:
            return np.where(self.instance_properties.label == label)[0][0]
        else:
            data = self.instance_properties.copy()
            data['arg'] = np.arange(0, data.shape[0])
            return data.loc[label].arg

    def cost2matrix(self, source_x: Sequence = [], target_y: Sequence = []):
        cost = self.cost(source_x, target_y)
        if not len(source_x):
            source_x = np.unique(cost.index_x)
        if not len(target_y):
            target_y = np.unique(cost.index_y)
        mx = np.zeros((len(source_x), len(target_y), 2))
        source_x = list(source_x)
        target_y = list(target_y)
        for i in range(0, cost.shape[0]):
            x, y, d1, d2 = cost.iloc[i, 0:4]
            id_x = source_x.index(x)
            id_y = target_y.index(y)
            mx[id_x, id_y, :] = [d1, d2]
        return mx
