# Hungarian algorithm (Kuhn-Munkres) for solving the linear sum assignment
# problem. Taken from scikit-learn. Based on original code by Brian Clapper,
# adapted to NumPy by Gael Varoquaux.
# Further improvements by Ben Root, Vlad Niculae and Lars Buitinck.
#
# Copyright (c) 2008 Brian M. Clapper <bmc@clapper.org>, Gael Varoquaux
# Author: Brian M. Clapper, Gael Varoquaux
# License: 3-clause BSD
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import math

from .mask_feature import MaskFeature
from analyser.config import CELL_TRACKE_PROPERTY, TRACE_IMAGE_PROPERTY
from .sort import Sort, KalmanBoxTracker, action_iou_batch, behavioral_decision
from .cell import Cell
from analyser import common, config
from .cell_image import CellImage
from analyser.multi_fluorescent_image_feature import FluorescentClassification
from .distance import find_nearnest_points
from analyser.config import CELL_TRACKE_PROPERTY


class Tracer(np.ndarray):
    """Object tracking is performed according to the input mask, 
    and the dimension order of the input image is frame*width*heigh(*channel)
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
        self.frame_number = self.shape[0]
        self.obj_number = 0
        self.maskobj = {}
        self.traced_image = None
        self.obj_property, self.trace_calendar = self.__init_tracing_table()
        self.distance = None
        self.props = None
        self.coords = None
    # def __init__(self, obj):
    #     # see InfoArray.__array_finalize__ for comments
    #     if obj is None:
    #         return
    #     self.frame_number = self.shape[0]
    #     self.obj_number = 0
    #     self.maskobj = {}
    #     self.traced_image = None
    #     self.obj_property, self.trace_calendar = self.__init_tracing_table()
    #     self.distance = None
    #     self.props = None

    def __frame_name(self, number, name='frame_', length=3):
        return name+str(number).zfill(length)

    def __init_tracing_table(self):
        frame_columns_name = [self.__frame_name(x) for x in np.arange(0, self.frame_number)]
        obj_property = pd.DataFrame(columns=CELL_TRACKE_PROPERTY, dtype=np.int16)
        trace_calendar = pd.DataFrame(columns=frame_columns_name, dtype=np.int16)
        return obj_property, trace_calendar

    def mask(self, frame: Optional[int] = None):
        """Returns the mask_feature object of the frame_th image
        ----------
        Args:
        frame: int, indicates frame number
        ----------
        Returns:
        MaskFeature: Return the frame_th mask.
        """
        if frame is None:
            return self.maskobj
        else:
            if frame not in self.maskobj.keys():
                self.maskobj[frame] = MaskFeature(self.__array__()[frame])
            return self.maskobj[frame]

    def tracing(self, **args):
        """Returns the mask_feature object of the frame_th image
        ----------
        Args:
        frame: int, indicates frame number
        ----------
        Returns:
        MaskFeature: Return the frame_th mask.
        """
        mot_tracker = Sort(**args)  # create instance of the SORT tracker
        KalmanBoxTracker.count = 0
        total_frames = 0
        output_d = []
        for frame in tqdm(range(0, self.frame_number), position=0):
            img = self.mask(frame)
            #  feed the [center_x, center_y, orientation, major_axis,
            #  minor_axis, label] into tracer
            dets = img.instance_properties.iloc[:, list(range(1, 6))+[0]].values
            total_frames += 1
            trackers = mot_tracker.update(np.array(dets))
            for d in trackers:
                output_d.append([frame]+list(d))
        # output order: [frame, center_x, center_y, orientation, long_axis, 
        # short_axis, id, label]
        output_d = np.array(output_d)
        self._asgine_feature(output_d)
        return output_d

    def _asgine_feature(self, output_d: np.array):
        """Based on the output of the tracker, update the result to the pandas tabel
        ----------
        Args:
        output_d: ndarray that output from tracer. The order of the elements is [frame, center_x, center_y, orientation, long_axis, short_axis, id, label]
        ----------
        Returns:
        Update self.obj_property
        """
        for i in range(output_d.shape[0]):
            new_id = int(output_d[i, -2])
            frame = int(output_d[i, 0])
            label = int(output_d[i, -1])
            self.trace_calendar.loc[new_id, self.__frame_name(frame)] = label
            if new_id not in self.obj_property.index:
                self.obj_property.loc[new_id, common.OBJ_START] = frame
            self.obj_property.loc[new_id, common.OBJ_END] = frame
        self.obj_property[common.OBJ_LIFE_SPAN] = self.obj_property[common.OBJ_END] -\
            self.obj_property[common.OBJ_START] + 1
        self.obj_property = self.obj_property.loc[self.obj_property[common.OBJ_LIFE_SPAN] > 1]
        self.trace_calendar = self.trace_calendar.loc[self.obj_property.index]
        self.obj_number = self.obj_property.shape[0]
        self.obj_property[common.OBJ_TABEL_ARG] = np.arange(0, self.obj_number)
        self.obj_property[common.OBJ_ID] = self.obj_property.index
        self.obj_property[common.CELL_GENERATION] = 1

    def update_traced_image(self):
        """Update the mask according to the identify after the tracking is completed.
        ----------
        Returns:
        traced_image: a ndarray that same size as original mask but with updated labels.
        """
        traced_image = np.zeros(self.__array__().shape)
        for frame in range(0, self.frame_number):
            d = self.trace_calendar.loc[:, self.__frame_name(frame)].dropna()
            for k, v in d.items():
                traced_image[frame][self.__array__()[frame] == v] = k
        self.traced_image = traced_image
        return traced_image

    def identify2arg(self, identify: int):
        """Given identify return arg
        """
        return self.obj_property.loc[identify, common.OBJ_TABEL_ARG]

    def arg2identify(self, arg: int):
        """Given arg return identify
        """
        return self.obj_property.iloc[arg][common.OBJ_ID]

    def index2label(self, index, frame):
        """Given identify return image label at frame.
        """
        return int(self.trace_calendar.loc[index, self.__frame_name(frame)])

    def label2index(self, label: int, frame: int):
        """Given image label at frame return identify.
        """
        data = self.trace_calendar.loc[self.trace_calendar[
            self.__frame_name(frame)] == label]
        if len(data) > 0:
            return data.index[0]
        else:
            return None
        # return self.trace_calendar.loc[self.trace_calendar[
        #     self.__frame_name(frame)] == label].index[0]

    def labels2indexs(self, label: Sequence, frame: int):
        """Given image label at frame return identify.
        ----------
        Args:
        label:
        frame: int,
        ----------
        Returns:
        """
        data = self.trace_calendar[self.__frame_name(frame)].copy()
        data["identity"] = self.obj_property[common.OBJ_ID]
        data.set_index(data[self.__frame_name(frame)])
        for x in label:
            if x in data.index:
                data.loc[x] = None
        return data.loc[label].identity

    def center(self, index, frame):
        """Returns the coordinates of the center point of the target index at frame.
        ----------
        Args:
        ----------
        Returns:
        """
        label = self.index2label(index, frame)
        return self.maskobj[frame].get_centers([label]).values[0]

    def orientation(self, index, frame):
        label = self.index2label(index, frame)
        return self.maskobj[frame].instance_properties.loc[label, common.IMAGE_ORIENTATION]

    def contour(self, index, frame):
        """Returns the coordinates of the center point of the target index at frame.
        ----------
        Args:
        ----------
        Returns:
        """
        label = self.index2label(index, frame)
        return self.maskobj[frame].get_outline([label]).values[0]

    # def connect_generation(self):
    #     """
    #     """
    #     # props = ct.run_cell_time_props()
    #     # cells = ct.create_cells()
    #     for f in range(0, self.frame_number):
    #         end_cell = list(self.obj_property.loc[self.obj_property.end_time == f].arg)
    #         start_cell = list(self.obj_property.loc[self.obj_property.start_time == (f+1)].arg)
    #         if len(end_cell) and len(start_cell):
    #             bb_x = []
    #             for v in end_cell:
    #                 bb_x.append(self.coords[v, f].T)
    #             bb_y = []
    #             for v in start_cell:
    #                 bb_y.append(self.coords[v, f+1].T)

    #             cost = action_iou_batch(bb_x, bb_y)
    #             connection, division, fusion = behavioral_decision(cost)
    #             # print(f, connection, "\n",division, "\n",fusion)
    #             if connection:
    #                 pass
    #             if division:
    #                 for k, v in division.items():
    #                     mother = self.obj_property.iloc[end_cell[k]].identity
    #                     daughter1 = self.obj_property.iloc[start_cell[v[0]]].identity
    #                     daughter2 = self.obj_property.iloc[start_cell[v[1]]].identity
    #                     generation = self.obj_property.loc[mother, CELL_TRACKE_PROPERTY[1]]+1
    #                     # update values          
    #                     self.obj_property.loc[mother, CELL_TRACKE_PROPERTY[6:9]] = [True, daughter1, daughter2]
    #                     self.obj_property.loc[[daughter1, daughter2], CELL_TRACKE_PROPERTY[4]] = mother
    #                     self.obj_property.loc[[daughter1, daughter2], CELL_TRACKE_PROPERTY[1]] = generation
    #             if fusion:
    #                 for k, v in fusion.items():
    #                     # store index
    #                     mother = self.obj_property.iloc[end_cell[v[0]]].identity
    #                     father = self.obj_property.iloc[end_cell[v[1]]].identity
    #                     daughter = self.obj_property.iloc[start_cell[k]].identity
    #                     generation = max(self.obj_property.loc[[mother, father], CELL_TRACKE_PROPERTY[1]])+1
    #                     self.obj_property.loc[mother, CELL_TRACKE_PROPERTY[9:12]] = [True, father, daughter]
    #                     self.obj_property.loc[father, CELL_TRACKE_PROPERTY[9:12]] = [True, mother, daughter]
    #                     self.obj_property.loc[daughter, [common.CELL_GENERATION, common.CELL_MOTHER, common.CELL_FATHER]] = [generation, mother, father]

    def features_to_3dmatrix(self):
        """Convert the traced results into a 3d matrix with the size of
        number of cells x number of frame x number of attributes.
        """
        trace_image_property = TRACE_IMAGE_PROPERTY.copy()
        if common.IMAGE_COORDINATE in TRACE_IMAGE_PROPERTY:
            # If the cell coordinate outline is also in the attribute,
            # create a separate table
            trace_image_property.remove(common.IMAGE_COORDINATE)
            coords = np.zeros((self.obj_number,  # object number
                               self.frame_number,  # frame number
                               common.IMAGE_CONTOURS_LENGTH,  # conours number
                               2))  # [x, y]
        props = np.zeros((self.obj_number,
                          self.frame_number,
                          len(trace_image_property)))
        for i in range(0, self.frame_number):
            img = self.mask(i)
            data = img.instance_properties.set_index(common.IMAGE_LABEL, drop=False)
            label_id_maps = self.trace_calendar.iloc[:, i].dropna()
            arg_id = self.obj_property.loc[label_id_maps.index, common.CELL_TABEL_ARG].values
            props[arg_id, i, :] = data.loc[label_id_maps.values, trace_image_property]
            aaa = np.array(list(data.loc[label_id_maps.values, common.IMAGE_COORDINATE]))
            coords[arg_id, i, :, :] = np.moveaxis(aaa, [0, 1, 2], [0, 2, 1])
        self.props = props
        self.coords = coords
        return props, coords

    def coords_3d(self):
        """Convert the traced results into a 3d matrix with the size of
        number of cells x number of frame x number of attributes.
        """
        if common.IMAGE_COORDINATE in TRACE_IMAGE_PROPERTY:
            coords = np.zeros((self.obj_number,  # object number
                               self.frame_number,  # frame number
                               common.IMAGE_CONTOURS_LENGTH,  # conours number
                               2))  # [x, y]
        for i in range(0, self.frame_number):
            img = self.mask(i)
            data = img.instance_properties.set_index(common.IMAGE_LABEL, drop=False)
            label_id_maps = self.trace_calendar.iloc[:, i].dropna()
            arg_id = self.obj_property.loc[label_id_maps.index, common.CELL_TABEL_ARG].values
            aaa = np.array(list(data.loc[label_id_maps.values, common.IMAGE_COORDINATE]))
            coords[arg_id, i, :, :] = np.moveaxis(aaa, [0, 1, 2], [0, 2, 1])
        self.coords = coords
        return coords

    # def create_single_cell_by_id(self, cell_index):
    #     trace_feature = self.obj_property.loc[cell_index].values
    #     start_time, end_time = self.obj_property.loc[cell_index, [2, 3]].astype(int)
    #     arg = int(self.obj_property.loc[cell_index].arg)
    #     prop = self.props[arg, start_time:end_time+1, :]
    #     coord = self.coords[arg, start_time:end_time+1, :, :]
    #     return Cell(trace_feature, prop, coord)

    def distance_3d(self, **args):
        """Convert the traced cells distance into a 3d matrix with the size of
        number of frame x number of objects x number of objects x  4[center dist, nearnest dist, nearnest point x, nearnest point y].
        """
        distance = np.zeros((self.frame_number,
                             self.obj_number,
                             self.obj_number,
                             4))  # [center dist, nearnest dist, nearnest point x, nearnest point y]
        for i in trange(0, self.frame_number):
            labels = self.trace_calendar.iloc[:, i].dropna()
            cell_idx = self.obj_property.loc[labels.index].arg
            mk = self.maskobj[i]
            index = mk.label2index(labels.values)
            # map index to cell_id
            cost = mk.cost(index, index, **args)
            map = pd.DataFrame(np.array([labels, cell_idx, index]).T,
                               columns=["label", "cell_idx", "index"],
                               index=index)
            distance[i,
                     list(map.loc[list(cost.index_x.astype(int))].cell_idx.astype(int)),
                     list(map.loc[list(cost.index_y.astype(int))].cell_idx.astype(int)),
                     :] = cost[config.OBJ_DISTANCE_COLUMNS]
        self.distance = distance
        return distance

    def prediction(self, image: np.array, n_components: int):
        """Calculate the cell type frame by frame, and return the maximum value.
        ----------
        Args:
        image: the orginal image inluce fluorescent information.
        n_components: the number of categories.
        ----------
        Returns:
        prediction: The clustering results.
        """
        prediction = self.trace_calendar.copy()
        prediction = prediction.fillna(-1)
        if image.shape[0] != self.__array__().shape[0]:
            return None
        for f in trange(0, image.shape[0]):
            # prediction cell type over frame.
            cellimage = CellImage(image[f], mask=self.maskobj[f])
            data = cellimage.get_fluorescent_intensity()
            cluster = FluorescentClassification(data)
            _, _ = cluster.predition_data_type(
                n_components=n_components,
                init_params='kmeans')
            cluster.data.loc[-1, "channel_prediction"] = -1
            prediction[self.__frame_name(f)] = cluster.data.loc[prediction[self.__frame_name(f)].values.astype(int)].channel_prediction.values
        self.obj_property["channel_prediction"] = prediction.replace(-1, None).mode(axis=1)[0].values
        return prediction

    def relations2objs(self, x_index: int, y_index: int, frame: int):
        """Calculate the distance between the specified two cells in space, time, and angle.
        ----------
        Args:
        x_index: int, identity of cell 1.
        y_index: int, identity of cell 2.
        frame: int, frame number.
        ----------
        Returns:
        center_dist: float, center distance.
        near_dist: float, nearest distance.
        angel_x: float, the angle of cell 1' nearest point to it's center.
        angel_y: float, the angle of cell 2' nearest point to it's center.
        time_gap: float, the time gap since last division.
        """
        center_dist, near_dist, nearnest_point_x_index, nearnest_point_y_index\
            = self.distance[frame, self.identify2arg(x_index),
                            self.identify2arg(y_index)]
        center_x = self.center(x_index, frame)
        near_x = self.contour(x_index, frame)[:, int(nearnest_point_x_index)]
        center_y = self.center(y_index, frame)
        near_y = self.contour(y_index, frame)[:, int(nearnest_point_y_index)]
        orientation_x = self.orientation(x_index, frame)
        orientation_y = self.orientation(y_index, frame)
        angel_x = self.angle_to_the_major_axis(center_x[0],
                                               center_x[1],
                                               near_x[0],
                                               near_x[1],
                                               orientation_x)
        # print(center_y, near_y, orientation_y)
        angel_y = self.angle_to_the_major_axis(center_y[0],
                                               center_y[1],
                                               near_y[0],
                                               near_y[1],
                                               orientation_y)
        time_gap = self.obj_property.loc[x_index, "start_time"] -\
            self.obj_property.loc[y_index, "start_time"]
        return center_dist, near_dist, angel_x, angel_y, time_gap

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

    # def fusion_cell_features(self):
    #     fusioned_cells = self.obj_property.loc[(~self.obj_property.mother.isna()) & (~self.obj_property.father.isna())].copy()
    #     fusioned_parents = None
    #     for cell in fusioned_cells.index:
    #         print(cell)
    #         mother_index, father_index, frame = fusioned_cells.loc[cell, ['mother', 'father', 'start_time']].astype(np.int16)
    #         # mother_id = self.cells[mother_index].indentify
    #         # father_id = self.cells[father_index].indentify
    #         frame = frame-1

    #         # exchange m & f
    #         if (self.obj_property.loc[mother_index].channel_prediction == self.obj_property.loc[father_index].channel_prediction):
    #             print("error")
    #         elif self.obj_property.loc[mother_index].channel_prediction > self.obj_property.loc[father_index].channel_prediction:
    #             c = mother_index
    #             mother_index = father_index
    #             father_index = c

    #         # assgin son' features
    #         center_distance, n_distance, angle_0, angle_1, timegap = self.relations2objs(mother_index, father_index, frame)
    #         fusioned_cells.loc[cell, 'center_distance'] = center_distance
    #         fusioned_cells.loc[cell, 'nearnest_distance'] = n_distance
    #         # fusioned_cells.loc[cell, 'start_nearnest_distance'] = start_distance
    #         fusioned_cells.loc[cell, 'angle_x'] = angle_0
    #         fusioned_cells.loc[cell, 'angle_y'] = angle_1
    #         fusioned_cells.loc[cell, 'timegap'] = timegap

    #         # from mothers perspective:
    #         frame_x = int(max(self.obj_property.loc[[mother_index, father_index]].start_time))
    #         mf_cf = self.neighbor_objects_freatures(mother_index, father_index, frame_x)
    #         mf_cf.loc[:, 'fusion_type'] = 'm'
    #         if fusioned_parents is None:
    #             fusioned_parents = mf_cf
    #         else:
    #             fusioned_parents = pd.concat([fusioned_parents, mf_cf])
    #         # from fathers perspective:
    #         ff_cf = self.neighbor_objects_freatures(father_index, mother_index, frame_x)
    #         ff_cf.loc[:, 'fusion_type'] = 'f'
    #         if fusioned_parents is None:
    #             fusioned_parents = ff_cf
    #         else:
    #             fusioned_parents = pd.concat([fusioned_parents, ff_cf])
    #     return fusioned_cells, fusioned_parents

    def neighbor_objects_freatures(self, x_index, y_index, frame, radius=200):
        """
        x_index: identity
        y_index: identity
        """
        nc = list(self.neighbor_objects(x_index, radius=radius))
        if y_index not in nc:
            nc += [y_index]
        fusion_parent_cells = self.obj_property.loc[nc].copy()
        for y in nc:
            if self.obj_property.loc[y].end_time < frame:
                continue
            center_distance, n_distance, angle_x, angle_y, timegap = self.relations2objs(x_index, y, frame)
            fusion_parent_cells.loc[y, [common.CENTER_DISTANCE,
                                        common.NEARNEST_DISTANCE,
                                        "angle_x",
                                        "angle_y",
                                        "timegap"]] = [center_distance, n_distance, angle_x, angle_y, timegap]
        fusion_parent_cells.loc[:, 'flag'] = False
        fusion_parent_cells.loc[y_index, 'flag'] = True
        fusion_parent_cells.loc[:, 'ref'] = x_index
        return fusion_parent_cells

    def neighbor_objects(self, index: int, frame: Optional[int] = None, radius: float = 100):
        """Calculate neighbor regions within a radius of a given object.
        ----------
        Args:
        index: int, identity of source object.
        frame: int, frame number, if none, returen the first frame of source object.
        radius: float, radius
        ----------
        Returns:
        y_index: the list of neighbor regions.
        """
        if frame is None:
            frame = int(self.obj_property.loc[index].start_time)
        x_label = self.index2label(index, frame)
        y_labels = self.maskobj[frame].nearnest_radius(x_label, radius)
        y_index = [self.label2index(y, frame) for y in y_labels]
        return list(filter(None, y_index))
        # return y_index
