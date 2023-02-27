# Hungarian algorithm (Kuhn-Munkres) for solving the linear sum assignment
# problem. Taken from scikit-learn. Based on original code by Brian Clapper,
# adapted to NumPy by Gael Varoquaux.
# Further improvements by Ben Root, Vlad Niculae and Lars Buitinck.
#
# Copyright (c) 2008 Brian M. Clapper <bmc@clapper.org>, Gael Varoquaux
# Author: Brian M. Clapper, Gael Varoquaux
# License: 3-clause BSD
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import math

from .mask_feature import MaskFeature
from analyser.config import CELL_TRACKE_PROPERTY, CELL_IMAGE_PROPERTY
from .sort import Sort, KalmanBoxTracker, action_iou_batch, behavioral_decision
from .cell import Cell
from analyser import common
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
        self.cell_number = 0
        self.maskobj = {}
        self.traced_image = None
        self.cell_property, self.trace_calendar = self.__init_tracing_table()
        self.cells = []
        self.distance = None
        self.props = None

    def __frame_name(self, number, name='frame_', length=3):
        return name+str(number).zfill(length)

    def __init_tracing_table(self):
        frame_columns_name = [self.__frame_name(x) for x in np.arange(0, self.frame_number)]
        cell_property = pd.DataFrame(columns=CELL_TRACKE_PROPERTY, dtype=np.int16)
        trace_calendar = pd.DataFrame(columns=frame_columns_name, dtype=np.int16)
        return cell_property, trace_calendar

    def _get_maskfeature_obj_by_frame(self, frame):
        if frame not in self.maskobj.keys():
            self.maskobj[frame] = MaskFeature(self.__array__()[frame])
        return self.maskobj[frame]

    def tracing(self, **args):
        mot_tracker = Sort(**args)  # create instance of the SORT tracker
        KalmanBoxTracker.count = 0
        total_frames = 0
        output_d = []
        for frame in tqdm(range(0, self.frame_number), position=0):
            img = self._get_maskfeature_obj_by_frame(frame)
            dets = img.instance_properties.iloc[:, list(range(1, 6))+[0]].values
            total_frames += 1
            trackers = mot_tracker.update(np.array(dets))
            for d in trackers:
                output_d.append([frame]+list(d))
        output_d = np.array(output_d)  # [frame, center_x, center_y, orientation, long_axis,short_axis, id, label]
        self.asgine_feature(output_d)
        return output_d

    def asgine_feature(self, output_d):
        for i in range(output_d.shape[0]):
            new_id = int(output_d[i, -2])
            frame = int(output_d[i, 0])
            label = int(output_d[i, -1])
            self.trace_calendar.loc[new_id, self.__frame_name(frame)] = label
            if new_id not in self.cell_property.index:
                self.cell_property.loc[new_id, common.CELL_START] = frame
            self.cell_property.loc[new_id, common.CELL_END] = frame
        self.cell_number = self.cell_property.shape[0]
        self.cell_property[common.CELL_LIFE_SPAN] = self.cell_property[common.CELL_END] - self.cell_property['start_time'] +1
        self.cell_property[common.CELL_TABEL_ARG] = np.arange(0, self.cell_number)
        self.cell_property[common.CELL_ID] = self.cell_property.index
        self.cell_property[common.CELL_GENERATION] = 1

    def update_traced_image(self):
        traced_image = np.zeros(self.__array__().shape)
        for frame in range(0, self.frame_number):
            d = self.trace_calendar.loc[:, self.__frame_name(frame)].dropna()
            for k, v in d.items():
                traced_image[frame][self.__array__()[frame] == v] = k
        self.traced_image = traced_image
        return traced_image

    def connect_generation(self):
        # props = ct.run_cell_time_props()
        # cells = ct.create_cells()
        for f in range(0, self.frame_number):
            end_cell = list(self.cell_property.loc[self.cell_property.end_time == f].arg)
            start_cell = list(self.cell_property.loc[self.cell_property.start_time == (f+1)].arg)
            if len(end_cell) and len(start_cell):
                bb_x = []
                for v in end_cell:
                    # bb_x.append(self.cells[v].contours[f])
                    bb_x.append(self.coords[v, f].T)
                bb_y = []
                for v in start_cell:
                    # bb_y.append(self.cells[v].contours[f+1])
                    bb_y.append(self.coords[v, f+1].T)

                cost = action_iou_batch(bb_x, bb_y)
                connection, division, fusion = behavioral_decision(cost)
                # print(f, connection, "\n",division, "\n",fusion)
                if connection:
                    pass
                if division:
                    for k, v in division.items():
                        mother = self.cell_property.iloc[end_cell[k]].identity
                        daughter1 = self.cell_property.iloc[start_cell[v[0]]].identity
                        daughter2 = self.cell_property.iloc[start_cell[v[1]]].identity
                        generation = self.cell_property.loc[mother, CELL_TRACKE_PROPERTY[1]]+1
                        # update values          
                        self.cell_property.loc[mother, CELL_TRACKE_PROPERTY[6:9]] = [True, daughter1, daughter2]
                        self.cell_property.loc[[daughter1, daughter2], CELL_TRACKE_PROPERTY[4]] = mother
                        self.cell_property.loc[[daughter1, daughter2], CELL_TRACKE_PROPERTY[1]] = generation
                if fusion:
                    for k, v in fusion.items():
                        # store index
                        mother = self.cell_property.iloc[end_cell[v[0]]].identity
                        father = self.cell_property.iloc[end_cell[v[1]]].identity
                        daughter = self.cell_property.iloc[start_cell[k]].identity
                        generation = max(self.cell_property.loc[[mother, father], CELL_TRACKE_PROPERTY[1]])+1
                        self.cell_property.loc[mother, CELL_TRACKE_PROPERTY[9:12]] = [True, father, daughter]
                        self.cell_property.loc[father, CELL_TRACKE_PROPERTY[9:12]] = [True, mother, daughter]
                        self.cell_property.loc[daughter, [common.CELL_GENERATION, common.CELL_MOTHER, common.CELL_FATHER]] = [generation, mother, father]

    def create_cells(self):
        all_cells = []
        for cell_id in self.cell_property[common.CELL_TABEL_ARG]:
            cell = self.create_single_cell_by_id(cell_id)
            all_cells.append(cell)
        self.cells = all_cells
        return all_cells

    def cell_features_to_3dmatrix(self):
        """Convert the traced results into a 3d matrix with the size of
        number of cells x number of frame x number of attributes.
        """
        cell_image_property = CELL_IMAGE_PROPERTY.copy()
        if common.IMAGE_COORDINATE in CELL_IMAGE_PROPERTY:
            cell_image_property.remove(common.IMAGE_COORDINATE)
            coords = np.zeros((self.cell_number, self.frame_number, 2, common.IMAGE_CONTOURS_LENGTH))
        props = np.zeros((self.cell_number, self.frame_number, len(cell_image_property)))
        for i in range(0, self.frame_number):
            img = self._get_maskfeature_obj_by_frame(i)
            data = img.instance_properties.set_index(common.IMAGE_LABEL, drop=False)
            label_id_maps = self.trace_calendar.iloc[:, i].dropna()
            arg_id = self.cell_property.loc[label_id_maps.index, common.CELL_TABEL_ARG].values
            props[arg_id, i, :] = data.loc[label_id_maps.values, cell_image_property]
            aaa = np.array(list(data.loc[label_id_maps.values, common.IMAGE_COORDINATE]))
            coords[arg_id, i, :, :] = aaa
        self.props = props
        self.coords = coords
        return props, coords

    def create_single_cell_by_id(self, cell_index):
        trace_feature = self.cell_property.loc[cell_index].values
        start_time, end_time = self.cell_property.loc[cell_index, [2, 3]].astype(int)
        arg = int(self.cell_property.loc[cell_index].arg)
        prop = self.props[arg, start_time:end_time+1, :]
        coord = self.coords[arg, start_time:end_time+1, :, :]
        return Cell(trace_feature, prop, coord)

    def cell_distance_3d(self):
        """Convert the traced cells distance into a 3d matrix with the size of
        number of cells x number of frame x 2(2 center and near).
        """
        distance = np.zeros((self.frame_number, self.cell_number, self.cell_number, 4))
        for i in trange(0, self.frame_number):
            labels = self.trace_calendar.iloc[:, i].dropna()
            cell_idx = self.cell_property.loc[labels.index].arg
            mk = self.maskobj[i]
            index = mk.label2index(labels.values)
            # map index to cell_id
            cost = mk.cost(index, index)
            map = pd.DataFrame(np.array([labels, cell_idx, index]).T,
                                columns= ["label", "cell_idx", "index"],
                                index=index)
            distance[i, 
                list(map.loc[list(cost.index_x.astype(int))].cell_idx.astype(int)), 
                list(map.loc[list(cost.index_y.astype(int))].cell_idx.astype(int)),
                :] = cost[['center_dist', 'nearnest_dis', 'nearnest_point_x_index', 'nearnest_point_y_index']]
        self.distance = distance
        return distance

    def prediction(self, image, n_components: int):
        prediction = self.trace_calendar.copy()
        prediction = prediction.fillna(-1)
        if image.shape[0] != self.__array__().shape[0]:
            return None
        for f in trange(0, image.shape[0]):
            cellimage = CellImage(image[f], mask=self.maskobj[f])
            data = cellimage.get_fluorescent_intensity()
            cluster = FluorescentClassification(data)
            _, _ = cluster.predition_data_type(
                n_components=n_components,
                init_params='kmeans')
            cluster.data.loc[-1, "channel_prediction"] = -1
            # print(cluster.data.loc[prediction[self.__frame_name(f)].values.astype(int)].channel_prediction)
            prediction[self.__frame_name(f)] = cluster.data.loc[prediction[self.__frame_name(f)].values.astype(int)].channel_prediction.values
        # print(prediction)
        self.cell_property["channel_prediction"] = prediction.replace(-1, None).mode(axis=1)[0].values
        return prediction

    def relations2cell(self, x_index, y_index, frame):
        center_dist, near_dist, nearnest_point_x_index, nearnest_point_y_index= self.distance[frame, x_index, y_index]
        center_x = self.get_centers(x_index, frame)
        near_x = self.get_coords(x_index, frame, int(nearnest_point_x_index))
        center_y = self.get_centers(y_index, frame)
        near_y = self.get_coords(y_index, frame, int(nearnest_point_y_index))
        orientation_x = self.get_orientation(x_index, frame)
        orientation_y = self.get_orientation(y_index, frame)
        angel_x = self.included_angle_to_the_major_axis(center_x[0], center_x[1], near_x[0], near_x[1], orientation_x)
        angel_y = self.included_angle_to_the_major_axis(center_y[0], center_y[1], near_y[0], near_y[1], orientation_y)
        time_gap = self.cell_property.loc[x_index, "start_time"] - self.cell_property.loc[y_index, "start_time"]
        return center_dist, near_dist, angel_x, angel_y, time_gap

    def get_centers(self, index, frame):
        label = self.index2label(index, frame)
        return self.maskobj[frame].get_centers([label]).values[0]

    def get_coords(self, index, frame, point_index):
        label = self.index2label(index, frame)
        return self.maskobj[frame].get_outline([label]).values[0][:, point_index]

    def get_orientation(self, index, frame):
        label = self.index2label(index, frame)
        return self.maskobj[frame].instance_properties.loc[label, "orientation"]

    def index2label(self, index, frame):
        return int(self.trace_calendar.loc[index, self.__frame_name(frame)])

    def label2index(self, label, frame):
        return self.trace_calendar.loc[self.trace_calendar[self.__frame_name(frame)]==label].index[0]

    def included_angle_to_the_major_axis(self, x1, y1, x2, y2, angle2):
        angle1 = math.atan2(y2-y1, x2-x1)
        included_angle = angle1-angle2
        included_angle = included_angle - np.pi*2*math.floor(included_angle/(2 * np.pi))
        if abs(included_angle) > np.pi:
            included_angle = included_angle-np.pi*2
        return included_angle

    def fusion_cell_features(self):
        fusioned_cells = self.cell_property.loc[(~self.cell_property.mother.isna()) & (~self.cell_property.father.isna())].copy()
        fusioned_parents = None
        for cell in fusioned_cells.index:
            print(cell)
            mother_index, father_index, frame = fusioned_cells.loc[cell, ['mother', 'father', 'start_time']].astype(np.int16)
            # mother_id = self.cells[mother_index].indentify
            # father_id = self.cells[father_index].indentify
            frame = frame-1

            # exchange m & f
            if (self.cell_property.loc[mother_index].channel_prediction == self.cell_property.loc[father_index].channel_prediction):
                print("error")
            elif self.cell_property.loc[mother_index].channel_prediction > self.cell_property.loc[father_index].channel_prediction:
                c = mother_index
                mother_index = father_index
                father_index = c

            # assgin son' features
            center_distance, n_distance, angle_0, angle_1, timegap = self.relations2cell(mother_index, father_index, frame)
            fusioned_cells.loc[cell, 'center_distance'] = center_distance
            fusioned_cells.loc[cell, 'nearnest_distance'] = n_distance
            # fusioned_cells.loc[cell, 'start_nearnest_distance'] = start_distance
            fusioned_cells.loc[cell, 'angle_x'] = angle_0
            fusioned_cells.loc[cell, 'angle_y'] = angle_1
            fusioned_cells.loc[cell, 'timegap'] = timegap

            # from mothers perspective:
            frame_x = int(max(self.cell_property.loc[[mother_index, father_index]].start_time))
            mf_cf = self.surrounding_cell_freatures(mother_index, father_index, frame_x)
            mf_cf.loc[:, 'fusion_type'] = 'm'
            if fusioned_parents is None:
                fusioned_parents = mf_cf
            else:
                fusioned_parents = pd.concat([fusioned_parents, mf_cf])
            # from fathers perspective:
            ff_cf = self.surrounding_cell_freatures(father_index, mother_index, frame_x)
            ff_cf.loc[:, 'fusion_type'] = 'f'
            if fusioned_parents is None:
                fusioned_parents = ff_cf
            else:
                fusioned_parents = pd.concat([fusioned_parents, ff_cf])
        return fusioned_cells, fusioned_parents

    def surrounding_cell_freatures(self, x_index, y_index, frame, radius=200):
        nc = list(self.neighbor_cells(x_index, radius=radius))
        if y_index not in nc:
            nc += [y_index]
        fusion_parent_cells = self.cell_property.loc[nc].copy()
        for y in nc:
            if self.cell_property.loc[y].end_time < frame:
                continue
            center_distance, n_distance, angle_x, angle_y, timegap = self.relations2cell(x_index, y, frame)
            fusion_parent_cells.loc[y, ["center_distance",
                                        "nearnest_distance",
                                        "angle_x",
                                        "angle_y",
                                        "timegap"]] = [center_distance, n_distance, angle_x, angle_y, timegap]
        fusion_parent_cells.loc[:, 'flag'] = False
        fusion_parent_cells.loc[y_index, 'flag'] = True
        fusion_parent_cells.loc[:, 'ref'] = x_index
        return fusion_parent_cells

    def neighbor_cells(self, index, radius=100):
        frame = int(self.cell_property.loc[index].start_time)
        x_label = self.index2label(index, frame)
        y_labels = self.maskobj[frame].nearnest_radius(x_label, radius)
        y_index = [self.label2index(y, frame) for y in y_labels]
        return y_index
