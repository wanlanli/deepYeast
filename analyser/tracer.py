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
from .mask_feature import MaskFeature
from analyser.config import CELL_TRACKE_PROPERTY, CELL_IMAGE_PROPERTY
from .sort import Sort, KalmanBoxTracker, action_iou_batch, action_dicision
from .cell import Cell
from tqdm import tqdm
OVERLAP_VMIN = 0.1
OVERLAP_VMAX = 0.75


#f*x*y*c
def __frame_name(number, name='frame_', length=3):
    return name+str(number).zfill(length)

# class Tracer(np.ndarray):
#     def __new__(cls, mask: np.ndarray):
#         # Input array is an already formed ndarray instance first cast to be our class type
#         obj = np.asarray(mask).view(cls)
#         # Finally, we must return the newly created object:
#         return obj

#     def __array_finalize__(self, obj):
#         # see InfoArray.__array_finalize__ for comments
#         if obj is None:
#             return
#         self.frame_number = self.shape[0]
#         self.cell_property, self.trace_calendar = self.__init_tracing_feature_data()
#         self.maskobj = {}
#         self.cell_number = 0
#         self.cells = []
#         self.props = None
#         self.tracingdata = None


class Tracer(object):
    def __init__(self, data) -> None:
        """Input masked movie to tracing
        """
        self.mask = data
        self.frame_number = self.mask.shape[0]
        self.cell_property, self.trace_calendar = self.__init_tracing_feature_data()
        self.maskobj = {}
        self.cell_number = 0
        self.cells = []
        self.props = None
        self.tracingdata = None

    def __init_tracing_feature_data(self):
        frame_columns_name = [__frame_name(x) for x in np.arange(0, self.frame_number)]
        cell_property = pd.DataFrame(columns=CELL_TRACKE_PROPERTY, dtype=np.int16)
        trace_calendar = pd.DataFrame(columns=frame_columns_name, dtype=np.int16)
        return cell_property, trace_calendar

    def _get_maskfeature_obj(self, frame):
        if frame not in self.maskobj.keys():
            self.maskobj[frame] = MaskFeature(self.mask[frame])
        return self.maskobj[frame]

    def tracing(self, **args):
        mot_tracker = Sort(**args)  # create instance of the SORT tracker
        KalmanBoxTracker.count = 0
        total_frames = 0
        output_d = []
        for frame in tqdm(range(0, self.frame_number), position=0):
            img = self._get_maskfeature_obj(frame)
            dets = img.instance_properties.iloc[:, list(range(1, 6))+[0]].values
            total_frames += 1
            trackers = mot_tracker.update(np.array(dets))
            for d in trackers:
                output_d.append([frame]+list(d))
        output_d = np.array(output_d)  # [frame, center_x, center_y,orientation,long_axis,short_axis, id, label]
        self.asgine_feature(output_d)
        return output_d

    def asgine_feature(self, output_d):
        for i in range(output_d.shape[0]):
            new_id = int(output_d[i, -2])
            frame = int(output_d[i, 0])
            label = int(output_d[i, -1])
            self.trace_calendar.loc[new_id, __frame_name(frame)] = label
            if new_id not in self.cell_property.index:
                self.cell_property.loc[new_id, 'start_time'] = frame
            self.cell_property.loc[new_id, 'end_time'] = frame
        self.cell_number = self.cell_property.shape[0]    
        self.cell_property['life_time'] = self.cell_property['end_time'] - self.cell_property['start_time'] +1
        self.cell_property['arg'] = np.arange(0, self.cell_number)
        self.cell_property['indentify'] = self.cell_property.index
        self.cell_property['generation'] = 1

    def plot_tracing(self):
        tracing_data = np.zeros(self.mask.shape)
        for frame in range(0, self.frame_number):
            d = self.trace_calendar.loc[:, __frame_name(frame)].dropna()
            for k, v in d.items():
                tracing_data[frame][self.mask[frame] == v] = k
        self.tracingdata = tracing_data
        return tracing_data

    def run_cell_time_props(self):
        props = np.zeros((self.cell_number, self.frame_number, len(CELL_IMAGE_PROPERTY)))
        for i in range(0, self.frame_number):
            img = self._get_maskfeature_obj(i)
            data = img.region.set_index("label", drop=False)
            label_id_maps = self.trace_calendar.iloc[:, i].dropna()
            arg_id = self.cell_property.loc[label_id_maps.index, 'arg'].values
            props[arg_id, i, :] = data.loc[label_id_maps.values, CELL_IMAGE_PROPERTY]
        self.props = props
        return props

    def cell_contours_over_time(self, cell_id):
        cell_coords = {}
        start_time, end_time = self.cell_property.iloc[cell_id, [4,5]]
        for f in range(int(start_time), int(end_time+1)):
            label = self.props[cell_id, f, 0]
            if label == 0:
                cell_coords[f] = []
                continue
            coord = self.maskobj[f]._coordinates(label)
            cell_coords[f] = coord
        return cell_coords

    def create_cells(self):
        all_cells = []
        for i, v in enumerate(self.cell_property.indentify):
            coord = self.cell_contours_over_time(i)
            x = self.cell_property.iloc[i].values
            f = self.props[i, :, :]
            cell = Cell(x, f, coord)
            all_cells.append(cell)
        self.cells = all_cells
        return all_cells

    def connect_generation(self):
        # props = ct.run_cell_time_props()
        # cells = ct.create_cells()
        for f in range(0, self.frame_number):
            end_cell = list(self.cell_property.loc[self.cell_property.end_time == f].arg)
            start_cell = list(self.cell_property.loc[self.cell_property.start_time == (f+1)].arg)
            if len(end_cell) and len(start_cell):
                bb_x = []
                for v in end_cell:
                    bb_x.append(self.cells[v].contours[f])
                bb_y = []
                for v in start_cell:
                    bb_y.append(self.cells[v].contours[f+1])

                cost = action_iou_batch(bb_x, bb_y)
                connection, division, fusion = action_dicision(cost)
                # print(f, connection, "\n",division, "\n",fusion)
                if connection:
                    pass
                if division:
                    for k, v in division.items():
                        mother = end_cell[k]
                        son1 = start_cell[v[0]]
                        son2 = start_cell[v[1]]

                        generation = self.cell_property.iloc[mother, 1]+1
                        # update values
                        self.cell_property.iloc[mother, 6:9] = [True, son1, son2]
                        self.cell_property.iloc[[son1, son2], 2] = mother
                        self.cell_property.iloc[[son1, son2], 1] = generation

                        self.cells[mother].is_divided = True
                        self.cells[mother].sub_1 = son1
                        self.cells[mother].sub_2 = son2

                        self.cells[son1].mother = mother
                        self.cells[son1].generation = generation
                        self.cells[son2].mother = mother
                        self.cells[son2].generation = generation

                if fusion:
                    for k, v in fusion.items():
                        # store index
                        mother = end_cell[v[0]]
                        father = end_cell[v[1]]
                        son = start_cell[k]
                        generation = max(self.cell_property.iloc[[mother, father], 1])+1

                        self.cell_property.iloc[mother, 9:12] = [True, father, son]
                        self.cell_property.iloc[father, 9:12] = [True, mother, son]
                        self.cell_property.iloc[son, 1:4] = [generation, mother, father]

                        self.cells[mother].is_fusioned = True
                        self.cells[mother].spouse = father
                        self.cells[mother].son = son
                        self.cells[father].is_fusioned = True
                        self.cells[father].spouse = mother
                        self.cells[father].son = son
                        self.cells[son].generation = generation
                        self.cells[son].mother = mother
                        self.cells[son].father = father


# class CellTracer(Tracer):
#     def __init__(self, img, mask) -> None:
#         super().__init__(mask)
#         self.img = img

    # def __background_threshold(self, thresdhold=5):
    #     """Return the mean values of masked piexes values
    #     """
    #     masked = self.img[1:]*(self.mask.X[None, :, :] == 0)
    #     bg_threshold = np.zeros(masked.shape[0])
    #     for i in range(0, masked.shape[0]):
    #         value = self._get_value(masked[i])
    #         floor = np.percentile(value, thresdhold)
    #         celling = np.percentile(value, 100-thresdhold)
    #         value = value[(value > floor) & (value < celling)]
    #         bg_threshold[i] = np.mean(value)
    #     self.background = bg_threshold
    #     return bg_threshold

    # def get_background(self):
    #     if self.background is None:
    #         return self.__background_threshold()
    #     else:
    #         return self.background

    # def fluorescent_singal(self, line=90):
    #     data = self.mask.region.copy()
    #     for index in data.index:
    #         label = data.loc[index, 'label']
    #         cell_mask = self.mask.get_region_by_id(label)
    #         for i in range(1, self.img.shape[0]):
    #             chi = self._get_value(cell_mask*self.img[i])
    #             data.loc[index, "ch%d" % i] = np.percentile(chi, line)
    #     bg = self.get_background()
    #     for i in range(1, self.img.shape[0]):
    #         data['bg%d' % i] = bg[i-1]
    #     return data

