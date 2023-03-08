from analyser.tracer import Tracer
from analyser import common
from analyser.config import CELL_TRACKE_PROPERTY
# from analyser.distance import predition_data_type
from analyser.distance import find_nearnest_points
from analyser.sort import Sort, KalmanBoxTracker, action_iou_batch, behavioral_decision
import math
import numpy as np
import pandas as pd
from analyser.cell import Cell


class CellTracer(Tracer):

    # def __init__(self, obj) -> None:
    #     super().__init__(obj)

    def create_cells(self):
        all_cells = []
        for cell_id in self.obj_property[common.CELL_TABEL_ARG]:
            cell = self.create_single_cell_by_id(cell_id)
            all_cells.append(cell)
        self.cells = all_cells
        return all_cells

    def create_single_cell_by_id(self, cell_index):
        trace_feature = self.obj_property.loc[cell_index].values
        start_time, end_time = self.obj_property.loc[cell_index, [2, 3]].astype(int)
        arg = int(self.obj_property.loc[cell_index].arg)
        prop = self.props[arg, start_time:end_time+1, :]
        coord = self.coords[arg, start_time:end_time+1, :, :]
        return Cell(trace_feature, prop, coord)

    def fusion_cell_features(self):
        """
        mother_index, father_index: identity of parents
        """
        fusioned_cells = self.obj_property.loc[(~self.obj_property.mother.isna())
                                               & (~self.obj_property.father.isna())].copy()
        fusioned_parents = None
        for cell in fusioned_cells.index:
            mother_index, father_index, frame = \
                fusioned_cells.loc[cell,
                                   [common.CELL_MOTHER, common.CELL_FATHER, common.CELL_START]
                                   ].astype(np.int16)
            # mother_id = self.cells[mother_index].indentify
            # father_id = self.cells[father_index].indentify
            frame = frame-1

            # exchange m & f
            mother_index, father_index = self.__check_parent_order(mother_index, father_index)
            # if (self.obj_property.loc[mother_index].channel_prediction == self.obj_property.loc[father_index].channel_prediction):
            #     print("error")
            # elif self.obj_property.loc[mother_index].channel_prediction > self.obj_property.loc[father_index].channel_prediction:
            #     c = mother_index
            #     mother_index = father_index
            #     father_index = c

            # assgin son' features
            center_distance, n_distance, angle_0, angle_1, timegap = \
                self.relations2objs(mother_index, father_index, frame)
            fusioned_cells.loc[cell,
                               [common.CENTER_DISTANCE,
                                common.NEARNEST_DISTANCE,
                                common.ANGLE_POINT_CENTER[0],
                                common.ANGLE_POINT_CENTER[1],
                                common.TIME_GAP
                                ]] = [center_distance, n_distance, angle_0, angle_1, timegap]
            # fusioned_cells.loc[cell, 'start_nearnest_distance'] = start_distance
            # fusioned_cells.loc[cell, common.ANGLE_POINT_CENTER[0]] = angle_0
            # fusioned_cells.loc[cell, common.ANGLE_POINT_CENTER[1]] = angle_1
            # fusioned_cells.loc[cell, common.TIME_GAP] = timegap

            # from mothers perspective:
            frame_x = int(max(self.obj_property.loc[[mother_index, father_index]].start_time))
            mf_cf = self.neighbor_objects_freatures(mother_index, father_index, frame_x)
            mf_cf.loc[:, 'fusion_type'] = 'm'
            # if fusioned_parents is None:
            #     fusioned_parents = mf_cf
            # else:
            fusioned_parents = pd.concat([fusioned_parents, mf_cf])
            # from fathers perspective:
            ff_cf = self.neighbor_objects_freatures(father_index, mother_index, frame_x)
            ff_cf.loc[:, 'fusion_type'] = 'f'
            # if fusioned_parents is None:
            #     fusioned_parents = ff_cf
            # else:
            fusioned_parents = pd.concat([fusioned_parents, ff_cf])
        return fusioned_cells, fusioned_parents

    def __check_parent_order(self, mother_index, father_index):
        if (self.obj_property.loc[mother_index].channel_prediction == self.obj_property.loc[father_index].channel_prediction):
            print("Warning: The parent has same prediction type!")
        elif self.obj_property.loc[mother_index].channel_prediction > self.obj_property.loc[father_index].channel_prediction:
                # c = mother_index
                # mother_index = father_index
                # father_index = c
            return father_index, mother_index
        else:
            return mother_index, father_index

    def connect_generation(self):
        """Scan the video in time order, compare the cells that appear and
        disappear in adjacent frames, calculate possible correlations, and
        update the obj_property.
        """
        # props = ct.run_cell_time_props()
        # cells = ct.create_cells()
        for f in range(0, self.frame_number):
            # print(f)
            end_cell = list(self.obj_property.loc[self.obj_property[common.CELL_END] == f].arg)
            start_cell = list(self.obj_property.loc[self.obj_property[common.CELL_START] == (f+1)].arg)
            if len(end_cell) and len(start_cell):
                # print(start_cell, end_cell)
                cost = self.__cal_cell_connection(end_cell, start_cell, f)
                connection, division, fusion = behavioral_decision(cost)
                if connection:
                    pass
                if division:
                    for k, v in division.items():
                        mother = self.obj_property.iloc[end_cell[k]][common.CELL_ID]
                        daughter1 = self.obj_property.iloc[start_cell[v[0]]][common.CELL_ID]
                        daughter2 = self.obj_property.iloc[start_cell[v[1]]][common.CELL_ID]
                        self.__update_division_key(mother, daughter1, daughter2)
                if fusion:
                    for k, v in fusion.items():
                        mother = self.obj_property.iloc[end_cell[v[0]]][common.CELL_ID]
                        father = self.obj_property.iloc[end_cell[v[1]]][common.CELL_ID]
                        daughter = self.obj_property.iloc[start_cell[k]][common.CELL_ID]
                        self.__update_fusion_key(mother, father, daughter)

    def __cal_cell_connection(self, end_cell: list, start_cell: list, frame: int):
        """According to the given list of cells, calculate the iou
        ----------
        Args:
        end_cell: list, the list of cell stoped at frame.
        start_cell: list, the list of cells started from frame.
        frame: int.
        ----------
        Returns:
        cost, np.array, the IOU matrix with shape len(end_cell) * len(start_cell)
        """
        bb_x = []
        for v in end_cell:
            # print("e:", v, self.coords[v, frame])
            bb_x.append(self.coords[v, frame])
        bb_y = []
        for v in start_cell:
            # print("s:", v, self.coords[v, frame+1])
            bb_y.append(self.coords[v, frame+1])
        cost = action_iou_batch(bb_x, bb_y)
        return cost

    def __update_fusion_key(self, mother: int, father: int, daughter: int):
        """Index assignment properties of parents and daughter based on fused.
        ----------
        Args:
        mother: int, the identity of mother.
        father: int, the identity of father.
        daughter: int, the identity of daughter.
        ----------
        Returns:
        update self.obj_property values.
        """
        generation = max(self.obj_property.loc[[mother, father], common.CELL_GENERATION])+1
        self.obj_property.loc[mother, [common.CELL_FUSION_FLAGE,
                                       common.CELL_SPOUSE,
                                       common.CELL_SON]
                              ] = [True, father, daughter]
        self.obj_property.loc[father, [common.CELL_FUSION_FLAGE,
                                       common.CELL_SPOUSE,
                                       common.CELL_SON]
                              ] = [True, mother, daughter]
        self.obj_property.loc[daughter, [common.CELL_GENERATION,
                                         common.CELL_MOTHER,
                                         common.CELL_FATHER]
                              ] = [generation, mother, father]

    def __update_division_key(self, mother: int, daughter1: int, daughter2: int):
        """Index assignment properties of parents and daughter based on divison.
        ----------
        Args:
        mother: int, the index of mother.
        daughter1: int, the index of daughter1.
        daughter2: int, the index of daughter2.
        ----------
        Returns:
        update self.obj_property values.
        """
        generation = self.obj_property.loc[mother, common.CELL_GENERATION]+1
        # update values
        self.obj_property.loc[mother, CELL_TRACKE_PROPERTY[6:9]] = [True, daughter1, daughter2]
        self.obj_property.loc[[daughter1, daughter2], CELL_TRACKE_PROPERTY[4]] = mother
        self.obj_property.loc[[daughter1, daughter2], common.CELL_GENERATION] = generation

    # def __getstate__(self):
    #     return self.maskobj
    #     # return (self.frame_number,
    #     #         self.obj_number,
    #     #         self.maskobj,
    #     #         self.traced_image,
    #     #         self.obj_property,)
    #     #         self.trace_calendar,
    #     #         self.distance,
    #     #         self.props)

    # def __setstate__(self, d):
    #     self.maskobj = d
    # #     self.obj_number = d[1]
    # #     self.maskobj = d[2]
    # #     self.traced_image = d[3]
    # #     self.obj_property = d[1]
    # #     # self.trace_calendar = d[5]
    # #     # self.distance = d[6]
    # #     # self.props = d[7]

    # # def __setstate__(self, dict):
    # #     fh = open(dict['name'])  # reopen file
    # #     self.name = dict['name']
    # #     self.file = fh
