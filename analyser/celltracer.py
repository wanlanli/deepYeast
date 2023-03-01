from analyser.tracer import Tracer
from analyser import common
from analyser.config import CELL_TRACKE_PROPERTY
from analyser.distance import predition_data_type
from analyser.distance import find_nearnest_points
from analyser.sort import Sort, KalmanBoxTracker, action_iou_batch, behavioral_decision
import math
import numpy as np
import pandas as pd


class CellTracer(Tracer):
    def __init__(self) -> None:
        super().__init__()

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

    def connect_generation(self):
        """Scan the video in time order, compare the cells that appear and
        disappear in adjacent frames, calculate possible correlations, and
        update the obj_property.
        """
        # props = ct.run_cell_time_props()
        # cells = ct.create_cells()
        for f in range(0, self.frame_number):
            end_cell = list(self.obj_property.loc[self.obj_property[common.CELL_END] == f].arg)
            start_cell = list(self.obj_property.loc[self.obj_property[common.CELL_START] == (f+1)].arg)
            if len(end_cell) and len(start_cell):
                cost = self.__cal_cell_connection(start_cell, end_cell, f)
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
            bb_x.append(self.coords[v, frame].T)
        bb_y = []
        for v in start_cell:
            bb_y.append(self.coords[v, frame+1].T)
        cost = action_iou_batch(bb_x, bb_y)
        return cost

    def __update_fusion_key(self, mother: int, father: int, daughter: int):
        """Index assignment properties of parents and daughter based on fused.
        ----------
        Args:
        mother: int, the index of mother.
        father: int, the index of father.
        daughter: int, the index of daughter.
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
