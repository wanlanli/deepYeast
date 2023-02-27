from .tracer import Tracer
from analyser.distance import predition_data_type
from .distance import find_nearnest_points
import math
import numpy as np
import pandas as pd


class CellTracer(Tracer):
    def __init__(self, img, data) -> None:
        super().__init__(data)
        self.img = img
        self.background = None
        self.distance = None

    def fusioned_cell_features(self):
        fusioned_cells = self.cell_property.loc[(~self.cell_property.mother.isna()) & (~self.cell_property.father.isna())].copy()
        fusioned_parents = None
        for cell in fusioned_cells.index:
            print(cell)
            mother_index, father_index, frame = fusioned_cells.loc[cell, ['mother', 'father', 'start_time']].astype(np.int16)
            # mother_id = self.cells[mother_index].indentify
            # father_id = self.cells[father_index].indentify
            frame = frame-1

            # exchange m & f
            if (self.cell_property.iloc[mother_index].pred == self.cell_property.iloc[father_index].pred):
                print("error")
            elif self.cell_property.iloc[mother_index].pred > self.cell_property.iloc[father_index].pred:
                c = mother_index
                mother_index = father_index
                father_index = c

            # assgin son' features
            center_distance, n_distance, start_distance, angle_0, angle_1, timegap = self.mating_features(mother_index, father_index, frame)
            fusioned_cells.loc[cell, 'fusion_center_distance'] = center_distance
            fusioned_cells.loc[cell, 'fusion_point_distance'] = n_distance
            fusioned_cells.loc[cell, 'start_nearnest_distance'] = start_distance
            fusioned_cells.loc[cell, 'angle_0'] = angle_0
            fusioned_cells.loc[cell, 'angle_1'] = angle_1

            # from mothers perspective:
            mf_cf = self.surrounding_cell_freatures(mother_index, father_index, frame)
            mf_cf.loc[:, 'fusion_type'] = 'm'
            if fusioned_parents is None:
                fusioned_parents = mf_cf
            else:
                fusioned_parents = pd.concat([fusioned_parents, mf_cf])
            # from fathers perspective:
            ff_cf = self.surrounding_cell_freatures(father_index, mother_index, frame)
            ff_cf.loc[:, 'fusion_type'] = 'f'
            if fusioned_parents is None:
                fusioned_parents = ff_cf
            else:
                fusioned_parents = pd.concat([fusioned_parents, ff_cf])
        return fusioned_cells, fusioned_parents

    def mating_features(self, x_index, y_index, frame):
        ce_0 = np.array(self.cells[x_index].get_center(frame))
        ce_1 = np.array(self.cells[y_index].get_center(frame))
        # if ((ce_0[0] > 0) & (ce_1[0] > 0)):
        center_distance = np.sqrt(np.sum(np.square(ce_0-ce_1)))
        # else:
        #     center_distance = -1

        c_0 = self.cells[x_index].get_contours(int(self.cells[x_index].start_time))
        c_1 = self.cells[y_index].get_contours(int(self.cells[y_index].start_time))
        # if not (len(c_0)>0 & len(c_1)>0):
        #     start_distance = -1
        # else:
        start_distance, id_0, id_1 = find_nearnest_points(c_0, c_1)

        c_0 = self.cells[x_index].get_contours(frame)
        c_1 = self.cells[y_index].get_contours(frame)
        nearest_distance, id_0, id_1 = find_nearnest_points(c_0, c_1)
        angle_0 = self.included_angle_to_the_major_axis(ce_0[0], ce_0[1], c_0[id_0][0], c_0[id_0][1], self.cells[x_index].orientation[frame])
        angle_1 = self.included_angle_to_the_major_axis(ce_1[0], ce_1[1], c_1[id_1][0], c_1[id_1][1], self.cells[y_index].orientation[frame])

        time_gap = self.cells[x_index].start_time - self.cells[y_index].start_time
        return center_distance, nearest_distance, start_distance, angle_0, angle_1, time_gap
