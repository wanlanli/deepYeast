from .tracer import Tracer
from analyser.distance import predition_data_type
from .distance import find_nearnest_points
import math
import numpy as np
import pandas as pd


class CellTracer(Tracer):
    def __init__(self, image, mask) -> None:
        self.image = image
        self.tracer = Tracer(mask)

    def _get_value(self, data):
        flatten = data.flatten()
        flatten = flatten[flatten > 0]
        return flatten

    def get_region_by_id(self, frame, label=None, index=None, pad=-1):
        """Returen region mask by label. Add padding for better plot images.
        """
        if (label is None) and index:
            label = int(self.cells[index].get_label(frame))
        mask = self.mask[frame] == label
        if pad < 0:
            return mask
        else:
            bbox = self.cells[index].get_bbox(frame)
            pad_mask = mask[bbox[0]-pad:bbox[2]+pad, bbox[1]-pad:bbox[3]+pad]
            return pad_mask

    def __background_threshold(self, thresdhold=5):
        """Return the mean values of masked piexes values
        """
        masked = self.img[:, 1:, :, :]*(self.mask[:, None, :, :] == 0)
        bg_threshold = np.zeros(masked.shape[1])
        for i in range(0, masked.shape[1]):
            value = self._get_value(masked[:, i, :, :])
            floor = np.percentile(value, thresdhold)
            celling = np.percentile(value, 100-thresdhold)
            value = value[(value > floor) & (value < celling)]
            bg_threshold[i] = np.mean(value)
        self.background = bg_threshold
        return bg_threshold

    def get_background(self):
        if self.background is None:
            return self.__background_threshold()
        else:
            return self.background

    def fluorescent_singal(self, line=90):
        for index in range(0, self.cell_number):
            data = self.cells[index]
            indentify = data.indentify
            frame = int(data.start_time)
            label = int(data.get_label(frame))
            cell_mask = self.get_region_by_id(frame, label, index)
            for i in range(1, self.img.shape[1]):
                chi = self._get_value(cell_mask*self.img[frame, i, :, :])
                self.cell_property.loc[indentify, "ch%d" % i] = np.percentile(chi, line)
        bg = self.get_background()
        for i in range(1, self.img.shape[1]):
            self.cell_property['bg%d' % i] = bg[i-1]
        return self.cell_property

    def prediction_cell_type(self, line=90, **arg):
        _ = self.fluorescent_singal(line)
        self.cell_property['ch1_norm'] = np.log(self.cell_property['ch1'])/np.log(self.cell_property['bg1'])
        self.cell_property['ch2_norm'] = np.log(self.cell_property['ch2'])/np.log(self.cell_property['bg2'])
        self.cell_property['pred'], _ = predition_data_type(self.cell_property[['ch1_norm', 'ch2_norm']], **arg)

    def neighbor_cells_distance(self, trg, radius, frame=None):
        nc = self.neighbor_cells(trg[0], radius)
        if frame is None:
            frame = int(self.cell_property.iloc[trg].start_time)
        distance = self.cells_distance(trg, nc, frame)
        return distance

    def neighbor_cells(self, index, radius=100):
        frame = int(self.cell_property.iloc[index].start_time)
        center = self.cells[index].get_center(frame)
        mask = self.generate_mask(center[0], center[1], radius=radius, w=self.img.shape[2], h=self.img.shape[3])
        neighbors = np.unique(mask*self.tracingdata[frame])
        neighbors = neighbors[neighbors != 0]
        return np.array(self.cell_property.loc[neighbors, 'arg']).astype(np.int16)

    def generate_mask(self, cx=50, cy=50, radius=10, w=100, h=100):
        x, y = np.ogrid[0: w, 0: h]
        mask = ((x-cx)**2 + (y-cy)**2) <= radius**2
        return mask

    def __init_distance(self):
        self.distance = np.zeros([self.cell_number, self.cell_number, self.frame_number, 2])
        self.distance[:, :, :, :] = -1

    def cells_distance(self, arg_x, arg_y, frame):
        if self.distance is None:
            self.__init_distance()
        # arg_x = np.array(self.cell_property.loc[id_x].arg)
        # arg_y = np.array(self.cell_property.loc[id_y].arg)
        for x in arg_x:
            # index_x = int(self.cell_property.loc[x].arg)
            for y in arg_y:
                # index_y = int(self.cell_property.loc[y].arg)
                if self.distance[x, y, frame, 0] > 0:
                    continue
                center_dist, nearnest_dis = self.two_regions_distance(x, y, frame)
                self.distance[x, y, frame, :] = [center_dist, nearnest_dis]
        data = self.distance[arg_x]
        data = data[:, arg_y]
        data = data[:, :, frame]
        return data

    def two_regions_distance(self, index_x, index_y, frame):
        """Given two regions' label, return 2 types distance between 2 regions.
        """
        # index_x = int(self.cell_property.loc[id_x].arg)
        # index_y = int(self.cell_property.loc[id_y].arg)
        coods_x = self.cells[index_x].contours[frame]
        coods_y = self.cells[index_y].contours[frame]
        nearnest_dis, _, _ = find_nearnest_points(coods_x, coods_y)
        center_x = np.array(self.cells[index_x].get_center(frame))
        center_y = np.array(self.cells[index_y].get_center(frame))
        center_dist = np.sqrt(np.sum(np.square(center_x - center_y)))
        return center_dist, nearnest_dis

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

    def included_angle_to_the_major_axis(self, x1, y1, x2, y2, angle2):
        angle1 = math.atan2(y2-y1, x2-x1)
        included_angle = angle1-angle2
        included_angle = included_angle - np.pi*2*math.floor(included_angle/(2 * np.pi))
        if abs(included_angle) > np.pi:
            included_angle = included_angle-np.pi*2
        return included_angle

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

    def surrounding_cell_freatures(self, x_index, y_index, frame, radius=200):
        nc = list(self.neighbor_cells(x_index, radius=radius))
        if y_index not in nc:
            nc += [y_index]
        fusion_parent_cells = self.cell_property.iloc[nc].copy()
        for i in nc:
            if self.cells[i].end_time < frame:
                continue
            identity = int(self.cell_property.iloc[i].indentify)
            center_distance, n_distance, start_distance, angle_0, angle_1, timegap = self.mating_features(x_index, i, frame)
            fusion_parent_cells.loc[identity, 'fusion_center_distance'] = center_distance
            fusion_parent_cells.loc[identity, 'fusion_point_distance'] = n_distance
            fusion_parent_cells.loc[identity, 'start_nearnest_distance'] = start_distance
            fusion_parent_cells.loc[identity, 'angle_0'] = angle_0
            fusion_parent_cells.loc[identity, 'angle_1'] = angle_1
        fusion_parent_cells.loc[:, 'flag'] = False
        fusion_parent_cells.loc[int(self.cell_property.iloc[y_index].indentify), 'flag'] = True
        fusion_parent_cells.loc[:, 'ref'] = x_index
        return fusion_parent_cells
