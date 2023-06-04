from typing import Optional, Sequence, Union
import math
import random


from .config import CELL_IMAGE_PROPERTY, CELL_TRACKE_PROPERTY, TIP_ANGLE
import analyser.common as common
import numpy as np


class Cell(object):
    def __init__(self, cell_attr, image_attr_overtime, coord_attr, footprint):
        """
        """
        self.attribute_data = cell_attr
        self.attribute_time_series = image_attr_overtime
        self.contours = coord_attr
        feature_list = CELL_TRACKE_PROPERTY + \
            [common.OBJ_TABEL_ARG, common.CHANNEL_PREDICTION]
        for i in range(0, len(feature_list)):
            key = feature_list[i]
            setattr(self, key, cell_attr[i])

        # for i in range(0, len(CELL_IMAGE_PROPERTY)):
        #     key = CELL_IMAGE_PROPERTY[i]
        #     setattr(self, key, image_attr_overtime[i, :])

    def get_attribute(self, name: str, frame: Optional[int] = None):
        if name in CELL_IMAGE_PROPERTY:
            if frame is None:
                return self.attribute_time_series[:, CELL_IMAGE_PROPERTY.index(name)]
            else:
                return self.attribute_time_series[frame, CELL_IMAGE_PROPERTY.index(name)]
        else:
            print("no attribute named %s" % name)
            return None

    def label(self, frame: Union[int, Sequence] = None):
        """frame:
        """
        if frame is None:
            return self.attribute_time_series[:, CELL_IMAGE_PROPERTY.index(common.IMAGE_LABEL)]
        elif isinstance(frame, int):
            frame = self._check_frame(frame)
            return self.attribute_time_series[frame, CELL_IMAGE_PROPERTY.index(common.IMAGE_LABEL)]
        else:
            frames = [self._check_frame(f) for f in frame]
            return self.attribute_time_series[frames, CELL_IMAGE_PROPERTY.index(common.IMAGE_LABEL)]

    def _check_frame(self, frame):
        """Convert global index to local index
        """
        avaliabel_frame_list = np.where(self.attribute_time_series[:, 0] != 0)[0]
        nearnest_index = np.argmin(abs(avaliabel_frame_list+self.start_time-frame))
        return int(avaliabel_frame_list[nearnest_index])

    def __singe_center(self, frame: int):
        """
        """
        i = self._check_frame(frame)
        return self.attribute_time_series[i,
                [CELL_IMAGE_PROPERTY.index(common.IMAGE_CENTER_LIST[0]),
                    CELL_IMAGE_PROPERTY.index(common.IMAGE_CENTER_LIST[1])]]

    def center(self, frame: Union[int, Sequence] = None):
        if frame is None:
            return self.attribute_time_series[:,
                    [CELL_IMAGE_PROPERTY.index(common.IMAGE_CENTER_LIST[0]),
                     CELL_IMAGE_PROPERTY.index(common.IMAGE_CENTER_LIST[1])]]
        elif type(frame) is int:
            return self.__singe_center(frame)
        else:
            coords = []
            for f in frame:
                coord = self.__singe_center(f)
                coords.append(coord)
            return np.array(coords)

    def orientation(self, frame):
        i = self._check_frame(frame)
        att_index = CELL_IMAGE_PROPERTY.index(common.IMAGE_ORIENTATION)
        return self.attribute_time_series[i, att_index]

    def contours2angle(self, frame):
        frame = self._check_frame(frame)
        coords = self.contours[frame]
        center = self.center(frame)
        orient = self.orientation(frame)
        angle = np.zeros(coords.shape[0])
        for i in range(coords.shape[0]):
            angle[i] = self.single_point_angle_to_the_major_axis(
                center[0],
                center[1],
                coords[i, 0],
                coords[i, 1],
                orient,
            )
        return angle

    def single_point_angle_to_the_major_axis(self, x1, y1, x2, y2, angle2):
        """
        x1: center x
        y1: center y
        x2: target x
        y2: target y
        angle2: orientation
        """
        angle1 = math.atan2(y2-y1, x2-x1)
        included_angle = angle1-angle2
        included_angle = included_angle - \
            np.pi*2*math.floor(included_angle/(2 * np.pi))
        if abs(included_angle) > np.pi:
            included_angle = included_angle-np.pi*2
        return included_angle

    def random_index(self, rate: list):
        """
        rate: 每个点出现的概率
        """
        start = 0
        index = 0
        randnum = random.randint(1, sum(rate))
        for index, scope in enumerate(rate):
            start += scope
            if randnum <= start:
                break
        return index

    def random_position_stage1(self, rate: float, frame):
        """
        rate: 尖端的概率, range(0, 1)
        """
        frame = self._check_frame(frame)
        coords = self.contours[frame]
        angle = self.contours2angle(frame)*180/np.pi
        is_tips = (abs(angle) < TIP_ANGLE) | (abs(angle) > 180 - TIP_ANGLE)
        length = is_tips.sum()
        rate_list = np.zeros(is_tips.shape)
        rate_list[is_tips] = rate/length
        rate_list[~is_tips] = (1-rate)/(is_tips.shape[0]-length)
        rate_list = (rate_list*100).astype(int)
        index = self.random_index(rate_list)
        return coords[index], index

    def random_position_stage2(self, gd_rate_list: list, frame: int):
        """
        rate: 根据所有坐标点的浓度，表示出现响应的概率，列表概率和=1
        """
        if np.sum(gd_rate_list) == 0:
            gd_rate_list = gd_rate_list+1
        else:
            gd_rate_list = gd_rate_list/np.sum(gd_rate_list)  # 0-1normalize
        frame = self._check_frame(frame)
        coords = self.contours[frame]
        gd_rate_list = (gd_rate_list*100).astype(int)
        index = self.random_index(gd_rate_list)
        return coords[index], index

    # def set_attributes(self):
    #     pass

    # def update_attributes(self):
    #     pass

    # def set_properties_over_time(self, f=[], v=[], c=[]):
    #     for i in range(0, len(f)):
    #         key = CELL_TRACKE_PROPERTY[i]
    #         setattr(self, key, f[i])
    #     for i in range(0, v.shape[1]):
    #         key = CELL_IMAGE_PROPERTY[i]
    #         setattr(self, key, v[:, i])
    #     self.contours = c

    # def get_label(self, frame):
    #     i = self._check_frame(frame)
    #     if i > -1:
    #         return int(self.label[i])
    #     else:
    #         print("ERROR!")

    # def get_center(self, frame):
    #     i = self._check_frame(frame)
    #     if isinstance(self.centroid_0[i], float):
    #         return (self.centroid_0[i], self.centroid_1[i])
    #     else:
    #         print("ERROR!")
    #         step = 1
    #         while(step < MIN_HITS+1):
    #             print("flag", frame-step)
    #             if frame-step >= self.start_time:
    #                 c = (self.centroid_0[i-step], self.centroid_1[i-step])
    #             if ((not isinstance(c[0], float)) & (i+step <= self.end_time)):
    #                 c = self.contours[i+step]
    #             if isinstance(c[0], float):
    #                 return c
    #                 break
    #             step += 1
    #         # return c
    #         # return (self.centroid_0[i-1], self.centroid_1[i-1])
    #         return (-1, -1)

    # def get_orientation(self, frame):
    #     i = self._check_frame(frame)
    #     if i > -1:
    #         return self.orientation[i]
    #     else:
    #         print("ERROR!")

    # def get_contours(self, frame):
    #     if ((frame > self.start_time - 1) & (frame < self.end_time + 1)):
    #         c = self.contours[frame]
    #         if not len(c):
    #             print("gt co")
    #             step = 1
    #             while(step < MIN_HITS+1):
    #                 print("flag", frame-step)
    #                 if frame-step >= self.start_time:
    #                     c = self.contours[frame-step]
    #                 if (not len(c)) & (frame+step <= self.end_time):
    #                     c = self.contours[frame+step]
    #                 if len(c):
    #                     break
    #                 step += 1
    #         return c
    #     return []

    def get_axis_major_length(self, frame):
        i = self._check_frame(frame)
        if i > -1:
            return self.axis_major_length[i]
        else:
            print("ERROR!")

    def get_axis_minor_length(self, frame):
        i = self._check_frame(frame)
        if i > -1:
            return self.axis_minor_length[i]
        else:
            print("ERROR!")

    def get_area(self, frame):
        i = self._check_frame(frame)
        if i > -1:
            return self.area[i]
        else:
            print("ERROR!")

    def get_bbox(self, frame):
        i = self._check_frame(frame)
        if i > -1:
            return (self.bbox_0[i], self.bbox_1[i], self.bbox_2[i], self.bbox_3[i])
        else:
            print("ERROR!")

    def get_eccentricity(self, frame):
        i = self._check_frame(frame)
        if i > -1:
            return self.eccentricity[i]
        else:
            print("ERROR!")

    def get_semantic(self, frame):
        i = self._check_frame(frame)
        if i > -1:
            return self.semantic[i]
        else:
            print("ERROR!")

    def get_instance(self, frame):
        i = self._check_frame(frame)
        if i > -1:
            return self.instance[i]
        else:
            print("ERROR!")


def create_cells_by_type(celltracer, types: list):
    all_cells = {}
    for t in types:
        all_cells[t] = []
    for t in types:
        data = celltracer.obj_property.loc[
            celltracer.obj_property[common.CHANNEL_PREDICTION] == t]
        for cell_arg in data[common.CELL_TABEL_ARG]:
            cell_id = celltracer.obj_property.iloc[int(cell_arg)][common.CELL_ID]
            cell = create_single_cell_by_id(celltracer, int(cell_id))
            all_cells[t].append(cell)
    return all_cells


def create_cells(celltracer):
    all_cells = []
    for cell_arg in celltracer.obj_property[common.CELL_TABEL_ARG]:
        cell_id = celltracer.obj_property.iloc[int(cell_arg)][common.CELL_ID]
        cell = create_single_cell_by_id(celltracer, int(cell_id))
        all_cells.append(cell)
    return all_cells


def create_single_cell_by_id(celltracer, cell_index):
    trace_feature = celltracer.obj_property.loc[cell_index].values
    footprint = celltracer.trace_calendar.loc[cell_index]
    start_time, end_time, arg = celltracer.obj_property.loc[
        cell_index, [common.OBJ_START,
                     common.OBJ_END,
                     common.OBJ_TABEL_ARG]].astype(int)
    prop = celltracer.props[arg, start_time:end_time+1, :]
    coord = celltracer.coords[arg, start_time:end_time+1, :, :]
    return Cell(trace_feature, prop, coord, footprint)
