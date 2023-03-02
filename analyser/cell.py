from typing import Optional

from .config import CELL_IMAGE_PROPERTY, CELL_TRACKE_PROPERTY, MIN_HITS
import analyser.common as common
import numpy as np


class Cell(object):
    def __init__(self, cell_attr, image_attr_overtime, coord_attr):
        """
        """
        self.attribute_data = cell_attr
        self.attribute_time_point = image_attr_overtime
        self.contours = coord_attr

        for i in range(0, len(CELL_TRACKE_PROPERTY)):
            key = CELL_TRACKE_PROPERTY[i]
            setattr(self, key, cell_attr[i])
        # for i in range(0, len(CELL_IMAGE_PROPERTY)):
        #     key = CELL_IMAGE_PROPERTY[i]
        #     setattr(self, key, image_attr_overtime[i, :])

    def get_attribute(self, name: str, frame: Optional[int] = None):
        if name in CELL_IMAGE_PROPERTY:
            if frame is None:
                return self.attribute_time_point[:, CELL_IMAGE_PROPERTY.index(name)]
            else:
                return self.attribute_time_point[frame, CELL_IMAGE_PROPERTY.index(name)]
        else:
            print("no attribute named %s" % name)
            return None

    def label(self, frame: Optional[int] = None):
        if frame is None:
            return self.attribute_time_point[:, CELL_IMAGE_PROPERTY.index(common.IMAGE_LABEL)]
        else:
            return self.attribute_time_point[frame, CELL_IMAGE_PROPERTY.index(common.IMAGE_LABEL)]

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

    # def __check_frame(self, frame):
    #     # if (frame < self.start_time) or (frame > self.end_time):
    #     #     return -1
    #     # else:
    #     #     return int(frame) # - self.start_time)
    #     if (frame >= self.start_time) and (frame <= self.end_time):
    #         return int(frame)
    #     else:
    #         return int(np.argmin(abs(np.arange(self.start_time, self.end_time+1)-frame))+self.start_time)

    # def get_label(self, frame):
    #     i = self.__check_frame(frame)
    #     if i > -1:
    #         return int(self.label[i])
    #     else:
    #         print("ERROR!")

    def get_center(self, frame):
        i = self.__check_frame(frame)
        if isinstance(self.centroid_0[i], float):
            return (self.centroid_0[i], self.centroid_1[i])
        else:
            print("ERROR!")
            step = 1
            while(step < MIN_HITS+1):
                print("flag", frame-step)
                if frame-step >= self.start_time:
                    c = (self.centroid_0[i-step], self.centroid_1[i-step])
                if ((not isinstance(c[0], float)) & (i+step <= self.end_time)):
                    c = self.contours[i+step]  
                if isinstance(c[0], float):
                    return c
                    break
                step += 1
            # return c
            # return (self.centroid_0[i-1], self.centroid_1[i-1])
            return (-1, -1)

    def get_orientation(self, frame):
        i = self.__check_frame(frame)
        if i > -1:
            return self.orientation[i]
        else:
            print("ERROR!")

    def get_contours(self, frame):
        if ((frame > self.start_time - 1) & (frame < self.end_time + 1)):
            c = self.contours[frame]
            if not len(c):
                print("gt co")
                step = 1
                while(step < MIN_HITS+1):
                    print("flag", frame-step)
                    if frame-step >= self.start_time:
                        c = self.contours[frame-step]
                    if (not len(c)) & (frame+step <= self.end_time):
                        c = self.contours[frame+step]  
                    if len(c):
                        break
                    step += 1
            return c
        return []

    def get_axis_major_length(self, frame):
        i = self.__check_frame(frame)
        if i > -1:
            return self.axis_major_length[i]
        else:
            print("ERROR!")

    def get_axis_minor_length(self, frame):
        i = self.__check_frame(frame)
        if i > -1:
            return self.axis_minor_length[i]
        else:
            print("ERROR!")

    def get_area(self, frame):
        i = self.__check_frame(frame)
        if i > -1:
            return self.area[i]
        else:
            print("ERROR!")

    def get_bbox(self, frame):
        i = self.__check_frame(frame)
        if i > -1:
            return (self.bbox_0[i], self.bbox_1[i], self.bbox_2[i], self.bbox_3[i])
        else:
            print("ERROR!")

    def get_eccentricity(self, frame):
        i = self.__check_frame(frame)
        if i > -1:
            return self.eccentricity[i]
        else:
            print("ERROR!")

    def get_semantic(self, frame):
        i = self.__check_frame(frame)
        if i > -1:
            return self.semantic[i]
        else:
            print("ERROR!")

    def get_instance(self, frame):
        i = self.__check_frame(frame)
        if i > -1:
            return self.instance[i]
        else:
            print("ERROR!")