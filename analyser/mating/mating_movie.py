import numpy as np

from analyser.cell.cells import Cells
from analyser.image_measure.meaure import ImageMeasure


class Mating(Cells):
    def __init__(self, image: np.array, cells) -> None:
        super().__init__(image, cells)

        self.center = None
        self.p = None
        self.m = None
        self.measure = {}

    def init_center(self, key):
        mask = self.image[-1, :, :, 3] == key
        labels = np.unique(mask[:, :]*self.image[-1, :, :, 4])
        if len(labels) == 2:
            self.center = labels[1]
            parents = self.cells[self.center].parents
            if parents is None:
                return
            else:
                if (self.cells[parents[0]].type != self.cells[parents[1]].type):
                    if self.cells[parents[0]].type == 1:
                        self.p = parents[0]
                        self.m = parents[1]
                    else:
                        self.p = parents[1]
                        self.m = parents[0]
                else:
                    print("Error! parents have same types")
                    self.p = parents[0]
                    self.m = parents[1]

    def neiber(self, k):
        s_time = self.cells[k].start
        e_time = self.cells[k].end
        candidates = self.features.loc[(self.features['end'] > s_time) &
                                       (self.features['start'] < e_time) &
                                       (self.features.parents.isna())
                                       ]
        return candidates

    def init_measure(self):
        for f in range(0, self.frame_number):
            self.measure[f] = ImageMeasure(self.image[f, :, :, 4])

    def angles(self, t, candidates):
        data = np.zeros((len(candidates), self.frame_number, 2))
        for i in range(0, len(candidates)):
            c = candidates[i]
            s_time = max(self.cells[t].start, self.cells[c].start)
            e_time = min(self.cells[t].end, self.cells[c].end)
            for f in range(s_time, e_time):
                measure_obj = self.measure[f]
                if (f in self.cells[t].frames) & (f in self.cells[c].frames):
                    angel_x, angel_y = measure_obj.two_regions_angle(measure_obj.label2index(t),
                                                                    measure_obj.label2index(c),)
                    data[i, f, :] = [angel_x, angel_y]
        return data

    def distance(self, t, candidates):
        data = np.zeros((len(candidates), self.frame_number, 4))
        for i in range(0, len(candidates)):
            c = candidates[i]
            s_time = max(self.cells[t].start, self.cells[c].start)
            e_time = min(self.cells[t].end, self.cells[c].end)
            for f in range(s_time, e_time):
                measure_obj = self.measure[f]
                if (f in self.cells[t].frames) & (f in self.cells[c].frames):
                    dist = measure_obj.ditance_matrix([measure_obj.label2index(t)],
                                                    [measure_obj.label2index(c)],)
                    data[i, f] = dist[0, 0]
        return data
