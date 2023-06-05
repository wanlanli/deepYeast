import numpy as np

from analyser.cell.cells import Cells


class Mating(Cells):
    def __init__(self, image: np.array, cells) -> None:
        super().__init__(image, cells)

        self.center = None
        self.p = None
        self.m = None

    def init_center(self, key):
        mask = self.image[-1, :, :, 3] == key
        labels = np.unique(mask[:, :]*self.image[-1, :, :, 4])
        if len(labels) == 2:
            self.center = labels[1]
            parents = self.cells[self.center].parents
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

    def neibor(self):
        self.p_candidates = self.candidate(self.m)
        self.m_candidates = self.candidate(self.p)

    def candidate(self, k):
        s_time = self.cells[k].start
        e_time = self.cells[k].end
        k_type = self.cells[k].type
        candidates = self.features.loc[(self.features['end'] > s_time) &
                                       (self.features['start'] < e_time) &
                                       (self.features['type'] != k_type)]
        return candidates.id
