import pandas as pd


from analyser import common


class Cell(object):
    def __init__(self, label):
        self.id = label
        self.frames = []
        self.start = None
        self.end = None
        self.life = None
        self.generation = 1

        self.division = False
        self.daughter_vg = None

        self.ancient = None
        self.sister = None

        self.parents = None

        self.fusion = False
        self.daughter = None
        self.spouse = None

        self.type = None

        self.features = pd.DataFrame(columns=common.CELL_PROP_COLUMNS)

    def update(self, frame, values):
        self.features.loc[frame] = values

    def centers(self, frame=None):
        if frame is None:
            frame = self.features.index
        return self.features.loc[frame, common.IMAGE_CENTER_LIST]

    def area(self, frame=None):
        if frame is None:
            frame = self.features.index
        return self.features.loc[frame, common.IMAGE_AREA]

    def major_axis(self, frame=None):
        if frame is None:
            frame = self.features.index
        return self.features.loc[frame, common.IMAGE_MAJOR_AXIS]

    def is_out_of_border(self, shape=(150, 150)):
        """
        """
        bbox = self.features.loc[:, common.IMAGE_BOUNDING_BOX_LIST]
        min_row = min(bbox.iloc[:, 0]) == 0
        min_col = min(bbox.iloc[:, 1]) == 0
        max_row = max(bbox.iloc[:, 2]) == shape[0]
        max_col = max(bbox.iloc[:, 3]) == shape[1]
        out_of_border = min_row | min_col | max_row | max_col
        return out_of_border
