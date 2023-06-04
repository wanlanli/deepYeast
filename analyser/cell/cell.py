import pandas as pd


from analyser import common


class Cell(object):
    def __init__(self, label):
        self.id = label
        self.frames=[]
        self.start=None
        self.end=None
        self.life=None
        self.generation = None
        self.mother = None
        self.father = None
        self.features=pd.DataFrame(columns=common.CELL_PROP_COLUMNS)

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
        