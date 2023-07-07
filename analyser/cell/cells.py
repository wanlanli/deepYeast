import numpy as np
import pandas as pd

from analyser.utils import flatten_nonzero_value

coloumns = ['id', 'start', 'end', 'life', 'generation', 'division', 'daughter_vg', 'ancient', 'sister', 'parents', 'fusion', 'daughter', 'spouse', 'type']


class Cells():
    """Measure cell image with flourencent properties.

    Parameters
    ----------
    img : 3D matrix,dtype:int
        frames x width x height x channels. The first channel most be reference channel.
    mask: 2D matrix,dtype:int
        predicted mask.
    """
    def __init__(self, image: np.array, cells) -> None:
        """image: frame*wide*heigh*5 channels(DIC, GFP, mCherry, segmentation, traced)
            cells: list of Cell objects
        """
        self.image = image
        self.cells = cells

        self.frame_number = self.image.shape[0]
        self.__background = None
        self.classifier = None
        self.__fluorescent_intensity = None
        self.features = None
        self.pixel_resolution = 1

    def set_pixel_resolution(self, x):
        self.pixel_resolution = x

    def set_type(self, pred):
        for k, v in pred.items():
            self.cells[k].type = v

    def init_propoerties(self):
        features = pd.DataFrame(columns=coloumns)
        for key in self.cells.keys():
            features.loc[key] = [getattr(self.cells[key], a) for a in coloumns]
        self.features = features

    def __background_threshold(self, thresdhold: int = 5):
        """Get the background threshold for every channel.
        Params:
        thresdhold: percentile of data not taken into account
        """
        masked = self.image[:, :, :, 1:3]*(self.image[:, :, :, [3]] == 0)
        bg_threshold = np.zeros((self.frame_number,  2))
        for i in range(0, 2):
            for f in range(0, self.frame_number):
                value = flatten_nonzero_value(masked[f, :, :, i])
                if value.sum() == 0:
                    bg_threshold[f, i] = 0
                else:
                    floor = np.percentile(value, thresdhold)
                    celling = np.percentile(value, 100-thresdhold)
                    value = value[(value >= floor) & (value <= celling)]
                    bg_threshold[f, i] = np.mean(value)
        return bg_threshold

    def background(self):
        if self.__background is None:
            self.__background = self.__background_threshold()
        return self.__background

    def instance_fluorescent_intensity(self, measure_line=90):
        """Get induvidual cell fluorescent intensity.
        Params:
        measure_line: take the percentile measure_line as the intensity
        """
        data = [] #columns=['label', 'frame', 'ch_1', 'ch_2', 'bg_1', 'bg_2'])
        for f in range(0, self.frame_number):
            label_list = np.unique(self.image[f, :, :, 4])[1:]
            for label in label_list:
                label = int(label)
                cell_mask = self.image[f, :, :, 4] == label
                data.append(
                    [label, f,
                     np.percentile(flatten_nonzero_value(cell_mask*self.image[f, :, :, 1]), measure_line),
                     np.percentile(flatten_nonzero_value(cell_mask*self.image[f, :, :, 2]), measure_line),
                     self.background()[f, 0],
                     self.background()[f, 1],
                     ])
        data = pd.DataFrame(data, columns=['label', 'frame', 'ch_1', 'ch_2', 'bg_1', 'bg_2'])
        return data

    def fluorescent_intensity(self, **args):
        if self.__fluorescent_intensity is None:
            self.__fluorescent_intensity = self.instance_fluorescent_intensity(**args)
        return self.__fluorescent_intensity
