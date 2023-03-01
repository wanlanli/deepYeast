from typing  import Union, Sequence

import numpy as np
import pandas as pd


from analyser.mask_feature import MaskFeature
from analyser.utils import flatten_nonzero_value


class CellImage():
    """Measure cell image with flourencent properties.

    Parameters
    ----------
    img : 3D matrix,dtype:int
        frames x width x height x channels. The first channel most be reference channel.
    mask: 2D matrix,dtype:int
        predicted mask.
    """
    def __init__(self, image: np.array, mask: Union[np.array, MaskFeature], order: Sequence = None) -> None:
        self.image = image
        if type(mask) == MaskFeature:
            self.mask = mask
        else:
            self.mask = MaskFeature(mask)
        self.background = None
        self.classifier = None
        self.fluorescent_intensity = None
        self.channel_number = self.image.shape[2]

    def get_cell_image(self, label: int, channel: Union[int, Sequence] = None):
        pass

    def __background_threshold(self, thresdhold: int = 5):
        """Get the background threshold for every channel.
        Params:
        thresdhold: percentile of data not taken into account
        """
        masked = self.image[:, :, 1:]*(self.mask.__array__()[:, :, None] == 0)
        bg_threshold = np.zeros(self.channel_number)
        for i in range(0, self.channel_number-1):
            value = flatten_nonzero_value(masked[:, :, i])
            if value.sum() == 0:
                bg_threshold[i] = 0
            else:
                floor = np.percentile(value, thresdhold)
                celling = np.percentile(value, 100-thresdhold)
                value = value[(value >= floor) & (value <= celling)]
                bg_threshold[i] = np.mean(value)
        self.background = bg_threshold
        return bg_threshold

    def get_background(self):
        if self.background is None:
            return self.__background_threshold()
        else:
            return self.background

    def instance_fluorescent_intensity(self, measure_line=90):
        """Get induvidual cell fluorescent intensity.
        Params:
        measure_line: take the percentile measure_line as the intensity
        """
        label_list = list(self.mask.get_region_label_list())
        data = pd.DataFrame(data=label_list, columns=['label'], index=label_list)
        for label in label_list:
            label = int(label)
            cell_mask = self.mask.get_instance_mask(label)
            for i in range(1, self.channel_number):
                chi = flatten_nonzero_value(cell_mask*self.image[:, :, i])
                data.loc[label, "ch%d" % i] = np.percentile(chi, measure_line)
        bg = self.get_background()
        for i in range(1, self.channel_number):
            data['bg%d' % i] = bg[i-1]
        self.fluorescent_intensity = data
        return data

    def get_fluorescent_intensity(self, **args):
        if self.fluorescent_intensity is None:
            return self.instance_fluorescent_intensity(**args)
        else:
            return self.fluorescent_intensity.iloc[:, 0:(self.channel_number*2+1)]

    # def cell_classification(self, **args):
    #     data = self.get_fluorescent_intensity()
    #     clusterobj = FluorescentClassification(data)
    #     pred, _ = clusterobj.predition_data_type(**args)
    #     self.fluorescent_intensity["channel_prediction"] = pred
    #     self.classifier = clusterobj
    #     return self.fluorescent_intensity
