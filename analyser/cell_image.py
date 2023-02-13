from typing  import Union, Sequence

import numpy as np
from sklearn.mixture import GaussianMixture

from analyser.mask_feature import MaskFeature
from analyser.utils import flatten_nonzero_value

class CellImage():
    """Measure cell image with flourencent properties.

    Parameters
    ----------
    img : 3D matrix,dtype:int
        channels x width x height. The first channel most be reference channel.
    mask: 2D matrix,dtype:int
        predicted mask.
    """
    def __init__(self, image: np.array, mask: np.array, order: Sequence = None) -> None:
        self.image = image
        self.mask = MaskFeature(mask)
        _ = self.mask.get_instance_properties()
        self.background = None
        self.classifier = None
        self.fluorescent_intensity = None
        self.channel_number = self.image.shape[0]

    def get_cell_image(self, label: int, channel: Union[int, Sequence] = None):
        pass

    def __background_threshold(self, thresdhold: int = 5):
        """Get the background threshold for every channel.
        Params:
        thresdhold: percentile of data not taken into account
        """
        masked = self.image[1:]*(self.mask[None, :, :] == 0)
        bg_threshold = np.zeros(self.channel_number)
        for i in range(0, self.channel_number):
            value = flatten_nonzero_value(masked[i])
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

    def instance_fluorescent_intensity(self, measure_line=90):
        """Get induvidual cell fluorescent intensity.
        Params:
        measure_line: take the percentile measure_line as the intensity
        """
        data = self.mask.instance_properties[["label"]].copy()
        for index in data.index:
            label = data.loc[index, 'label']
            cell_mask = self.mask.get_instance_mask(label)
            for i in range(1, self.channel_number+1):
                chi = flatten_nonzero_value(cell_mask*self.image[i])
                data.loc[index, "ch%d" % i] = np.percentile(chi, measure_line)
        bg = self.get_background()
        for i in range(1, self.channel_number+1):
            data['bg%d' % i] = bg[i-1]
        self.fluorescent_intensity = data
        return data

    def get_fluorescent_intensity(self, **args):
        if self.fluorescent_intensity is None:
            return self.instance_fluorescent_intensity(**args)
        else:
            return self.fluorescent_intensity.iloc[:,0:(self.channel_number*2+1)]

    # def cell_classification(self, **args):
    #     data = self.get_fluorescent_intensity()
    #     clusterobj = FluorescentClassification(data)
    #     pred, _ = clusterobj.predition_data_type(**args)
    #     self.fluorescent_intensity["channel_prediction"] = pred
    #     self.classifier = clusterobj
    #     return self.fluorescent_intensity


# class FluorescentClassification():
#     def __init__(self, data) -> None:
#         self.data = data.copy()
#         self.channel_number = int((data.shape[1]-1)/2)
#         self.model = None
    
#     def normalize_inensity(self):
#         for i in range(1, self.channel_number+1):
#             self.data['ch%d_norm'%i] = np.log(self.data['ch%d'%i])/np.log(self.data['bg%d' %i ])
#         return self.data

#     def predition_data_type(self, **args):
#         self.normalize_inensity()
#         data = self.data[['ch%d_norm'%i for i in range(1, self.channel_number+1)]]
#         clustering = GaussianMixture(**args).fit(data)
#         data_pred = clustering.predict(data)
#         class_map = rename_classes(data, clustering.means_)
#         data_pred = [class_map[x] for x in data_pred]
#         self.model = clustering
#         self.data["channel_prediction"] = data_pred
#         return data_pred, clustering
    
#     # def predition_data_type_2(self, **args):
#     #     self.normalize_inensity()
#     #     data = self.data[['ch%d_norm'%i for i in range(1, self.channel_number+1)]]
#     #     pred = data.copy()
#     #     for i in range(1, self.channel_number+1):
#     #         ch_data = data['ch%d_norm'%i].to_numpy().reshape(-1,1)
#     #         clustering = GaussianMixture(n_components=2, init_params='kmeans').fit(ch_data)
#     #         data_pred = clustering.predict(ch_data)
#     #         class_map = rename_1_ch(clustering.means_)
#     #         data_pred = [class_map[x] for x in data_pred]
#     #         pred['ch%d_norm'%i] = data_pred
#     #     self.data["channel_prediction_2"] = np.sum([pred['ch%d_norm'%i]*2**(i-1) for i in range(1,self.channel_number+1)],axis=0)
#     #     self.model = clustering
#     #     return self.data["channel_prediction_2"], clustering

# # def rename_1_ch(cluster_centers):
# #     if cluster_centers[0]<cluster_centers[1]:
# #         class_map = {0:0,1:1}
# #     else:
# #         class_map = {0:1,1:0}
# #     return class_map


# def rename_classes(data, cluster_centers):
#     coords = __get_bounding_points(data)
#     class_map = {}
#     for i, c in enumerate(cluster_centers):
#         dist = np.sqrt(np.square(coords - c).sum(axis=1))
#         label = np.argmin(dist)
#         class_map[i] = label
#     return class_map


# def __get_bounding_points(data): 
#     box_min = data.min()
#     box_max = data.max()
#     coords = np.array([box_min, box_max]).T
#     import itertools
#     coords_index = np.flip(np.array(list(itertools.product([0, 1], repeat=coords.shape[0]))),axis=1)
#     out = []
#     for i in range(coords.shape[0]):
#         out.append(coords[i, coords_index[:, i]])
#     out = np.array(out).T
#     return out
