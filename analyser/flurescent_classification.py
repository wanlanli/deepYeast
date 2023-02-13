import numpy as np
from sklearn.mixture import GaussianMixture


class FluorescentClassification():
    def __init__(self, data) -> None:
        self.data = data.copy()
        self.channel_number = int((data.shape[1]-1)/2)
        self.model = None
    
    def normalize_inensity(self):
        for i in range(1, self.channel_number+1):
            self.data['ch%d_norm'%i] = np.log(self.data['ch%d'%i])/np.log(self.data['bg%d' %i ])
        return self.data

    def predition_data_type(self, **args):
        self.normalize_inensity()
        data = self.data[['ch%d_norm'%i for i in range(1, self.channel_number+1)]]
        clustering = GaussianMixture(**args).fit(data)
        data_pred = clustering.predict(data)
        class_map = rename_classes(data, clustering.means_)
        data_pred = [class_map[x] for x in data_pred]
        self.model = clustering
        self.data["channel_prediction"] = data_pred
        return data_pred, clustering
    
    # def predition_data_type_2(self, **args):
    #     self.normalize_inensity()
    #     data = self.data[['ch%d_norm'%i for i in range(1, self.channel_number+1)]]
    #     pred = data.copy()
    #     for i in range(1, self.channel_number+1):
    #         ch_data = data['ch%d_norm'%i].to_numpy().reshape(-1,1)
    #         clustering = GaussianMixture(n_components=2, init_params='kmeans').fit(ch_data)
    #         data_pred = clustering.predict(ch_data)
    #         class_map = rename_1_ch(clustering.means_)
    #         data_pred = [class_map[x] for x in data_pred]
    #         pred['ch%d_norm'%i] = data_pred
    #     self.data["channel_prediction_2"] = np.sum([pred['ch%d_norm'%i]*2**(i-1) for i in range(1,self.channel_number+1)],axis=0)
    #     self.model = clustering
    #     return self.data["channel_prediction_2"], clustering


def rename_classes(data, cluster_centers):
    coords = __get_bounding_points(data)
    class_map = {}
    for i, c in enumerate(cluster_centers):
        dist = np.sqrt(np.square(coords - c).sum(axis=1))
        label = np.argmin(dist)
        class_map[i] = label
    return class_map

def __get_bounding_points(data): 
    box_min = data.min()
    box_max = data.max()
    coords = np.array([box_min, box_max]).T
    import itertools
    coords_index = np.flip(np.array(list(itertools.product([0, 1], repeat=coords.shape[0]))),axis=1)
    out = []
    for i in range(coords.shape[0]):
        out.append(coords[i, coords_index[:, i]])
    out = np.array(out).T
    return out