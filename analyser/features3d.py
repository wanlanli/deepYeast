from .mask_feature import CellSignal
from .tracer import CellTracer
import os
import numpy as np
from .config import PROP_NAMES
import pandas as pd
from tqdm import trange


class Cell3D():
    def __init__(self, img: np.array, mask: np.array, is_tracing=True) -> None:
        self.img = img
        self.mask = mask

        # init tracing
        self.tracer = CellTracer(self.mask)
        if is_tracing:
            self.tracer.tracing()
        self.number = self.tracer.cell_number
        # init propers
        self.slicer = None
        self._init_frame_features()
        self.frame_number = self.img.shape[0]

        # overall cell features
        self.props = None
        # overall cell distances
        self.distance = None


class CellSignal3D():
    """Measure video properties.

    Parameters
    ----------
    img : 4D matrix over time,dtype:int
        time point x channels x width x height. The first channel most be
        reference channel.
    mask: 3D matrix,dtype:int
        predicted mask.
    """
    def __init__(self, img: np.array, mask: np.array, is_tracing=True) -> None:
        self.img = img
        self.mask = mask

        # init tracing
        self.tracer = CellTrace(self.mask)
        self.cell_number = None
        if is_tracing:
            self.tracer.tracing()
            self.cell_number = self.tracer.cell_number

        # init propers
        self.slicer = None
        self._init_frame_features()
        self.frame_number = self.img.shape[0]

        # overall cell features
        self.props = None
        # overall cell distances
        self.distance = None

    def _init_frame_features(self):
        if self.slicer is None:
            features_3d = {}
            for frame in range(0, self.img.shape[0]):
                cell_frame = CellSignal(self.img[frame], self.mask[frame])
                features_3d[frame] = cell_frame
            self.slicer = features_3d

    def _distance_2d_to_3d(self, frame):
        mask = self.slicer[frame].mask
        frame_cell_labels = mask.labels
        label_map = self.tracer.trace.iloc[:, [frame]].copy()
        label_map['ids'] = label_map.index
        label_map = label_map.set_index("frame_%03d" % frame)
        cost = mask.all_by_all_distance()

        for i in range(0, cost.shape[0]):
            label_x = frame_cell_labels[cost.iloc[i, 0]]
            if label_x not in label_map.index:
                continue
            label_y = frame_cell_labels[cost.iloc[i, 1]]
            if label_y not in label_map.index:
                continue
            cell_id_x = label_map.loc[label_x, 'ids']
            cell_id_y = label_map.loc[label_y, 'ids']
            if cell_id_x > cell_id_y:
                a = cell_id_x
                cell_id_x = cell_id_y
                cell_id_y = a
            self.distance[frame, :, cell_id_x, cell_id_y] = cost.iloc[i, [2,3]].values

    def __init_3d_features(self):
        self.distance = np.zeros([self.frame_number, 2, self.cell_number, self.cell_number])
        self.distance[:, :, :, :] = -1
        self.props = np.zeros([self.cell_number, len(PROP_NAMES), self.frame_number])

    def run_pipeline(self):
        self.__init_3d_features()
        for frame in trange(0, self.tracer.frame_number):
            maskfeature = self.slicer[frame].mask
            data = maskfeature.region.set_index('label', drop=False)
            # for not exist labels, save as 0
            data.loc[0] = 0
            data.loc[-1] = 0
            data = data.loc[:, PROP_NAMES]
            label_list = self.tracer.trace.iloc[:, frame]
            # save region properties as tracer order
            self.props[:, :, frame] = data.loc[label_list]
            # store distance matrix as tracer order
            self._distance_2d_to_3d(frame)
        # return self.props, self.distance

    def save_result_to_disk(self, path, name):
        self.tracer.trace_feature.to_csv(os.path.join(path, name+"_trace_feature.csv"))
        self.tracer.trace.to_csv(os.path.join(path, name+"_trace.csv"))
        np.save(os.path.join(path, name+"_3dfeature"), self.props)
        np.save(os.path.join(path, name+"_3ddistance"), self.distance)

    def load_result_from_disk(self, path, name):
        self.slicer = self._init_frame_features()
        self.tracer = CellTrace(self.mask)
        self.tracer.trace = pd.read_csv(os.path.join(path, name+"_trace.csv"), index_col=0)
        self.tracer.trace_feature = pd.read_csv(os.path.join(path, name+"_trace_feature.csv"), index_col=0)
        self.cell_number = self.tracer.trace.shape[0]

        self.props = np.load(os.path.join(path, name+"_3dfeature.npy"), allow_pickle=True)
        self.distance = np.load(os.path.join(path, name+"_3ddistance.npy"), allow_pickle=True)