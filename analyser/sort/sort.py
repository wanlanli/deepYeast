"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import absolute_import, division

import numpy as np


from analyser.sort.kalman_filter import KalmanFilter


np.random.seed(0)
DIMENTION = 5


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)


def area_ratio_batch(ar_test, ar_gt):
    return np.expand_dims(ar_test, 1)/np.expand_dims(ar_gt, 0)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, input_bbox, dimention=DIMENTION):
        """
        Initialises a tracker using initial bounding box.[x,y,r,l,s,label]
        """
        self.dimention = dimention
        bbox = input_bbox[:dimention]
        self.label = input_bbox[dimention]
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=dimention*2, dim_z=dimention)
        self.kf.F = self._init_f(dimention*2, dimention)
        self.kf.H = self._init_h(dimention*2, dimention)
        self.kf.R[2:, 2:] *= 1.
        self.kf.P[dimention:, dimention:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 100.
        self.kf.Q[dimention:, dimention:] *= 100.

        self.kf.x[:dimention] = np.array(bbox).reshape(dimention, 1)  # convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def _init_f(self, dim_x, dim_z):
        kf_f = np.zeros((dim_x, dim_x))
        kf_f[np.arange(dim_x), np.arange(dim_x)] = 1
        kf_f[np.arange(0, dim_z), np.arange(dim_z, dim_x)] = 1
        return kf_f

    def _init_h(self, dim_x, dim_z):
        kf_h = np.zeros((dim_z, dim_x))
        kf_h[np.arange(0, dim_z), np.arange(0, dim_z)] = 1
        return kf_h

    def update(self, input_bbox):
        """
        Updates the state vector with observed bbox.
        """
        bbox = input_bbox[:self.dimention]
        self.label = input_bbox[self.dimention]
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(np.array(bbox).reshape(self.dimention, 1))  # convert_bbox_to_z(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(np.array(self.kf.x[:self.dimention]).reshape(1, self.dimention))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:self.dimention].reshape(1, self.dimention)  # convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,
                                     trackers,
                                     iou_threshold=0.3,
                                     area_threshold=0.5):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, DIMENTION), dtype=int)

    # print(detections.shape, trackers.shape)
    iou_matrix = iou_batch(detections[:, 0:4], trackers[:, :4])
    area_ratio_matrix = area_ratio_batch(detections[:, 4], trackers[:, 4])
    # print(iou_matrix.shape)
    # print(area_ratio_matrix.shape)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if ((iou_matrix[m[0], m[1]] >= iou_threshold)
            and (area_ratio_matrix[m[0], m[1]] >= area_threshold)
            and (area_ratio_matrix[m[0], m[1]] <= 1/area_threshold)):
            matches.append(m.reshape(1, 2))
        else:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        # if (iou_matrix[m[0], m[1]] < iou_threshold) or (area_ratio_matrix < area_threshold):
        #     unmatched_detections.append(m[0])
        #     unmatched_trackers.append(m[1])
        # else:
        #     matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, area_threshold=0.5):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.area_threshold = area_threshold
        self.trackers = []  # traced objects at frame `frame_count`
        self.frame_count = 0

    def update(self, input_dets=np.empty((0, DIMENTION+1))):
        """
        Params:
          input_dets - a numpy array of detections in the format [[x1,y1,x2,y2,score,label],[x1,y1,x2,y2,score,label],...]
          Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
          Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        dets = input_dets[:, :DIMENTION+1]
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks_pred = np.zeros((len(self.trackers), DIMENTION+1))
        to_del = []
        ret = []
        for t, trk in enumerate(trks_pred):
            # print(t, trk)
            pos = self.trackers[t].predict()[0]
            trk[:DIMENTION] = pos
            trk[DIMENTION] = 0
            # [pos[0], pos[1], pos[2], pos[3], pos[4], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks_pred = np.ma.compress_rows(np.ma.masked_invalid(trks_pred))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks_pred, self.iou_threshold, self.area_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(input_dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(input_dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)

        # clean unmached trk
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1, trk.label])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            # new object missed in the begin frame modify
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, DIMENTION))


def run_tracing(images: np.array, **args):
    """
    """
    mot_tracker = Sort(**args)  # create instance of the SORT tracker
    KalmanBoxTracker.count = 0
    total_frames = 0
    output_d = []
    from analyser.image_measure.meaure import ImageMeasure
    for frame in range(0, images.shape[0]):
        img = images[frame]
        maskobj = ImageMeasure(img)
        dets = maskobj.instance_properties.iloc[:, list(range(7, 11))+[6]+[0]].values
        total_frames += 1
        trackers = mot_tracker.update(np.array(dets))
        for d in trackers:
            output_d.append([frame]+list(d))
    output_d = np.array(output_d).astype(np.uint16)
    traced_image = _asgine_feature(output_d, images).astype(np.uint16)
    return output_d, traced_image


def _asgine_feature(output_d: np.array, data):
    traced_image = np.zeros(data.shape)
    for i in range(output_d.shape[0]):
        new_id = int(output_d[i, -2])
        frame = int(output_d[i, 0])
        label = int(output_d[i, -1])
        traced_image[frame][data[frame] == label] = new_id
    return traced_image


if __name__ == "__main__":
    from skimage.io import imread
    images = imread("/home/wli6/006_0015.tif").astype(np.uint16)[:, :, :, 3]
    output_d, traced_image = run_tracing(images)
