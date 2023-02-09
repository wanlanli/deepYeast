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
import numpy as np
import math
from shapely.geometry import Polygon
from filterpy.kalman import KalmanFilter
OVERLAP_VMIN = 0.1
OVERLAP_VMAX = 0.75

DIVISION_LABEL = 3
FUSION_LABEL = 2
np.random.seed(0)

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])#
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_tests, bb_gts):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2,x3,y3,x4,y4]
    """
    o = np.zeros((len(bb_tests),len(bb_gts)))
    for i, bb_test in enumerate(bb_tests):
        bb_test = convert_x_to_bbox(bb_test).reshape((4,2),order='F')
        for j ,bb_gt in enumerate(bb_gts):
            bb_gt = convert_x_to_bbox(bb_gt).reshape((4,2),order='F')
            a_shape = Polygon(bb_test)
            b_shape = Polygon(bb_gt)
            interaction = a_shape.intersection(b_shape).area
            union = a_shape.union(b_shape).area
            o[i,j] = interaction/union
    return (o)

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    # w = bbox[2] - bbox[0]
    # h = bbox[3] - bbox[1]
    # x = bbox[0] + w/2.
    # y = bbox[1] + h/2.
    # s = w * h    #scale is just area
    # r = w / float(h)
    # return np.array([x, y, s, r]).reshape((4, 1))
    x1,x2,_,_,y1,y2,y3,y4 = bbox
    
    x = (x1+x2)/2
    y = (y1+y2)/2
    theta = math.atan((x2-x1)/(y1-y2))
    minor = (x2-x1)/math.sin(theta)
    major = (y3-y4)/math.sin(theta)
    return np.array([x,y,theta,major,minor]).reshape((5,1))

def convert_x_to_bbox(det):
    """
    Takes a bounding box in the centre form [x,y,r,l,s] and returns it in the form
      [x1,x2,x3,x4,y1,y2,y3, y4] where 4 top points of the bounding box.
    """
    x,y,orientation,axis_major_length,axis_minor_length = det[0:5]

    sin = math.sin(orientation)
    cos = math.cos(orientation)

    x1 = x - sin*0.5*axis_minor_length
    y1 = y + cos*0.5*axis_minor_length
    
    x2 = x + sin*0.5*axis_minor_length
    y2 = y - cos*0.5*axis_minor_length
    
    x3 = x + cos*0.5*axis_major_length
    y3 = y + sin*0.5*axis_major_length

    x4 = x - cos*0.5*axis_major_length
    y4 = y - sin*0.5*axis_major_length
    return np.array([x1,x3,x2,x4,y1,y3,y2,y4]).reshape(1,8)


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, input_bbox):
      """
      Initialises a tracker using initial bounding box.[x,y,r,l,s,label]
      """
      bbox = input_bbox[:5]
      self.label = input_bbox[5]
      # define constant velocity model
      self.kf = KalmanFilter(dim_x=10, dim_z=5) 
      self.kf.F = np.array([[1,0,0,0,0,1,0,0,0,0],
                            [0,1,0,0,0,0,1,0,0,0],
                            [0,0,1,0,0,0,0,1,0,0],
                            [0,0,0,1,0,0,0,0,1,0],
                            [0,0,0,0,1,0,0,0,0,1],
                            [0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,1]])
      self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0]])

      self.kf.R[2:,2:] *= 1.
      self.kf.P[5:,5:] *= 1000. #give high uncertainty to the unobservable initial velocities
      self.kf.P *= 10.
      self.kf.Q[-1,-1] *= 100.
      self.kf.Q[5:,5:] *= 100.
      
      self.kf.x[:5] = np.array(bbox).reshape(5,1) # convert_bbox_to_z(bbox)
      self.time_since_update = 0
      self.id = KalmanBoxTracker.count
      KalmanBoxTracker.count += 1
      self.history = []
      self.hits = 0
      self.hit_streak = 0
      self.age = 0

    def update(self, input_bbox):
      """
      Updates the state vector with observed bbox.
      """
      bbox = input_bbox[:5]
      self.label = input_bbox[5]
      self.time_since_update = 0
      self.history = []
      self.hits += 1
      self.hit_streak += 1
      self.kf.update(np.array(bbox).reshape(5,1))# convert_bbox_to_z(bbox))

    def predict(self):
      """
      Advances the state vector and returns the predicted bounding box estimate.
      """
      # if((self.kf.x[6]+self.kf.x[2])<=0):
      #   self.kf.x[6] *= 0.0
      self.kf.predict()
      self.age += 1
      if(self.time_since_update>0):
        self.hit_streak = 0
      self.time_since_update += 1
      # self.history.append(convert_x_to_bbox(self.kf.x))
      self.history.append(np.array(self.kf.x[:5]).reshape(1,5))
      return self.history[-1]

    def get_state(self):
      """
      Returns the current bounding box estimate.
      """
      return self.kf.x[:5].reshape(1,5) #convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  # print(detections.shape, trackers.shape)
  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
      """
      Sets key parameters for SORT
      """
      self.max_age = max_age
      self.min_hits = min_hits
      self.iou_threshold = iou_threshold
      self.trackers = []
      self.frame_count = 0

  def update(self, input_dets=np.empty((0, 6))):
    """
    Params:
      input_dets - a numpy array of detections in the format [[x1,y1,x2,y2,score,label],[x1,y1,x2,y2,score,label],...]
      Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
      Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    dets = input_dets[:,:6]
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 6))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4],0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(input_dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(input_dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    # clean unmached trk
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1, trk.label])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))


def action_iou_batch(bb_tests, bb_gts):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2,x3,y3,x4,y4]
    """
    o = np.zeros((len(bb_tests),len(bb_gts)))
    for i, bb_test in enumerate(bb_tests):
        for j, bb_gt in enumerate(bb_gts):
            a_shape = Polygon(bb_test)
            b_shape = Polygon(bb_gt)
            intersection = a_shape.intersection(b_shape).area
            union = a_shape.union(b_shape).area
            x = a_shape.area
            # print(intersection/union)
            if (intersection/union) < OVERLAP_VMIN:
                # no overlap
                o[i, j] = 0
            elif intersection/union > OVERLAP_VMAX:
                # matched cell
                o[i, j] = 1
            elif (intersection/x > OVERLAP_VMAX) & (x/union < OVERLAP_VMAX):
                # fusion
                o[i, j] = FUSION_LABEL
            elif (x/union > OVERLAP_VMAX) & (intersection/x < OVERLAP_VMAX):
                # division
                o[i, j] = DIVISION_LABEL
            else:
                o[i, j] = 4
    return o


def action_dicision(cost):
    """Divided/fusioned cell match
    """
    connection = {}
    for x in range(0, cost.shape[0]):
        if np.sum(cost[x, :] == 1) == 1:
            y = np.where(cost[x, :] == 1)[0]
            connection[x] = y
    division = {}
    for x in range(0, cost.shape[0]):
        if np.sum(cost[x, :] == DIVISION_LABEL) == 2:
            y = np.where(cost[x, :] == DIVISION_LABEL)[0]
            division[x] = y
    fusion = {}
    for y in range(0, cost.shape[1]):
        if np.sum(cost[:, y] == FUSION_LABEL) == 2:
            x = np.where(cost[:, y] == FUSION_LABEL)[0]
            fusion[y] = x
    return connection, division, fusion
