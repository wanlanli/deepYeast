import pandas as pd
import numpy as np

from analyser.image_measure.meaure import ImageMeasure
from analyser.cell.cell import Cell


FLAG_DIVISION = 1
FLAG_FUSION = 2
FLAG_OTHER = 3
IOU_THRESHOLD = 0.3
OVERLAP_THRESHOLD = 0.7


def step_1(traced_image):
    """Go through all cells
    """
    cell = {}
    for i in range(0, traced_image.shape[0]):
        # labels = np.unique(traced_image[i])[1:]
        img = traced_image[i]
        maskobj = ImageMeasure(img)
        labels = maskobj.instance_properties.label
        for j in labels:
            if j not in cell.keys():
                cell[j] = Cell(j)
            cell[j].update(i, maskobj.instance_property(label=j)[1:])
    return cell


def step_2(cell, life_span=5, shape=(150, 150)):
    """screen cells
    """
    data = pd.DataFrame(columns=["start", "end"])
    for i in cell.keys():
        # if cell[i].is_out_of_border(shape):
        #     continue
        # else:
        cell[i].frames = list(cell[i].features.index)
        cell[i].start = cell[i].features.index.min()
        cell[i].end = cell[i].features.index.max()
        data.loc[i] = [cell[i].start, cell[i].end]
    data["life"] = data["end"] - data["start"]+1
    data = data[data.life > life_span]
    return data


def fusion_iou(x, y):
    interaction = np.sum(x & y)
    iou = interaction/np.sum(x | y)
    overlap_i = interaction/np.sum(x)
    overlap_j = interaction/np.sum(y)
    overlap = max(overlap_i, overlap_j)
    return [iou, overlap, overlap_i, overlap_j]


def judge_status(iou, overlap, overlap_i, overlap_j,
                 iou_threshold, overlap_threshold):
    if (iou > iou_threshold) & (overlap > overlap_threshold):
        if overlap_i > overlap_j:
            return FLAG_FUSION
        else:
            return FLAG_DIVISION
    else:
        return FLAG_OTHER


def step_3(cell, filter, traced_image):
    """fusion & division map
    """
    fusion_data = pd.DataFrame(0, index=filter.index, columns=filter.index)
    for i in filter.index:
        next_id = filter.loc[(filter.start > filter.loc[i].end-3) &
                             (filter.start < filter.loc[i].end+3)]
        for j in next_id.index:
            c_i = traced_image[cell[i].end, :, :] == i
            c_j = traced_image[cell[j].start, :, :] == j
            iou, overlap, overlap_i, overlap_j = fusion_iou(c_i, c_j)
            fusion_data.loc[i, j] = judge_status(iou,
                                                 overlap,
                                                 overlap_i,
                                                 overlap_j,
                                                 IOU_THRESHOLD,
                                                 OVERLAP_THRESHOLD)
    return fusion_data


def step_4(cell, fusion_data):
    """connection
    """
    for i in fusion_data.index:
        if np.sum(fusion_data.loc[i] == FLAG_DIVISION) == 2:
            # fusion_data.loc[fusion_data.loc[i] == FLAG_DIVISION]
            # print("division")
            a, b = fusion_data.loc[i, fusion_data.loc[i] == FLAG_DIVISION].index

            cell[i].division = True
            cell[i].daughter_vg = [a, b]

            cell[a].generation = cell[i].generation+1
            cell[a].ancient = i
            cell[a].sister = b

            cell[b].generation = cell[i].generation+1
            cell[b].ancient = i
            cell[b].sister = a

        elif np.sum(fusion_data.loc[:, i] == FLAG_FUSION) == 2:
            # fusion_data.loc[fusion_data.loc[i] == FLAG_DIVISION]
            a, b = fusion_data.loc[fusion_data.loc[:, i] == FLAG_FUSION, i].index

            cell[i].parents = [a, b]
            cell[i].generation = cell[a].generation+1

            cell[a].fusion = True
            cell[a].daughter = i
            cell[a].spouse = b

            cell[b].fusion = True
            cell[b].daughter = i
            cell[b].spouse = a
    return cell


def generate_cells(traced_image):
    init_cells = step_1(traced_image)
    filter_data = step_2(init_cells, shape=traced_image.shape[1:3])
    cells = {}
    for key in filter_data.index:
        cells[key] = init_cells[key]
    fusion_data = step_3(cells, filter_data, traced_image)
    cells = step_4(cells, fusion_data)
    return cells


def clean_traced_image(cells_id, traced_image):
    cleaned = traced_image.copy()
    labels = np.unique(traced_image)[1:]
    for label in labels:
        if label not in cells_id:
            cleaned[cleaned == label] = 0
    return cleaned
