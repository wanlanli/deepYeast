import math
import numpy as np
import matplotlib.pyplot as plt


def main_axis(center, ori, length):
    orit = np.zeros((center.shape[0], 4))
    for i in range(0, center.shape[0]):
        y0 = center[i, 0]
        x0 = center[i, 1]
        orientation = ori[i]
        axis_major_length = length[i]
        x2 = x0 - math.sin(orientation) * 0.5 * axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * axis_major_length
        x3 = x0 + math.sin(orientation) * 0.5 * axis_major_length
        y3 = y0 + math.cos(orientation) * 0.5 * axis_major_length
        orit[i, 1] = x2
        orit[i, 0] = y2
        orit[i, 3] = x3
        orit[i, 2] = y3
    return orit


def plot_cells_with_angle(img_m, a, b):
    p_t, p_s, angel_x, angel_y = img_m.two_regions_angle(a,b)
    # ca = img_m.centers(index=a)
    # cb = img_m.centers(index=b)

    data = img_m.instance_properties.iloc[[a,b]][[
        'centroid_0', 'centroid_1', 'orientation', 'axis_major_length']]
    axis_p = main_axis(np.array(data.iloc[:, 0:2]), list(data.iloc[:, 2]), list(data.iloc[:, 3]))

    print(angel_x*180/np.pi, angel_y*180/np.pi,)
    plt.imshow(img_m.instance_mask(index=a) | img_m.instance_mask(index=b))
    plt.scatter(p_t[1],p_t[0], c='g')
    plt.scatter(p_s[1],p_s[0], c='r')
    for i in range(0, axis_p.shape[0]):
        plt.plot([axis_p[i, 3],axis_p[i,1]], [axis_p[i, 2],axis_p[i,0]], c='w', linestyle='--')
