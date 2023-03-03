import matplotlib.pyplot as plt


def plot_cells(image, ct=None, ot=None, tip=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image, 'gray')
    if ct is not None:
        ax.scatter(ct[:, 1], ct[:, 0], c='y')
    if ot is not None:
        for i in range(0, ot.shape[0]):
            ax.plot(ot[i][1], ot[i][0], label=i)
        ax.legend()
    if tip is not None:
        for i in range(0, tip.shape[0]):
            ax.scatter(tip[i, :, 1], tip[i, :, 0], label=i, marker='*')

# def plot_fusion_cells()
#     for i in range(son.shape[0]):
#         data = son.iloc[i]
#         mother = int(data.mother)
#         father = int(data.father)
#         frame = int(data.start_time)
        
#         common_first_time = int(max(ter.obj_property.loc[[mother, father], 'start_time']))
#         common_last_time = frame - 1
        
#         label = ter.trace_calendar.loc[mother][common_first_time]
#         label_y = ter.trace_calendar.loc[father][common_first_time]
#         x_list = ter.maskobj[common_first_time].nearnest_radius(label, 90)
        
#         ct = ter.maskobj[common_first_time].get_centers(x_list).values
#         ot = ter.maskobj[common_first_time].get_outline(x_list).values
#         # tip = np.array([ot[0][:,t_x], ot[1][:,t_y]])
#         # tip = np.array([[ot[0][:,t_x[i]], ot[i+1][:,t_y[i]]] for i in range(0, len(y_label))])

#         img = ter.maskobj[common_first_time].get_cells(x_list)
#         img = img + ter.maskobj[common_first_time].get_cells([label])*2
#         img = img + ter.maskobj[common_first_time].get_cells([label_y])
#         plot_cells(img, ct, ot)
