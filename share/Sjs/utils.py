import os

import numpy as np
from skimage.io import imread
from skimage.color import label2rgb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from analysis.multi_fluorescent_image_feature import FluorescentImage, FluorescentClassification
from analysis.utils import file_traverse
from postprocess.post_process_utils import post_process_panoptic

LABELMAP = {0:[0], 1:[1, 3], 2:[2]}
LABELCOLOR = {0:'w', 1:'r', 2:'g'}

def load_model(model_dir, config_file):
    import yaml
    from config_yml import ExperimentOptions
    from trainer.train import DeepCellModule
    mode = 'test'
    num_gpus=1
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    configs = ExperimentOptions(config)
    configs.model_options.backbone.drop_path_keep_prob=1
    cellmodel = DeepCellModule(mode, model_dir, configs, num_gpus)
    return cellmodel


def load_image(image_path):
    """Read image and adjust the position of the image channels, the reference channel is at the first place, and the fluorescence images are at the back
    """
    image = imread(image_path)
    image = np.moveaxis(image, -1, 0)
    image[:,:,:] = image[[2,0,1],:,:]
    return image


def plot_result(image, data, mask=None, sub_figsize=5, basename=''):
    f = sns.jointplot(data=data, x="ch1_norm", y="ch2_norm",  hue='channel_prediction')
    f.savefig('tmp.png')
    plt.close(f.fig)
    fig, axs = plt.subplots(2, 3, figsize=(sub_figsize*3, sub_figsize*2))
    for i in range(0, image.shape[0]):
        axs[0, i].imshow(image[i], cmap="gray")
        show_data =  data.loc[data.channel_prediction.isin(LABELMAP[i])]
        color = LABELCOLOR[i]
        axs[0, i].scatter(show_data['centroid_1'], show_data['centroid_0'],c=color,marker='.')
        axs[0, 0].scatter(show_data['centroid_1'], show_data['centroid_0'],c=color,marker='.')    
    if mask:
        axs[1, 0].imshow(label2rgb(mask))
    axs[1, 1].imshow(imread('tmp.png')) 

    sum_table = __overview_table(data)
    table = axs[1, 2].table(cellText=sum_table.values,
                            rowLabels=sum_table.index,
                            colLabels=sum_table.columns,
                            loc='center')
    table.set_fontsize(20)
    table.scale(0.5, 3) 
    
    # set title
    fig.suptitle(basename)    
    axs[0, 0].set_title("Reference")
    axs[0, 1].set_title("CH1")
    axs[0, 2].set_title("CH2")
    axs[1, 0].set_title("Mask")
    axs[1, 1].set_title("Clustering")
    # close axis
    for i in range(0, 2):
        for j in range(0, 3):
            axs[i, j].axis("off")
    return fig, sum_table


def __overview_table(data):
    """Convert stats to tabular form.
    """
    columns = ['cell', 'tetrad']
    sum_table = pd.DataFrame(index=LABELMAP.keys(), columns=columns)
    for i, key in LABELMAP.items():
        for j in range(0, len(columns)):
            if j == 0:
                sum_table.loc[i, columns[j]] = sum((data.channel_prediction.isin(key)) & (data.semantic.isin([1,2])))
            else:
                sum_table.loc[i, columns[j]] = sum((data.channel_prediction.isin(key)) & (data.semantic.isin([3,4])))
    sum_table.index = ['gray', 'green', 'red']
    return sum_table


def batch_processing(root_dir, model_dir, config_file, save_path, area_threshold=1000):
    model = load_model(model_dir, config_file)
    file_list = file_traverse(root_dir, file_regular=r".*.tif$")
    all_tabel = pd.DataFrame(columns=['cell_gray','tetrad_gray','cell_green','terad_green','cell_red','tetrad_red'],)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path,'figs')):
        os.mkdir(os.path.join(save_path,'figs'))
    for file in tqdm(file_list):
        image = load_image(file)
        basename = os.path.basename(file)[:-4]
        output = model.predict(image[0])
        mask = post_process_panoptic(output["panoptic_pred"][0].numpy(), area_threshold=area_threshold)
        fimageobj = FluorescentImage(image, mask)
        data = fimageobj.cell_classification(n_components=4, init_params='kmeans')
        data[['centroid_0','centroid_1','semantic']] = fimageobj.mask.instance_properties[['centroid_0','centroid_1','semantic']]
        fig, tabel = plot_result(image, data, basename=basename)
        all_tabel.loc[basename] =tabel.values.flatten()
        all_tabel.to_csv(os.path.join(save_path, "summary_table"+".csv"))
        fig.savefig(os.path.join(save_path,'figs', basename+'.png'))

