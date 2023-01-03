
# # import tensorflow as tf
# # import functools
# # import sys
# # sys.path.append("./models/")
# # import orbit
import os
import re
# # from deeplab2.trainer.train_lib import create_deeplab_model
# # from deeplab2.model.loss import loss_builder
# # from deeplab2.trainer import evaluator as evaluator_lib
# # from deeplab2.trainer import runner_utils
# # from deeplab2.data import dataset
# # from absl import logging
# # from google.protobuf import text_format
# # from deeplab2 import config_pb2
# # from deeplab2.trainer import distribution_utils
# # import SimpleITK as sitk
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from skimage.measure import regionprops
# # from scipy import ndimage as ndi
# # from skimage.morphology import binary_erosion, binary_dilation, remove_small_objects
# # from skimage.util import img_as_ubyte
# # from scipy.optimize import linear_sum_assignment
# # from skimage.measure import regionprops_table
# # from skimage.color import label2rgb
# # import pandas as pd


# color_dict = {0:"b",1:"g",2:"r"}

# # class CellSegModel():
# #     def __init__(self, config_files, mode, flags_model_dir,num_gpus) -> None:
# #         logging.info('Reading the config file.')
# #         with tf.io.gfile.GFile(config_files, 'r') as proto_file:
# #             self.config = text_format.ParseLines(proto_file, config_pb2.ExperimentOptions())

# #         combined_model_dir = os.path.join(flags_model_dir, self.config.experiment_name)
# #         self.model_dir = combined_model_dir

# #         self.strategy = distribution_utils.create_strategy(None, num_gpus)
# #         logging.info('Using strategy %s with %d replicas', type(self.strategy),
# #                 self.strategy.num_replicas_in_sync)
# #         self.mode = mode
# #         if 'eval' in self.mode:
# #             self.dataset_name = self.config.eval_dataset_options.dataset
# #         if num_gpus > 1:
# #             raise ValueError(
# #                 'Using more than one GPU for evaluation is not supported.')
# #         else:
# #             self.dataset_name = self.config.train_dataset_options.dataset

# #         self.num_classes = dataset.MAP_NAME_TO_DATASET_INFO[self.dataset_name].num_classes
# #         self.ignore_label = dataset.MAP_NAME_TO_DATASET_INFO[self.dataset_name].ignore_label
# #         self.ignore_depth = dataset.MAP_NAME_TO_DATASET_INFO[self.dataset_name].ignore_depth
# #         self.class_has_instances_list = (
# #             dataset.MAP_NAME_TO_DATASET_INFO[self.dataset_name].class_has_instances_list)
    
# #     def load_model(self):
# #         trainer = None
# #         evaluator = None
# #         with self.strategy.scope():
# #             deeplab_model = create_deeplab_model(
# #                 self.config, dataset.MAP_NAME_TO_DATASET_INFO[self.dataset_name])
# #             losses = loss_builder.DeepLabFamilyLoss(
# #                 loss_options=self.config.trainer_options.loss_options,
# #                 num_classes=self.num_classes,
# #                 ignore_label=self.ignore_label,
# #                 ignore_depth=self.ignore_depth,
# #                 thing_class_ids=self.class_has_instances_list)
# #             global_step = orbit.utils.create_global_step()
# #             # if 'train' in mode:
# #             #       trainer = trainer_lib.Trainer(config, deeplab_model, losses, global_step)
# #             if 'eval' in self.mode:
# #                 evaluator = evaluator_lib.Evaluator(self.config, deeplab_model, losses,
# #                                             global_step, self.model_dir)
        
# #         checkpoint_dict = dict(global_step=global_step)
# #         checkpoint_dict.update(deeplab_model.checkpoint_items)
# #         if trainer is not None:
# #             checkpoint_dict['optimizer'] = trainer.optimizer
# #             if trainer.backbone_optimizer is not None:
# #                 checkpoint_dict['backbone_optimizer'] = trainer.backbone_optimizer
# #         checkpoint = tf.train.Checkpoint(**checkpoint_dict)
        
        
# #         # Define items to load from initial checkpoint.
# #         init_dict = deeplab_model.checkpoint_items
# #         if (not self.config.model_options
# #             .restore_semantic_last_layer_from_initial_checkpoint):
# #             del init_dict[common.CKPT_SEMANTIC_LAST_LAYER]
# #         if (not self.config.model_options
# #             .restore_instance_last_layer_from_initial_checkpoint):
# #             for layer_name in _INSTANCE_LAYER_NAMES:
# #                 if layer_name in init_dict:
# #                     del init_dict[layer_name]
# #         init_fn = functools.partial(runner_utils.maybe_load_checkpoint,
# #                                 self.config.model_options.initial_checkpoint,
# #                                 init_dict)

# #         checkpoint_manager = tf.train.CheckpointManager(
# #         checkpoint,
# #         directory=self.model_dir,
# #         max_to_keep=self.config.trainer_options.num_checkpoints_to_keep,
# #         step_counter=global_step,
# #         checkpoint_interval=self.config.trainer_options.save_checkpoints_steps,
# #         init_fn=init_fn)

# #         controller = orbit.Controller(
# #         strategy=self.strategy,
# #         trainer=None,#trainer,
# #         evaluator=evaluator,
# #         global_step=global_step,
# #         steps_per_loop=self.config.trainer_options.steps_per_loop,
# #         checkpoint_manager=checkpoint_manager,
# #         summary_interval=self.config.trainer_options.save_summaries_steps,
# #         summary_dir=os.path.join(self.model_dir, 'train'),
# #         eval_summary_dir=os.path.join(self.model_dir, 'eval'))
# #         self.model = deeplab_model
# #         return deeplab_model


# #     def refresh_model(self):
# #         self.model = self.load_model()
# #         return self.model

# # def load_model(config_files, mode, flags_model_dir,num_gpus):
# #     ## loading configs
# #     logging.info('Reading the config file.')
# #     with tf.io.gfile.GFile(config_files, 'r') as proto_file:
# #         config = text_format.ParseLines(proto_file, config_pb2.ExperimentOptions())

# #     combined_model_dir = os.path.join(flags_model_dir, config.experiment_name)
# #     model_dir = combined_model_dir

# #     strategy = distribution_utils.create_strategy(None, num_gpus)
# #     logging.info('Using strategy %s with %d replicas', type(strategy),
# #                strategy.num_replicas_in_sync)

# #     if 'eval' in mode:
# #         dataset_name = config.eval_dataset_options.dataset
# #     if num_gpus > 1:
# #           raise ValueError(
# #               'Using more than one GPU for evaluation is not supported.')
# #     else:
# #         dataset_name = config.train_dataset_options.dataset

# #     num_classes = dataset.MAP_NAME_TO_DATASET_INFO[dataset_name].num_classes
# #     ignore_label = dataset.MAP_NAME_TO_DATASET_INFO[dataset_name].ignore_label
# #     ignore_depth = dataset.MAP_NAME_TO_DATASET_INFO[dataset_name].ignore_depth
# #     class_has_instances_list = (
# #       dataset.MAP_NAME_TO_DATASET_INFO[dataset_name].class_has_instances_list)

# #     trainer = None
# #     evaluator = None
# #     with strategy.scope():
# #         deeplab_model = create_deeplab_model(
# #             config, dataset.MAP_NAME_TO_DATASET_INFO[dataset_name])
# #         losses = loss_builder.DeepLabFamilyLoss(
# #             loss_options=config.trainer_options.loss_options,
# #             num_classes=num_classes,
# #             ignore_label=ignore_label,
# #             ignore_depth=ignore_depth,
# #             thing_class_ids=class_has_instances_list)
# #         global_step = orbit.utils.create_global_step()
# #         # if 'train' in mode:
# #         #       trainer = trainer_lib.Trainer(config, deeplab_model, losses, global_step)
# #         if 'eval' in mode:
# #               evaluator = evaluator_lib.Evaluator(config, deeplab_model, losses,
# #                                           global_step, model_dir)
    
# #     checkpoint_dict = dict(global_step=global_step)
# #     checkpoint_dict.update(deeplab_model.checkpoint_items)
# #     if trainer is not None:
# #         checkpoint_dict['optimizer'] = trainer.optimizer
# #         if trainer.backbone_optimizer is not None:
# #             checkpoint_dict['backbone_optimizer'] = trainer.backbone_optimizer
# #     checkpoint = tf.train.Checkpoint(**checkpoint_dict)
    
    
# #     # Define items to load from initial checkpoint.
# #     init_dict = deeplab_model.checkpoint_items
# #     if (not config.model_options
# #         .restore_semantic_last_layer_from_initial_checkpoint):
# #         del init_dict[common.CKPT_SEMANTIC_LAST_LAYER]
# #     if (not config.model_options
# #         .restore_instance_last_layer_from_initial_checkpoint):
# #         for layer_name in _INSTANCE_LAYER_NAMES:
# #             if layer_name in init_dict:
# #                 del init_dict[layer_name]
# #     init_fn = functools.partial(runner_utils.maybe_load_checkpoint,
# #                               config.model_options.initial_checkpoint,
# #                               init_dict)

# #     checkpoint_manager = tf.train.CheckpointManager(
# #       checkpoint,
# #       directory=model_dir,
# #       max_to_keep=config.trainer_options.num_checkpoints_to_keep,
# #       step_counter=global_step,
# #       checkpoint_interval=config.trainer_options.save_checkpoints_steps,
# #       init_fn=init_fn)

# #     controller = orbit.Controller(
# #       strategy=strategy,
# #       trainer=None,#trainer,
# #       evaluator=evaluator,
# #       global_step=global_step,
# #       steps_per_loop=config.trainer_options.steps_per_loop,
# #       checkpoint_manager=checkpoint_manager,
# #       summary_interval=config.trainer_options.save_summaries_steps,
# #       summary_dir=os.path.join(model_dir, 'train'),
# #       eval_summary_dir=os.path.join(model_dir, 'eval'))
# #     return deeplab_model


# # def calculate_loss(model, inputs, outputs):
# #     loss_dict = model._loss(inputs, outputs)
# #     return loss_dict


# # def save_array_as_png(data, name):
# #     sitk.WriteImage(sitk.GetImageFromArray(data.astype(np.uint16)), name)


# # def load_image(path):
# #     image = sitk.GetArrayFromImage(sitk.ReadImage(path))
# #     return image


def file_traverse(file_path, file_regular=r'.*', **kwarg):
    """
    Parameters
    ----------
    file_path : str, file path
    file_regular : regular expression of target file
    Returns
    path_list : list, the list of target file's absolute file path.
    -------
    """
    path = os.path.abspath(file_path)
    if (not os.path.isdir(path)):
        return [path]
    else:
        path_list = []
        for root, _, files in os.walk(path, topdown=False):
            for file in files:
                abs_path = os.path.join(root, file)
                if (not re.match(file_regular, abs_path) is None):
                    path_list.append(abs_path)
        path_list.sort()
        return path_list


# ## prostprocessing of predictions
# def remove_small_noise_regions(array, threshold = 300):
#     data = np.array(array).astype(np.int16).copy()
#     props = regionprops(data)
#     for region in props:
#         if region.area < threshold:
#             data[data==region.label]=0
#     return data

# def post_process_prediction(img, threshold = 300):
#     data = np.array(img).astype(np.int16).copy()
#     props = regionprops(data)
#     for region in props:
#         if region.area < threshold:
#             data[data==region.label]=0
#         else:
#             close_img = open_close_operation(data==region.label)
#             data[data == region.label]=0
#             data[close_img] = region.label
#     return data

# def open_close_operation(img):
#     from skimage.morphology import disk  # noqa
#     footprint = disk(7)
#     fill = ndi.binary_fill_holes(img)
#     open_img = binary_erosion(fill)
#     close_img = binary_dilation(open_img)
#     return close_img

# def clean(data):
#     img = data.copy()
#     mask = img >0
#     cleaned = remove_small_objects(mask, 100)
#     img[~cleaned]=0
#     return img

# def plot_evaluation_result(img, y_true, y_pred, value, y_pred_heatmap=None, name="unknown",save_path=''):
#     num = 4
#     fig, axs = plt.subplots(1,num,figsize=(8*num,8))
#     ## plot number1 GT
#     # plot background
#     axs[0].imshow(img, cmap="gray")
#     # plot gt mask
#     axs[0].imshow(y_true, alpha=0.7,)
#     # plot gt center
#     for region in regionprops(y_true):
#         axs[0].plot(region.centroid[1], region.centroid[0], '.r', markersize=30)
#     axs[0].set_title("GT: target number %d"%len(np.unique(y_true)))
    
#     ## plot number2 pred
#     # plot backgourd
#     axs[1].imshow(img, cmap="gray")
#     #plot pred mask
#     axs[1].imshow(y_pred,alpha=0.7)
#     # plot pred center
#     if y_pred_heatmap is not None:
#         axs[1].imshow( y_pred_heatmap, cmap='hot',alpha=0.3)
#     axs[1].set_title("y_pred: prediction number %d"%len(np.unique(y_pred)))
    
#     ##plot errors
#     axs[2].imshow((y_pred==0).astype(np.int8) - (y_true==0).astype(np.int8), cmap = "PiYG")
#     axs[2].set_title("errors: y_pred - y_true")
    
#     axs[3].axis([0, 1, 0, 1])
#     t = ("\n").join(['PQ: %6.4f'%value["pq"], 'SQ: %6.4f'%value["sq"], 'RQ: %6.4f'%value["rq"], 'tp: %6.4f'%value["tp"], 'fn: %6.4f'%value["fn"], 'fp: %6.4f'%value["fp"],'precision: %6.4f'%value["precision"], 'recall: %6.4f'%value["recall"]],)
#     text_kwargs = dict(ha='left', va='center', fontsize=30, color='black',)
#     axs[3].text(0,0.6,t, wrap=True,**text_kwargs)
#     axs[3].axis("off")
#     plt.suptitle(name)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()


# def calc_measure_value(result):
#     pq, sq, rq, tp, fn, fp = result
#     precision = tp/(tp+fp)
#     recall = tp/(tp+fn)
#     value = {"pq":pq, "sq":sq, "rq":rq, "tp":tp, "fn":fn, "fp":fp, "precision":precision, "recall":recall}
#     return value


# def cost_iou(res_0, res_1):
#     intersection = np.logical_and(res_0, res_1)
#     union = np.logical_or(res_0, res_1)
#     iou_score = np.sum(intersection) / np.sum(union)
#     return iou_score


# def uint16_to_rgb(img):
#     data = np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8)
#     # data[:,:,0] = img_as_ubyte(img)
#     data[:,:,0] = ((img - img.min())/(img.max()-img.min())*255).astype(np.uint8)
#     data[:,:,1] = data[:,:,2] = data[:,:,0]
#     return data



# def region_props(img):
#     props = regionprops_table(img, properties=('label','centroid',
#                                                  'orientation',
#                                                  'axis_major_length',
#                                                  'axis_minor_length',
#                                                  'area',
#                                                'bbox','eccentricity',
#                                                 ))
#     data = pd.DataFrame(props)
#     data["semantic"] = data["label"]//1000
#     data["instance"] = data["label"]%1000
#     return data

# def max_match_region(img_0, img_1):
#     uniq_0 = np.unique(img_0)[1:]
#     uniq_1 = np.unique(img_1)[1:]
#     cost_array = pd.DataFrame(data=0, index=uniq_0, columns=uniq_1)
#     for i in cost_array.index:
#         for j in cost_array.columns:
#             res_0 = img_0==i
#             res_1 = img_1==j
#             cost = cost_iou(res_0, res_1)
#             cost_array.loc[i,j] = cost
#     row_ind, col_ind = linear_sum_assignment(cost_array, maximize=True)
#     cost_sum = cost_array.values[row_ind, col_ind]
#     return uniq_0[row_ind], uniq_1[col_ind], cost_array, cost_sum


# def flaten_positive_value(data):
#     flatten = data.flatten()
#     flatten = flatten[flatten>0]
#     return flatten

# def overview_table(cell_ch_pred):
#     index = [0,1,2]
#     columns = ['cell', 'tetrad']
#     sum_table = pd.DataFrame(index=index,columns=columns)
#     for i in index:
#         for j in range(0, len(columns)):
#             if j == 0:
#                 sum_table.loc[i,columns[j]] = sum((cell_ch_pred.pred_type_2 == i) & (cell_ch_pred.semantic==1))
#             else:
#                 sum_table.loc[i,columns[j]] = sum((cell_ch_pred.pred_type_2 == i) & ((cell_ch_pred.semantic==3) | (cell_ch_pred.semantic==4)))
#     return sum_table

# def plot_pred_channel_fig(org_img, panapitc, cell_ch_pred, sum_table,name='', save_path=None):
#     index = [0,1,2]
#     columns = ['cell', 'tetrad']
#     num=3
#     fig, axs = plt.subplots(2,num,figsize=(8*num,8*2))
#     axs[0,0].imshow(org_img[0], cmap="gray")
#     axs[0,1].imshow(org_img[1], cmap="gray")
#     axs[0,2].imshow(org_img[2], cmap="gray")

#     axs[0,0].scatter(cell_ch_pred["centroid-1"], cell_ch_pred["centroid-0"], c = cell_ch_pred["pred_type_2"].map(color_dict))
#     axs[0,1].scatter(cell_ch_pred.loc[cell_ch_pred.pred_type_2 == 1,"centroid-1"], cell_ch_pred.loc[cell_ch_pred.pred_type_2 == 1, "centroid-0"], c = 'g')
#     axs[0,2].scatter(cell_ch_pred.loc[cell_ch_pred.pred_type_2 == 2,"centroid-1"], cell_ch_pred.loc[cell_ch_pred.pred_type_2 == 2, "centroid-0"], c = 'r')

#     axs[1,1].imshow(label2rgb(panapitc))
#     axs[1,2].imshow(panapitc)


#     table = axs[1,0].table(cellText=sum_table.values,
#                           rowLabels=index,
#                           #rowColours=colors,
#                           colLabels=columns,
#                           loc='center',)
#     table.set_fontsize(28)
#     table.scale(1,2)

#     for i in range(0, 2):
#         for j in range(0, num):
#             axs[i,j].axis("off")

#     fig.suptitle(name)
#     plt.tight_layout()
#     if save_path is not None:
#         plt.savefig(os.path.join(save_path,name+".png"))
#         plt.close('all')