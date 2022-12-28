class ConfigBase:
    def __init__(self, config) -> None:
        for key, v in config.items():
            setattr(self, key, v)


class ExperimentOptions:
    def __init__(self, config) -> None:
        self.model_options = ModelOptions(config['model_options'])
        self.trainer_options = TrainerOptions(config['trainer_options'])
        self.evaluator_options = EvaluatorOptions(config['evaluator_options'])
        self.train_dataset_options = DatasetOptions(config['train_dataset_options'])
        self.eval_dataset_options = DatasetOptions(config['eval_dataset_options'])


class ModelOptions:
    def __init__(self, config) -> None:
        self.initial_checkpoint = config['initial_checkpoint']
        self.backbone = BackboneOptions(config["backbone"])
        self.meta_architecture = "panoptic_deeplab"
        self.panoptic_deeplab = PanopticOptions(config['panoptic_deeplab'])
        self.decoder = DecoderOptions(config['decoder'])
        self.restore_semantic_last_layer_from_initial_checkpoint = True
        self.restore_instance_last_layer_from_initial_checkpoint = True

class BackboneOptions:
    def __init__(self, backbone) -> None:
        self.name = backbone['name']
        self.output_stride = backbone['output_stride']
        self.stem_width_multiplier = backbone['stem_width_multiplier']
        self.backbone_width_multiplier = backbone['backbone_width_multiplier']
        self.backbone_layer_multiplier = backbone['backbone_layer_multiplier']
        # self.use_squeeze_and_excite = 
        self.drop_path_keep_prob = backbone['drop_path_keep_prob']
        self.drop_path_schedule = backbone['drop_path_schedule']
        # self.use_sac_beyond_stride = 


class PanopticOptions:
    def __init__(self, config) -> None:
        self.low_level = ConfigBase(config['low_level'])
        self.instance = InstanceOptions(config['instance'])
        self.semantic_head = SemanticOptions(config['semantic_head'])


class DecoderOptions(ConfigBase):
    def __init__(self, config) -> None: 
        self.decoder_conv_type = b"depthwise_separable_conv".decode('utf-8')
        self.atrous_rates = []
        self.aspp_channels = 256
        self.aspp_use_only_1x1_proj_conv = True
        super(DecoderOptions, self).__init__(config)


class InstanceOptions:
    def __init__(self, config) -> None:
        self.enable = True
        for key, v in config.items():
            if key == "instance_decoder_override":
                self.instance_decoder_override = DecoderOptions(config['instance_decoder_override'])
            else:
                setattr(self, key, ConfigBase(v))
        self.center_head.head_conv_type = "depthwise_separable_conv"
        self.regression_head.head_conv_type = "depthwise_separable_conv"


class SemanticOptions(ConfigBase):
    def __init__(self, config) -> None:
        self.head_conv_type = "depthwise_separable_conv"
        super(SemanticOptions, self).__init__(config)


class TrainerOptions:
    def __init__(self, config) -> None:
        self.save_checkpoints_steps = config['save_checkpoints_steps']
        self.save_summaries_steps = config['save_checkpoints_steps']
        self.steps_per_loop = config['steps_per_loop']
        self.num_checkpoints_to_keep = 1
        self.loss_options = LossOptions(config['loss_options'])
        self.solver_options = SolverOptions(config['solver_options'])


class LossOptions:
    def __init__(self, config) -> None:
        self.semantic_loss = SingleLossOptions(config['semantic_loss'])
        self.center_loss = SingleLossOptions(config['center_loss'])
        self.regression_loss = SingleLossOptions(config['regression_loss'])


class SingleLossOptions(ConfigBase):
    def __init__(self, config) -> None:
        self.top_k_percent = float(1)
        super(SingleLossOptions, self).__init__(config)


class SolverOptions(ConfigBase):
    def __init__(self, config) -> None:
        # self.use_gradient_clipping = False
        self.learning_policy = "poly"
        self.optimizer = "adam"
        self.use_sync_batchnorm = True
        self.batchnorm_momentum = float(0.99)
        self.batchnorm_epsilon = float(0.001)
        self.weight_decay = float(0)
        self.poly_end_learning_rate = float(0)
        self.poly_learning_power = float(0.9)
        super(SolverOptions, self).__init__(config)


class EvaluatorOptions(ConfigBase):
    def __init__(self, config) -> None:
        self.keep_k_centers = 400
        self.add_flipped_images = False
        self.eval_scales = []
        self.use_tf_function = True
        self.raw_panoptic_format = 'two_channel_png'
        self.convert_raw_to_eval_ids = False
        super(EvaluatorOptions, self).__init__(config)


class DatasetOptions:
    def __init__(self, config) -> None:
        self.decode_groundtruth_label = True
        self.sigma = float(8)
        self.resize_factor = None
        self.thing_id_mask_annotations = False
        self.max_thing_id = 128
        self.augmentations = AugmentationsOptions()
        for key, v in config.items():
            if isinstance(v, dict):
                setattr(self, key, ConfigBase(v))
            else:
                setattr(self, key, v)

class AugmentationsOptions:   
    def __init__(self) -> None:
        self.min_scale_factor = 1
        self.max_scale_factor = 1
        self.scale_factor_step_size = 1
        self.autoaugment_policy_name = None