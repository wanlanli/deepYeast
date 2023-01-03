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