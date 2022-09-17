from easydict import EasyDict


cfg = EasyDict()

cfg.device = 'cuda'  # 'cpu'
cfg.workers = 2
cfg.batch_size = 8
cfg.image_size = 64
cfg.nc = 1
cfg.nz = 100
cfg.ngf = 64
cfg.ndf = 64
cfg.epochs = 50
cfg.lr = 0.0001
cfg.w_gp = 10

cfg.penalty_coef = 10  # lambda
cfg.n_critic = 5
cfg.alpha = 0.0001
cfg.beta1 = 0
cfg.beta2 = 0.9

cfg.log_metrics = True
cfg.experiment_name = 'with_layer_norm'

# cfg.evaluate_on_train_set = True
# cfg.evaluate_before_training = True

cfg.load_saved_model = False
cfg.checkpoints_dir = f'../saved_files/checkpoints/{cfg.experiment_name}'
cfg.epoch_to_load = 9
cfg.save_model = True
cfg.epochs_saving_freq = 1
