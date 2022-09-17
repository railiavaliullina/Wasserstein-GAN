from easydict import EasyDict


cfg = EasyDict()

cfg.dataset_path = '../data/'
cfg.preprocessed_dataset_path = cfg.dataset_path + 'preprocessed_data/'
cfg.vocab_size = 50000
cfg.load_preprocessed_data = True
