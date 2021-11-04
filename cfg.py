import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.img_width = 256
Cfg.img_height = 256
Cfg.b1 = 0.5
Cfg.b2  = 0.999
Cfg.decay_epoch=100 # epoch from which to start lr decay
Cfg.n_cpu = 8
Cfg.lr = 0.0002
Cfg.n_epochs = 200
Cfg.epoch=0
Cfg.batch_size = 1
Cfg.sample_interval = 500
Cfg.gpu_id = '3'
Cfg.train_path = "datasets/cityscapes/train/*.jpg"
Cfg.val_path = "datasets/cityscapes/val/*.jpg"
Cfg.test_path = "datasets/cityscapes/val/*.jpg"
Cfg.TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')
Cfg.ck = os.path.join(_BASE_DIR, 'images')
Cfg.tb = os.path.join(_BASE_DIR, 'tb')

