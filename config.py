from yacs.config import CfgNode as CN

cfg = CN()
cfg.MODEL = CN()
cfg.MODEL.MODEL1 = ""
cfg.MODEL.MODEL2 = ""
cfg.OPTIMIZER = CN()
cfg.OPTIMIZER.BATCH_SIZE = 1
cfg.OPTIMIZER.EPOCHS = 1
cfg.OPTIMIZER.LR = 1.
cfg.OPTIMIZER.TYPE = ""
cfg.OPTIMIZER.LR_DECAY_STAGES = [60, 120, 180]
cfg.OPTIMIZER.LR_DECAY_RATE = 0.1
cfg.OPTIMIZER.WEIGHT_DECAY = 0.0005
cfg.OPTIMIZER.MOMENTUM = 0.9
cfg.DATASET = ""
cfg.SAVE_DIR = ""
cfg.KL_WEIGHT = 0.5
cfg.FEA_WEIGHT = 0.01 
