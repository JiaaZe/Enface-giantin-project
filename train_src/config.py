batch_size = 8
epochs = 200

drop_rate = 0
use_batchNorm = True
use_sharp = True
use_plus = True
kernel_initializer = 'he_normal'


data_gen = False
# 0 no use; 1 ReduceLROnPlateau; 2 LearningRateScheduler
LR_reduce_mode = 0
ReduceLROnPlateau_factor = 0.8
ReduceLROnPlateau_patience = 10

# learning rate schedule
lr_start = 3e-4  # -3
lr_end = 1e-4  # -4
lr_decay = (lr_end / lr_start) ** (1. / epochs)
