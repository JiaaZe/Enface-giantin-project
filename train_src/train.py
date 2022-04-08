import os
from datetime import datetime

from TFRecord_func import TFRecord_to_data
from functions import make_model_input, clear_blank_mask
from model import U_Net, U_Net_PlusPlus
from config import *
from metrics import *

from tensorflow.python.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

tfrecord_train_path = "../data/train"
tfrecord_test_path = "../data/test"
train_image, train_mask, _ = TFRecord_to_data(tfrecord_train_path)
test_image, test_mask, _ = TFRecord_to_data(tfrecord_test_path)

X_train_tmp, _ = make_model_input(train_image, do_norm=True, data_shape=(-1, 256, 256, 1))
y_train_tmp, _ = make_model_input(train_mask, do_norm=False, data_shape=(-1, 256, 256, 1))
X_test, _ = make_model_input(test_image, do_norm=True, data_shape=(-1, 256, 256, 1))
y_test, _ = make_model_input(test_mask, do_norm=False, data_shape=(-1, 256, 256, 1))

print("X_train shape:{}".format(X_train_tmp.shape))
print("y_train shape:{}".format(y_train_tmp.shape))
print("X_test shape:{}".format(X_test.shape))
print("y_test shape:{}".format(y_test.shape))

X_train, y_train = clear_blank_mask(X_train_tmp, y_train_tmp, ratio_of_blank=0.2)

print("Clear blank mask")
print("X_train shape:{}".format(X_train.shape))
print("y_train shape:{}".format(y_train.shape))
print("X_test shape:{}".format(X_test.shape))
print("y_test shape:{}".format(y_test.shape))

print("X_train dataType:{}".format(X_train.dtype))
print("y_train dataType:{}".format(y_train.dtype))
print("X_test dataType:{}".format(X_test.dtype))
print("y_test dataType:{}".format(y_test.dtype))


class LRTensorBoard(TensorBoard):
    # ssh -L 16006:120.0.0.1:6006 ecsyuphd@10.97.40.58
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


opt = Adam(lr=lr_start)
if use_plus:
    model = U_Net_PlusPlus(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1,
                           filters=[32, 64, 128, 256, 512],
                           use_batchNorm=use_batchNorm,
                           dp_rate=drop_rate,
                           kernel_initializer=kernel_initializer,
                           # kernel_regularizer=None,
                           )
else:
    model = U_Net(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1,
                  filters=[32, 64, 128, 256, 512],
                  use_batchNorm=use_batchNorm,
                  dp_rate=drop_rate,
                  kernel_initializer=kernel_initializer,
                  # kernel_regularizer=None,
                  use_sharp=use_sharp)

model.compile(optimizer=opt,
              loss=bce_dice_loss,
              metrics=["binary_crossentropy", mean_iou, dice_coef],
              run_eagerly=True,
              )
model.summary()
callback_info = "_{}".format(kernel_initializer)

if LR_reduce_mode == 1:
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=ReduceLROnPlateau_factor,
                                     patience=ReduceLROnPlateau_patience,
                                     min_lr=lr_end, verbose=1, mode="min")
    callback_info += "_reduceLR_{}per{}".format(ReduceLROnPlateau_factor, ReduceLROnPlateau_patience)
elif LR_reduce_mode == 2:
    lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e, verbose=1)
    callback_info += "_LRdecay:{}".format(lr_decay)
else:
    ...

if use_batchNorm:
    callback_info += "_use-bn".format(use_batchNorm)
if drop_rate > 0:
    callback_info += "_droprate:{}".format(drop_rate)
if data_gen:
    callback_info += "_datagen"

if use_plus:
    model_name = "unet++"
else:
    if use_sharp:
        model_name = "sharpunet"
    else:
        model_name = "unet"

model_path_prefix = "../model/{}_{}_512_epoch:{}_batch:{}_lr:{}{}/".format(
    datetime.now().strftime("%m-%d|%H-%M-%S"), model_name,
    epochs, batch_size, lr_start, callback_info)
if not os.path.exists(model_path_prefix):
    os.makedirs(model_path_prefix)
callbacks = [
    ModelCheckpoint(monitor='val_mean_iou',
                    filepath=model_path_prefix + 'valLoss:{val_loss:.4f}_cur_epoch:{epoch:02d}_'
                                                 'trainBC:{binary_crossentropy:.4f}_trainDice:{dice_coef:.4f}_'
                                                 'trainMeanIoU:{mean_iou:.4f}_valBC:{val_binary_crossentropy:.4f}_'
                                                 'valDice:{val_dice_coef:.4f}_valMeanIoU:{val_mean_iou:.4f}.h5',
                    save_best_only=True,
                    mode='max',
                    verbos=1, ),
    LRTensorBoard(log_dir=model_path_prefix + 'logs'),
]
if LR_reduce_mode != 0:
    callbacks.append(lr_scheduler)

if data_gen:

    ver_X_train = tf.image.flip_up_down(X_train)
    ver_y_train = tf.image.flip_up_down(y_train)

    hor_X_train = tf.image.flip_left_right(X_train)
    hor_y_train = tf.image.flip_left_right(y_train)

    dia_X_train = tf.image.flip_left_right(ver_X_train)
    dia_y_train = tf.image.flip_left_right(ver_y_train)

    new_X_train = tf.concat([X_train, ver_X_train, hor_X_train, dia_X_train], axis=0)
    new_y_train = tf.concat([y_train, ver_y_train, hor_y_train, dia_y_train], axis=0)


else:
    new_X_train = X_train
    new_y_train = y_train

model.fit(new_X_train, new_y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          callbacks=callbacks)
#
# datagen = ImageDataGenerator(horizontal_flip=True)
#
# image_datagen = datagen.flow(tf.concat([X_train, ver_X_train], axis=0), shuffle=False, seed=100)
# mask_datagen = datagen.flow(tf.concat([y_train, ver_X_train], axis=0), shuffle=False, seed=100)
#
# train_generator = zip(image_datagen, mask_datagen)
#
# model.fit_generator(train_generator,
#                     epochs=epochs,
#                     steps_per_epoch=len(X_train) // batch_size,
#                     validation_data=(X_test, y_test),
#                     callbacks=callbacks)
