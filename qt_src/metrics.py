from keras import backend as K
from keras.losses import binary_crossentropy
import numpy as np

smooth = 1.


# Metric function

def dice_coef(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def jacard(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    iou = intersection / union
    return iou


# from U-Net++ paper
def mean_iou(y_ture, y_pred):
    y_ture = K.cast(y_ture, 'int32')
    prec = []
    for t in np.arange(0.4, 1.0, 0.05):
        y_pred_ = K.cast((y_pred > t), 'int32')
        iou = jacard(y_ture, y_pred_)
        if not np.isnan(iou.numpy()):
            prec.append(iou)
        else:
            print("nan")
    if len(prec) == 0:
        prec = [0]
    return K.mean(K.stack(prec), axis=0)


# Loss function

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


# from U-Net++ paper
def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)
