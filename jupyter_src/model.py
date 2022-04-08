import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, concatenate, Conv2DTranspose, DepthwiseConv2D

smooth = 1.


# Metric function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def jacard(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    return intersection / union


def get_kernel():
    """
    See https://setosa.io/ev/image-kernels/
    """

    k1 = np.array([[0.0625, 0.125, 0.0625],
                   [0.125, 0.25, 0.125],
                   [0.0625, 0.125, 0.0625]])

    # Sharpening Spatial Kernel, used in paper
    k2 = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

    k3 = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

    return k1, k2, k3


def build_sharp_blocks(layer):
    """
    Sharp Blocks
    """
    # Get number of channels in the feature
    in_channels = layer.shape[-1]
    # Get kernel
    _, w, _ = get_kernel()
    # Change dimension
    w = np.expand_dims(w, axis=-1)
    # Repeat filter by in_channels times to get (H, W, in_channels)
    w = np.repeat(w, in_channels, axis=-1)
    # Expand dimension
    w = np.expand_dims(w, axis=-1)
    return w


def SharpUNet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, opt='adam'):
    # Unet with sharp Blocks in skip connections

    # Kernel size for sharp blocks
    kernel_size = 3

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = inputs
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
    conv6 = Dropout(0.4)(conv6)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv6)

    # Skip connection 1
    # 1. Get sharpening kernel weights(1, H, W, channels)
    W1 = build_sharp_blocks(conv5)
    # 2. Build depthwise convolutional layer with random weights
    sb1 = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    # 3. Pass input to layer
    conv5 = sb1(conv5)
    # 4. Set filters as layer weights
    sb1.set_weights([W1])
    # 5. Dont update weights
    sb1.trainable = False

    up7 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv6), conv5], axis=3)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Dropout(0.4)(conv7)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)

    # Skip connection 2
    W2 = build_sharp_blocks(conv4)
    sb2 = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv4 = sb2(conv4)
    sb2.set_weights([W2])
    sb2.trainable = False

    up8 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7), conv4], axis=3)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv8)

    # Skip connection 3
    W3 = build_sharp_blocks(conv3)
    sb3 = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv3 = sb3(conv3)
    sb3.set_weights([W3])
    sb3.trainable = False

    up9 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8), conv3], axis=3)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)

    # Skip connection 4
    W4 = build_sharp_blocks(conv2)
    sb4 = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv2 = sb4(conv2)
    sb4.set_weights([W4])
    sb4.trainable = False

    up10 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9), conv2], axis=3)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up10)
    conv10 = Dropout(0.1)(conv10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv10)

    # Skip connection 5
    W5 = build_sharp_blocks(conv1)
    sb5 = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    conv1 = sb5(conv1)
    sb5.set_weights([W5])
    sb5.trainable = False

    up11 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10), conv1], axis=3)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up11)
    conv11 = Dropout(0.1)(conv11)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)

    conv12 = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

    model = Model(inputs=[inputs], outputs=[conv12])

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[jacard, dice_coef])

    return model
