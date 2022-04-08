from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, concatenate, Conv2DTranspose, DepthwiseConv2D, \
    BatchNormalization, Activation
from keras.regularizers import l2
import numpy as np


def standard_block(input_tensor, stage, filter, dropout_rate=0, kernel_size=3, use_batchNorm=True,
                   kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-4)):
    x = Conv2D(filter, (kernel_size, kernel_size), name='conv' + stage + '_1', padding='same',
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input_tensor)
    if use_batchNorm:
        x = BatchNormalization(name='bn' + stage + '_1')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name='drop' + stage + '_1')(x)
    x = Activation('relu', name='relu' + stage + '_1')(x)

    x = Conv2D(filter, (kernel_size, kernel_size), name='conv' + stage + '_2', padding='same',
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    if use_batchNorm:
        x = BatchNormalization(name='bn' + stage + '_2')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name='drop' + stage + '_2')(x)
    x = Activation('relu', name='relu' + stage + '_2')(x)
    return x


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


def U_Net(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, filters=[32, 64, 128, 256, 512], use_batchNorm=True, dp_rate=0,
          use_sharp=False, num_class=1, kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-4)):
    pool_size = (2, 2)
    len_filters = len(filters)
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="input")
    encoder_conv = []
    print(inputs)
    # Encoder
    for i, filter in enumerate(filters):
        stage = "{}1".format(i + 1)
        if i == 0:
            conv = standard_block(inputs, stage=stage, filter=filters[i], use_batchNorm=use_batchNorm,
                                  dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer)
            pool = MaxPooling2D(pool_size, strides=(2, 2))(conv)
        elif i == len_filters - 1:
            conv = standard_block(pool, stage=stage, filter=filters[i], use_batchNorm=use_batchNorm,
                                  dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer)
        else:
            conv = standard_block(pool, stage=stage, filter=filters[i], use_batchNorm=use_batchNorm,
                                  dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer)
            pool = MaxPooling2D(pool_size, strides=(2, 2))(conv)
        encoder_conv.append(conv)
    # Decoder
    for i, filter in enumerate(filters[:-1][::-1]):
        stage = "{}{}".format(len_filters - i - 1, i + 2)
        if use_sharp:
            W = build_sharp_blocks(conv)
            sharp_block = DepthwiseConv2D(kernel_size=3, use_bias=False, padding='same',
                                          name="depthWiseConv{}".format(stage))
            conv = sharp_block(conv)
            sharp_block.set_weights([W])
            sharp_block.trainable = False
        up = Conv2DTranspose(filter, (2, 2), strides=(2, 2), name="up{}".format(stage), padding="same")(conv)
        conv = concatenate([up, encoder_conv[-2 - i]], name="merge{}".format(stage))
        conv = standard_block(conv, stage=stage, filter=filter, use_batchNorm=use_batchNorm, dropout_rate=dp_rate,
                              kernel_initializer='glorot_uniform')
    unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer=kernel_initializer,
                         padding='same', kernel_regularizer=kernel_regularizer)(conv)

    model = Model(inputs=inputs, outputs=unet_output)
    return model


def U_Net_PlusPlus(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, filters=[32, 64, 128, 256, 512], use_batchNorm=True, dp_rate=0,
                   num_class=1, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), deep_supervision=False):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="input")

    # Handle Dimension Ordering for different backends

    # conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    conv1_1 = standard_block(inputs, stage="11", filter=filters[0], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    # conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    conv2_1 = standard_block(pool1, stage="21", filter=filters[1], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12')
    # conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])
    conv1_2 = standard_block(conv1_2, stage="12", filter=filters[0], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)

    # conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    conv3_1 = standard_block(pool2, stage="31", filter=filters[2], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22')
    # conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])
    conv2_2 = standard_block(conv2_2, stage="22", filter=filters[1], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)

    up1_3 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13')
    # conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])
    conv1_3 = standard_block(conv1_3, stage="13", filter=filters[0], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)

    # conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    conv4_1 = standard_block(pool3, stage="41", filter=filters[3], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32')
    # conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])
    conv3_2 = standard_block(conv3_2, stage="32", filter=filters[2], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)

    up2_3 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23')
    # conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])
    conv2_3 = standard_block(conv2_3, stage="23", filter=filters[1], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)

    up1_4 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14')
    # conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])
    conv1_4 = standard_block(conv1_4, stage="14", filter=filters[0], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)

    # conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])
    conv5_1 = standard_block(pool4, stage="51", filter=filters[4], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)

    up4_2 = Conv2DTranspose(filters[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42')
    # conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])
    conv4_2 = standard_block(conv4_2, stage="42", filter=filters[3], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)

    up3_3 = Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33')
    # conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])
    conv3_3 = standard_block(conv3_3, stage="33", filter=filters[2], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)

    up2_4 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24')
    # conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])
    conv2_4 = standard_block(conv2_4, stage="24", filter=filters[1], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)

    up1_5 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15')
    # conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])
    conv1_5 = standard_block(conv1_5, stage="15", filter=filters[0], use_batchNorm=use_batchNorm,
                             dropout_rate=dp_rate, kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)

    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4',
                              kernel_initializer=kernel_initializer, padding='same',
                              kernel_regularizer=kernel_regularizer)(conv1_5)

    if deep_supervision:
        nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1',
                                  kernel_initializer=kernel_initializer, padding='same',
                                  kernel_regularizer=kernel_regularizer)(conv1_2)
        nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2',
                                  kernel_initializer=kernel_initializer, padding='same',
                                  kernel_regularizer=kernel_regularizer)(conv1_3)
        nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3',
                                  kernel_initializer=kernel_initializer, padding='same',
                                  kernel_regularizer=kernel_regularizer)(conv1_4)
        model = Model(inputs=inputs, outputs=[nestnet_output_1,
                                              nestnet_output_2,
                                              nestnet_output_3,
                                              nestnet_output_4])
    else:
        model = Model(inputs=inputs, outputs=[nestnet_output_4])

    return model
