'''DenseMapNet - a tiny network for fast disparity estimation
from stereo images

DenseMapNet class is where the actual model is built

Atienza, R. "Fast Disparity Estimation using Dense Networks".
International Conference on Robotics and Automation,
Brisbane, Australia, 2018.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling2D 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import plot_model

from util.utilLoss import *
import numpy as np
# def unnormalize_disp(norm_disp):
#     max_d = 256/2.0
#     return norm_disp * max_d

# def pixel_mae(y_true, y_pred):
#     y = unnormalize_disp(y_true)
#     y_ = unnormalize_disp(y_pred)
#     return tensorflow.reduce_mean(tensorflow.abs(y - y_))

def loadModel(backboneName = 'resnet50', labels=5, input_shape=(224, 224, 3), activation_output='sigmoid'):
    lr=1e-3
    left = Input(shape=input_shape)
    right = Input(shape=input_shape)
    
    dropout = 0.2
    # left image as reference
    x = Conv2D(filters=16, kernel_size=5, padding='same')(left)
    xleft = Conv2D(filters=1,
                    kernel_size=5,
                    padding='same',
                    dilation_rate=2)(left)

    # left and right images for disparity estimation
    xin = tensorflow.keras.layers.concatenate([left, right])
    xin = Conv2D(filters=32, kernel_size=5, padding='same')(xin)

    # image reduced by 8
    x8 = MaxPooling2D(8)(xin)
    x8 = BatchNormalization()(x8)
    x8 = Activation('relu', name='downsampled_stereo')(x8)

    dilation_rate = 1
    y = x8
    # correspondence network
    # parallel cnn at increasing dilation rate
    for i in range(4):
        a = Conv2D(filters=32,
                    kernel_size=5,
                    padding='same',
                    dilation_rate=dilation_rate)(x8)
        a = Dropout(dropout)(a)
        y = keras.layers.concatenate([a, y])
        dilation_rate += 1

    dilation_rate = 1
    x = MaxPooling2D(8)(x)
    # disparity network
    # dense interconnection inspired by DenseNet
    for i in range(4):
        x = keras.layers.concatenate([x, y])
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv2D(filters=64,
                    kernel_size=1,
                    padding='same')(y)

        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(filters=16,
                    kernel_size=5,
                    padding='same',
                    dilation_rate=dilation_rate)(y)
        y = Dropout(dropout)(y)
        dilation_rate += 1
    
    # disparity estimate scaled back to original image size
    x = keras.layers.concatenate([x, y], name='upsampled_disparity')
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=1, padding='same')(x)
    x = UpSampling2D(8)(x)
    # if not self.settings.nopadding:
    #     x = ZeroPadding2D(padding=(2, 0))(x)

    # left image skip connection to disparity estimate
    x = keras.layers.concatenate([x, xleft])
    y = BatchNormalization()(x)
    y = Activation('relu')(y)
    y = Conv2D(filters=16, kernel_size=5, padding='same')(y)

    x = keras.layers.concatenate([x, y])
    y = BatchNormalization()(x)
    y = Activation('relu')(y)
    y = Conv2DTranspose(filters=1, kernel_size=9, padding='same')(y)

    # prediction
    yout = Activation(activation_output, name='disp_out')(y)

    # densemapnet model
    model = Model([left, right],yout)
    
    # if self.settings.otanh:
    #     self.model.compile(loss='binary_crossentropy',
    #                        optimizer=RMSprop(lr=lr))
    # else:
    model.compile(loss={'disp_out': smooth_grad_loss},
                optimizer=RMSprop(lr=lr, clipnorm=1),
                metrics={'disp_out': pixel_mae(activation_output)})

    print("DenseMapNet Model:")
    model.summary()
    plot_model(model, to_file='densemapnet.png', show_shapes=True)

    return model
