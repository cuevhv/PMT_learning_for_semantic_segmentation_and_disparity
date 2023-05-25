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
from tensorflow.keras.layers import MaxPooling2D, Add
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import concatenate
from tensorflow_addons.layers.optical_flow import CorrelationCost
from util.utilLoss import *
import numpy as np
# def unnormalize_disp(norm_disp):
#     max_d = 256/2.0
#     return norm_disp * max_d

# def pixel_mae(y_true, y_pred):
#     y = unnormalize_disp(y_true)
#     y_ = unnormalize_disp(y_pred)
#     return tensorflow.reduce_mean(tensorflow.abs(y - y_))

def Getcorr1d(x, y, kernel_size = 3, displacement = 1, stride_1 = 1, stride_2 = 1, padding = 0):
    input_a = tf.keras.backend.permute_dimensions(x, pattern=(0,3,1,2))
    input_b = tf.keras.backend.permute_dimensions(y, pattern=(0,3,1,2))
    print('input shape: ', input_a.shape, input_b.shape)
    if padding == -1:
        padding = displacement + (kernel_size-1)/2
    out = CorrelationCost(kernel_size, displacement, stride_1, stride_2, padding, data_format='channels_first')([input_a, input_b])
    print('out shape: ', out.shape)
    out = tf.keras.backend.permute_dimensions(out, pattern=(0,2,3,1))
    print('out shape: ', out.shape)
    return out

def pool_delated(x, pool_n = 4):
    dropout = 0.2
    dilation_rate = 1
    y = x
    # correspondence network
    # parallel cnn at increasing dilation rate
    for i in range(pool_n):
        a = Conv2D(filters=32,
                    kernel_size=5,
                    padding='same',
                    dilation_rate=dilation_rate)(x)
        a = Dropout(dropout)(a)
        y = concatenate([a, y])
        dilation_rate += 1
    return y

def reduce_net(input_shape):
    input_tensor = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=5, padding='same')(input_tensor)
    x = MaxPooling2D(8)(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='downsampled_stereo')(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model

def reduce_img(x, name):
    x = Conv2D(filters=32, kernel_size=5, padding='same')(x)
    x = MaxPooling2D(8)(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name=name+'downsampled_stereo')(x)
    return x


#-----------------------------------------------------------------------------------------------------------------------
def loadModel(backboneName = 'resnet50', labels=5, input_shape=(224, 224, 3), activation_output='sigmoid'):
    lr=1e-3
    left = Input(shape=input_shape)
    right = Input(shape=input_shape)
    edges = Input(shape=(input_shape[0], input_shape[1], 1))
    dropout = 0.2

    reduce_network = reduce_net(input_shape=input_shape)
    # image reduced by 8
    #x8r = reduce_img(right, 'right_')
    x8r = reduce_network(right)
    y = pool_delated(x8r, pool_n = 4)
    
    #x8l = reduce_img(left, 'left_')
    x8l = reduce_network(left)
    x = pool_delated(x8l, pool_n = 4)
    # disparity network
    # dense interconnection inspired by DenseNet

    window_size = int(y.shape[2]/1.92)/2
    print(window_size)
    dilation_rate = 1
    for i in range(4):
        if i == 0:
            x = Getcorr1d(x, y, displacement=window_size, padding=-1, kernel_size=3)
        else:
            x = concatenate([x, y])
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
    x = concatenate([x, y], name='upsampled_disparity')
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=1, padding='same')(x)
    x = UpSampling2D(8)(x)
    # if not self.settings.nopadding:
    #     x = ZeroPadding2D(padding=(2, 0))(x)


    # edge
    # yedge = Conv2D(filters=1,
    #                 kernel_size=5,
    #                 padding='same')(left)
    # yedge = tf.subtract(edges, yedge)
    yedge = Conv2D(filters=128,
                    kernel_size=5,
                    padding='same')(left)
    yedge = Conv2D(filters=16,
                    kernel_size=5,
                    padding='same')(yedge)
    ycon = Conv2DTranspose(filters=16,
                    kernel_size=5,
                    padding='same')(yedge)
    yedge = Conv2DTranspose(filters=1,
                    kernel_size=3,
                    padding='same')(ycon)
    yedge = Activation('sigmoid', name='disp_edge')(yedge)

    # left image skip connection to disparity estimate
    xleft = Conv2D(filters=1,
                    kernel_size=5,
                    padding='same',
                    dilation_rate=2)(left)

    x = concatenate([x, xleft])
    x = Conv2D(filters=16, kernel_size=1)(x)
    x = Add()([x, ycon])
    y = BatchNormalization()(x)
    y = Activation('relu')(y)
    y = Conv2D(filters=16, kernel_size=5, padding='same')(y)

    x = concatenate([x, y])
    y = BatchNormalization()(x)
    y = Activation('relu')(y)
    y = Conv2DTranspose(filters=1, kernel_size=9, padding='same')(y)

    # prediction
    yout = Activation(activation_output, name='disp_out')(y)

    # y = Conv2D(filters=32, kernel_size=3, padding='same')(yout)
    # y1 = Conv2D(filters=32, kernel_size=3, padding='same')(y)
    # y = Conv2D(filters=32, kernel_size=3, padding='same')(y1)
    # y = Conv2DTranspose(filters=32, kernel_size=3, padding='same')(y)
    # y = Add()([y1, y])
    # y = Conv2DTranspose(filters=32, kernel_size=3, padding='same')(y)
    # y = Conv2DTranspose(filters=1, kernel_size=3, padding='same')(y)
    # yedge = Activation('sigmoid', name='disp_edge')(y)
            

    # densemapnet model
    model = Model([left, right, edges], [yout, yedge])
    
    # if self.settings.otanh:
    #     self.model.compile(loss='binary_crossentropy',
    #                        optimizer=RMSprop(lr=lr))
    # else:
    model.compile(loss={'disp_out': 'mse', 'disp_edge': 'mse'},
                optimizer=RMSprop(lr=lr),
                metrics={'disp_out': pixel_mae(activation_output), 'disp_edge': 'mae'})

    print("DenseMapNet Model:")
    model.summary()
    plot_model(model, to_file='densemapnet.png', show_shapes=True)

    return model
