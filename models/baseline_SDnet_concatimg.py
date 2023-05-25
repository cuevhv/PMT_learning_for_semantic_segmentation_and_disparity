import tensorflow as tf

# from keras import backend as K
# K.set_image_data_format('channels_last')

#from corr1d import correlation1d

from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50V2, Xception
from tensorflow.keras.layers import Activation, Add, Dense, Lambda
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Concatenate

from tensorflow_addons.layers.optical_flow import CorrelationCost
#helper Functions

from util.utilLoss import *

def __get_normalization_axis():
    # if K.image_data_format() == 'channels_last':
    return 3
    # return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------


def UpSampling2DBilinear(size, name = '', align_corners=True):
    #return Lambda(lambda x: tf.image.resize(x, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=align_corners))
    if name:
        return Lambda(lambda x: tf.image.resize(x, size, method=tf.image.ResizeMethod.BILINEAR), name=name)
    else:
        return Lambda(lambda x: tf.image.resize(x, size, method=tf.image.ResizeMethod.BILINEAR))

#----------------------------------------------------------------------------------------------------------------------------------------------------
def loadBackbone(input_tensor, backboneName = 'resnet50'):
    if backboneName == 'resnet50':
        return ResNet50V2(include_top=False, weights='imagenet', input_tensor=input_tensor)
    if backboneName == 'resnet101':
        return resnet_v2.ResNet101V2(include_top=False, weights='imagenet', input_tensor=input_tensor)
    if backboneName == 'xception':
        return Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)

#----------------------------------------------------------------------------------------------------------------------------------------------------
def getBranches(inputs, pooling_dims, name):
    bn_axis = __get_normalization_axis()
    branches = [inputs]

    for pooling in pooling_dims:
        b = AveragePooling2D((pooling, pooling), (pooling, pooling))(inputs)
        b = Conv2D(32, 3, padding='same', use_bias=False)(b)
        b = BatchNormalization(axis=-1)(b)
        b =  Activation('relu')(b)
        b = UpSampling2DBilinear([pooling_dims[0], pooling_dims[0]])(b)
        branches.append(b)
    b = Concatenate(name=name+'_concatBranch')(branches)
    return b

#----------------------------------------------------------------------------------------------------------------------------------------------------
def Getcorr1d(x, y, kernel_size = 3, displacement = 1, stride_1 = 1, stride_2 = 1, padding = 0):
    input_a = tf.keras.backend.permute_dimensions(x, pattern=(0,3,1,2))
    input_b = tf.keras.backend.permute_dimensions(y, pattern=(0,3,1,2))
    print('input shape: ', input_a.shape, input_b.shape)
    if padding == -1:
        padding = displacement + (kernel_size-1)/2
    out = CorrelationCost(kernel_size, displacement, stride_1, stride_2, padding, data_format='channels_first')([input_a, input_b])
    print 'out shape: ', out.shape
    out = tf.keras.backend.permute_dimensions(out, pattern=(0,2,3,1))
    print 'out shape: ', out.shape
    return out
    #return Lambda(lambda branches: correlation1d(branches[0], branches[1], kernel_size, displacement, stride_1, stride_2, padding), name=name+"_corr1d")#([x, y])

#----------------------------------------------------------------------------------------------------------------------------------------------------
def SiblingNet(backboneName = 'resnet50', input_shape=(224, 224, 3), show_model=False):
    input_tensor = Input(shape=input_shape)
    backbone = loadBackbone(input_tensor, backboneName)

    #for layer in backbone.layers:
    #    layer.trainable = False

    b0 = backbone.get_layer("conv1_conv").output #128x128
    b1 = backbone.get_layer("conv2_block1_out").output #64x64
    b2 = backbone.get_layer("conv2_block3_out").output #32x32
    b3 = backbone.get_layer("conv3_block4_out").output #16x16
    b4 = backbone.get_layer("conv4_block6_out").output #8x8

    # b1_concat = getBranches(b1, [64, 32, 16, 8], name='b1')
    b2_concat = getBranches(b2, [32, 16, 8], name='b2') #32x32
    # b3_concat = getBranches(b3, [16, 8], name='b3')

    # b1_concat = getBranches(b1, [128, 64, 32, 16, 8], name='b1')
    # b2_concat = getBranches(b2, [64, 32, 16, 8], name='b2')
    # b3_concat = getBranches(b3, [32, 16, 8], name='b3')

    backbone_output = backbone.output #8x8
    outputs = [b0, b1, b2, b3, b4, b2_concat, backbone_output]

    model = Model(inputs=input_tensor, outputs=outputs)
    if show_model:
        model.summary()
        plot_model(model, to_file=SiblingNet.__name__+'_model.png')

    return model

#----------------------------------------------------------------------------------------------------------------------------------------------------
def conv2D_block(feature1, feature2, filter_size):
    bn_axis = __get_normalization_axis()
    conv2d_output = Concatenate()([feature1, feature2])
    conv2d_output = Conv2DTranspose(filters=filter_size, kernel_size=3, strides=2, padding='same')(conv2d_output)
    conv2d_output = BatchNormalization(axis=bn_axis)(conv2d_output)
    return Activation('relu')(conv2d_output)

#----------------------------------------------------------------------------------------------------------------------------------------------------
def loadModel(backboneName = 'resnet50', labels=5, input_shape=(224, 224, 3), activation_output='sigmoid'):
    bn_axis = __get_normalization_axis()
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    sibling_network = SiblingNet(backboneName=backboneName, input_shape=input_shape, show_model=False)
    a_b0, a_b1, a_b2, a_b3, a_b4, a_pyramidB_2, a_backbone_output = sibling_network(input_a)
    b_b0, b_b1, b_b2, b_b3, b_b4, b_pyramidB_2, b_backbone_output = sibling_network(input_b)

    PyramidBranch_a = [a_pyramidB_2]
    PyramidBranch_b = [b_pyramidB_2]
    branches_a = [a_b1, a_b2, a_b3]

    for i in branches_a:
        print i
    #corr1d
    for i in range(len(PyramidBranch_a)):
        window_size = int(PyramidBranch_a[i].shape[2]/1.92)/2
        corr = Getcorr1d(PyramidBranch_a[i], PyramidBranch_b[i], displacement=window_size, padding=-1)
        # branches_a[i] = Conv2D(128,1)(branches_a[i])
        # branches_a[i] =  Activation('relu')(branches_a[i])

    corr = Conv2D(128,1)(corr)
    corr = Activation('relu')(corr)

    branches_a[1] = Conv2D(32,1)(corr)
    branches_a[1] = Activation('relu')(branches_a[1])
    branches_a[1] = Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same')(branches_a[1])
    branches_a[1] = Activation('relu')(branches_a[1])

    branches_a[0] = Concatenate(name='concat_b_1')([branches_a[1], branches_a[0]])

    branches_a[0] = Activation('relu')(branches_a[0])
    branches_a[0] = Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same')(branches_a[0])
    branches_a[0] = Activation('relu')(branches_a[0])
    #out 2

    print('a: ', a_b0)
    print('branches: ', branches_a[0])
    a_b0 = Concatenate(name='concat_b_0')([a_b0, branches_a[0]])
    # branches_a[0] = Concatenate(name='concat_b_0')([a_b0, branches_a[0]])
    # out = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same')(a_b0)

    if activation_output:
        out = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', activation=activation_output)(a_b0)
        # out = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', activation=activation_output)(branches_a[0])
    else:
        out = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same')(a_b0)

    #out = Lambda(lambda x: tf.image.resize(x, (input_shape[0], input_shape[1]), method=tf.image.ResizeMethod.BILINEAR))(out)
    out = Lambda(lambda x: x * 10, name='disp_out')(out)

    print 'branches out: ', out

    seg_branch = Conv2D(1024,1)(a_backbone_output)
    seg_branch = BatchNormalization(axis=bn_axis)(seg_branch)
    seg_branch = Activation('relu')(seg_branch)

    seg_branch = conv2D_block(a_b4, seg_branch, 512)
    seg_branch = conv2D_block(a_b3, seg_branch, 256)
    seg_branch = conv2D_block(a_b2, seg_branch, 256)

    seg_branch = Add()([a_b1, seg_branch])
    seg_branch = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(seg_branch)
    seg_branch = BatchNormalization(axis=bn_axis)(seg_branch)
    seg_branch = Activation('relu')(seg_branch)

    a_b0 = Conv2D(64, 1)(a_b0)
    a_b0 = Activation('relu')(a_b0)

    seg_branch = Add()([a_b0, seg_branch])
    seg_branch = Conv2DTranspose(filters=labels, kernel_size=3, strides=2, padding='same', activation='softmax', name='seg_out')(seg_branch)


    print 'branches out: ', seg_branch

    model = Model(inputs = [input_a, input_b], outputs=[out, seg_branch])
    model.summary()

    plot_model(model, to_file=loadModel.__name__+'_model.png')

    opt = 'adam'
    opt = optimizers.Adam(lr=0.001, decay=0.01)

    # if load_weights:
    #     model.load_weights(load_weights+'.h5')
    #     print('weights loaded')

    model.compile(optimizer=opt, loss={'disp_out': dynamic_huber_loss, 'seg_out': 'categorical_crossentropy'},
                                metrics={'disp_out': pixel_mae(activation_output), 'seg_out': 'accuracy'})

    return model

#----------------------------------------------------------------------------------------------------------------------------------------------------

# def dynamic_huber_loss(y_true, y_pred):
#     #delta = 0.2*tf.reduce_max(tf.math.abs(y_true - y_pred),  axis=(1,2,3), keepdims=True)
#     delta = 0.2*tf.reduce_max(tf.math.abs(y_true - y_pred))
#     return tf.keras.losses.Huber(y_true, y_pred, delta)

