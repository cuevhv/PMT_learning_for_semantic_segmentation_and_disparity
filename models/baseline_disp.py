#
import tensorflow as tf

#keras

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

    for layer in backbone.layers:
        layer.trainable = False

    b0 = backbone.get_layer("conv1_conv").output #128x128
    b1 = backbone.get_layer("conv2_block1_out").output #64x64
    b2 = backbone.get_layer("conv2_block3_out").output #32x32
    b3 = backbone.get_layer("conv3_block4_out").output #16x16
    b4 = backbone.get_layer("conv4_block6_out").output #8x8

    # b1_concat = getBranches(b1, [64, 32, 16, 8], name='b1')
    b2_concat = getBranches(b2, [32, 16, 8], name='b2')
    # b3_concat = getBranches(b3, [16, 8], name='b3')

    # b1_concat = getBranches(b1, [128, 64, 32, 16, 8], name='b1')
    # b2_concat = getBranches(b2, [64, 32, 16, 8], name='b2')
    # b3_concat = getBranches(b3, [32, 16, 8], name='b3')



    model = Model(inputs=input_tensor, outputs=[b0, b1, b2, b3, b2_concat])
    if show_model:
        model.summary()
        plot_model(model, to_file=SiblingNet.__name__+'_model.png')

    return model

#----------------------------------------------------------------------------------------------------------------------------------------------------


def loadModel(backboneName = 'resnet50', labels=5, input_shape=(224, 224, 3), activation_output='sigmoid'):

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    sibling_network = SiblingNet(backboneName=backboneName, input_shape=input_shape, show_model=False) 
    a_b0, a_b1, a_b2, a_b3, a_pyramidB_2 = sibling_network(input_a)
    b_b0, b_b1, b_b2, b_b3, b_pyramidB_2 = sibling_network(input_b)

    PyramidBranch_a = [a_pyramidB_2]
    PyramidBranch_b = [b_pyramidB_2]
    branches_a = [a_b1, a_b2, a_b3]

    for i in branches_a:
        print i
    #corr1d
    for i in range(len(PyramidBranch_a)):
        window_size = int(PyramidBranch_a[i].shape[2]/1.92)/2
        corr = Getcorr1d(PyramidBranch_a[i], PyramidBranch_b[i], displacement=window_size, padding=-1)
        # branches_a[i] = Conv2D(32,1)(branches_a[i])
        # branches_a[i] =  Activation('relu')(branches_a[i])
    
    branches_a[1] = Concatenate(name='concat_b_2')([corr, branches_a[1]])

    branches_a[1] = Conv2D(32,1)(branches_a[1])
    branches_a[1] = Activation('relu')(branches_a[1]) #concatenate 1
    branches_a[1] = Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same')(branches_a[1])
    branches_a[1] = Activation('relu')(branches_a[1])

    
    branches_a[0] = Concatenate(name='concat_b_1')([branches_a[1], branches_a[0]])
    branches_a[0] = Conv2D(32,1)(branches_a[0]) #concatenate 2
    branches_a[0] = Activation('relu')(branches_a[0])
    branches_a[0] = Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same')(branches_a[0])
    branches_a[0] = Activation('relu')(branches_a[0])
    #out 2

    a_b0 = Concatenate(name='concat_b_0')([a_b0, branches_a[0]])
    a_b0 = Conv2D(32,1)(a_b0) #concatenate 3
    if activation_output:
        out = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', name="branch_out", activation=activation_output)(a_b0)
    else:
        out = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', name="branch_out")(a_b0)
    print 'branches out: ', out    
  
    model = Model(inputs = [input_a, input_b], outputs=[out])
    model.summary()
    plot_model(model, to_file=loadModel.__name__+'_model.png')
    
    opt = 'adam'
    opt = optimizers.Adam(lr=0.001, decay=0.01)


    model.compile(optimizer=opt, loss=['mse'], metrics=['accuracy'])
    return model

#----------------------------------------------------------------------------------------------------------------------------------------------------

