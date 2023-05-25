import tensorflow as tf

# from keras import backend as K
# K.set_image_data_format('channels_last')

#from corr1d import correlation1d

from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50V2, Xception, DenseNet121
from tensorflow.keras.layers import Activation, Add, Dense, Lambda, UpSampling2D 
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.utils import plot_model, multi_gpu_model
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


def UpSampling2DBilinear(size, name = '', align_corners=True, mode='bilinear'):
    #return Lambda(lambda x: tf.image.resize(x, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=align_corners))
    if name:
        return Lambda(lambda x: tf.image.resize(x, size, method=tf.image.ResizeMethod.BILINEAR if mode == 'bilinear' else tf.image.ResizeMethod.NEAREST_NEIGHBOR), name=name)
    else:
        return Lambda(lambda x: tf.image.resize(x, size, method=tf.image.ResizeMethod.BILINEAR if mode == 'bilinear' else tf.image.ResizeMethod.NEAREST_NEIGHBOR))

#----------------------------------------------------------------------------------------------------------------------------------------------------
def loadBackbone(input_tensor, backboneName = 'resnet50'):
    if backboneName == 'resnet50':
        return ResNet50V2(include_top=False, weights='imagenet', input_tensor=input_tensor)
    if backboneName == 'resnet101':
        return resnet_v2.ResNet101V2(include_top=False, weights='imagenet', input_tensor=input_tensor)
    if backboneName == 'xception':
        return Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)
    if backboneName == 'densenet':
        return DenseNet121(include_top=False, weights='imagenet', input_tensor=input_tensor)
#----------------------------------------------------------------------------------------------------------------------------------------------------
def getBranches(inputs, pooling_dims, name):
    bn_axis = __get_normalization_axis()
    branches = [inputs]

    for pooling in pooling_dims:
        b = AveragePooling2D((pooling, pooling), (pooling, pooling))(inputs)
        b = Conv2D(32, 3, padding='same', use_bias=False, kernel_initializer=tf.initializers.he_normal()())(b)
        b = BatchNormalization(axis=-1)(b)
        b =  Activation('relu')(b)
        b = UpSampling2DBilinear([pooling_dims[0], pooling_dims[0]])(b)
        branches.append(b)
    b = Concatenate(name=name+'_concatBranch')(branches)
    return b

#----------------------------------------------------------------------------------------------------------------------------------------------------
def getBranches2(inputs, pool_dim_h, pool_dim_w, name):
    bn_axis = __get_normalization_axis()
    branches = [inputs]
    pooling_dims = zip(pool_dim_h, pool_dim_w)

    for ph, pw in pooling_dims:
        b = AveragePooling2D((ph, pw), (ph, pw))(inputs)
        b = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False, kernel_initializer=tf.initializers.he_normal())(b)
        b = BatchNormalization(axis=-1)(b)
        b =  Activation('relu')(b)
        b = UpSampling2DBilinear([inputs.shape[1], inputs.shape[2]])(b)
        branches.append(b)
    b = Concatenate(name=name+'_concatBranch')(branches)
    return b

#----------------------------------------------------------------------------------------------------------------------------------------------------
def Getcorr1d(x, y, kernel_size = 3, displacement = 1, stride_1 = 1, stride_2 = 1, padding = 0):
    # input_a = tf.keras.backend.permute_dimensions(x, pattern=(0,3,1,2))
    # input_b = tf.keras.backend.permute_dimensions(y, pattern=(0,3,1,2))
    input_a = x
    input_b = y
    # print('input shape: ', input_a.shape, input_b.shape)
    if padding == -1:
        padding = displacement + (kernel_size-1)/2
    out = CorrelationCost(kernel_size, displacement, stride_1, stride_2, int(padding), data_format='channels_last')([input_a, input_b])
    # print ('out shape: ', out.shape)
    # out = tf.keras.backend.permute_dimensions(out, pattern=(0,2,3,1))
    # print ('out shape: ', out.shape)
    return out
    
#----------------------------------------------------------------------------------------------------------------------------------------------------
def SiblingNet(backboneName = 'resnet50', input_shape=(224, 224, 3), show_model=False):
    input_tensor = Input(shape=input_shape)
    backbone = loadBackbone(input_tensor, backboneName)

    #for layer in backbone.layers:
    #    layer.trainable = False

    b0 = backbone.get_layer("conv1/relu").output #128x128
    b1 = backbone.get_layer("pool2_conv").output #64x64
    b2 = backbone.get_layer("pool3_conv").output #32x32
    b3 = backbone.get_layer("pool4_conv").output #16x16
    b4 = backbone.output #8x8

    # b2_concat = getBranches(b2, [32, 16, 8], name='b2') #32x32
    # b0_concat = getBranches(b0, [128, 64, 32, 16, 8], name='b0')

    #pooling_val_h = [input_shape[0]/(2**i) for i in range(1,5+1)]
    #pooling_val_w = [input_shape[1]/(2**i) for i in range(1,5+1)]
    pooling_val_h = [256/(2**i) for i in range(1,5+1)]
    pooling_val_w = [256/(2**i) for i in range(1,5+1)]
    #raw_input(input_shape)
    b2_concat = getBranches2(b2, pooling_val_h[2:], pooling_val_w[2:], name='b2') #32x32
    b0_concat = getBranches2(b0, pooling_val_h, pooling_val_w, name='b0')

    backbone_output = backbone.output #8x8
    outputs = [b0, b1, b2, b3, b4, b2_concat, b0_concat, backbone_output]

    model = Model(inputs=input_tensor, outputs=outputs)
    if show_model:
        model.summary()
        plot_model(model, to_file=SiblingNet.__name__+'_model.png')

    return model

#----------------------------------------------------------------------------------------------------------------------------------------------------
def conv2D_BA(x, filter, kernel, strides=1, dilation_rate = 1, padding='same'):
    bn_axis = __get_normalization_axis()
    x = tf.keras.layers.Conv2D(filters=filter,
                strides=strides,
                kernel_size=kernel,
                dilation_rate=dilation_rate,
                padding=padding,
                kernel_initializer=tf.initializers.he_normal())(x)
    
    x = BatchNormalization(axis=bn_axis)(x)
    return Activation('relu')(x)

#----------------------------------------------------------------------------------------------------------------------------------------------------
def conv2DT_BA(x, filters, kernel, padding, strides=1):
    bn_axis = __get_normalization_axis()
    x = Conv2DTranspose(filters=filters, strides=strides, kernel_size=kernel, padding=padding)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    return Activation('relu')(x)
#----------------------------------------------------------------------------------------------------------------------------------------------------

def Conv2DownUp(x, filters, kernel, padding, stride = 1, lastLayer=True):
    x1 = conv2D_BA(x, filters, kernel, padding=padding)
    x2 = conv2D_BA(x1, filters, kernel, padding=padding)
    x = conv2D_BA(x2, filters, kernel, padding=padding)
    x = conv2DT_BA(x, filters, kernel, strides=1, padding=padding)
    x = Add()([x2, x])
    x = conv2DT_BA(x, filters, kernel, strides=1, padding=padding)
    x = Add()([x1, x])
    if not lastLayer:
        return x
    x = conv2DT_BA(x, filters, kernel, strides=1, padding=padding)
    return x
#----------------------------------------------------------------------------------------------------------------------------------------------------
def conv2D_block(feature1, feature2, filters):
    bn_axis = __get_normalization_axis()
    conv2d_output = Concatenate()([feature1, feature2])
    conv2d_output = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(conv2d_output)
    conv2d_output = BatchNormalization(axis=bn_axis)(conv2d_output)
    return Activation('relu')(conv2d_output)
#----------------------------------------------------------------------------------------------------------------------------------------------------
def group_of_layers(intermediate_dim, padding):
    shape = intermediate_dim.shape
    x_input = Input(shape=(shape[1], shape[2], shape[3]))
    x = conv2DT_BA(x_input, 64, 5, padding, strides=1)
    return Model(x_input, x)
#----------------------------------------------------------------------------------------------------------------------------------------------------

def loadModel(backboneName = 'resnet50', labels=5, input_shape=(224, 224, 3), activation_output='sigmoid', weights_path='', n_output='all', strategy=''):
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():    
        bn_axis = __get_normalization_axis()
        # input_a = Input(shape=input_shape)
        # input_b = Input(shape=input_shape)

        input_a = Input(shape=input_shape, name="left")
        input_b = Input(shape=input_shape, name="right")

        padding = 'same'
        sibling_network = SiblingNet(backboneName=backboneName, input_shape=input_shape, show_model=False)
        a_b0, a_b1, a_b2, a_b3, a_b4, a_pyramidB_2, a_pyramidB_0, a_backbone_output = sibling_network(input_a)
        b_b0, b_b1, b_b2, b_b3, b_b4, b_pyramidB_2, b_pyramidB_0, b_backbone_output = sibling_network(input_b)

        xleft1 = conv2D_BA(input_a, 1, 5, dilation_rate = 2, padding='same')
        xleft3 = conv2D_BA(input_a, 1, 5, dilation_rate = 2, padding='same')
        
        xleft2 = conv2D_BA(input_a, 1, 5, dilation_rate = 2, padding='same')

        x = Concatenate()([a_b4, b_b4])
        x = UpSampling2D(2)(x)
        x = Conv2D(filters=64, kernel_size=1, kernel_initializer=tf.initializers.he_normal(), activation='relu')(x)
        x = Conv2DownUp(x, 32, 3, padding)
        x1 = UpSampling2D(2)(x)
        seg_branch = Conv2DownUp(x1, 32, 3, padding, lastLayer=False)
        seg_branch = Conv2DTranspose(filters=labels, kernel_size=3, padding=padding)(seg_branch)
        seg_branch = UpSampling2D(8)(seg_branch)

        seg_branch = UpSampling2DBilinear([input_a.shape[1], input_a.shape[2]], mode='nearest')(seg_branch)
        #raw_input()

        seg_branch = Activation('softmax', name='seg_out')(seg_branch)
        
        
        #window_size = int(a_pyramidB_2.shape[2]/1.92)/2
        window_size = 8
        # print(window_size, a_pyramidB_2.shape[2])
        # raw_input()
        y = Getcorr1d(a_pyramidB_2, b_pyramidB_2, kernel_size=1, displacement=window_size, padding=-1)
        #y = Concatenate()([a_pyramidB_2, b_pyramidB_2])
        y = Conv2D(filters=128, kernel_size=1, kernel_initializer=tf.initializers.he_normal(), activation='relu')(y)
        y1 = Conv2DownUp(x1, 128, 3, padding)
        
        y1 = UpSampling2DBilinear([y.shape[1], y.shape[2]])(y1)
        #raw_input()
        
        y = Concatenate()([y1, y])
        y = Conv2DownUp(y, 64, 3, padding)
        y2 = UpSampling2D(8)(y)

        xleft2 = UpSampling2DBilinear([y2.shape[1], y2.shape[2]])(xleft2)
        #raw_input()

        disp_out = Concatenate()([y2, xleft2])
        disp_out = Conv2D(filters=64, kernel_size=1, kernel_initializer=tf.initializers.he_normal(), activation='relu')(disp_out)
        disp_out = Conv2DownUp(disp_out, 64, 5, padding = padding, lastLayer=False)
        disp_out = Conv2DTranspose(filters=1, kernel_size=5, padding=padding)(disp_out)

        disp_out = UpSampling2DBilinear([input_a.shape[1], input_a.shape[2]])(disp_out)
        #raw_input()
        
        if activation_output:
            disp_out = Activation(activation_output, name='disp_out2')(disp_out)
        else:
            disp_out = Activation(None, name='disp_out2')(disp_out)
        
        x = UpSampling2D(4)(x)
        y3 = UpSampling2D(2)(y)


        x = UpSampling2DBilinear([y3.shape[1], y3.shape[2]])(x)
        #raw_input()

        x = Concatenate()([x, y3])
        x = Conv2D(filters=64, kernel_size=1, kernel_initializer=tf.initializers.he_normal(), activation='relu')(x)
        x = Conv2DownUp(x, 64, 5, padding)

        x = UpSampling2DBilinear([a_b1.shape[1], a_b1.shape[2]])(x)
        #raw_input()

        x = Concatenate()([x, a_b1])
        x = Conv2D(filters=64, kernel_size=1, kernel_initializer=tf.initializers.he_normal(), activation='relu')(x)
        x = conv2DT_BA(x, filters=32, kernel = 3, strides=2, padding='same')
        x3 = x
        
        x = UpSampling2DBilinear([a_b0.shape[1], a_b0.shape[2]])(x)
        #raw_input()

        x = Concatenate()([x, a_b0])
        x = Conv2D(filters=32, kernel_size=1, kernel_initializer=tf.initializers.he_normal(), activation='relu')(x)
        x = conv2DT_BA(x, filters=32, kernel = 3, strides=2, padding='same')
        
        xleft1 = UpSampling2DBilinear([x.shape[1], x.shape[2]])(xleft1)
        #raw_input()

        x = Concatenate()([x, xleft1])
        x = Conv2D(filters=32, kernel_size=1, kernel_initializer=tf.initializers.he_normal(), activation='relu')(x)
        seg_branch2 = Conv2DownUp(x, 32, 5, padding = padding, lastLayer=False)
        seg_branch2 = Conv2DTranspose(filters=labels, kernel_size=5, padding=padding)(seg_branch2)
        seg_branch2 = Activation('softmax')(seg_branch2)

        seg_branch2 = UpSampling2DBilinear([input_a.shape[1], input_a.shape[2]], mode='nearest')(seg_branch2)
        #raw_input()

        seg_branch2 = Add(name='seg_out2')([0.9*seg_branch2, 0.1*seg_branch])


        window_size = int(a_b0.shape[2]/1.92)/2
        #y4 = Getcorr1d(a_b0, b_b0, kernel_size=3, displacement=window_size, padding=-1)
        y4 = Concatenate()([a_pyramidB_0, b_pyramidB_0])
        y4 = Conv2D(filters=128, kernel_size=1, kernel_initializer=tf.initializers.he_normal(), activation='relu')(y4)
        y = UpSampling2D(4)(y)
        
        y = UpSampling2DBilinear([y4.shape[1], y4.shape[2]])(y)
        #raw_input()

        y = Concatenate()([y4, y])
        y5 = Conv2DownUp(x3, 64, 3, padding)

        y = UpSampling2DBilinear([y5.shape[1], y5.shape[2]])(y)
        #raw_input()

        y = Concatenate()([y5, y])
        y = Conv2DownUp(y, 64, 3, padding)
        y = UpSampling2D(2)(y)

        xleft3 = UpSampling2DBilinear([y.shape[1], y.shape[2]])(xleft3)
        #raw_input()
        disp_out2 = Concatenate()([y, xleft3])


        disp_out2 = Conv2D(filters=64, kernel_size=1, kernel_initializer=tf.initializers.he_normal(), activation='relu')(disp_out2)
        disp_out2 = Conv2DownUp(disp_out2, 64, 5, padding = padding, lastLayer=False)
        disp_out2 = Conv2DTranspose(filters=1, kernel_size=5, padding=padding)(disp_out2)
        if activation_output:
            disp_out2 = Activation(activation_output)(disp_out2)
        else:
            disp_out2 = Activation(None)(disp_out2)

        disp_out2 = UpSampling2DBilinear([input_a.shape[1], input_a.shape[2]])(disp_out2)
        #raw_input()

        disp_out2 = Add(name='disp_out')([0.8*disp_out2, 0.2*disp_out])

        opt = 'adam'
        opt = optimizers.Adam(lr=0.001, decay=0.0)#lr=0.001

        inputs = [input_a, input_b]

        if n_output == 'all':
            outputs = [disp_out2, disp_out, seg_branch, seg_branch2]     
            
            losses = {'disp_out': masked_MAE,
                        'disp_out2': masked_MAE,
                        'seg_out': weightedCrossEntropy,#'categorical_crossentropy',
                        'seg_out2': weightedCrossEntropy}#'categorical_crossentropy'}
            metrics = {'disp_out': pixel_mae(activation_output),
                        'disp_out2': pixel_mae(activation_output),
                        'seg_out': 'accuracy',
                        'seg_out2': 'accuracy'}
        
        if n_output == 'disp_only':
            outputs = [disp_out2, disp_out]     
            
            losses = {'disp_out': masked_MAE,
                        'disp_out2': masked_MAE}
            metrics = {'disp_out': pixel_mae(activation_output),
                        'disp_out2': pixel_mae(activation_output)}

        if n_output == 'seg_only':
            outputs = [seg_branch, seg_branch2]     
            
            losses = {'seg_out': weightedCrossEntropy, #'categorical_crossentropy',
                        'seg_out2': weightedCrossEntropy} #'categorical_crossentropy'}
            metrics = {'seg_out': 'accuracy',
                        'seg_out2': 'accuracy'}

        model = Model(inputs = inputs, outputs=outputs)   
        plot_model(model, to_file=loadModel.__name__+'_model.png')

        if weights_path:
            model.load_weights(weights_path)
            print('weights loaded')
        #model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer=opt, loss=losses, metrics=metrics)
    
        #model.summary()
        return model

#----------------------------------------------------------------------------------------------------------------------------------------------------

# def dynamic_huber_loss(y_true, y_pred):
#     #delta = 0.2*tf.reduce_max(tf.math.abs(y_true - y_pred),  axis=(1,2,3), keepdims=True)
#     delta = 0.2*tf.reduce_max(tf.math.abs(y_true - y_pred))
#     return tf.keras.losses.Huber(y_true, y_pred, delta)
