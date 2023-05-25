import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

weights = tf.Variable(np.array([ 0.22151023, 1.5008001, 0.5451698, 4.3949537, 8.908529, 4.4717636, 
                            25.78316, 9.138443, 0.16472627, 0.5120778, 0.51006365, 60.924007,
                            396.55142, 0.8575901, 26.111387, 500.89224, 17.682869, 608.4612, 172.94113], dtype=np.float32), dtype=tf.float32)
weights = tf.reshape(weights, [1,1,1,19])        
def unnormalize_disp(norm_disp, activation):
    max_d = 256/2.0
    if activation == 'tanh':
        return tf.where(norm_disp != -10, (norm_disp + 10) * max_d / 20.0, 0)
    if activation == 'sigmoid':
        return norm_disp * max_d
    else:
        return norm_disp

# def dynamic_huber_loss(y_true, y_pred):
#     mask = tf.cast((y_true > 0), tf.float32)
#     error = y_true - y_pred
#     abs_error = tf.abs(error * mask)
#     #return tf.reduce_mean(abs_error)
#     delta = 0.1*tf.reduce_max(abs_error)
#     delta = tfp.stats.percentile(tf.abs(error), q = 50)
#     cond  = tf.keras.backend.abs(error) < delta

#     squared_loss = 0.5 * tf.keras.backend.square(error)
#     linear_loss  = delta * abs_error - 0.5 * delta * delta
    
#     return tf.reduce_mean(tf.where(cond, squared_loss, linear_loss))

def pixel_mae(activation):
    def MAE(y_true, y_pred):
        mask = tf.cast(y_true > 0, tf.float32)
        y = unnormalize_disp(y_true, activation)
        y_ = unnormalize_disp(y_pred, activation)
        return tf.reduce_mean(tf.cast(tf.abs(y * mask - y_ * mask) > 3.0, tf.float32))
    return MAE
# def multiple_huber_loss(mask):
#     def loss(y_true, y_pred):
#         losses = 0.0
#         for i in range(mask.shape[-1]):
#             y = tf.math.multiply(mask[:,:,:,i], y_true)
#             y_ = tf.math.multiply(mask[:,:,:,i], y_true)
#             losses = dynamic_huber_loss(y_true, y_pred) * mask[:,:,:,i]
#         return losses
#     return loss

def multiple_huber_loss(labels):
    def loss(y_true, y_pred):
        losses = 0
        for i in range(labels):
            losses += dynamic_huber_loss(tf.expand_dims(y_true[:,:,:,i], axis=-1), tf.expand_dims(y_pred[:,:,:,i], axis=-1)) 
        return losses
    return loss

def get_gradient(img, diff):
    m, n = -1, -1
    if diff == 'down':
        i, j = 1, 0
    if diff == 'right':
        i, j = 0, 1
    return tf.abs(img[:, 0:m-1-i, 0:n-1-j, :]-img[:, 0+i:m-1, 0+j:n-1, :])
    
    
def loss_gradient(y_true, y_pred):
    mean = 0
    for diff in ['down', 'right']:
        gt_gradient = get_gradient(y_true, diff)
        pred_gradient = get_gradient(y_pred, diff)
        gt_gradient = tf.exp(1 - gt_gradient)
        mean += tf.reduce_mean(tf.multiply(pred_gradient, gt_gradient))
    return mean

def multiple_losses(labels):
    def loss(y_true, y_pred):
        multi_loss = 0
        split = tf.split(y_true, labels+1, -1)
        gt = split[0]
        # for i in range(1, labels+1):
        #     # mask = tf.expand_dims(seg[:,:,:,0], axis = -1)
        #     # indx = tf.where(mask == 0)
        #     indx = tf.where(split[i] == 1)
        #     y = tf.gather_nd(gt, indx)
        #     y_ = tf.gather_nd(y_pred, indx)
        #     shape = y.shape
        #     if shape != 0:    
        #         multi_loss += tf.reduce_mean(tf.abs(y - y_))
        return 0 * multi_loss + 1 * tf.reduce_mean(tf.abs(gt - y_pred)) + loss_gradient(gt, y_pred)
    return loss

def smooth_grad_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred)) + loss_gradient(y_true, y_pred)

def pixel_mae_mult_out(activation):
    def MAE(y_true, y_pred):
        y = unnormalize_disp(tf.expand_dims(y_true[:,:,:,0], axis=-1), activation)
        y_ = unnormalize_disp(y_pred, activation)
        return tf.reduce_mean(tf.abs(y - y_))
    return MAE

def area_loss(y_true, y_pred):
    kernel = tf.ones((5,5,1,1))/25.0
    y = tf.nn.conv2d(y_true,
                        filters=kernel,
                        strides=2,
                        padding = "SAME")
    y_ = tf.nn.conv2d(y_pred,
                        filters=kernel,
                        strides=2,
                        padding = "SAME")

    return tf.reduce_mean(tf.square(y_true - y_pred)) + 0.5 * tf.reduce_mean(tf.abs(y_ - y))

def masked_MAE(y_true, y_pred):
    b, h, w, c = y_pred.shape
    #h = 100
    mask1 = tf.cast(y_true > 0, tf.float32)
    #mask2 = tf.cast(y_true == 0, tf.float32) 
    #m1 = tf.cast(y_true[:, :int(h/2), :, :] > -1, tf.float32) 
    #m2 = tf.cast(y_true[:, int(h/2):, :, :] > 0, tf.float32)
   # m1 = tf.cast(y_true[:, :int(h/2), :, :] == 0, tf.float32)
    m1 = tf.cast(y_true[:, :int(h/6), :, :] >= 0, tf.float32)
    m2 = tf.cast(y_true[:, int(h/6):, :, :] < 0, tf.float32) 
    mask2 = tf.concat((m1,m2), axis=1)
    abs_error = tf.abs(y_true * mask1 - y_pred * mask1)# + 0.01 * tf.abs(y_true * mask2 - y_pred * mask2)
    return tf.reduce_mean(abs_error)

def weightedCrossEntropy(y_true, y_pred):

    # scale predictions so that the class probas of each sample sum to 1
    # y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # calc
    loss = y_true * (tf.math.log(y_pred) * weights)
    loss = -tf.reduce_sum(loss, -1) 
    return loss
