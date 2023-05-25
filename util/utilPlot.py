import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#----------------------------------------------------------------------------------------------------------------------------------------------------
def plot_learning_curves(hist, net_type):
    keys = hist.history.keys()
    print(keys)
    if 'disp' in net_type:
        plt.subplot(2, 1, 1)
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Disparity learning curve')
        plt.ylabel('Loss')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1, x2, 0, 0.15))
        plt.xlabel('Epoch')
        plt.legend(['Training set', 'val set'], loc='upper right')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(hist.history['MAE'])
        plt.plot(hist.history['val_MAE'])
        plt.title('Disparity MAE curve')
        plt.ylabel('MAE')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1, x2, 0, 5.0))
        plt.xlabel('Epoch')
        plt.legend(['Training set', 'val set'], loc='upper left')
        plt.grid(True)
    else:
    # if net_type == 'segdisp':
        plt.subplot(2, 1, 1)
        if 'disp_out_loss' in keys:
            plt.plot(hist.history['disp_out_loss'])
            plt.plot(hist.history['val_disp_out_loss'])
            name = 'disp '
        else:
            plt.plot(np.array(hist.history['loss']) - np.array(hist.history['seg_out_loss']))
            plt.plot(np.array(hist.history['val_loss']) - np.array(hist.history['val_seg_out_loss']))
            name = ''


        plt.title('Disparity learning curve')
        plt.ylabel('Loss')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1, x2, 0, 2.0))
        plt.xlabel('Epoch')
        plt.legend([name+'Training set', name+'val set'], loc='upper right')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(hist.history['seg_out_loss'])
        plt.plot(hist.history['val_seg_out_loss'])
        plt.title('Segmentation learning curve')
        plt.ylabel('Loss')
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1, x2, 0, 2.0))
        plt.xlabel('Epoch')
        plt.legend(['Segmentation Training set', 'Segmentation val set'], loc='upper left')
        plt.grid(True)

            # else:
            #     plt.plot(hist.history['loss'])
            #     plt.plot(hist.history['val_loss'])
            #     plt.title('learning curve')
            #     plt.ylabel('Loss')
            #     x1,x2,y1,y2 = plt.axis()
            #     plt.axis((x1, x2, 0, 2.0))
            #     plt.xlabel('Epoch')
            #     plt.legend(['Training set', 'val set'], loc='upper right')

            #     plt.grid(True)
    # plt.subplot(2, 1, 2)
    # plt.plot(history.history['branch_out_loss'])
    # plt.plot(history.history['val_branch_out_loss'])
    # plt.title('model '+keys[1])
    # plt.ylabel(keys[1])
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.grid(True)

#----------------------------------------------------------------------------------------------------------------------------------------------------
def gtPlot(left, right, seg, disp):
    plt.figure()
    images = [tf.keras.preprocessing.image.array_to_img(left),
              tf.keras.preprocessing.image.array_to_img(right),
              np.argmax(seg, axis=-1), disp]
    max_depth = disp.shape[1]
    dispNorm = np.copy(disp)
    i = 221
    for j in range(len(images)):
        plt.subplot(i+j)
        plt.imshow(np.squeeze(images[j]))
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    plt.close()

#----------------------------------------------------------------------------------------------------------------------------------------------------
def cvPlot(left, right, seg, disp):
    cv2.imshow('left', left)
    cv2.imshow('right', right)
    cv2.imshow('seg', seg)
    dispNorm = (disp*255).astype(np.uint8)
    cv2.imshow('disp post', dispNorm)

    cv2.waitKey(0)

#----------------------------------------------------------------------------------------------------------------------------------------------------
def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

#----------------------------------------------------------------------------------------------------------------------------------------------------
def display_data(display_list, title):
    plt.figure(figsize=(15, 15))
    #fig, axs = plt.subplots(3, 2, figsize=(6, 9))
    fig, axs = plt.subplots(4, 2, figsize=(6, 9))
    ax = axs.ravel()
    disps = []
    for i in range(len(display_list)+1):
        if i == len(display_list):
            if len(disps) > 0:
                mask = disps[0] > 0
                #img = ax[i].imshow(np.abs(disps[0]-disps[1])*mask, cmap='jet')
                img = ax[i].imshow(ErrorColorImg(disps[1], disps[0]))
                img.set_clim(0,70)
        else:
            ax[i].set_title(title[i])
            img = ax[i].imshow(display_list[i], cmap='jet')
            if 'disp' in title[i]:
                img.set_clim(0,70)
                disps.append(display_list[i])
            if 'seg' in title[i]:
                img.set_clim(0,19)#8)
            # if 'gradient' in title[i]:
            #     img.set_clim(0,1)
        colorbar(img)
        ax[i].axis('off')

#----------------------------------------------------------------------------------------------------------------------------------------------------
def display_data_many_disp(display_list, title):
    n = len(display_list)
    fig, axs = plt.subplots(2, n/2, figsize=(3*n/2, 6))
    ax = axs.ravel()
    for i in range(n):
        ax[i].set_title(title[i])
        img = ax[i].imshow(display_list[i])
        img.set_clim(0,50)
        colorbar(img)
        ax[i].axis('off')

#----------------------------------------------------------------------------------------------------------------------------------------------------
def ErrorColorImg(pred, gt):
    label_colors = np.array([[0., 0.0, 1.0  ],
                            [1.0, 0.0, 0],])
                            #[1., 0., 0.0  ]])
    disp_n = [0, 3]#, 6]
    zeros = (gt > 0.0)
    error = np.expand_dims(np.abs(pred*zeros - gt*zeros), axis=-1)
    r = np.zeros_like(error)
    g = np.zeros_like(error)
    b = np.zeros_like(error)
    #print('dp1, ', np.mean(error > 3))
    for l in range(0, len(disp_n)):
        idx = (error > disp_n[l])
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.concatenate([r, g, b], axis=-1)
    #print(rgb.shape)
    return rgb#(torch.abs(pred*zeros - gt*zeros) > 4.0/100.0) * 1.0