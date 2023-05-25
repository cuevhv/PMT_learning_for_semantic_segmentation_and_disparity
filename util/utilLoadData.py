#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import cv2
import gc
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.utils import shuffle
import util.utilIOPfm as readIO
from tqdm import tqdm
from scipy import signal
np.random.seed(233)
    
    
def cropImage(img, crop_h, crop_w, centered='center', start_crop_x=0, start_crop_y=0):
   
    if len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    if crop_h > 0:
        if centered == 'center':
            min_h, max_h = (h-crop_h)/2, (h+crop_h)/2
            min_w, max_w = (w-crop_w)/2, (w+crop_w)/2
        
        elif centered == 'bottom':
            return img[h-256: h, (w-256)/2: (w+256)/2]
        else:
            min_h, max_h = 0, crop_h
            min_w, max_w = 0, crop_w
       
        if centered == 'random':
            min_h = start_crop_y
            max_h = start_crop_y+crop_h
            min_w = start_crop_x
            max_w = start_crop_x+crop_w
        return img[min_h: max_h, min_w: max_w]
    else:
        return img

        
#--------------------------------------------------------------------------------
def load_images(array_x_files, array_y_files, config, activation_output, crop, centered):
    data_size = len(array_x_files)
    X = [[] for i in range(len(array_x_files[0]))]
    Y = [[] for i in range(len(array_y_files[0]))]
    
    color = image.load_img(array_x_files[0][0])
    color = image.img_to_array(color)
    if len(color.shape) == 3:
        h, w, _ = color.shape
    else:
        h, w = color.shape
    
    if np.random.choice([0, 1], p = [0.3, 0.7]):
        y_start = h-crop[0] - 50
    else:
        y_start = 0
    start_crop_y = np.random.randint(y_start, h - crop[0] + 1, data_size)
    start_crop_x = np.random.randint(0, w - crop[1] + 1, data_size)

    if config.dataset == 'garden':
        count = np.zeros(10)
        labels = 8+1
        f = 640
        b = 0.03
    for idx in tqdm(range(data_size)):
        for i in range(len(array_y_files[idx])):
            if i == 1: #seg
                if config.dataset == 'garden':
                    im = Image.open(array_y_files[idx][i])
                    seg = np.array(im)
                    # if np.sum(seg == 9)/float(seg.shape[0]*seg.shape[1]) >= 0.6:
                    #     continue
                uniquelabels = np.unique(seg)
                count[uniquelabels] += 1
                seg = cropImage(seg, crop[0], crop[1], centered, start_crop_x[idx], start_crop_y[idx])

                # plt.show()
                segment = np.zeros((seg.shape[0], seg.shape[1], labels))
                for j in range(labels):
                    if config.dataset == 'garden':
                        segment[:,:,j] = (seg == j+1).astype(np.uint8)
                Y[1].append(segment)

            if i == 0: #disp
                disp = readIO.read(array_y_files[idx][i])
                disp = cropImage(disp, crop[0], crop[1], centered, start_crop_x[idx], start_crop_y[idx])
                if config.dataset == 'garden': 
                    disp = np.where(disp > 0, f*b* 1/disp, 0)
                min_d = 0
                max_d = disp.shape[1]/2.0
                if activation_output == 'tanh':
                    disp[disp > max_d] = max_d
                    disp = np.where(disp != 0, 20 * disp/float(max_d) - 10, -10)
                elif activation_output == 'sigmoid':
                    disp[disp > max_d] = max_d
                    disp = disp/max_d
                    disp[disp > max_d] = max_d
                Y[0].append(disp)
       

        for i in range(len(array_x_files[idx])):
            color = image.load_img(array_x_files[idx][i])                
            color = image.img_to_array(color)
            color = cropImage(color, crop[0], crop[1], centered, start_crop_x[idx], start_crop_y[idx]) #height, width
            # if i == 0:
            #     leftC = color
            X[i].append(color)
        # import tensorflow as tf
        # plt.figure()
        # plt.imshow(tf.keras.preprocessing.image.array_to_img(leftC))
        # plt.figure()
        # plt.imshow(tf.keras.preprocessing.image.array_to_img(color))
        # plt.figure()
        # plt.imshow(disp)
        # scaled_intensity = (0.2125*leftC[:,:,0] + 0.7154*leftC[:,:,1] + 0.0721*leftC[:,:,2])/255.0
        # plt.figure()
        # gradients1 = laplacian(scaled_intensity, sigma=0.9, th=0.9, smooth=True)
        # plt.imshow(gradients1)
        
        # plt.figure()
        # gradients2 = laplacian(disp, sigma=0.9, th=0.3, smooth=False)
        # plt.imshow(gradients2)
        # gradients2[gradients2 == 0] = -20
        # plt.figure()
        # plt.imshow(np.equal(gradients1, gradients2))
        # plt.show()
        
    for i in range(len(X)):
        X[i] = preprocess_input(np.asarray(X[i]))

    for i in range(len(Y)):
        if i == 0: #disp
            Y[i] = np.expand_dims(np.asarray(Y[i]), axis=-1)
        if i == 1:
            Y[i] = np.array(Y[i])
    del color, disp, seg, segment
    gc.collect()
    return X, Y

def laplacian(img, sigma=0.9, th=0.4, smooth=True):
    # img = np.copy(image)
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    kernel_sobel_h = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    kernel_sobel_v = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    g_kernel = gaussian_kernel(9, sigma)
    if smooth:
        img = signal.convolve2d(img, g_kernel, boundary='symm', mode='same')

    horizontal_grad = signal.convolve2d(img, kernel_sobel_h, boundary='symm', mode='same')
    vertical_grad = signal.convolve2d(img, kernel_sobel_v, boundary='symm', mode='same')
    magnitude = np.sqrt(np.square(horizontal_grad) + np.square(vertical_grad))
    print (np.min(magnitude), np.max(magnitude))
    grad = signal.convolve2d(magnitude, kernel, boundary='symm', mode='same')
    grad[grad<=0] = 0
    grad = 1/(1+np.exp(-10*grad/1+1))
    grad[grad<=th] = 0
    grad[grad>th] = 1
    return grad

def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2.0), size // 2.0, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    if verbose:
        plt.imshow(kernel_2D, interpolation='none',cmap='gray')
        plt.title("Image")
        plt.show()
    return kernel_2D
    
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

# ----------------------------------------------------------------------------
def load_singe_image(array_x_files, array_y_files, config):
    X = []
    Y = []

    img_x = array_x_files
    #img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
    img_y = array_y_files

    if config.resize[0] != -1:
        img_x = cv2.resize(img_x, (config.resize[0], config.resize[1]), interpolation = cv2.INTER_CUBIC)
        img_y = cv2.resize(img_y, (config.resize[0], config.resize[1]), interpolation = cv2.INTER_CUBIC)

    if config.equ != 'none':
        img_x = utilImage.apply_equalization(img_x, config.equ)

    if config.norm == 'minmax':
        img_x = cv2.normalize(img_x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    #cv2.imshow("Img_x", img_x)
    #cv2.imshow("Img_y", img_y)
    #cv2.waitKey(0)

    X.append( img_x )
    Y.append( img_y )

    Y = np.asarray(Y).astype('float32') #/ 255.
    Y = Y > 0.1
    #Y = np.logical_not(Y)   # TODO - comprobar a cambiar !!!!!!!!!!!!!
    # Si se cambia aquí hay que cambiar la función de evaluación (la inversión) y  el cval del aumentado
    Y = Y.astype('float32')


    X = np.asarray(X).astype('float32')

    # Without equalization
    MEAN = 133.752400094
    STD = 41.4346141506
    if config.equ == 'hsv':
        MEAN = 118.439312038
        STD = 69.3948067857

    if config.norm == '255':
        X = X / 255.
    elif config.norm == 'standard':
        X = (X - np.mean(X)) / (np.std(X) + 0.00001)
    elif config.norm == 'mean':
        X = X - np.mean(X)
    elif config.norm == 'fstandard':
        X = (X - MEAN) / STD
    elif config.norm == 'fmean':
        X = X - MEAN
    elif config.norm == 'frgb':
        X[..., 0] -= 103.939
        X[..., 1] -= 116.779
        X[..., 2] -= 123.68

    util.print_stats('X', X)
    util.print_stats('Y', Y)
    #print('Sample of X values:', X[0,X.shape[1]/2,:10])
    #print('Sample of Y values:', Y[0,X.shape[1]/2,:10])

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 3 if len(X.shape) == 4 else 1)
    Y = Y.reshape(Y.shape[0], Y.shape[1], Y.shape[2], 1)

    return X, Y


class LazyFileLoader:
    def __init__(self, array_x_files, array_y_files, page_size, ptr_load_images_func, load_images_config, activation_output, crop, centered):
        assert len(array_x_files) > 0
        assert len(array_x_files)== len(array_y_files)
        self.array_x_files = array_x_files
        self.array_y_files = array_y_files
        self.ptr_load_images_func = ptr_load_images_func
        self.load_images_config = load_images_config
        self.pos = 0
        self.activation_output = activation_output
        self.crop = crop
        self.centered = centered
        if page_size <= 0:
            self.page_size = len(array_x_files)
        else:
            self.page_size = page_size

    def __len__(self):
        return len(self.array_x_files)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def truncate_to_size(self, truncate_to):
        self.array_x_files = self.array_x_files[0:truncate_to]
        self.array_y_files = self.array_y_files[0:truncate_to]

    def set_x_files(self, array_x_files, array_y_files):
        self.array_x_files = array_x_files
        self.array_y_files = array_y_files

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def set_pos(self, pos):
        self.pos = pos

    def shuffle(self):
        self.array_x_files, self.array_y_files = \
                shuffle(self.array_x_files, self.array_y_files, random_state=0)

    def next(self):
        psize = self.page_size
        if self.pos + psize >= len(self.array_x_files):  # last page?
            if self.pos >= len(self.array_x_files):
                raise StopIteration
            else:
                psize = len(self.array_x_files) - self.pos

        print('Loading page from {} to {}...'.format(self.pos, self.pos + psize))
        page_data = self.ptr_load_images_func(
                                                    self.array_x_files[self.pos:self.pos + psize],
                                                    self.array_y_files[self.pos:self.pos + psize],
                                                    self.load_images_config,
                                                    self.activation_output,
                                                    self.crop,
                                                    self.centered)
        self.pos += self.page_size

        return page_data

