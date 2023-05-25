import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from skimage import filters
#custom
import util.utilIOPfm as readIO

#------------------------------------------------------------------------------
def cropImage(img, crop_h, crop_w, centered=True):
    if len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape

    if centered:
        min_h, max_h = (h-crop_h)/2, (h+crop_h)/2
        min_w, max_w = (w-crop_w)/2, (w+crop_w)/2
    else:
        min_h, max_h = 0, crop_h
        min_w, max_w = 0, crop_w

    return img[min_h: max_h, min_w: max_w]

#------------------------------------------------------------------------------
def preProcData(left, right, seg, disp, crop_h, crop_w, activation_output, dataset_name, centered=True):
    import matplotlib.pyplot as plt
    labels = 0
    colorL = cropImage(left, crop_h, crop_w, centered)#/255.0
    colorR = cropImage(right, crop_h, crop_w, centered)#/255.0
    segCrop = cropImage(seg, crop_h, crop_w, centered)
    disp = cropImage(disp, crop_h, crop_w, centered)

    if dataset_name == 'garden':
        labels = 8+1
        f = 640
        b = 0.03
        disp = np.where(disp > 0, f*b* 1/disp, 0)

    segment = np.zeros((segCrop.shape[0], segCrop.shape[1], labels))
    for i in range(labels):
        segment[:,:,i] = (segCrop == i+1).astype(np.uint8)
    #segCrop -= 1
    #categorical = tf.keras.utils.to_categorical(segCrop, labels)

    if activation_output == 'tanh':
        min_d = 0
        max_d = disp.shape[1]/2.0
        disp[disp > max_d] = max_d
        disp = np.where(disp != 0, 20 * disp/float(max_d) - 10, -10)
        # plt.figure()
        # plt.imshow(dispNorm)
        # plt.figure()
        # plt.hist(dispNorm)
        # plt.show()
    elif activation_output == 'sigmoid':
        min_d = 0
        max_d = disp.shape[1]/2.0
        disp[disp > max_d] = max_d
        disp = disp/max_d
    else:
        min_d = 0
        max_d = disp.shape[1]/2.0
        disp[disp > max_d] = max_d

    return colorL, colorR, segment, disp

#----------------------------------------------------------------------------------------------------------------------------------------------------
def dataLoader(n_data, colorL_list, colorR_list, disp_list, seg_list, dataset_name, activation_output, preprocess, crop_h, crop_w, edges=False, centered=True):
    left_data = []
    right_data = []
    disp_data = []
    seg_data = []
    edges_left = []
    edges_disp = []
    count = np.zeros(10)

    for i in tqdm(range(0,len(seg_list))):
        if len(seg_data) >= n_data:
            break
        if dataset_name == 'sceneflow':
            disp = readIO.read(disp_list[i]).astype(float)
            seg = readIO.read(seg_list[i])[:,:,::-1]
        if dataset_name == 'city':
            disp = cv2.imread(disp_list[i], -1)
            disp =  ( disp - 1. ) / 256.0
            seg = readIO.read(seg_list[i])[:,:,::-1]
        if dataset_name == 'garden':
            im = Image.open(seg_list[i])
            seg = np.array(im)
            if np.sum(seg == 9)/float(seg.shape[0]*seg.shape[1]) >= 0.6:
                continue
            labels = np.unique(seg)
            count[labels] += 1

            disp = readIO.read(disp_list[i])

        colorL = image.load_img(colorL_list[i])
        colorL = image.img_to_array(colorL)

        colorR = image.load_img(colorR_list[i])
        colorR = image.img_to_array(colorR)

        if preprocess:
            c1, c2, s, d = preProcData(colorL, colorR, seg, disp, crop_h, crop_w, activation_output, dataset_name, centered=centered)
        else:
            c1, c2, s, d = colorL, colorR, seg, disp

        if edges:
            left_post = preprocess_input(np.expand_dims(c1, axis=0))

            from skimage.color import rgb2gray
            e_left = filters.sobel(rgb2gray(c1/255.0))
            e_disp = filters.sobel(d)
            if np.min(e_disp) < 0 or np.max(e_disp) > 1:
                raw_input('error')
            edges_disp.append(e_left)
            edges_left.append(e_disp)
            
        left_data.append(c1)
        right_data.append(c2)
        disp_data.append(d)
        seg_data.append(s)
        #gtPlot(c1, c2, s, d)

    left_data = preprocess_input(np.asarray(left_data))
    right_data = preprocess_input(np.asarray(right_data))
    disp_data = np.expand_dims(np.asarray(disp_data), axis=-1)
    # seg_data = np.expand_dims(np.asarray(seg_data), axis=-1)
    seg_data = np.array(seg_data)

    edges_left = np.expand_dims(np.array(edges_left), axis=-1)
    edges_disp = np.array(edges_disp)
    print('labels: ', count)
    if edges:
        return left_data, right_data, disp_data, seg_data, edges_left, edges_disp
    else:
        return left_data, right_data, disp_data, seg_data, [], []
