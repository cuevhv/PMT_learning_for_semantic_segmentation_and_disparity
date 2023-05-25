#!/usr/bin/python

# Obtain disparity files from dispersion ".exr" files. With option to show example results.

import OpenEXR
import Imath
import array
import numpy as np
import csv, os, sys
import time
import datetime
import h5py
import matplotlib.pyplot as plt
import glob
import cv2

# ----------------------------------------------------------------------------
def read_files(path):
    L_name = path+'/depth0*_L.exr'
    L_list = [i for i in sorted(glob.glob(L_name))]
    R_name = path+'/depth0*_R.exr'
    R_list = [i for i in sorted(glob.glob(R_name))]
    return L_list, R_list

# ----------------------------------------------------------------------------
def __disp(Z):
    fl = (35.0/32.0)* 752.0
    return 0.3 * fl / Z + 0.00001

# ----------------------------------------------------------------------------
def exr2numpy(exr, maxvalue=1.,normalize=True):
    """ converts 1-channel exr-data to 2D numpy arrays """
    if type(exr) is not np.ndarray:
        file = OpenEXR.InputFile(exr)

        # Compute the size
        dw = file.header()['dataWindow']
        sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        # Read the three color channels as 32-bit floats
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        (R) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R") ]

        # create numpy 2D-array
        img = np.zeros((sz[1],sz[0],3), np.float64)

        # normalize
        data = np.array(R)
        data[data > maxvalue] = maxvalue
    else:
        data = exr
        img = np.zeros((data.shape[0],data.shape[1],3), np.float64)

    dispar = __disp(data)

    if normalize:
        normalized_img = data/6.0
        normalized_img = np.array(normalized_img).reshape(img.shape[0],-1)
        #data /= np.max(data)

    depth_img = np.array(data).reshape(img.shape[0],-1)
    disp_img = np.array(dispar).reshape(img.shape[0],-1)  # !
    translated_disp = 0*np.ones(disp_img.shape)

    for i in range(disp_img.shape[0]):
        for j in range(disp_img.shape[1]):
            if j+disp_img[i,j] < disp_img.shape[1]:
                if depth_img[i,j] < 6:
                    translated_disp[i, j+disp_img[i,j].astype(np.uint8)] = depth_img[i,j]


    if normalize:
        return depth_img, disp_img, translated_disp, normalized_img

    else:
        return depth_img, disp_img, translated_disp, 0

# ----------------------------------------------------------------------------
def recover_class(depth_Nin, scale = 1):
    depth_N = depth_Nin.copy()
    if np.max(np.max(depth_N)) > 1:
        depth_N /= 255.0
    return scale*depth_N

# ----------------------------------------------------------------------------
def get_inverse(xin):
    x = xin.copy()
    if np.max(np.max(x)) > 1:
        x -= 255.0
    else:
        x -= 1

    x[x < 0] *=-1.0
    return x

# ----------------------------------------------------------------------------
def compare(img_1, img_2):
    dif = abs(img_1-img_2)
    print(np.sum(np.sum(dif))/(img_1.shape[0]*img_1.shape[1]))

    plt.subplot(1,3,1)
    plt.imshow(img_1)
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(img_2)
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(dif)
    plt.colorbar()
    plt.show()

# ----------------------------------------------------------------------------
def save_pfm(filename, image, scale = 1):
    file = open(filename, 'w')
    color = None

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)  
    file.close()

# ----------------------------------------------------------------------------
def exportToPfm(disp_dataL, path_exr):
    paths_segm = path_exr.split('/')
    curr_folder = path_init + paths_segm[2] + "/disp"
    os.system("mkdir -p " + curr_folder)
    #new_path = curr_folder + '/disp' + paths_segm[4].split('depth')[1][:-4] + '.pfm'
    #save_pfm(new_path, disp_dataL)
    new_path = curr_folder + '/disp' + paths_segm[4].split('depth')[1][:-4] + '.png'
    cv2.imwrite(new_path, disp_dataL)


## Use to obtain disparity .pfm
plot = False
only_show = False
left_exr, right_exr = [], []
path_init = "../ROSeS_depth_4_types_ORIGINAL/"

for i in range(37):
    left_exr_aux, right_exr_aux = read_files(path_init + str(i) + "/depth")
    left_exr += left_exr_aux; right_exr += right_exr_aux 


for i in range(len(left_exr)):
    depth_dataL, disp_dataL, translated_dispL, depth_N_L = exr2numpy(left_exr[i], maxvalue=6.0, normalize=True)
    if not only_show:
        exportToPfm(disp_dataL, left_exr[i])
        print(left_exr[i])
    else:
        cv2.imwrite('depth11.png', disp_dataL/752*255)
        depth_dataR, disp_dataR, translated_dispR, depth_N_R = exr2numpy(right_exr[i], maxvalue=6.0, normalize=True)
        #compare(translated_dispR, depth_dataL)
        inverted_depth = get_inverse(depth_N_L)

if plot and only_show:
    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(depth_dataL)
    plt.title('Depth L')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(np.abs(depth_dataL-translated_dispR))

    plt.title('Disparity L')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(depth_N_L)

    plt.title('normalized depth')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(get_inverse(depth_N_L))
    plt.title('inverted normalized depth')
    plt.colorbar()

    plt.show()

