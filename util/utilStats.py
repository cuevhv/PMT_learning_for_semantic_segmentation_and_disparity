from PIL import Image
from util.utilCityscape import ImgId2trainId
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
def getDatasetStats(colorL_list, seg_list, args, labels):
    
    print('training data length: ', len(colorL_list))
    save_segment = []
    save_color = []

    pxl_total = 0
    pxl_perClass = np.array([0]*labels, dtype=np.float32)
    for i in tqdm(range(len(colorL_list))):
        seg = Image.open(seg_list[i])
        seg = np.array(seg)
        if args.dataset == 'kitti':
            segment = ImgId2trainId(seg, labels)
        #segment = np.zeros((seg.shape[0], seg.shape[1], labels))
        pxl_total += seg.shape[0] * seg.shape[1]
        pxl_perClass += np.sum(segment, axis=(0,1))
        # if np.any(segment[:,:,18]):
        #     plt.imshow(np.argmax(segment, -1))
        #     plt.show()
        # for j in range(labels):
        #     segment[:,:,j] = (seg == j).astype(np.uint8)
        #     pxl_perClass[j] += np.sum(segment[:,:,j])
    #         if j == 6 and np.sum(segment[:,:,j]) > 0:
    #             save_segment.append(np.sum(segment[:,:,j]))
    #             save_color.append(colorL_list[i])
    #         #     print(pxl_perClass[j])
    #         #     plt.figure()
    #         #     plt.imshow(segment.argmax(axis=-1))
    #         #     plt.show()
    # indx = np.argsort(save_segment)[::-1]
    # save_color = np.array(save_color)
    # save_segment = np.array(save_segment)
    # print(save_segment[indx])
    # for file_name in save_color[indx][:12]:
    #     print(file_name)
    
        # left = Image.open(colorL_list[i])
        # left = np.array(left)
        # plt.figure()
        # plt.subplot(211)
        # plt.imshow(left)
        # plt.subplot(212)
        # plt.imshow(segment.argmax(axis=-1))
        # plt.show()

    #print(['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'])
    total_pixels = np.sum(pxl_perClass)
    np.set_printoptions(suppress=True)
    print(pxl_perClass)
    print(np.argsort(pxl_perClass))
    print(pxl_perClass/pxl_total)
    print(total_pixels/(labels*pxl_perClass))
    input()
    ''' weights [  0.22151023   1.5008001    0.5451698    4.3949537    8.908529
   4.4717636   25.78316      9.138443     0.16472627   0.5120778
   0.51006365  60.924007   396.55142      0.8575901   26.111387
 500.89224     17.682869   608.4612     172.94113   ]'''

def getLoaderStats(dataLoader, labels):
    pxl_total = 0
    pxl_perClass = []
    pxl_perClass = [0]*labels

    for data in tqdm(dataLoader):
        seg_out = data[1]['seg_out']
        img_out = data[0]['left']
        for i in range(seg_out.shape[0]):
            pxl_total += seg_out[i].shape[0] * seg_out[i].shape[1]
            seg = seg_out[i].argmax(axis=-1)
            img = np.asarray(tf.keras.preprocessing.image.array_to_img(np.squeeze(img_out[i])))
            
            for j in range(labels):
                pxl_perClass[j] += np.sum(seg == j)

            # plt.figure()
            # plt.subplot(211)
            # plt.imshow(seg)
            # plt.subplot(212)
            # plt.imshow(img)
            # plt.show()
            # plt.close()
        print(pxl_perClass)
    print(['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'])
    print(np.array(pxl_perClass))
    print(np.array(pxl_perClass)/pxl_total)
    print('total pixels: ', pxl_total)
    print(pxl_total/((labels) * np.array(pxl_perClass)))
    input()