from torch.utils.data import Dataset, DataLoader
from util.utilCityscape import ImgCol2id, ImgId2trainId
import os
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from skimage import io, transform
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import util.utilIOPfm as readIO
import cv2
from skimage import filters
from skimage.measure import label as sklabel
from skimage.color import rgb2gray
from skimage.morphology import dilation, square
from matplotlib import cm
import pandas as pd
from scipy import ndimage as ndi


# import imgaug as ia
# from imgaug import augmenters as iaa
class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, array_x_files, array_y_files, n_labels, max_d, datasetName, normalize,
                 output_activation='sigmoid', only_test=False, transform=None, hdf5='', class_lbl_list='',
                 transform_color=None, to_tensor=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hdf5 = hdf5  # '/home/hanz/Documents/phd/joint_learning/jointsegdispnet/results/test3.hdf5'
        if self.hdf5:
            import h5py
            print('using h5py')
            self.file_f = h5py.File(self.hdf5, mode='r')
            print('datasize:', len(self.file_f['left']))
        else:
            assert len(array_x_files) > 0
            assert len(array_x_files) == len(array_y_files)
        self.array_x_files = array_x_files
        self.array_y_files = array_y_files
        self.datasetName = datasetName
        self.normalize = normalize
        self.output_activation = output_activation
        self.transform = transform
        self.transform_color = transform_color
        self.to_tensor = to_tensor
        self.n_labels = n_labels
        self.max_d = max_d
        self.only_test = only_test
        self.f = 640
        self.b = 0.03
        self.class_lbl_list = class_lbl_list
        if self.class_lbl_list:
            self.class_lbl_csv = pd.read_csv(class_lbl_list)
            if self.datasetName == 'garden':
                self.balance_class = np.array([0, 2, 3, 4, 5, 6, 7])
            elif self.datasetName == 'roses':
                self.balance_class = np.array([0, 1]) # TODO                
            else:
                self.balance_class = np.array([3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18])
            self.class_count = np.zeros(self.n_labels + 1)  # np.zeros_like(self.balance_class)
            for i in self.balance_class:
                self.class_lbl_csv['count_' + str(i)] = np.ones(len(self.class_lbl_csv))

    def get_indx_per_class(self):
        max_count = np.max(self.class_count[self.balance_class])
        # print(self.balance_class)
        if np.all(self.class_count[self.balance_class] == max_count):
            # random_choice = np.random.choice(self.balance_class)
            rand_idx = torch.randint(0, len(self.balance_class), (1,)).item()
            random_choice = self.balance_class[rand_idx]
            class_indx = np.where(self.balance_class == random_choice)[0][0]
        else:
            sample_classes_list = np.where(self.class_count[self.balance_class] < max_count)[0]
            # class_indx = np.random.choice(sample_classes_list)
            rand_idx = torch.randint(0, len(sample_classes_list), (1,)).item()
            class_indx = sample_classes_list[
                rand_idx]  # torch.multinomial(torch.tensor(sample_classes_list).float(), 1).item()

        csv_indx = str(self.balance_class[class_indx])
        count_csv_name = 'count_' + csv_indx
        imgs_indx = self.class_lbl_csv[self.class_lbl_csv[csv_indx] == 1]['n'].to_numpy()

        # I count the images I used in total and not per class
        img_prob = self.class_lbl_csv[count_csv_name][imgs_indx].to_numpy()
        max_count = np.max(img_prob)

        if np.all(img_prob == max_count):
            # indx = np.random.choice(imgs_indx)
            rand_idx = torch.randint(0, len(imgs_indx), (1,)).item()
            indx = imgs_indx[rand_idx]  # torch.multinomial(torch.tensor(imgs_indx).float(), 1).item()
        else:
            sample_classes_list = np.where(img_prob < max_count)[0]
            # indx = np.random.choice(imgs_indx[sample_classes_list])
            rand_idx = torch.randint(0, len(imgs_indx[sample_classes_list]), (1,)).item()
            indx = imgs_indx[sample_classes_list][rand_idx]

        self.class_lbl_csv[count_csv_name][indx] += 1
        if self.datasetName == 'garden':
            lowest_classes = []
        if self.datasetName == 'roses':
            lowest_classes = [] # TODO
        else:
            lowest_classes = [14, 15, 16]
        if self.balance_class[class_indx] in lowest_classes:
            self.class_count[self.balance_class[class_indx]] += 0.5
        else:
            self.class_count[self.balance_class[class_indx]] += 1

        # print(indx, self.balance_class[class_indx], 'iloc')
        # print(self.class_lbl_csv.iloc[[indx], :20])
        # print(self.class_lbl_csv.iloc[[indx], 20:])
        # print(self.balance_class[class_indx])
        # print(self.class_count)
        # print(imgs_indx.shape, img_prob.shape)
        # print('indx', indx)
        # print('count 1', self.class_count)
        return indx, self.balance_class[class_indx]

    def __len__(self):
        if self.hdf5:
            return len(self.file_f['left'])
        else:
            return len(self.array_x_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.class_lbl_list:
            idx, class_indx = self.get_indx_per_class()

        if self.hdf5:
            left_image = self.file_f['left'][idx].astype(np.float32) / 255.0
            right_image = self.file_f['right'][idx].astype(np.float32) / 255.0
            seg_image = self.file_f['seg'][idx].astype(np.float32)
            disp_image = self.file_f['disp'][idx].astype(np.float32)
            left_edges = self.file_f['disp'][idx].astype(np.float32)
        else:
            left_image = (io.imread(self.array_x_files[idx][0]))[:,:,:3]  # /255.0).astype(np.float32)
            right_image = (io.imread(self.array_x_files[idx][1]))[:,:,:3]  # /255.0).astype(np.float32)
            # left_image = (left_image - self.normalize[0])/self.normalize[1]
            # right_image = (right_image - self.normalize[0])/self.normalize[1]
            seg = Image.open(self.array_y_files[idx][1])
            seg = np.array(seg)
            left_edges = filters.sobel(rgb2gray(left_image))
            inst = np.array(Image.open(self.array_y_files[idx][2]))
            edge_sobel = filters.sobel(inst)
            left_edges = ((edge_sobel > 0) * 1).astype(np.float32)
            # plt.subplot(411)
            # plt.imshow(left_image)
            # plt.subplot(412)
            # plt.imshow(inst)
            # plt.subplot(413)
            # edges = (edge_sobel > 0)*1
            # edges = 1-dilation(edges, square(3))
            # blobs = sklabel(edges, 8)
            # label_objects, nb_labels = ndi.label(edges)
            # print(nb_labels)
            # plt.imshow(edges)
            # plt.subplot(414)
            # plt.imshow(label_objects, cmap=plt.get_cmap('terrain'))
            # plt.show()

            if self.datasetName == 'garden':
                disp_image = readIO.read(self.array_y_files[idx][0])
                with np.errstate(invalid='ignore', divide='ignore'):
                    disp_image = np.where(disp_image > 0, self.f * self.b * 1 / disp_image, 0)

            if self.datasetName == 'roses': #TODO
                disp_image = readIO.read(self.array_y_files[idx][0])
                with np.errstate(invalid='ignore', divide='ignore'):
                    disp_image = np.where(disp_image > 0, self.f * self.b * 1 / disp_image, 0)

            if self.datasetName == 'kitti' or self.datasetName == 'cityscapes':# or self.datasetName == 'roses':
                disp = cv2.imread(self.array_y_files[idx][0], -1)
                disp_image = disp.astype(np.float32) / 256.0
                seg_image = ImgId2trainId(seg, self.n_labels).astype(np.float32)

            if self.output_activation != 'linear':
                disp_image[disp_image > self.max_d] = self.max_d
            # disp_image = np.round(disp_image)
            if self.output_activation == 'sigmoid':
                disp_image = disp_image / self.max_d
                # delta = 10/(2.0-0.1); np.floor(delta*np.floor(a/delta)) 
                # plt.imshow(disp_image)
                # plt.show()
            if self.output_activation == 'tanh':
                disp_image = np.where(disp_image != 0, 2 * disp_image / float(self.max_d) - 1, -1)

            if self.datasetName != 'kitti' and self.datasetName != 'cityscapes': # and self.datasetName != 'roses':
                seg_image = np.zeros((seg.shape[0], seg.shape[1], self.n_labels), dtype=np.float32)
                for j in range(self.n_labels):
                    if self.datasetName == 'roses':
                        # Each dimension for each class
                        threshold_background = 128
                        seg_binary = np.zeros(seg.shape)
                        seg_binary[seg > threshold_background] = 1 # background or not #TODO set threshold
                        seg_binary = seg_binary[:,:,2] # Delete last dimension, only 1 channel
                        seg_image[:, :, j] = (seg_binary == j ).astype(np.uint8)   
                    else:
                        seg_image[:, :, j] = (seg == j + 1).astype(np.uint8)

        if self.datasetName == 'kitti':
            if not np.all(left_image.shape[0:2] == disp_image.shape[0:2]):
                seg_image = transform.resize(seg_image, (left_image.shape[0], left_image.shape[1]))
                disp_image = transform.resize(disp_image, (left_image.shape[0], left_image.shape[1]))

        sample = {'left': left_image,
                  'right': right_image,
                  'seg': seg_image,
                  'disp': np.expand_dims(disp_image, axis=-1),
                  'edges': np.expand_dims(left_edges, axis=-1)}

        if self.class_lbl_list:
            sample['class_indx'] = class_indx
        else:
            sample['class_indx'] = -1

        if self.transform:
            sample = self.transform(sample)
            # plt.subplot(3,2,1)
            # old = sample['left']
            # plt.imshow(np.transpose(sample['left'], (0,1,2)))
            # plt.subplot(3,2,2)
            # old2 = sample['right']
            # plt.imshow(np.transpose(sample['right'], (0,1,2)))

            if self.transform_color and torch.multinomial(torch.tensor([0.1, 0.9]), 1).item():
                # import time
                # s = time.time()
                sample['left'] = Image.fromarray(sample['left'])
                # sample['left'] = np.array(self.transform_color(sample['left'])) #transforms.ToTensor()(self.transform_color(sample['left']))

                sample['right'] = Image.fromarray(sample['right'])
                # sample['right'] = np.array(self.transform_color(sample['right'])) #transforms.ToTensor()(self.transform_color(sample['right']))
                sample['left'], sample['right'] = self.adjust_brightess(sample['left'], sample['right'])
                sample['left'], sample['right'] = np.array(sample['left']), np.array(sample['right'])
                # print(time.time()-s)
            # plt.subplot(3,2,3)
            # plt.imshow(np.transpose(sample['left'], (0,1,2)))
            # plt.subplot(3,2,5)
            # plt.imshow(np.transpose(np.abs(old-sample['left']), (0,1,2)))
            # plt.subplot(3,2,4)
            # plt.imshow(np.transpose(sample['right'], (0,1,2)))
            # plt.subplot(3,2,6)
            # plt.imshow(np.transpose(np.abs(old2-sample['right']), (0,1,2)))
            # plt.show()

        sample['left'] = ((sample['left'] / 255.0 - self.normalize[0]) / self.normalize[1]).astype(np.float32)
        sample['right'] = ((sample['right'] / 255.0 - self.normalize[0]) / self.normalize[1]).astype(np.float32)
        sample = self.to_tensor(sample)

        if self.only_test:
            _, h, w = sample['left'].shape
            sample['seg'] = torch.zeros((self.n_labels + 1, h, w))
            sample['disp'] = torch.zeros((1, h, w))
            metadata = [self.array_x_files[idx][0], self.array_x_files[idx][1]]
        else:
            if self.hdf5:
                metadata = self.hdf5
            elif self.datasetName == 'kitti':
                metadata = [self.array_x_files[idx][0], self.array_x_files[idx][1]]
            else:
                metadata = [self.array_y_files[idx][0], self.array_y_files[idx][1]]
        sample['meta'] = metadata

        return sample

    def adjust_brightess(self, left, right, brightness=0.5, contrast=0.2, saturation=0.5):
        # if torch.multinomial(torch.tensor([0.5,0.5]), 1).item():
        lower = max(0, 1 - brightness)
        upper = brightness + 1
        b = ((upper - lower) * torch.rand((1)) + lower).item()
        left = TF.adjust_brightness(left, b)
        right = TF.adjust_brightness(right, b)

        # if torch.multinomial(torch.tensor([0.5,0.5]), 1).item():
        lower = max(0, 1 - contrast)
        upper = contrast + 1
        c = ((upper - lower) * torch.rand((1)) + lower).item()
        left = TF.adjust_contrast(left, c)
        right = TF.adjust_contrast(right, c)

        # if torch.multinomial(torch.tensor([0.5,0.5]), 1).item():
        lower = max(0, 1 - saturation)
        upper = saturation + 1
        s = ((upper - lower) * torch.rand((1)) + lower).item()
        left = TF.adjust_saturation(left, s)
        right = TF.adjust_saturation(right, s)

        if torch.multinomial(torch.tensor([0.1, 0.9]), 1).item():
            sigma = 0.15 + torch.rand((1)).item() * 1.15
            left = left.filter(ImageFilter.GaussianBlur(radius=sigma))
            right = right.filter(ImageFilter.GaussianBlur(radius=sigma))

        return left, right

# class colorAugment(object):
#     def __init__(self, aug):
#         self.aug = aug

#     def __call__(self, sample):
#         import time
#         t = time.time()
#         rand = torch.randint(0, 1000, (1,)).item()
#        # print(rand)
#         ia.random.seed(rand)
#         transform_color = self.aug.to_deterministic()
#         images = [sample['left'],
#                 sample['right'],
#                 sample['disp'],
#                 sample['seg'],
#                 sample['edges']]

#         import matplotlib.pyplot as plt
#         import time
#         # plt.subplot(3,2,1)
#         # old1 = images[0]
#         # plt.imshow(old1)
#         # plt.subplot(3,2,2)
#         # old2 = images[1]
#         # plt.imshow(images[1])

#         images[0] = transform_color(image=images[0])
#         images[1] = transform_color(image=images[1])
#         print('time aug:', time.time()-t)
#         # plt.subplot(3,2,3)
#         # plt.imshow(images[0])
#         # plt.subplot(3,2,4)
#         # plt.imshow(images[1])
#         # plt.subplot(3,2,5)
#         # plt.imshow(np.abs(images[0]-old1))
#         # plt.subplot(3,2,6)
#         # plt.imshow(np.abs(images[1]-old2))
#         # plt.show()

#         return {'left': images[0], 'right': images[1],
#                 'disp': images[2], 'seg': images[3],
#                 'edges': images[4]}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, datasetName, is_down=False, sliceandSwitch=False, augment_DoubleLeftImg=False,
                 focusPerson=False, resizeImg=False, flipHorizontal=False):
        assert isinstance(output_size, (int, tuple, list))
        self.is_down = is_down
        self.resizeImg = resizeImg
        self.sliceandSwitch = sliceandSwitch
        self.augment_DoubleLeftImg = augment_DoubleLeftImg
        self.flipHorizontal = flipHorizontal
        self.focusPerson = focusPerson
        self.datasetName = datasetName
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        if self.datasetName == 'kitti' or self.datasetName == 'cityscapes':
            self.class_count = np.ones(20)
            self.balance_class = np.array([3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18])
        elif self.datasetName == 'garden':
            self.class_count = np.ones(10)
            self.balance_class = np.array([0, 2, 3, 4, 5, 6, 7])
            # self.prob_class[[0, 1, 2, 8, 10]] = 0
            # self.prob_class[[11, 12]] = 1
            # self.prob_class /= self.prob_class.sum()
            # print('start', self.class_count)

    def __call__(self, sample):
        images = [sample['left'],
                  sample['right'],
                  sample['disp'],
                  sample['seg'],
                  sample['edges']]
        self.class_indx = sample['class_indx']
        # plt.subplot(131)
        # plt.imshow(sample['seg'].argmax(-1))
        # plt.subplot(132)
        # plt.imshow(sample['seg'].argmax(-1) == self.class_indx)
        if self.output_size[0] == 0:
            # scale = 0.5 #2
            # h, w = images[0].shape[:2]
            # dim = (round(w*scale), round(h*scale))
            # for i in range(len(images)):
            #     #images[i] = images[i][256+128:512+128, 512+256:1024+512-256]
            #     #h, w = images[i].shape[:2]
            #     #dim = (round(w*scale), round(h*scale))
            #     images[i] = cv2.resize(images[i] if i!=2 else images[i]*scale, dim, interpolation = cv2.INTER_AREA if i<2 else cv2.INTER_NEAREST)
            #     if len(images[i].shape) < 3:
            #        images[i] = np.expand_dims(images[i], -1)
            # return {'left': images[0], 'right': images[1],
            #     'disp': images[2], 'seg': images[3],
            #     'edges': images[4]}

            return sample
        if self.resizeImg:
            if self.datasetName == 'kitti':
                upper = 1.5
                lower = 0.90
            if self.datasetName == 'cityscapes':
                upper = 1.2 if self.output_size[0] < 512 else 1.5
                lower = np.ceil(self.output_size[0] / 1024 * 100) / 100  # 0.25 if self.output_size[0] < 512 else 0.50
            if self.datasetName == 'garden':
                upper = 1.2
                lower = 1.
            if self.datasetName == 'roses':
                upper = 1.2
                lower = 1.                
            if torch.multinomial(torch.tensor([0.2, 0.8]), 1).item():  # and self.datasetName in ['kitti', 'cityscapes']:
                scale = round(((upper-lower) * torch.rand(1) + lower).item(), 2) #scale [.25-.75] two decimals
                #print(scale)
                h, w = images[0].shape[:2]
                dim = (round(w * scale), round(h * scale))
                for i in range(len(images)):
                    # Reduction of dimension
                    if (len(images[i].shape) == 4):
                        # Reduction to 3 dimensions
                        images[i] = np.squeeze(images[i], axis=(3,))

                    images[i] = cv2.resize(images[i] if i != 2 else images[i] * scale, dim,
                                           interpolation=cv2.INTER_AREA if i < 2 else cv2.INTER_NEAREST)
                    if len(images[i].shape) < 3:
                        images[i] = np.expand_dims(images[i], -1)
        h, w = images[0].shape[:2]
        new_h, new_w = self.output_size
        # crop_top = torch.randint((h - new_h)//2 -20, (h - new_h)//2 + 20, (1,))
        # crop_left = torch.randint((w - new_w)//2 -10, (w - new_w)//2 + 10, (1,))

        if self.is_down:
            crop_top = (h - new_h)
            crop_left = (w - new_w) // 2
        else:
            if torch.multinomial(torch.tensor([0.2, 0.8]), 1).item() and self.datasetName == 'kitti':
                y_start = max(h - new_h - 100, 0)
            else:
                y_start = 0
            crop_left = -1
            crop_top = -1
            if self.focusPerson:
                crop_left, crop_top = self.cropPerson(images[3])
            if crop_left == -1:
                crop_top = torch.randint(y_start, h - new_h + 1, (1,))
                crop_left = torch.randint(0, w - new_w + 1, (1,))

        if self.sliceandSwitch:
            w, h, c = images[0].shape
            divisor = float(torch.randint(2, 6, (1,)))
            div_img = int(w / divisor)

        for i in range(len(images)):
            images[i] = images[i][crop_top: crop_top + new_h,
                        crop_left: crop_left + new_w]  # images[i] = [h,w,c]
            if self.sliceandSwitch:
                slice1 = images[i][:div_img, :, :]
                slice2 = images[i][div_img:, :, :]
                images[i] = np.concatenate((slice2, slice1), axis=0)

        if self.augment_DoubleLeftImg and torch.multinomial(torch.tensor([0.9, 0.1]), 1).item():
            images[0] = images[0][:, ::-1].copy()  # flip horizontally the left image
            images[1] = images[0]  # right = left
            images[2] = np.zeros_like(images[2]) + 0.0001
            images[3] = images[3][:, ::-1].copy()
            images[4] = images[4][:, ::-1].copy()

        if self.flipHorizontal and torch.multinomial(torch.tensor([0.5, 0.5]), 1).item() and self.datasetName == 'cityscapes':
            temp_left = images[0][:, ::-1].copy()
            images[0] = images[1][:, ::-1].copy()
            images[1] = temp_left
            translated_disp = np.zeros_like(images[2])
            translated_seg = np.zeros_like(images[3])
            r = np.arange(0, images[2].shape[0], 1)
            c = np.arange(0, images[2].shape[1], 1)
            cv, rv = np.meshgrid(c, r)
            cv_disp = (cv - images[2].squeeze(-1)).astype(np.int)
            cv_disp[cv_disp < 0] = 0
            cv_disp = cv_disp.flatten()
            images[2][rv.flatten(), cv_disp.flatten(), :] = images[2][rv.flatten(), cv.flatten(), :]
            images[3][rv.flatten(), cv_disp.flatten(), :] = images[3][rv.flatten(), cv.flatten(), :]
            images[2][:, -10:] = 0
            images[3][:, -20:, :] = 0
            mask = (np.sum(images[2], axis=2) == 0) * 1
            # plt.subplot(1,3,1), plt.imshow(mask)
            # plt.subplot(1,3,2), plt.imshow(images[3].argmax(-1))
            # plt.subplot(1, 3, 3), plt.imshow(images[2].squeeze(-1))
            # plt.show()
            images[3][:, :, -1] = mask
            images[3][:, :, :-1] *= (1 - mask[:, :, None])
            images[2] = images[2][:, ::-1, :].copy()
            images[3] = images[3][:, ::-1, :].copy()

        if False:  # flip everything
            for i in range(len(images)):
                plt.subplot(len(images), 2, (i + 1) * 2 - 1)
                plt.imshow(images[i].squeeze() if i != 3 else images[i].argmax(-1))
                # images[i] = cv2.flip(images[i], 1)
                images[i] = images[i][:, ::-1].copy()
                if i == 2:
                    translated_disp = np.zeros_like(images[i])
                    translated_seg = np.zeros_like(images[3])
                    for k in range(images[i].shape[0]):
                        for j in range(images[i].shape[1]):
                            print(j + images[i][k, j][0].astype(np.uint8) < images[i].shape[1],
                                  j + images[i][k, j][0].astype(np.uint8), images[i].shape[1])
                            if j + images[i][k, j][0].astype(np.uint8) < images[i].shape[1]:
                                translated_seg[k, j + images[i][k, j][0].astype(np.uint8)] = images[3][:, ::-1][k, j]
                                translated_disp[k, j + images[i][k, j][0].astype(np.uint8)] = images[i][k, j]
                    images[i] = translated_disp
                if i == 3:
                    images[i] = translated_seg
                plt.subplot(len(images), 2, (i + 1) * 2)
                plt.imshow(images[i].squeeze() if i != 3 else images[i].argmax(-1))
                # if len(images[i].shape) == 2:
                #   images[i] = images[i][:,:,None]
            plt.show()

        return {'left': images[0], 'right': images[1],
                'disp': images[2], 'seg': images[3],
                'edges': images[4]}

    def cropPerson(self, seg):
        h, w, _ = seg.shape
        person = 0
        if self.datasetName == 'kitti' or self.datasetName == 'cityscapes' or self.datasetName == 'garden':
            count_class = np.any(seg, axis=(0, 1))
            current_classes = np.where(count_class == 1)[0]
            # self.prob_class += count_class
            class_indx = np.intersect1d(current_classes, self.balance_class)
            # print('current class', current_classes)
            # print(class_indx)
            # plt.subplot(121)
            # plt.imshow(seg.argmax(-1))
            # plt.subplot(133)
            # plt.imshow(seg.argmax(-1) == self.class_indx)
            # plt.show()
            if len(class_indx) or self.class_indx != -1:
                if self.class_indx == -1:
                    class_prob = 1 / self.class_count[class_indx]
                    class_prob /= np.sum(class_prob)
                    random_choice = np.random.choice(np.flatnonzero(class_prob == class_prob.max()))
                    random_choice = class_indx[random_choice]
                else:
                    random_choice = self.class_indx
                    # print('weighted by class_indx', random_choice)
                # random_choice = np.random.choice(class_indx, 1, p=class_prob)[0]
                # self.class_count[random_choice] += 1 #comment this
                # print('count 2', self.class_count)
                # print('choice', random_choice)
                person = random_choice
                # print(person)
                # else:
                #     print('nel')
                # if np.sum(seg[:,:,11]) > 0:
                #     person = 11
                # if np.sum(seg[:,:,12]) > 0:
                #     person = 12
                # if person:
                lbl_person = sklabel(seg[:, :, person])
                lbl = np.random.choice(np.arange(np.max(lbl_person)) + 1)
                indx = np.argwhere((lbl_person == lbl) * 1 > 0)
                r_min, c_min = np.min(indx, axis=0)
                r_max, c_max = np.max(indx, axis=0)
                # start_crop_y = (r_max - r_min)/2 + r_min - self.crop[0]/2
                # start_crop_x = (c_max - c_min)/2 + c_min - self.crop[1]/2
                start_crop_y = np.random.randint(min(r_max - self.output_size[0], r_min),
                                                 max(r_max - self.output_size[0], r_min) + 1)  #
                start_crop_x = np.random.randint(min(c_max - self.output_size[1], c_min),
                                                 max(c_max - self.output_size[1], c_min) + 1)  # c_min

                start_crop_y = int(max(min((start_crop_y, h - self.output_size[0])), 0))
                start_crop_x = int(max(min((start_crop_x, w - self.output_size[1])), 0))

                if self.class_indx == -1:
                    count_class = np.any(seg[start_crop_y:start_crop_y + self.output_size[0],
                                         start_crop_x:start_crop_x + self.output_size[1]], axis=(0, 1))
                    current_classes = np.where(count_class == 1)[0]
                    class_indx = np.intersect1d(current_classes, self.balance_class)
                    self.class_count[class_indx] += 1
                    # print('count 2', self.class_count)

                # if random_choice == 3:
                # plt.subplot(313)
                # plt.imshow(seg.argmax(-1)[start_crop_y:start_crop_y + self.output_size[0], start_crop_x:start_crop_x+self.output_size[1]] == random_choice)
                # plt.subplot(312)
                # plt.imshow(seg.argmax(-1)[start_crop_y:start_crop_y + self.output_size[0], start_crop_x:start_crop_x+self.output_size[1]])
                # plt.subplot(311)
                # plt.imshow(seg.argmax(-1))
                # plt.show()
                return start_crop_x, start_crop_y
            else:
                return -1, -1
        else:
            return -1, -1


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        images = [sample['left'],
                  sample['right'],
                  sample['disp'],
                  sample['seg'],
                  sample['edges']]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for i in range(len(images)):
            # Reduction of dimension to 3
            if (len(images[i].shape) == 4):
                images[i] = np.squeeze(images[i], axis=(3,))
            images[i] = images[i].transpose((2, 0, 1))
        return {'left': torch.from_numpy(images[0]),
                'right': torch.from_numpy(images[1]),
                'disp': torch.from_numpy(images[2]),
                'seg': torch.from_numpy(images[3]),
                'edges': torch.from_numpy(images[4])}


def generateDataloaders(training_script, test_script, crop, n_labels, max_d, output_activation, datasetName,
                        normalize_input, only_test, train):
    if datasetName == 'cityscapes':
        class_lbl_list = 'cityscapes_image_labels2.csv'
    elif datasetName == 'garden':
        class_lbl_list = 'garden_image_labels4.csv'
    elif datasetName == 'roses':
        class_lbl_list = ''#'roses_image_labels.csv'
    else:
        class_lbl_list = ''
    # birghtness = iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-20, 20))
    # contrast = iaa.contrast.GammaContrast(gamma=(0.8, 1.2))
    # saturation = iaa.MultiplySaturation(mul=(0.7, 1.2))
    # rand = torch.randint(0, 100, (1,)).item()
    # print(rand)
    # ia.random.seed(rand)
    # aug = iaa.Sequential([birghtness, contrast, saturation, iaa.Affine(rotate=(-25, 25))])

    if type(training_script) == list:
        trainCompressed = ''
        testCompressed = ''
        colorL, colorR, disp, seg, inst = training_script
        colorL_test, colorR_test, disp_test, seg_test, inst_test = test_script
        n_data = len(colorL)
        rep = 2
        n_augment = 1
        if datasetName == 'kitti':
            n_augment *= 5
        if test_script:
            colorL = [j for j in colorL for i in range(n_augment)]
            colorR = [j for j in colorR for i in range(n_augment)]
            disp = [j for j in disp for i in range(n_augment)]
            seg = [j for j in seg for i in range(n_augment)]
            inst = [j for j in inst for i in range(n_augment)]
            X_train = list(zip(colorL, colorR))
            Y_train = list(zip(disp, seg, inst))
            X_test = list(zip(colorL_test, colorR_test))
            Y_test = list(zip(disp_test, seg_test, inst_test))
    else:
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        trainCompressed = training_script
        testCompressed = test_script

        print('training size: ', len(X_train))
        print('test size: ', len(X_test))

    trainset = []
    # Only compulsory if training
    if train:
        # Changed from "GardenDataset" to adapt other datasets
        trainset = CustomDataset(X_train, Y_train, n_labels, max_d, datasetName, normalize_input,
                                output_activation=output_activation,
                                hdf5=trainCompressed, class_lbl_list=class_lbl_list,
                                transform=transforms.Compose([RandomCrop(crop, datasetName=datasetName,
                                                                        is_down=False, sliceandSwitch=False,
                                                                        augment_DoubleLeftImg=False, focusPerson=True,
                                                                        resizeImg=True, flipHorizontal=True), ]),
                                # colorAugment(aug)
                                transform_color=transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.5),
                                to_tensor=ToTensor(), )

    crop = [0, 0]
    testset = CustomDataset(X_test, Y_test, n_labels, max_d, datasetName, normalize_input,
                            output_activation=output_activation, only_test=only_test,
                            hdf5=testCompressed, class_lbl_list='',
                            transform=transforms.Compose([RandomCrop(crop, datasetName=datasetName,
                                                                     is_down=True, sliceandSwitch=False,
                                                                     augment_DoubleLeftImg=False, focusPerson=False),
                                                          ]),
                            to_tensor=ToTensor())

    return trainset, testset
