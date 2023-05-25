from torch.utils.data import Dataset, DataLoader
from util.utilCityscape import ImgCol2id, ImgId2trainId
import os
import torch
from torchvision import transforms
from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import util.utilIOPfm as readIO
import cv2 
from skimage import filters
from skimage.measure import label as sklabel
from skimage.color import rgb2gray
from skimage.morphology import dilation, square
from matplotlib import cm
from scipy import ndimage as ndi
import time
class GardenDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, array_x_files, array_y_files, n_labels, max_d, datasetName, normalize, output_activation ='sigmoid', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert len(array_x_files) > 0
        assert len(array_x_files)== len(array_y_files)
        self.array_x_files = array_x_files
        self.array_y_files = array_y_files
        self.datasetName = datasetName
        self.normalize = normalize
        self.output_activation = output_activation
        self.transform = transform
        self.n_labels = n_labels
        self.max_d = max_d
        self.f = 640
        self.b = 0.03
    def __len__(self):
        return len(self.array_x_files)

    def __getitem__(self, idx):
        # start_t = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        left_image = io.imread(self.array_x_files[idx][0])
        right_image = io.imread(self.array_x_files[idx][1])
        # left_image = (io.imread(self.array_x_files[idx][0])/255.0).astype(np.float32)
        # right_image = (io.imread(self.array_x_files[idx][1])/255.0).astype(np.float32)
        # left_image = (left_image - self.normalize[0])/self.normalize[1]
        # right_image = (right_image - self.normalize[0])/self.normalize[1]
        seg = Image.open(self.array_y_files[idx][1])
        seg = np.array(seg)
        left_edges = filters.sobel(rgb2gray(left_image))

        # inst = np.array(Image.open(self.array_y_files[idx][2]))
        # edge_sobel = filters.sobel(inst)
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
                disp_image = np.where(disp_image > 0, self.f*self.b*1/disp_image, 0)
            
            
        
        if self.datasetName == 'kitti':
            # import imageio
            # disp_image = imageio.imread(self.array_y_files[idx][0])
            
            disp = cv2.imread(self.array_y_files[idx][0], -1)
            disp_image = disp.astype(np.float32) / 256.0
            seg_image = ImgId2trainId(seg, self.n_labels)#.astype(np.float32)
            # print(seg_image)
            # if np.any(seg_image[:,:,10]):
            #     disp_image = disp_image + seg_image[:,:,10] * 0.0001 
                # plt.imshow(disp_image)
                # plt.show()
            '''
            import matplotlib.pyplot as plt
            from scipy.ndimage import convolve, median_filter
            from skimage.restoration import inpaint
            n = 15
            kernel = 1/float(n*n)*np.ones((n,n))
            mean_res = convolve(disp_image, kernel)
            median_res = median_filter(disp_image, size=2)

            mask = (disp_image == 0) & (seg != 5)
            mask[mask > 0] = 255
            translated_disp = np.zeros(disp_image.shape)
            translated_rgb = np.ones(left_image.shape)
            print(translated_rgb.shape)
            zeros = 0
            for i in range(disp_image.shape[0]):
                for j in range(disp_image.shape[1]):
                    if j == 814 and i == 371:
                        print('disp', disp_image[i,j])
                    if j-disp_image[i,j] > 0:
                        val = disp_image[i,j].astype(np.uint8)
                        translated_rgb[i, j-disp_image[i,j].astype(np.uint8), :] = left_image[i,j,:]
                        translated_disp[i, j-disp_image[i,j].astype(np.uint8)] = disp_image[i,j]
                    
            #conv_disp_image = cv2.inpaint(disp_image, mask.astype(np.uint8),3,cv2.INPAINT_TELEA)
            plt.figure()
            plt.imshow(left_image) 
            plt.figure()
            plt.imshow(right_image) 
            plt.figure()
            plt.imshow(disp_image)          
            plt.figure()
            plt.imshow(translated_rgb)
            # plt.figure()
            # plt.imshow(conv_disp_image)
            plt.figure()
            plt.imshow(mean_res)
            plt.figure()
            plt.imshow(median_res)
            plt.show()
            '''

        if self.output_activation != 'linear':
            disp_image[disp_image > self.max_d] = self.max_d
        #disp_image = np.round(disp_image)
        if self.output_activation == 'sigmoid':
            disp_image= disp_image/self.max_d
            # delta = 10/(2.0-0.1); np.floor(delta*np.floor(a/delta)) 
            # plt.imshow(disp_image)
            # plt.show()
        if self.output_activation == 'tanh':
            disp_image = np.where(disp_image != 0, 2 * disp_image/float(self.max_d) - 1, -1)

        if self.datasetName != 'kitti':
            seg_image = np.zeros((seg.shape[0], seg.shape[1], self.n_labels), dtype=np.float32)
            for j in range(self.n_labels):
                # if self.datasetName == 'kitti':
                    #seg_image[:,:,j] = (seg == j).astype(np.uint8)
                    # if j+1 == self.n_labels:
                    #     seg_image[:,:,0] = (seg != 6+1).astype(np.uint8) & (seg != 0+1).astype(np.uint8) 
                    #     seg_image[:,:,1] = (seg == 6+1).astype(np.uint8)
                    #     seg_image[:,:,2] = (seg == 0+1).astype(np.uint8)
                        #disp_image = disp_image * seg_image[:,:,0]#(seg == j+1).astype(np.uint8)
                #else:
                seg_image[:,:,j] = (seg == j+1).astype(np.uint8)
        
        sample = {'left': left_image,
                    'right': right_image,
                    'seg': seg_image,
                    'disp': disp_image,
                    'edges': left_edges}

        # sample = {'left': left_image,
        #             'right': right_image,
        #             'seg': seg_image,
        #             'disp': np.expand_dims(disp_image, axis=-1),
        #             'edges': np.expand_dims(left_edges, axis=-1)}
        # if self.transform:
        #     sample = self.transform(sample)
        
        
        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, datasetName, is_down=False, sliceandSwitch=False, augment_DoubleLeftImg=False, focusPerson=False):
        assert isinstance(output_size, (int, tuple, list))
        self.is_down = is_down
        self.sliceandSwitch = sliceandSwitch
        self.augment_DoubleLeftImg = augment_DoubleLeftImg
        self.focusPerson = focusPerson
        self.datasetName = datasetName
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # start = time.time()
        images = [sample['left'],
                sample['right'],
                sample['disp'],
                sample['seg'],
                sample['edges']]
        if self.output_size[0] == 0:
            return sample
        h, w = images[0].shape[:2]
        new_h, new_w = self.output_size
        # crop_top = torch.randint((h - new_h)//2 -20, (h - new_h)//2 + 20, (1,))
        # crop_left = torch.randint((w - new_w)//2 -10, (w - new_w)//2 + 10, (1,))
        
        if self.is_down:
            crop_top = (h-new_h)
            crop_left = (w-new_w)//2 
        else:
            if torch.multinomial(torch.tensor([0.2,0.8]), 1).item():
                y_start = h-new_h - 100
            else:
                y_start = 0
            crop_left = -1
            crop_top = -1
            if self.focusPerson:
                crop_left, crop_top = self.cropPerson(images[3])
            if crop_left == -1:
                crop_top = torch.randint(y_start, h - new_h+1, (1,))
                crop_left = torch.randint(0, w - new_w+1, (1,))

        if self.sliceandSwitch:
            w,h,c = images[0].shape
            divisor = float(torch.randint(2,6, (1,)))
            div_img = int(w/divisor)

        for i in range(len(images)):
            images[i] = images[i][crop_top: crop_top + new_h,
                        crop_left: crop_left + new_w] #images[i] = [h,w,c]
            if self.sliceandSwitch:
                slice1 = images[i][:div_img,:,:]
                slice2 = images[i][div_img:,:,:] 
                images[i] = np.concatenate((slice2, slice1), axis=0)   
        
        if self.augment_DoubleLeftImg and torch.multinomial(torch.tensor([0.8,0.2]), 1).item():
            images[1] = images[0] #right = left
            images[2] = np.abs(images[2] * 0 + 0.0001)
        #print('random crop: ', time.time()-start)
        return {'left': images[0], 'right': images[1],
                'disp': images[2], 'seg': images[3],
                'edges': images[4]}

    def cropPerson(self, seg):
        h, w, _ = seg.shape
        person = 0
        if self.datasetName == 'kitti':
            if np.sum(seg[:,:,11]) > 0:
                person = 11
            if np.sum(seg[:,:,12]) > 0:
                person = 12
            if person:
                # start = time.time()
                lbl_person = sklabel(seg[:,:,person])
                lbl = np.random.choice( np.arange(np.max(lbl_person))+1)
                indx = np.argwhere((lbl_person == lbl)*1 > 0)
                r_min, c_min = np.min(indx, axis=0)
                r_max, c_max = np.max(indx, axis=0)
                # start_crop_y = (r_max - r_min)/2 + r_min - self.crop[0]/2
                # start_crop_x = (c_max - c_min)/2 + c_min - self.crop[1]/2
                start_crop_y = np.random.randint(min(r_max - self.output_size[0], r_min), max(r_max - self.output_size[0], r_min)+1)#
                start_crop_x = np.random.randint(min(c_max - self.output_size[1], c_min), max(c_max - self.output_size[1], c_min)+1)#c_min

                start_crop_y = int(max(min((start_crop_y, h - self.output_size[0])), 0))
                start_crop_x = int(max(min((start_crop_x, w - self.output_size[1])), 0))
                
                # plt.subplot(313)
                # plt.imshow(lbl_person == lbl)
                # plt.subplot(312)
                # plt.imshow(seg.argmax(-1)[start_crop_y:start_crop_y + self.output_size[0], start_crop_x:start_crop_x+self.output_size[1]])
                # plt.subplot(311)
                # plt.imshow(seg.argmax(-1))
                # plt.show()
                # print('Crop person time: ', time.time()-start)
                return start_crop_x, start_crop_y
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
            images[i] = images[i].transpose((2, 0, 1))
        return {'left': torch.from_numpy(images[0]),
                'right': torch.from_numpy(images[1]),
                'disp': torch.from_numpy(images[2]),
                'seg': torch.from_numpy(images[3]),
                'edges': torch.from_numpy(images[4])}



def generateDataloaders(training_script, test_script, crop, n_labels, max_d, output_activation, datasetName, train, batch_size, normalize_input):
    colorL, colorR, disp, seg, inst = training_script
    colorL_test, colorR_test, disp_test, seg_test, inst_test = test_script
    n_data = len(colorL)

    rep = 2
    n_augment = 1
    if test_script:
        colorL = [j for j in colorL[:] for i in range(n_augment)]
        colorR = [j for j in colorR[:] for i in range(n_augment)]
        disp = [j for j in disp[:] for i in range(n_augment)]
        seg = [j for j in seg[:] for i in range(n_augment)]
        inst = [j for j in inst[:] for i in range(n_augment)]
        X_train = list(zip(colorL, colorR))
        Y_train = list(zip(disp, seg, inst))
        # X_test = list(zip(colorL_test[:2], colorR_test[:2]))
        # Y_test = list(zip(disp_test[:2], seg_test[:2]))
        X_test = list(zip(colorL_test[:], colorR_test[:]))
        Y_test = list(zip(disp_test[:], seg_test[:], inst_test[:]))
    else:
        split_data_ratio = 0.67#0.015#0.015#0.0002#0.67
        split = int(n_data*split_data_ratio)
        #split=4#40#254
        n = 0
        up_limit = -1#split+3#70

        if n == 0:
            X_train = list(zip(colorL[:split], colorR[:split]))
            Y_train = list(zip(disp[:split], seg[:split]))
        else:
            X_train = list(zip(colorL[split-n:split], colorR[split-n:split]))
            Y_train = list(zip(disp[split-n:split], seg[split-n:split]))

        if up_limit > -1:
            X_test = list(zip(colorL[split:up_limit], colorR[split:up_limit]))
            Y_test = list(zip(disp[split:up_limit], seg[split:up_limit]))
        else:
            X_test = list(zip(colorL[split:], colorR[split:]))
            Y_test = list(zip(disp[split:], seg[split:]))
    print('training size: ', len(X_train))
    print('test size: ', len(X_test))
    
    trainset = GardenDataset(X_train, Y_train, n_labels, max_d, datasetName, normalize_input, output_activation=output_activation,
                    transform=transforms.Compose([RandomCrop(crop, datasetName=datasetName,
                                                    is_down=False, sliceandSwitch=False, augment_DoubleLeftImg=False, focusPerson=True), 
                                                        ToTensor(),]))
    
    # crop = [0, 0]
    testset = GardenDataset(X_test, Y_test, n_labels, max_d, datasetName, normalize_input, output_activation=output_activation,
                    transform=transforms.Compose([RandomCrop(0, datasetName=datasetName,
                                                        is_down=True, sliceandSwitch=False, augment_DoubleLeftImg=False, focusPerson=False),
                                                        ToTensor()]))

    return trainset, testset
