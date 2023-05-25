import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import util.utilIOPfm as readIO
from util.utilTorchPlot import decode_segmap
from util.utilCityscape import ImgCol2id, ImgId2trainId

def computeMeanStd(dataloader, input_size):
    total_images = 0
    img_total = torch.zeros((3, input_size[0], input_size[1]))
    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        left = sample_batched['left']
        right = sample_batched['right']
        img_total += right.sum(0) + left.sum(0) 
        total_images += left.shape[0]
    total_images *= 2
    mean = torch.tensor((img_total[0,:,:].mean()/total_images,
            img_total[1,:,:].mean()/total_images,
            img_total[2,:,:].mean()/total_images))
    

    img_total = torch.zeros((3, input_size[0], input_size[1]))
    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        left = sample_batched['left']
        right = sample_batched['right']
        img_total += ((left - mean[None, :, None, None])**2).sum(0)
        img_total += ((right - mean[None, :, None, None])**2).sum(0)
    std = torch.sqrt(torch.tensor( (img_total[0,:,:].mean(),
                                    img_total[1,:,:].mean(),
                                    img_total[2,:,:].mean())) /total_images)
    print('mean: ', mean, 'std: ', std)
    
    # img_total = torch.zeros((3, input_size[0], input_size[1]))
    # for i_batch, sample_batched in enumerate(dataloader):
    #     left = sample_batched['left']
    #     right = sample_batched['right']
    #     standard_left = (left - mean[None, :, None, None])/std[None,:, None,None]
    #     standard_right = (right - mean[None, :, None, None])/std[None,:, None,None]
    #     img_total += standard_left.sum(0) + standard_right.sum(0)
    # new_mean = torch.tensor((img_total[0,:,:].mean()/total_images,
    #         img_total[1,:,:].mean()/total_images,
    #         img_total[2,:,:].mean()/total_images))

    # img_total = torch.zeros((3, input_size[0], input_size[1]))
    # for i_batch, sample_batched in enumerate(dataloader):
    #     left = sample_batched['left']
    #     right = sample_batched['right']
    #     standard_left = (left - mean[None, :, None, None])/std[None,:, None,None]
    #     standard_right = (right - mean[None, :, None, None])/std[None,:, None,None]
    #     img_total += ((standard_left - mean[None, :, None, None])**2).sum(0)
    #     img_total += ((standard_right - mean[None, :, None, None])**2).sum(0)
    # new_std = torch.sqrt(torch.tensor( (img_total[0,:,:].mean(),
    #                                 img_total[1,:,:].mean(),
    #                                 img_total[2,:,:].mean())) /total_images)
    # print(new_mean, new_std)
    # input()
    return mean, std


def computeDispStats(filepath, datasetName):
    max_disp = 200
    histBins = np.array(range(max_disp))
    histVal = np.zeros((max_disp))
    for data in tqdm(filepath):
        if datasetName == 'garden':
            f = 640
            b = 0.03
            disp_image = readIO.read(data)
            with np.errstate(invalid='ignore', divide='ignore'):
                disp_image = np.where(disp_image > 0, f*b*1/disp_image, 0)
                
        if datasetName == 'kitti':
            disp = cv2.imread(data, -1)
            disp_image = disp.astype(np.float32) / 256.0
        disp_image = disp_image.astype(np.int8)
        # plt.imshow(disp_image == 0)
        # plt.show()
        (dispValues, counts) = np.unique(disp_image, return_counts=True)

        for j in range(len(dispValues)):
            histVal[dispValues[j]] = counts[j]

    #shistVal = histVal/np.max(histVal[1:])
    plt.bar(histBins[1:], histVal[1:], width = 0.5, color='#0504aa',alpha=0.7)
    plt.xlim(min(histBins), max(histBins)+1)
    plt.show()

def testTrainloader(trainloader):
    print('len', len(trainloader))
    for q in range(2):
        print('new_ds')
        for i in trainloader:
            from models.torch_dsnet import testWarp
            left_torch = i['left'].cuda()
            right_torch = i['right'].cuda()
            disp_torch = i['disp'].cuda()
            seg_torch = i['seg'].cuda()
            print(disp_torch.shape, disp_torch.dtype)
            print(left_torch.shape)
            # disp_torch = torch.zeros((1,1,5,5)).cuda()
            # disp_torch[:,:,2,2] = -1.0
            # left_torch = torch.zeros((1,3,5,5)).cuda()
            # left_torch[:,:,2,2] = 1
            out = testWarp().cuda()(seg_torch, disp_torch)
            print(left_torch.shape)
            for j in range(left_torch.shape[0]):
                
                left = np.transpose(left_torch[j].cpu().numpy(), (1,2,0))
                right = np.transpose(i['right'][j].cpu().numpy(), (1,2,0))
                #disp_img = i['disp'][j].numpy().squeeze()
                disp_img = disp_torch[j].cpu().numpy().squeeze()
                seg = np.transpose(i['seg'][j].cpu().numpy(), (1,2,0))
                translated_disp_warp = np.transpose(out[j].cpu().numpy(), (1,2,0))
                disp_img *= -1.0
    #             print(np.unique(seg))
    #             plt.imshow(seg.argmax(0)); plt.show()
            #     print(disp_img.shape)
                translated_disp = np.zeros(left.shape)
                
                for m in range(disp_img.shape[0]):
                    for n in range(disp_img.shape[1]):
                        if n+disp_img[m,n] < disp_img.shape[1] and n+disp_img[m,n] > 0:
                            #print(j, disp_img[i,j].astype(np.uint8))
                            translated_disp[m, (n+disp_img[m,n]).astype(np.uint8)] += left[m,n]-0.5 if disp_img[m,n] != 0 else left[m,n]+0.25  
                
                plt.subplot(4,4,j+1)
                plt.imshow(left)
                plt.subplot(4,4,j+1+4)
                plt.imshow(translated_disp_warp.argmax(-1))
                plt.subplot(4,4,j+1+4*2)
                plt.imshow(translated_disp)
                plt.subplot(4,4,j+1+4*3)
                plt.imshow(right)
            plt.show()
            # print('-'*100)
            # pass

    # pxl_total = 0
    # pxl_perClass = np.array([0]*19, dtype=np.float32)
    # count = 0

        '''
    [0.22618429 0.04064565 0.08634054 0.01664703 0.00577606 0.01405406
    0.00180349 0.00871002 0.31257843 0.11185435 0.04563894 0.00265076
    0.00084605 0.07596903 0.00256571 0.         0.00009674 0.00000895
    0.00122538]

    [0.24082748 0.05215133 0.04873305 0.00232674 0.01640558 0.02055631
    0.00640275 0.00547152 0.29440923 0.08132281 0.13007322 0.00126544
    0.00072696 0.06397973 0.00315431 0.00234834 0.0000347  0.00012629
    0.00143004]
    '''
    # t_start = time.time()
    # for i, data in enumerate(trainloader, 0):
    #     # print('rank: {}, data: {}'.format(rank, i))
    #     # plt.subplot(211)
    #     # plt.imshow(data['left'][0].numpy().transpose((1,2,0)))
    #     # plt.subplot(212)
    #     # plt.imshow(data['left'][1].numpy().transpose((1,2,0)))
    #     # plt.savefig('{}_{}.png'.format(i, rank))
    #     # plt.close()
    #     print('image: ', i, time.time()-t_start)
    #     print('_'*10)
    #     t_start = time.time()
        #0.9 per image, 0.2 loading an image, random crop 0.02
        
    #     count += 1
    #     seg_out = data['seg']
    #     print(seg_out.shape)
    #     pxl_total += seg_out.shape[1] * seg_out.shape[2]
    #     pxl_perClass += np.sum(seg_out.numpy(), axis=(1,2))
    #     print(['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'])
    #     np.set_printoptions(suppress=True)
    #     print(np.array(pxl_perClass))
    #     print(np.array(pxl_perClass)/pxl_total)
    #     print(count, ' total pixels: ', pxl_total)
    #     print(pxl_total/(19.0 * np.array(pxl_perClass)+1e-8))
    #     print(1.0/np.log(pxl_perClass/pxl_total + 1.1))
    # input()

def getLoaderStats(dataLoader, labels):
    pxl_total = 0
    pxl_perClass = []
    pxl_perClass = [0]*labels

    for data in tqdm(dataLoader):
        seg_out = ['seg']
        for i in range(seg_out.shape[0]):
            pxl_total += seg_out[i].shape[1] * seg_out[i].shape[2]
            seg = seg_out[i].argmax(axis=1)
            
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


def getDatasetStats(colorL_list, seg_list, args, labels, ignore_class=None, use_parallel=True):
    import time
    print('training data length: ', len(colorL_list))
    save_segment = []
    save_color = []

    pxl_total = 0
    data_size = 100#len(seg_list)
    pxl_perClass = np.array([0]*labels, dtype=np.float32)
    total_pixels = np.zeros((data_size, labels+1))
    if use_parallel:
        from joblib import Parallel, delayed
        import multiprocessing as mp
        
        def get_row(i, seg_list, labels, args):
            print(i+1, data_size)
            seg = Image.open(seg_list[i])
            seg = np.array(seg)
            if args.datasetName == 'kitti' or args.datasetName == 'cityscapes':
                segment = ImgId2trainId(seg, labels)
                # segment = np.delete(segment, ignore_class-1, 2)
            return np.sum(segment, axis=(0,1))
        time_s = time.time()
        total_pixels[:,:] = Parallel(n_jobs=16)(delayed(get_row)(j, seg_list, labels, args) for j in range(data_size))
        print('time', time.time()-time_s)
        if ignore_class != None:
            total_pixels = np.delete(total_pixels, ignore_class-1, 1)
        pxl_perClass = np.sum(total_pixels, axis=0)
        pxl_total = np.sum(total_pixels)
    else:
        time_s = time.time()
        for i in tqdm(range(len(colorL_list))):
            seg = Image.open(seg_list[i])
            seg = np.array(seg)
            if args.datasetName == 'kitti' or args.datasetName == 'cityscapes':
                segment = ImgId2trainId(seg, labels)
                if ignore_class != None:
                    segment = np.delete(segment, ignore_class-1, 2)

            #segment = np.zeros((seg.shape[0], seg.shape[1], labels))
            pxl_total += seg.shape[0] * seg.shape[1]
            pxl_perClass += np.sum(segment, axis=(0,1))
        print('time', time.time()-time_s)
    #print(['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'])
    total_pixels = np.sum(pxl_perClass)
    np.set_printoptions(suppress=True)
    print(pxl_perClass)
    print(np.argsort(pxl_perClass))
    print(pxl_perClass/pxl_total)
    print(total_pixels/(labels*pxl_perClass))
    print(1.0/np.log(pxl_perClass/pxl_total + 1.1))
    name = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    plt.figure(figsize = (10,10))
    plt.bar(name, pxl_perClass/pxl_total); plt.xticks(rotation=90)
    plt.savefig('results/proportion_percentage.png')
    plt.close()

    plt.figure(figsize = (10,10))
    plt.bar(name, pxl_perClass); plt.xticks(rotation=90)
    plt.savefig('results/proportion_pixels.png')
    plt.close()
    input()


def evaluteGenerator(dataloader, args, labels, ignore_class=None, use_parallel=True):
    import time
    ''' [15898918.  2346597.  6459957.   801322.   395326.   787558.   136592.
   385380. 21379551.  6877418.  6904576.    57806.     8881.  4106593.
   134875.     7031.   199163.     5788.    20364.]
[17 15 12 18 11 14  6 16  7  4  5  3  1 13  2  9 10  0  8]
[0.23760335 0.03506901 0.09654163 0.01197546 0.005908   0.01176976
 0.00204132 0.00575936 0.31950934 0.10278042 0.10318629 0.00086389
 0.00013272 0.06137149 0.00201566 0.00010508 0.00297642 0.0000865
 0.00030433]
[  0.22151026   1.5008003    0.5451698    4.39495418   8.90852986
   4.47176395  25.78316061   9.13844381   0.16472626   0.51207786
   0.51006368  60.92401262 396.55145521   0.85759009  26.11138813
 500.89225909  17.68287018 608.46120831 172.94114485]
[ 3.43785012  7.89306788  5.57303577  9.42168533  9.93377002  9.43813623
 10.29185627  9.94705167  2.85460407  5.41606492  5.40618626 10.40634432
 10.47879391  6.68441947 10.29432318 10.4815542  10.202795   10.48340967
 10.46169475]'''
    
    save_segment = []
    save_color = []
    rep = 4
    data_size = len(dataloader)
    print('training data length: ', data_size)
    pxl_total = 0
    pxl_perClass = np.array([0]*labels, dtype=np.float32)
    total_pixels = np.zeros((rep*data_size, labels+1))
    if use_parallel:
        from joblib import Parallel, delayed
        import multiprocessing as mp
        
        def get_row(j, i, data):
            print(j, i+1, data_size)
            segment = data[i]['seg'].numpy().transpose((1,2,0))
            return np.sum(segment, axis=(0,1))
        time_s = time.time()
        for j in range(rep):
            total_pixels[j*data_size:(j+1)*data_size,:] = Parallel(n_jobs=1)(delayed(get_row)(j, i, dataloader) for i in range(data_size))
        print('time', time.time()-time_s)
        if ignore_class != None:
            print(total_pixels.shape)
            total_pixels = np.delete(total_pixels, ignore_class-1, 1)
            print(total_pixels.shape)
        pxl_perClass = np.sum(total_pixels, axis=0)
        pxl_total = np.sum(total_pixels)
    else:
        time_s = time.time()
        for data in tqdm(dataloader):
            segment = data['seg'].numpy()
            if ignore_class != None:
                segment = np.delete(segment, ignore_class-1, 0)

            #segment = np.zeros((seg.shape[0], seg.shape[1], labels))
            pxl_total += segment.sum()
            pxl_perClass += np.sum(segment, axis=(1,2))
        print('time', time.time()-time_s)
    #print(['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'])
    total_pixels = np.sum(pxl_perClass)
    np.set_printoptions(suppress=True)
    print(pxl_perClass)
    print(np.argsort(pxl_perClass))
    print(pxl_perClass/pxl_total)
    print(total_pixels/(labels*pxl_perClass))
    print(1.0/np.log(pxl_perClass/pxl_total + 1.1))
    name = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    plt.figure(figsize = (10,10))
    plt.bar(name, pxl_perClass/pxl_total); plt.xticks(rotation=90)
    plt.ylim(0, .50)
    plt.savefig('results/proportion_percentage.png')
    plt.close()

    plt.figure(figsize = (10,10))
    plt.bar(name, pxl_perClass); plt.xticks(rotation=90)
    plt.ylim(0, 5e6)
    plt.savefig('results/proportion_pixels.png')
    plt.close()
    input()


def count_classes_in_dataset(trainset, dataset='cityscapes', min_pxl=100):
    if dataset == 'garden':
        columns = ['n', 'Grass', 'Ground', 'Pavement', 'Hedge', 'Topiary', 'Rose', 'Obstacle', 'Tree', 'Background']
        remove_last = False
    elif dataset == 'roses':
        columns = ['Background', 'Branch']
        remove_last = False

    else:
        columns = ['n', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                   'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                   'bicycle']
        remove_last = True

    img_label_list = np.zeros((len(trainset), len(columns)))
    import pandas as pd
    # df = pd.read_csv('cityscapes_image_labels.csv')
    # for i in range(19):
    #     class_in_img = df[str(i)]==1
    #     pd.read_csv('cityscapes_image_labels.csv')
    #     print(df.shape)
    # input()

    def count_classes(i, data, min_pxl=100, remove_last=True):
        seg = data['seg'].cpu().numpy()
        seg = np.sum(seg, axis=(1,2))
        print(i, seg > min_pxl)
        count_class = (seg > min_pxl)*1 #np.any(seg, axis=(1,2))*1
       # print(count_class)
        #current_classes = np.where(count_class == 1)[0]
        print(np.append(i, count_class[:-1] if remove_last else count_class[:]))
        return np.append(i, count_class[:-1] if remove_last else count_class[:])

    from joblib import Parallel, delayed
    import multiprocessing as mp

    img_label_list[:] = Parallel(n_jobs=16)(delayed(count_classes)(idx, data, min_pxl, remove_last) for idx, data in enumerate(trainset, 0))
    
    # for idx, data in enumerate(trainset, 0):
    #     seg = data['seg'].cpu().numpy()
    #     seg = np.sum(seg, axis=(1,2))
    #     print(seg > 100)
    #     count_class = (seg > 100)*1 #np.any(seg, axis=(1,2))*1
    #     print(count_class)
    #     current_classes = np.where(count_class == 1)[0]
    #     img_label_list[idx] = np.append(idx, count_class[:-1])
    #     print(img_label_list[idx])
    df = pd.DataFrame(img_label_list, columns=columns) 
    df.to_csv('{}_image_labels_train_val.csv'.format(dataset))
    print(df)
    input()

def show_outputs(outputs, left, right, disp, seg, seg_full):
    n = 0
    seg_branch_both, disp_out, seg_branch, disp_out, seg_branch_right, at_d = outputs
    decode_seg = lambda x: np.transpose(decode_segmap(x.argmax(1).unsqueeze(1)).cpu().numpy()[n], (1,2,0))
    mask = (at_d[n].cpu().numpy().squeeze() > 0.7) * 1
    dif_seg = (seg_branch.argmax(1)-seg.argmax(1)) * (1-seg_full[:,-1])
    dif_seg = (np.abs(dif_seg.cpu().numpy()[n]) > 1) * 1
    
    plt.subplot(4,2,1)
    plt.imshow(np.transpose(left[n].cpu().numpy(), (1,2,0)), interpolation='none') 
    plt.subplot(4,2,2)
    plt.imshow(np.transpose(right[n].cpu().numpy(), (1,2,0)), interpolation='none') 

    plt.subplot(4,2,3)   
    plt.imshow(decode_seg(seg_branch), interpolation='none')#*(1-mask[:,:,None]))
    plt.subplot(4,2,4)
    #plt.imshow(dif_seg)
    plt.imshow(decode_seg(seg_branch_right), interpolation='none')#*mask[:,:,None])
    plt.subplot(4,2,5)
    plt.imshow(decode_seg(seg_branch_both), interpolation='none')
    plt.subplot(4,2,7)
    plt.imshow(decode_seg(seg_full), interpolation='none')
    #plt.imshow(at_d[n].cpu().numpy().squeeze(), interpolation='none')
    
    plt.subplot(4,2,8)
    plt.imshow(disp_out[n,:].cpu().numpy().squeeze(), interpolation='none')
    plt.subplot(4,2,6)
    plt.imshow(dif_seg, interpolation='none')
    #plt.imshow(disp[n,:].cpu().numpy().squeeze(), interpolation='none') #*(1-mask))
    #plt.imshow(np.abs(disp[n,:].cpu().numpy().squeeze()-disp_out[n,:].cpu().numpy().squeeze()))
    
    plt.show() 

def eval_seg_result(outputs, left, seg, label):
    from util.utilTorchPlot import decode_segmap
    #decode_segmap(seg, nc=21)
    from skimage import segmentation
    for k in range(left.shape[0]):
        softmax_output = torch.softmax(outputs[2], dim=1)
        sorted_output = torch.argsort(-softmax_output.detach().cpu(), dim=1)
        #decoded_seg = decode_segmap(outputs[2].detach().cpu().argmax(1).unsqueeze(1), nc=21).numpy()
        decoded_seg = decode_segmap(sorted_output[:,0].detach().cpu().unsqueeze(1), nc=21).numpy()
        decoded_seg2 = decode_segmap(sorted_output[:,1].detach().cpu().unsqueeze(1), nc=21).numpy()
        decoded_seg = np.transpose(decoded_seg, (0,2,3,1))
        decoded_seg2 = np.transpose(decoded_seg2, (0,2,3,1))
        left_batch = np.transpose(left[k,:].cpu().numpy(), (1,2,0))
        seg_output = outputs[2][k,:].argmax(0).cpu().numpy()
        seg_pred_wall = seg_output == label
        seg_wall = seg.cpu().numpy()[k,label,:]
        seg_batch = seg[k,:]
        if np.any(seg_batch[3,:,:].cpu().numpy()):
            pred_gt = seg_batch.cpu().numpy().argmax(0)
            seg_boundaries = segmentation.mark_boundaries(left_batch, seg_wall.astype(np.int), mode='thick', color=(1,0,0))
            seg_boundaries = segmentation.mark_boundaries(seg_boundaries, seg_pred_wall.astype(np.int), mode='thick', color=(0,0,1))
            seg_pred_wall1 = segmentation.mark_boundaries(decoded_seg[k,:], seg_wall.astype(np.int), mode='thick', color=(1,0,0))
            seg_pred_wall2 = segmentation.mark_boundaries(decoded_seg2[k,:], seg_wall.astype(np.int), mode='thick', color=(1,0,0))
            seg_pred_wall2 = segmentation.mark_boundaries(decoded_seg2[k,:], seg_wall.astype(np.int), mode='thick', color=(1,0,0))
            seg_k_pred = segmentation.mark_boundaries(softmax_output[k,label].detach().cpu(), seg_wall.astype(np.int), mode='thick', color=(1,0,0))
            plt.subplot(2,2,1), plt.imshow(seg_boundaries)
            plt.subplot(2,2,2), plt.imshow(seg_k_pred)
            #plt.subplot(3,1,2), plt.imshow(seg_wall)
            plt.subplot(2,2,3), plt.imshow(seg_pred_wall1)
            plt.subplot(2,2,4), plt.imshow(seg_pred_wall2*(pred_gt==3)[:,:,None])
            plt.show()

        # plt.imshow()
        # input()


def invertDisp(trainloader):
        print('len', len(trainloader))
        for q in range(2):
            print('new_ds')
            for i in trainloader:
                from models.torch_dsnet import testWarp
                left_torch = i['left']
                right_torch = i['right']
                disp_torch = i['disp']
                seg_torch = i['seg']

                for j in range(left_torch.shape[0]):
                    left = np.transpose(left_torch[j].cpu().numpy(), (1, 2, 0))
                    right = np.transpose(i['right'][j].cpu().numpy(), (1, 2, 0))
                    disp_img = disp_torch[j].cpu().numpy().squeeze()
                    seg = np.transpose(i['seg'][j].cpu().numpy(), (1, 2, 0)).argmax(-1)
                    translated_disp = np.zeros_like(disp_img)
                    translated_seg = np.zeros_like(seg) + 19

                    r = np.arange(0, disp_img.shape[0], 1)
                    c = np.arange(0, disp_img.shape[1], 1)
                    cv, rv = np.meshgrid(c, r)
                    cv_disp = (cv - disp_img).astype(np.int)
                    cv_disp[cv_disp < 0] = 0
                    cv_disp = cv_disp.flatten()
                    translated_disp[rv.flatten(), cv_disp.flatten()] = disp_img[rv.flatten(), cv.flatten()]
                    translated_seg[rv.flatten(), cv_disp.flatten()] = seg[rv.flatten(), cv.flatten()]
                    translated_disp[:, -20] = 0
                    translated_seg[:, -20] = 0
                    # print(translated_disp[:, cv], 'translated' )
                    # for m in range(disp_img.shape[0]):
                    #     for n in range(disp_img.shape[1]):
                    #         if disp_img.shape[1] > n - disp_img[m, n] > 0:
                    #             # print(j, disp_img[i,j].astype(np.uint8))
                    #             translated_disp[m, (n - disp_img[m, n]).astype(np.int)] = seg[m, n]

                    plt.subplot(4, 2, 1), plt.imshow(left)
                    plt.subplot(4, 2, 3), plt.imshow(right)
                    plt.subplot(4, 2, 5), plt.imshow(seg)
                    plt.subplot(4, 2, 7), plt.imshow(disp_img)
                    plt.subplot(4, 2, 2), plt.imshow(np.fliplr(right))
                    plt.subplot(4, 2, 4), plt.imshow(np.fliplr(left))
                    plt.subplot(4, 2, 6), plt.imshow(np.fliplr(translated_seg))
                    plt.subplot(4, 2, 8), plt.imshow(np.fliplr(translated_disp))
                    plt.show()
                # print('-'*100)
                # pass