import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from torchvision import utils
import util.utilCityscape as cityscapes 
import seaborn as sn
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvas
from PIL import Image
def showImg(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def plotBatchData(dataloader, normalize_input):
    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched['left'].size(),
        #                sample_batched['right'].size(),
        #                sample_batched['disp'].size(),
        #                sample_batched['seg'].size(),)
        plt.figure()
        show_img_batch(sample_batched, torch.tensor(normalize_input))
        # plt.axis('off')
        # plt.ioff()
        # plt.savefig('fig.png')
        # plt.close()
        break

def plotData(left, right, seg, disp):
    ax = plt.subplot(2, 2, 1)
    ax.imshow(left)
    ax = plt.subplot(2, 2, 2)
    ax.imshow(right)
    ax = plt.subplot(2, 2, 3)
    ax.imshow(np.argmax(seg, -1))
    ax = plt.subplot(2, 2, 4)
    ax.imshow(disp)
    plt.show()
    #Example
    # # for i in range(len(dataset)):
    # #     sample = dataset[i]
    # #     plotData(**sample)


def toJetColor(img):
    jet = cm.get_cmap('jet')
    img = jet(img[:,1,:,:].numpy())
    img = img[:,:,:,:3]
    img = img.transpose(0, 3, 1, 2)
    return torch.from_numpy(img).float()

def ErrorColorImg(pred, gt):
    label_colors = torch.tensor([[0., 0.0, 1.0  ],
                            [0., 1.0, 0],
                            [1., 0., 0.0  ]])
    disp_n = [0, 3, 6]
    zeros = (gt > 0.0)
    error = (torch.abs(pred*zeros - gt*zeros)[:,1,:,:]).unsqueeze(1)
    r = torch.zeros_like(error, dtype=torch.float32)
    g = torch.zeros_like(error, dtype=torch.float32)
    b = torch.zeros_like(error, dtype=torch.float32)
    
    for l in range(0, 3):
        idx = (error > disp_n[l]/100.0)
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    
    rgb = torch.cat([r, g, b], dim=1)
    return rgb#(torch.abs(pred*zeros - gt*zeros) > 4.0/100.0) * 1.0

def show_img_batch(sample, normalize_input, pred=None):
    """Show image with landmarks for a batch of samples."""
    images_batch = [[],[],[],[]]
    
    for key, val in sample.items():
        if key == 'seg':
            if val.shape[1] > 1:
                seg = torch.argmax(val, dim=1)
                seg = torch.unsqueeze(seg, 1)
                val = decode_segmap(seg, nc=21)
            else:
                val = decode_segmap(val, nc=21)
            
            images_batch[3] = val
        if key == 'left':
            images_batch[0] = val * normalize_input[1][None,:,None,None] + normalize_input[0][None,:,None,None]
        if key == 'right':
            images_batch[1] = val *normalize_input[1][None,:,None,None] + normalize_input[0][None,:,None,None]
        if key == 'disp':
            #print('dispGT')
            images_batch[2] = normalizeDisp(val)

    if type(pred) != type(None):
        images_batch+= [[],[],[],[],[]]
        for key, val in pred.items():
            if 'seg' in key:
                if val.shape[1] > 1:
                    seg = torch.argmax(val, dim=1)
                    seg = torch.unsqueeze(seg, 1)
                    val = decode_segmap(seg, nc=21)
                else:
                    val = torch.cat([val, val, val], dim=1)
                    #val = decode_segmap(val, nc=21)
                if key == 'seg':
                    images_batch[5] = val
                if key == 'seg2':
                    images_batch[8] = val
                if key == 'seg3':
                    images_batch[7] = val
            if key == 'disp':
                #print('disp1')
                images_batch[4] = normalizeDisp(val)
                #images_batch[6] = toJetColor(torch.abs(images_batch[4]*zeros - images_batch[2]*zeros)) 
                error = ErrorColorImg(images_batch[4], images_batch[2])
                images_batch[6] = error
            
            if key == 'disp2':
                #print('disp2')
                images_batch[7] = toJetColor(normalizeDisp(val))
                #error = ErrorColorImg(images_batch[7], images_batch[2])
                #images_batch[6] = error
            if key == 'edge':
                out_val = torch.sigmoid(val)
                images_batch[8] = torch.cat([out_val, out_val, out_val], dim=1)

        img = torch.cat((images_batch[0],
                            images_batch[1],
                            toJetColor(images_batch[2]),
                            toJetColor(images_batch[4]),
                            images_batch[6],
                            images_batch[3],
                            images_batch[5],
                            images_batch[7],
                            images_batch[8]), 0)
    else:
        img = torch.cat((images_batch[0],
                            images_batch[1],
                            images_batch[2],
                            images_batch[3]), 0)

    grid = utils.make_grid(img, scale_each=True,
                            nrow=images_batch[0].size(0),
                            padding=10)
    return grid
    
def normalizeDisp(img):
    max_disp = torch.max(img)
    # print('max disp: ', max_disp)
    if max_disp > 1:
        #print('YEAH!'*100)
        img /= 100.0
        img[img <0] = 0
    return torch.cat((img,img,img), 1)
        

def decode_segmap(image, nc=21):
    # label_colors = torch.tensor([[0, 0.8, 0], 
    #                         [0.3, 0.5, 0  ],
    #                         [0.7, 0.8, 0.9],
    #                         [0.5, 0.5, 0  ],
    #                         [0, 0.7, 0.7],
    #                         [0.9, 0, 0 ],
    #                         [0.2, 0.2, 0.9],
    #                         [0.3, 0.7, 0.1],
    #                         [0.1, 0.1, 0.1],])

    # label_colors = torch.tensor([[0.0, 0.0, 0.0], 
    #                         [1.0, 1.0, 1.0  ],
    #                         [0.7, 0.8, 0.9],
    #                         [0.5, 0.5, 0  ],
    #                         [0, 0.7, 0.7],
    #                         [0.9, 0, 0 ],
    #                         [0.2, 0.2, 0.9],
    #                         [0.3, 0.7, 0.1],
    #                         [0.1, 0.1, 0.1]])

    label_colors = (torch.tensor([[128, 64,128],
                                [244, 35,232],
                                [ 70, 70, 70],
                                [102,102,156],
                                [190,153,153],
                                [153,153,153],
                                [250,170, 30],
                                [220,220,  0],
                                [107,142, 35],
                                [152,251,152],
                                [ 70,130,180],
                                [220, 20, 60],
                                [255,  0,  0],
                                [  0,  0,142],
                                [  0,  0, 70],
                                [  0, 60,100],
                                [  0, 80,100],
                                [  0,  0,230],
                                [119, 11, 32],])/255.0).type(torch.float32)

    r = torch.zeros_like(image, dtype=torch.float32)
    g = torch.zeros_like(image, dtype=torch.float32)
    b = torch.zeros_like(image, dtype=torch.float32)


    for l in range(0, len(label_colors)):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    
    rgb = torch.cat([r, g, b], axis=1)
    return rgb


def showResults(dataset, outputs, net, normalize_input, outputType, show_statistics=False, name=''):
    # outputs = net(left, right)
    predData = {'disp': dataset['disp'], 'seg': dataset['seg']}
    
    if outputType == 'seg':
        predData['seg'] = outputs.detach().cpu()
    
    if outputType == 'disp':
        out_disp = outputs[:,0,:,:].unsqueeze(1)
        #dataset['left'] = warpDisp(dataset['disp'].cuda(), dataset['left'].cuda(), trainTime=False).detach().cpu()
        if warpedOutput:
            dataset['left'] = outputs[:,1:,:,:].detach().cpu()
        predData['disp'] = out_disp.detach().cpu()

    if outputType == 'fmetric':
        out_disp = outputs[:,0,:,:].unsqueeze(1)
        dataset['left'] = outputs[:,1:,:,:].detach().cpu()
        predData['disp'] = out_disp.detach().cpu()

    if outputType == 'both':
        out_disp = outputs[:,0,:,:].unsqueeze(1)
        if old_config:
            out_disp = torch.sigmoid(out_disp)
        #out_disp = ((disp != 0) * 1.0) * out_disp.detach().cpu()
        if warpedOutput:
            dataset['left'] = outputs[:,-3:,:,:].detach().cpu()
            out_pred = outputs[:,1:-3,:,:]
        else:
            dataset['left'] = warpDisp(out_disp, dataset['left'].cuda(), trainTime=False).detach().cpu()
            out_pred = outputs[:,1:,:,:]
        predData = {'disp': out_disp.detach().cpu(),
                    'seg': out_pred.detach().cpu()}

    #if outputType == 'all' or outputType == 'smallOutPair':
    if outputType == 'deeplab':
        predData['seg'] = outputs.detach().cpu()
        predData['seg2'] = outputs.detach().cpu()
        predData['disp'] = dataset['disp'].detach().cpu()
        predData['disp2'] = dataset['disp'].detach().cpu()
    elif outputType == 'pspnet':
        predData['disp'] = outputs.detach().cpu()
        predData['disp2'] = outputs.detach().cpu()
        predData['seg'] = dataset['seg'][:,:19].detach().cpu()
        predData['seg2'] = dataset['seg'][:,:19].detach().cpu()

    else:
        predData['disp'] = outputs[3].detach().cpu()
        predData['seg'] = outputs[2].detach().cpu()
        if outputType == 'ThreeOutPuts':
            predData['seg3'] = outputs[4].detach().cpu()
            predData['disp'] = outputs[5].detach().cpu()
            #predData['disp2'] = outputs[1].detach().cpu()
        else:
            predData['disp2'] = outputs[1].detach().cpu()
        if outputType == 'edgeOut':
            predData['edge'] = outputs[0].detach().cpu()
        else:
            predData['seg2'] = outputs[0].detach().cpu()
    
    # Temp coment
    # # plt.figure()
    # grid = show_img_batch(dataset, torch.tensor(normalize_input), predData)
    # grid = torch.nn.functional.interpolate(grid.unsqueeze(0), size=(int(grid.shape[1]/2), int(grid.shape[2]/2)), mode='bilinear').squeeze()
    # plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
    # if name:
    #     count=name
    # else:
    #     count=0
    # name='results/fig_{}.png'.format(count)
    # # while os.path.isfile(name):
    # #     count += 1
    # #     name='fig_{}.png'.format(count)
    # utils.save_image(grid, name)

    # plt.title('Batch from dataloader')
    # plt.close()

    if show_statistics:
        seg_gt = dataset['seg'][:,:19]
        max_out = predData['seg'].max(dim=1, keepdim=True)
        y_ = (predData['seg'] == max_out[0]) * 1.0
        acc_per_class = torch.sum(y_ * seg_gt , dim=(2,3)).float()
        acc_per_class_norm = acc_per_class/(torch.sum(seg_gt, dim=(2,3))+1e-8)

        name = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        for i in range(acc_per_class.shape[0]):
            # plt.subplot(211)
            # plt.bar(name, (torch.sum(seg_gt, dim=(2,3))[i] > 1)*1.0 - acc_per_class_norm[i], color='r'); plt.xticks(rotation=90)
            # plt.subplot(212)
            # plt.bar(name, torch.sum(seg_gt, dim=(2,3))[i], color='r'); plt.xticks(rotation=90)
            # plt.bar(name, acc_per_class[i]); plt.xticks(rotation=90)
            # plt.savefig('results/bar_{}_{}.png'.format(count, i))
            # plt.close()

            # print(dataset['meta'][0][i]) #disp
            # print(dataset['meta'][1][i]) #seg
            data_name = "KITTI_2015"
            if data_name in dataset['meta'][1][i]:
                seg_name = dataset['meta'][1][i].split("KITTI_2015/")[-1]
            elif "cityscapes" in dataset['meta'][1][i]:
                data_name = "cityscapes"
                seg_name = dataset['meta'][1][i].split("cityscapes/")[-1]
            elif "garden" in dataset['meta'][1][i]:
                data_name = "garden"
                seg_name = dataset['meta'][1][i].split("garden2018/")[-1]
            else:  # roses
                data_name = "roses"
                seg_name =  dataset['meta'][1][i].split("roses/")[-1]

            disp_name = os.path.join('results/{}'.format(data_name), "disp/"+seg_name)
            seg_name = os.path.join('results/{}'.format(data_name), "seg/"+seg_name)
            os.makedirs(os.path.dirname(seg_name), exist_ok=True)
            os.makedirs(os.path.dirname(disp_name), exist_ok=True)
        # classSeg = np.zeros_like(seg_pred)
        # values = np.unique(seg_pred.reshape(-1), axis=0)
        # for i in values:
        #     mask = (seg_pred == i) * cityscapes.trainId2label[i].id
        #     classSeg += mask
            if True and outputType != 'deeplab':
                disp = outputs if outputType == 'pspnet' else outputs[3]
                disp = disp.detach().cpu()[i].squeeze().numpy()
                disp = np.uint16(disp* 256)#* 256)
                disp = Image.fromarray(disp)
                #cv2.imwrite(save_path, disp)
                disp.save(disp_name)

                pred = predData['seg'].argmax(dim=1, keepdim=True)
                classSeg = np.zeros_like(pred[i])

                for j in range(acc_per_class.shape[1]):
                    mask = (pred[i] == j) * cityscapes.trainId2label[j].id
                    classSeg += mask.numpy()

                classSeg = Image.fromarray(classSeg.squeeze().astype('uint8'), mode='L')
                #seg_name  = 'results/seg/{}.png'.format(str(count))
                classSeg.save(seg_name)
            # plt.axis('off')
            # plt.ioff()
            # plt.savefig('fig.png')
            # plt.close()    

def plot_confusion_matrix(cm, classes, name='name', normalize=True):
    
    if normalize:
        cm = np.nan_to_num(cm)
        cm = cm/cm.sum(axis=1, keepdims=True)
        cm = np.round(cm, 2)

    fig = plt.Figure()
    
    df_cm = pd.DataFrame(cm, index = classes,
                  columns = classes)
    plt.figure(figsize = (10,10))
    hp = sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}).get_figure()  
    # canvas = FigureCanvas(hp)
    # canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    # X = np.array(canvas.renderer.buffer_rgba())
    # return X[:,:,:3]

    # plt.imshow(X[:,:,:3])
    plt.savefig('results/{}.png'.format(name))
    plt.close()
    # plt.show()
    
