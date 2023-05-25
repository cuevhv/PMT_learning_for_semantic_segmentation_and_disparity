import warnings

warnings.filterwarnings("ignore")
from torchConfig import configParser
import matplotlib
import torch.multiprocessing as mp
import torch.distributed as dist

matplotlib.use('Agg')
from util import utilTorchGate
import time
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import sys
import shutil
import numpy as np
import torch.optim as optim
import torch.nn as nn
# from torchsummary import summary
from util.utilTorchDataLoader import *
from util import utilLoad
import torch.nn.functional as F
from util.utilTorchLoss import Mean_Intersection_over_Union
# from util import utilTorchGate
# from util.lovasz_losses import lovasz_softmax
from torch.utils.data import Dataset, DataLoader
from util.utilTorchPlot import plotBatchData, show_img_batch, showResults, plot_confusion_matrix
from util.utilLoadNetwork import getNetwork
from util.utilTorchAnalysis import computeMeanStd, computeDispStats, testTrainloader, getDatasetStats, evaluteGenerator, \
   count_classes_in_dataset
# from util.utilTorchLoss import *
# from util.utilTorchAnalysis import show_outputs
from util.utilTorch_loadweight import load_checkpoint_and_params
from models_psmnet import process_input
from losses.multiLosses import lossSeg_fn, lossDisp_fn, lossEdge_fn
from tabulate import tabulate

try:
    import apex
    # from apex.parallel import DistributedDataParallel as DDP
    # New version
    from torch.nn.parallel import DistributedDataParallel as DDP
except ImportError:
    print('no apex module')


# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/23
# torch.manual_seed(223)
def save_checkpoint(state, old_loss, new_loss, old_D_error, new_D_error, filename='checkpoint'):
    epoch_save_name = filename + '.pth.tar'

    if new_loss > old_loss:
        new_loss = round(new_loss, 4)
        old_loss = round(old_loss, 4)
        old_D_error = round(old_D_error, 4)
        new_D_error = round(new_D_error, 4)
        state['best_metric'] = [new_D_error, new_loss]
        torch.save(state, epoch_save_name)
        if os.path.exists(filename + '_model_best_IOU{}_Derr{}.pth.tar'.format(old_loss, old_D_error)):
            os.remove(filename + '_model_best_IOU{}_Derr{}.pth.tar'.format(old_loss, old_D_error))
        shutil.copyfile(epoch_save_name, filename + '_model_best_IOU{}_Derr{}.pth.tar'.format(new_loss, new_D_error))
    else:
        torch.save(state, epoch_save_name)


def divideNetOutput(model, left, right, seg, disp):
    seg1_complete = torch.zeros_like(seg)
    seg2_complete = torch.zeros_like(seg)
    disp_complete = torch.zeros_like(disp)
    sub_crop = 0.5
    b, c, h, w = left.shape
    h_new = 256
    w_new = 512
    h_div = h / float(h_new) / sub_crop
    w_div = w / float(w_new) / sub_crop
    h_crop = int(h_new * sub_crop)
    w_crop = int(w_new * sub_crop)
    # from util.utilTorchPlot import decode_segmap
    # plt.figure()
    count = 1
    height = int(h_div - 1)
    width = int(w_div - 1)
    for i in range(int(h_div - 1)):
        for j in range(int(w_div - 1)):
            # if i*h_crop < 512 and i*h_crop > 128 and j*w_crop < 512: #i*h_crop < 256 and i*h_crop > 0 and j*w_crop < 256:#

            outputs = model(left[:, :, i * h_crop: i * h_crop + h_new, j * w_crop: j * w_crop + w_new],
                            right[:, :, i * h_crop: i * h_crop + h_new, j * w_crop: j * w_crop + w_new])

            seg1_complete[:, :, i * h_crop: i * h_crop + h_new, j * w_crop: j * w_crop + w_new] += sub_crop ** 2 * \
                                                                                                   outputs[0]
            disp_complete[:, :, i * h_crop: i * h_crop + h_new, j * w_crop: j * w_crop + w_new] += sub_crop ** 2 * \
                                                                                                   outputs[1]
            seg2_complete[:, :, i * h_crop: i * h_crop + h_new, j * w_crop: j * w_crop + w_new] += sub_crop ** 2 * \
                                                                                                   outputs[2]
    #         plt.figure(1)
    #         plt.subplot(height, width, count)

    #         decoded_output = decode_segmap(outputs[2].detach().cpu().argmax(1).unsqueeze(1), nc=21)
    #         plt.imshow(np.transpose(decoded_output[1].numpy(), (1,2,0)))
    #         plt.figure(2)
    #         plt.subplot(height, width, count)
    #         i1 = plt.imshow(torch.softmax(outputs[2], dim=1).detach().cpu().numpy()[1,15])
    #         i1.set_clim(0, 1)
    #         plt.figure(3)
    #         plt.subplot(height, width, count)
    #         i2=plt.imshow(torch.softmax(outputs[2], dim=1).detach().cpu().numpy()[1,2])
    #         i2.set_clim(0, 1)
    #         count+=1
    # plt.show()
    outputs = [seg1_complete, disp_complete, seg2_complete, disp_complete]
    return outputs


def netForward(left, right, disp, seg, seg_full, model, outputType, return_cpu=False):
    # outputs = divideNetOutput(model, left, right, seg, disp)
    if outputType == 'pspnet':
        outputs = process_input.process(left, right, model)

    elif 'deeplab' in outputType:
        left = left * 2 - 1
        h, w = left.shape[2:]
        left = F.pad(left, [0, 1, 0, 1])
        if outputType == 'deeplab_mod':
            right = F.pad(right, [0, 1, 0, 1])
            outputs = model(left, right)
        else:
            outputs = model(left)

    elif outputType == 'ThreeOutPutsDisp':
        outputs = model(left, right, disp)
    elif outputType == 'edgeOut':
        edge = utilTorchGate.compute_grad_mag(left, True, False)
        outputs = model(left, right, edge)
    elif outputType == 'hanet':
        h = torch.arange(0, 1024).unsqueeze(0).unsqueeze(2).expand(left.shape[0], -1,
                                                                   2048) // 8  # .unsqueeze(0).expand(s2.shape[0],-1,-1,-1)//8
        w = torch.arange(0, 2048).unsqueeze(0).unsqueeze(1).expand(left.shape[0], 1024,
                                                                   -1) // 16  # .unsqueeze(0).expand(s2.shape[0],-1,-1,-1)//16
        pos = (h.cuda(), w.cuda())
        outputs = model(left, right, pos)

    elif outputType == 'multitask':
        seg_full = seg_full.cuda()
        disp = disp.cuda()
        outputs = model(left, right, None, disp, seg_full.argmax(1))

    else:
        outputs = model(left, right)
    seg2 = None
    seg3 = None
    warped_right = None
    edge_ds = None
    if outputType in ['ThreeOutPuts', 'ThreeOutPutsDisp', 'ThreeOutPutsDispConsist']:
        seg1, disp1, seg2, _, seg3, warped_right = outputs
    elif outputType == 'deeplab':
        seg1 = F.interpolate(outputs, size=(h + 1, w + 1), mode='bilinear', align_corners=True)[..., :h, :w]
        disp1 = disp
        seg2 = None
    elif outputType == 'deeplab_mod':
        seg1 = F.interpolate(outputs[0], size=(h + 1, w + 1), mode='bilinear', align_corners=True)[..., :h, :w]
        seg2 = F.interpolate(outputs[2], size=(h + 1, w + 1), mode='bilinear', align_corners=True)[..., :h, :w]
        disp1 = F.interpolate(outputs[1], size=(h + 1, w + 1), mode='bilinear', align_corners=True)[..., :h, :w]
    elif outputType == 'pspnet':
        disp1 = outputs
        seg1 = seg.cuda()
    elif outputType == 'edgeOut':
        edge_ds, disp1, seg1, _ = outputs
    elif outputType == 'multitask':
        seg1, disp1, seg2, _, loss_disp, loss_seg1, loss_seg2 = outputs
        loss_seg1 = loss_seg1.mean()
        loss_seg2 = loss_seg2.mean()
        loss_disp = loss_disp.mean()
        return seg1.detach().cpu(), disp1.detach().cpu(), seg2.detach().cpu(), seg3, loss_disp, loss_seg1, loss_seg2, outputs
    else:
        seg1, disp1, seg2, _ = outputs

    if return_cpu:
        return seg1.detach().cpu(), disp1.detach().cpu(), seg2.detach().cpu(), seg3, warped_right, edge_ds, outputs
    else:
        return seg1, disp1, seg2, seg3, warped_right, edge_ds, outputs


def slideWindowInfer(left, right, disp, seg, model, outputType):
    # resize_val = (int(left.size()[2]*2), int(left.size()[3]*2))
    # left = F.interpolate(left, size=resize_val, mode='bilinear')
    # right = F.interpolate(right, size=resize_val, mode='bilinear')
    # disp = F.interpolate(disp, size=resize_val, mode='bilinear')
    # seg_full = F.interpolate(seg_full, size=resize_val, mode='nearest')
    # seg = F.interpolate(seg, size=resize_val, mode='nearest')
    window_r, window_c = 512, 512
    stride = 256
    # cells_r = np.ceil(window_r/stride).astype(np.int)
    # cells_c = np.ceil(window_c/stride).astype(np.int)
    # print(np.ceil(window_r/stride).astype(np.int), np.ceil(window_c/stride))
    _, _, r, c = left.shape
    n_wr = np.arange(0, r - (window_r - stride), stride)
    n_wc = np.arange(0, c - (window_c - stride), stride)

    print(n_wc, n_wr)
    print(n_wc + window_c, n_wr + window_r)
    # print(cells_c, cells_r)
    seg1 = torch.zeros_like(seg)
    seg2 = torch.zeros_like(seg)
    disp1 = torch.zeros_like(disp)
    for wr in n_wr:
        for wc in n_wc:
            # print('-'*10)
            if wr + window_r <= r and wc + window_c <= c:
                pass
                # print('current window: ', wr,wr+window_r,  wc,wc+window_c)
            else:
                # print('skipped current window: ', wr,wr+window_r,  wc,wc+window_c)
                wc = c - window_c
                wr = r - window_r
                # print('new current window: ', wr,wr+window_r,  wc,wc+window_c)

            seg1_p, disp1_p, seg2_p, seg3, _, _, outputs = netForward(left[:, :, wr:wr + window_r, wc:wc + window_c],
                                                                      right[:, :, wr:wr + window_r, wc:wc + window_c],
                                                                      disp[:, :, wr:wr + window_r, wc:wc + window_c],
                                                                      seg[:, :, wr:wr + window_r, wc:wc + window_c],
                                                                      model, outputType, return_cpu=True)
            seg1_p = F.softmax(seg1_p, dim=1)
            seg2_p = F.softmax(seg2_p, dim=1)
            seg1[:, :, wr:wr + window_r, wc:wc + window_c] += seg1_p
            seg2[:, :, wr:wr + window_r, wc:wc + window_c] += seg2_p
            disp1[:, :, wr:wr + window_r, wc:wc + window_c] = disp1_p

    return seg1, disp1, seg2, seg3, outputs


def networkOutput(data, model, CFG, freeze_bn=0, num_image=0):
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.eval()

    loss, loss_seg, loss_disp = 0, 0, 0
    max_disp = 1.0
    left = data['left']
    right = data['right']
    disp = data['disp']
    seg_full = data['seg']
    if CFG.datasetName in ['cityscapes', 'kitti']:
        seg = seg_full[:, :seg_full.shape[1] - 1]
    elif CFG.datasetName == 'garden':
        seg = seg_full
    elif CFG.datasetName == 'roses':
        seg = seg_full

    left = left.cuda()
    right = right.cuda()
    disp = disp.cuda()

    if CFG.edges:
        edge = data['edges'].float()
        edge = edge.cuda()
        left = torch.cat((left, edge), dim=1)

    slide_window = 0
    if slide_window:
        seg1, disp1, seg2, seg3, outputs = slideWindowInfer(left, right, disp, seg, seg_full, model, CFG.outputType)
    elif CFG.outputType == 'multitask':
        seg1, disp1, seg2, seg3, l_disp1, l_seg1, l_seg2, outputs = netForward(left, right, disp, seg, seg_full, model,
                                                                               CFG.outputType)
    else:
        seg1, disp1, seg2, seg3, warped_right, edge_ds, outputs = netForward(left, right, disp, seg, seg_full, model,
                                                                             CFG.outputType)

    # show_outputs(outputs, left, right, disp, seg, seg_full)

    # SEG LOSSES
    if not slide_window and CFG.outputType != 'multitask':
        pixelAcc1, conf_matrix1, l_seg1, dice_val, pixelPrec_1, pixelRecall_1, pixelF1_1, pixelBF1_1,  = lossSeg_fn(['cross_entropy'], seg_full, seg1, CFG, num_image=str(num_image) + "_seg1")
    elif CFG.outputType == 'multitask':
        pixelAcc1, conf_matrix1, _, dice_val = lossSeg_fn(['None'], seg_full, seg1, CFG)
    else:
        pixelAcc1, conf_matrix1, l_seg1, dice_val = lossSeg_fn(['None'], seg_full, seg1, CFG)

    loss_seg = loss_seg + l_seg1
    if CFG.outputType in ['smallOutPair', 'deeplab', 'edgeOut', 'pspnet']:
        pixelAcc2 = pixelAcc1
        conf_matrix2 = conf_matrix1
    elif CFG.outputType == 'multitask':
        pixelAcc2, conf_matrix2, _, _ = lossSeg_fn(['None'], seg_full, seg2, CFG)
        loss_seg = loss_seg + l_seg2
    else:
        pixelAcc2, conf_matrix2, l_seg2, _, pixelPrec_2, pixelRecall_2, pixelF1_2, pixelBF1_2 = lossSeg_fn(CFG.loss, seg_full, seg2, CFG, num_image=str(num_image) + "_seg2")
        loss_seg = loss_seg + l_seg2

    if CFG.outputType in ['ThreeOutPuts', 'ThreeOutPutsDisp', 'ThreeOutPutsDispConsist']:
        _, _, l_seg3, _ = lossSeg_fn(['cross_entropy'], seg_full, seg3, CFG)
        loss_seg = loss_seg + l_seg3

    # DISP LOSSES
    if CFG.outputType == 'multitask':
        err, val_pxl, _, smooth_val = lossDisp_fn(CFG.outputType, left, seg_full, disp, disp1, max_disp, CFG)
    else:
        err, val_pxl, l_disp1, smooth_val, dispRMSE, dispSqRel, BdispRMSE, BdispSqRel = lossDisp_fn(CFG.outputType, left, seg_full, disp, disp1, max_disp, CFG, num_image=str(num_image))
    loss_disp = loss_disp + l_disp1
    MAE = [err, val_pxl, err, val_pxl]
    pixelAcc = [pixelAcc1, pixelAcc2]
    pixelPrec = [pixelPrec_1, pixelPrec_2]
    pixelRecall = [pixelRecall_1, pixelRecall_2]
    pixelF1 = [pixelF1_1, pixelF1_2]
    pixelBF1 = [pixelBF1_1, pixelBF1_2]
    conf_matrix = np.concatenate((np.expand_dims(conf_matrix1, 0), np.expand_dims(conf_matrix2, 0)), 0)

    if CFG.outputType in ['smallOutWarp', 'ThreeOutPutsDispConsist']:
        loss_disp = loss_disp * 0.0
        photo_consistency_loss = nn.MSELoss()(warped_right, left)
        loss_disp = loss_disp + photo_consistency_loss

    # EDGE LOSSES
    if CFG.outputType == 'edgeOut':
        edge_loss = lossEdge_fn(data['edges'], edge_ds)
        smooth_val = edge_loss.item()
        loss = loss + edge_loss

    loss = loss + loss_disp + loss_seg
    if not slide_window:
        return loss, MAE, pixelAcc, outputs, loss_disp.item(), loss_seg.item(), np.array(
            [smooth_val, dice_val]), conf_matrix,  pixelPrec, pixelRecall, pixelF1, pixelBF1, dispRMSE, dispSqRel, BdispRMSE, BdispSqRel
    else:
        return loss, MAE, pixelAcc, outputs, loss_disp, loss_seg, np.array([smooth_val, dice_val]), conf_matrix


def train_model(datasetLoader, epoch, model, optimizer, cfg, rank=0, scaler=False):
    use_apex = cfg.f16
    accumulation_steps = cfg.acmt_grad
    model.train()
    if cfg.outputType == 'deeplab':
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.eval()
    Total_MAE = np.array([0.0, 0.0, 0.0, 0.0])
    TotalpixelAcc = np.array([0.0, 0.0])
    total_conf = np.zeros((2, cfg.n_labels, cfg.n_labels))
    total_loss, total_disp, total_seg, total_other_losses = 0, 0, 0, 0

    if rank == 0:
        start_t = time.time()
    optimizer.zero_grad()  # zero the parameter gradients
    model.zero_grad()
    for i, data in enumerate(datasetLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        if optimizer.__module__ == 'torch.optim.sgd':
            adjust_learning_rate(optimizer, epoch, i, len(datasetLoader))
        if scaler:
            with torch.cuda.amp.autocast():  # new
                loss, MAE, pixelAcc, _, loss_disp, loss_seg, other_losses, conf_matrix = networkOutput(data, model, cfg,
                                                                                                       cfg.freeze_bn)
        else:
            loss, MAE, pixelAcc, _, loss_disp, loss_seg, other_losses, conf_matrix, *rest = networkOutput(data, model, cfg,
                                                                                                   cfg.freeze_bn)

        loss = loss / accumulation_steps
        total_loss += loss.item()
        Total_MAE += MAE
        TotalpixelAcc += pixelAcc
        total_disp += loss_disp
        total_seg += loss_seg
        total_other_losses += other_losses
        total_conf += conf_matrix
        if i % 5 == 0 and rank == 0:
            print('[%d, %5d / %5d] loss: %.3f, disp loss: %.3f, seg loss: %.3f, MAE: %.3f, '
                  'MIoU1: %.3f, MIoU2: %.3f, time: %.3f, opt: %.5f' %
                  (epoch + 1, i + 1, len(datasetLoader), loss.item(), loss_disp, loss_seg,
                   MAE[0] / MAE[1],
                   Mean_Intersection_over_Union(conf_matrix[0])[0], Mean_Intersection_over_Union(conf_matrix[1])[0],
                   time.time() - start_t, optimizer.param_groups[0]['lr']))
            

            start_t = time.time()
        if use_apex:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                # torch.nn.utils.clip_grad_norm_(apex.amp.master_params(optimizer), 1)
        else:
            if scaler:
                print('scale', scaler.get_scale())
                scaler.scale(loss).backward()  
            else:
                loss.backward()
        if (i + 1) % accumulation_steps == 0 or i + 1 == len(datasetLoader):
            if scaler:
                scaler.step(optimizer)  
                scaler.update()  
            else:
                optimizer.step()
            optimizer.zero_grad()  # zero the parameter gradients
            model.zero_grad()

    return total_loss / (i + 1.0), total_disp / (i + 1.0), total_seg / (i + 1.0), Total_MAE[0] / Total_MAE[1], \
           Total_MAE[2] / Total_MAE[3], Mean_Intersection_over_Union(total_conf[0]), Mean_Intersection_over_Union(
        total_conf[1]), total_conf, total_other_losses / (i + 1.0)


# Returns string with mean and std of metric
def mainAndStd(measure):
    return str(np.round(np.mean(measure), 4)) + '±' + str(np.round(np.std(measure), 4))

def printResultsMetrics(epoch, div, datasetLoader, outputLoss, outputPixels, outputDisp, outputSegm, outputBranch, i = 0, final = False, std = False):
    if not final:
        print('\nEval [%d:%5d/%5d]' %
            (epoch + 1, div, len(datasetLoader) ) )

    # Error as standard deviation
    if std:
        tlos, D_Loss, S_Loss= outputLoss[0],outputLoss[1],outputLoss[2]
        Px_Prec, Px_Recall, Px_F1 =outputPixels[1], outputPixels[2],outputPixels[3] 
        d_RMSE, D_SqRel =  outputDisp[0],outputDisp[1]
        S_AvIoU, =  outputSegm
        
        headers = ['T_Loss', 'D_Loss', 'S_Loss', 'Px_Prec', 'Px_Recall', 'Px_F1', 'D_RMSE', 'D_SqRel', 'S_AvIoU']
        contentTable = [mainAndStd(tlos), mainAndStd(D_Loss), mainAndStd(S_Loss), mainAndStd(Px_Prec), mainAndStd(Px_Recall), mainAndStd(Px_F1), mainAndStd(d_RMSE), mainAndStd(D_SqRel), mainAndStd(S_AvIoU)] 

        px_F1, d_RMSE, d_SqRel = outputBranch[0],outputBranch[1],outputBranch[2]
        headers2 = ['Px_F1', 'D_RMSE', 'D_SqRel']
        contentTable2 = [mainAndStd(px_F1), mainAndStd(d_RMSE), mainAndStd(d_SqRel)]

        table = tabulate([contentTable], headers=headers, tablefmt='orgtbl')
        print(table, "\n" )

        table2 = tabulate([contentTable2], headers=headers2, tablefmt='orgtbl')
        print(table2, "\n" )



    else:
        headers = ['T_Loss', 'D_Loss', 'S_Loss', 'Px_Prec', 'Px_Recall', 'Px_F1', 'D_RMSE', 'D_SqRel', 'S_AvIoU']
        contentTable = [outputLoss[0],outputLoss[1],outputLoss[2],outputPixels[1], outputPixels[2],outputPixels[3],outputDisp[0],outputDisp[1],outputSegm] 

        headers2 = ['Px_F1', 'D_RMSE', 'D_SqRel']
        contentTable2 = [outputBranch[0],outputBranch[1],outputBranch[2]]

        table = tabulate([np.round(contentTable, 4)], headers=headers, tablefmt='orgtbl')
        print(table, "\n" )

        table2 = tabulate([np.round(contentTable2, 4)], headers=headers2, tablefmt='orgtbl')
        print(table2, "\n" )



def test_model(datasetLoader, epoch, model, cfg, show_results=False, showperStep=False, rank=0, train=False):
    model.eval()
    normalize_input = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    Total_MAE = np.array([0.0, 0.0, 0.0, 0.0])
    TotalpixelAcc = np.array([0.0, 0.0])
    # TotalpixelRecall, TotalpixelF1, TotalpixelBF1  = np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])
    TotalpixelRecall, TotalpixelF1, TotalpixelBF1, TotalpixelPrec = 0.0, 0.0, 0.0, 0.0
    TotaldispRMSE, TotaldispSqRel, TotalAvIoU = 0.0, 0.0, 0.0
    TotalBdispRMSE, TotalBdispSqRel = 0.0, 0.0
    
    # Totals to get final metrics on std and average
    all_loss = []
    all_loss_disp = []
    all_loss_seg = []
    all_pixelAcc = []
    all_pixelPrec = []
    all_pixelRecall = []
    all_pixelF1 = []
    all_dispRMSE = []
    all_dispSqRel = []
    all_avIoU = []
    all_pixelBF1 = []
    all_BdispRMSE = []
    all_BdispSqRel = []

    all_outputLosses, all_outputPixels, all_outputDisp, all_outputSegm, all_outputBranch = [], [], [], [], []


    total_conf = np.zeros((2, cfg.n_labels, cfg.n_labels))
    total_loss, total_disp, total_seg = 0, 0, 0
    # from tqdm import tqdm
    # file_log = tqdm(total=0, position=1, bar_format='{desc}')
    # tk0 = tqdm(datasetLoader, total=len(datasetLoader))
    with torch.no_grad():
        for i, data in enumerate(datasetLoader, 0):
            loss, MAE, pixelAcc, outputs, loss_disp, loss_seg, _, \
            conf_matrix, pixelPrec, pixelRecall, \
            pixelF1, pixelBF1, dispRMSE, dispSqRel, BdispRMSE, BdispSqRel  = networkOutput(data, model, cfg, cfg.freeze_bn, num_image="_img_" + str(i) )
            
            # from util.utilTorchAnalysis import eval_seg_result
            # eval_seg_result(outputs, left, seg, 3)
            total_loss += loss.item()
            Total_MAE += MAE

            TotalpixelAcc    += pixelAcc
            TotalpixelPrec   += max(pixelPrec) # Best segmentation
            TotalpixelRecall += max(pixelRecall)
            TotalpixelF1     += max(pixelF1)
            TotalpixelBF1    += max(pixelBF1)
            TotaldispRMSE    += dispRMSE
            TotaldispSqRel   += dispSqRel
            TotalBdispRMSE   += BdispRMSE
            TotalBdispSqRel  += BdispSqRel
            
            # avIoU = (Mean_Intersection_over_Union(conf_matrix[0])[0] + Mean_Intersection_over_Union(conf_matrix[1])[0])   / 2.0
            avIoU = max(Mean_Intersection_over_Union(conf_matrix[0])[0], Mean_Intersection_over_Union(conf_matrix[1])[0]) # Select only best segmentation. Second one
            TotalAvIoU       += avIoU
            
            total_disp += loss_disp
            total_seg += loss_seg
            total_conf += conf_matrix

            ## Average results
            div = i + 1.0

            outputLosses = total_loss / div, total_disp / div, total_seg / div
            outputPixels = np.mean(TotalpixelAcc / div), TotalpixelPrec / div, TotalpixelRecall / div, TotalpixelF1 / div
            outputDisp   = TotaldispRMSE / div, TotaldispSqRel / div
            outputSegm   = TotalAvIoU / div
            outputBranch = TotalpixelBF1 / div, TotalBdispRMSE / div, TotalBdispSqRel / div

            #######
            # Without average results
            all_loss.append(loss.item())
            all_loss_disp.append(loss_disp)
            all_loss_seg.append(loss_seg)

            all_pixelAcc.append(np.mean(pixelAcc))
            all_pixelPrec.append(max(pixelPrec))
            all_pixelRecall.append(max(pixelRecall))
            all_pixelF1.append(max(pixelF1))

            all_dispRMSE.append(dispRMSE)
            all_dispSqRel.append(dispSqRel)

            all_avIoU.append(avIoU)

            all_pixelBF1.append(max(pixelBF1))
            all_BdispRMSE.append(BdispRMSE)
            all_BdispSqRel.append(BdispSqRel)


            all_outputLosses = [all_loss , all_loss_disp , all_loss_seg ]
            all_outputPixels = [all_pixelAcc, all_pixelPrec, all_pixelRecall, all_pixelF1 ]
            all_outputDisp = [all_dispRMSE, all_dispSqRel ]
            all_outputSegm = [all_avIoU]
            all_outputBranch = [all_pixelBF1, all_BdispRMSE, all_BdispSqRel] 


            ## Iteration results
            if showperStep:
                outputLossesI = loss.item() , loss_disp , loss_seg
                outputPixelsI = np.mean(pixelAcc), max(pixelPrec), max(pixelRecall), max(pixelF1)
                outputDispI   = dispRMSE, dispSqRel
                outputSegmI   = avIoU
                outputBranchI = max(pixelBF1), BdispRMSE, BdispSqRel


                printResultsMetrics(epoch, div, datasetLoader, outputLossesI, 
                                    outputPixelsI, outputDispI, outputSegmI, outputBranchI)


            if (i % 10 == 0 and rank == 0) or showperStep:                
                print("\n#################\n Average results\n#################\n")
                printResultsMetrics(epoch, div, datasetLoader, outputLosses, 
                    outputPixels, outputDisp, outputSegm, outputBranch, final = True)
                print("\n")
                
            if showperStep and show_results:
                showResults(data, outputs, model, normalize_input, cfg.outputType, showperStep, name=i)
            
        if not showperStep and show_results:
            showResults(data, outputs, model, normalize_input, cfg.outputType)

    totalOutput  = (total_conf, all_outputLosses, all_outputPixels, all_outputDisp, all_outputSegm, all_outputBranch)


    if not train:
        return  totalOutput
    else:
        return total_loss / (i + 1.0), total_disp / (i + 1.0), total_seg / (i + 1.0), Total_MAE[0] / Total_MAE[1], \
                    Total_MAE[2] / Total_MAE[3], Mean_Intersection_over_Union(total_conf[0]), Mean_Intersection_over_Union(
                    total_conf[1]), total_conf

def changeWeightName(state_dict, model_dict_list):
    old_dict_list = np.array(list(state_dict.keys()))
    model_dict_list = np.array(model_dict_list)

    mask1 = np.isin(model_dict_list, old_dict_list)
    mask2 = np.isin(old_dict_list, model_dict_list)
    if not np.all(mask1) and not np.all(mask2):
        new_name = model_dict_list[~mask1][0]
        old_name = old_dict_list[~mask2][0]
        state_dict[new_name] = state_dict[old_name]
        del state_dict[old_name]

    return state_dict


def adjust_learning_rate(optimizer, epoch, itr, total_iter):
    base_lr = 0.005
    epoch_total = 2400

    T = epoch * total_iter + itr
    N = epoch_total * total_iter
    if epoch >= epoch_total:
        T = N - 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * (1 - T / float(N))
        # print(epoch, T, param_group['lr'])


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


def runNetwork(gpu_id, CFG):
    gpu = gpu_id
    if CFG.nodes:
        print('gpu', gpu_id)
        rank = CFG.nr * len(CFG.gpu_n.split(',')) + gpu_id
        print('rank', rank)
        print(CFG.world_size)
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=CFG.world_size, rank=rank)
    else:
        print("NO GPU")
        rank = 0
    CFG.outputType = CFG.output_type
    training_script = CFG.trainCompressed
    test_script = CFG.testCompressed
    colorL, colorR, disp, seg, inst, \
    colorL_test, colorR_test, disp_test, seg_test, inst_test = utilLoad.getTextDataset(CFG)

    if not os.path.isdir('results'):
        os.mkdir('results')
    if CFG.segWeight:
        print('weighting segmentation output')

    if CFG.datasetName == 'garden':
        CFG.n_labels = 9
        max_disp = 100.0  # 25.0
    elif CFG.datasetName == 'roses':
        CFG.n_labels = 2
        max_disp = 100.0  # 25.0        
    else:
        CFG.n_labels = 19  # 8
        max_disp = 100.0

    if CFG.output_activation == 'linear':
        max_disp = 1

    net = getNetwork(CFG)
    # computeDispStats(disp, CFG.datasetName)

    if rank == 0:
        print('batch_size: ', CFG.batch * torch.cuda.device_count())
        if torch.cuda.device_count() > 1 and CFG.nodes == 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            CFG.batch = torch.cuda.device_count() * CFG.batch
        elif CFG.nodes:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        else:
            print("only 1 GPU")

    print(CFG.backbone, 'backbone')
    print('aspp config: {}, hanet: {}, convDeconv last: {}, dropout: {}, abliation: {}, multitaskLoss: {}'.format(
        CFG.aspp, CFG.hanet,
        CFG.convDeconvOut,
        CFG.dropout,
        CFG.abilation,
        CFG.multaskloss))

    if False:  # not CFG.nodes:
        from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
        from sync_batchnorm.replicate import patch_replication_callback
        def replace_bn(m, name):
            for attr_str in dir(m):
                target_attr = getattr(m, attr_str)
                if type(target_attr) == torch.nn.BatchNorm2d:
                    # print('replaced: ', name, attr_str)
                    setattr(m, attr_str,
                            SynchronizedBatchNorm2d(target_attr.num_features, target_attr.eps, target_attr.momentum,
                                                    target_attr.affine))
            for n, ch in m.named_children():
                replace_bn(ch, n)

        def check_bn(m, name):
            for attr_str in dir(m):
                target_attr = getattr(m, attr_str)
                # print('replaced: ', attr_str)
                if type(target_attr) == SynchronizedBatchNorm2d:
                    print('sync: ', attr_str)
                if type(target_attr) == torch.nn.BatchNorm2d:
                    print('not: ', attr_str)
                    # setattr(m, attr_str, SynchronizedBatchNorm2d(target_attr.num_features, target_attr.eps, target_attr.momentum, target_attr.affine))
            for n, ch in m.named_children():
                check_bn(ch, n)
        # replace_bn(net, "net")
        # check_bn(net, "net")
    # for name, module in net.named_modules():
    #     if isinstance(module, nn.BatchNorm2d):
    #         # Get current bn layer
    #         print(name)
    #         bn = getattr(net, name)
    #         # Create new gn layer
    #         gn = SynchronizedBatchNorm2d
    #         # Assign gn
    #         print('Swapping {} with {}'.format(bn, gn))
    #         setattr(net, name, gn)
    if CFG.optimType == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)
    else:
        if CFG.net == 'deeplab':
            lr = 5e-06
        elif len(CFG.loss) > 2:
            lr = 5e-04
        else:
            lr = 0.0015
        optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-7, weight_decay=0)  # , weight_decay=0.01, momentum=0.9)

    if CFG.nodes:
        if CFG.f16:
            print('starting apex')
            net.cuda(gpu)
            ## Previous implementation, previous apex version
            # net = apex.parallel.convert_syncbn_model(net)
            # net = apex.parallel.convert_syncbn_model(net)
            # net = DDP(net, delay_allreduce=True) # Pertains to previous apex versions
            net = apex.parallel.DistributedDataParallel(net)
            net, optimizer = apex.amp.initialize(net, optimizer)
            net = DDP(net)

        else:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net.cuda(gpu)
            net = nn.parallel.DistributedDataParallel(net, device_ids=[gpu], find_unused_parameters=True)
    else:
        if CFG.f16:
            net.cuda(gpu)
            net, optimizer = apex.amp.initialize(net, optimizer)

        net = nn.DataParallel(net)
        # patch_replication_callback(net)
        net = net.cuda()
        # net.cuda(gpu)

    if training_script:
        workers = 0
    else:
        workers = 1 # Testing on low capabilities machine 
        #workers = 4
    # normalize_input = np.array([[0.3875, 0.4101, 0.3638], [0.2574, 0.2596, 0.2936]], dtype=np.float32)

    if 'efficientnet' in CFG.backbone or 'pspnet' in CFG.net:
        normalize_input = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]], dtype=np.float32)
    else:
        normalize_input = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    if not training_script:
        training_script = [colorL[:], colorR[:], disp[:], seg[:], inst[:]]
    if not test_script:
        test_script = [colorL_test[:], colorR_test[:], disp_test[:], seg_test[:], inst_test[:]]
    trainset, testset = generateDataloaders(training_script, test_script, CFG.crop, CFG.n_labels, max_disp, CFG.output_activation,
                                                                        CFG.datasetName, normalize_input,
                                                                        CFG.only_test, CFG.train)

    if CFG.nodes:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,
                                                                        num_replicas=CFG.world_size,
                                                                        rank=rank)

        test_sampler = torch.utils.data.distributed.DistributedSampler(testset,
                                                                       num_replicas=CFG.world_size,
                                                                       rank=rank)

    print("Num_workers:", workers)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CFG.batch,
                                              shuffle=False if CFG.nodes else True,
                                              sampler=train_sampler if CFG.nodes else None, num_workers=workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=CFG.batch,
                                             # CFG.batch//2 if CFG.batch >= 8 else 2,
                                             shuffle=test_sampler, sampler=None if CFG.nodes else None,
                                             num_workers=workers, pin_memory=True)

    # testloader = torch.utils.data.DataLoader(testset, batch_size=1 if CFG.nodes or CFG.datasetName == 'kitti' else 2,
    #                                          # CFG.batch//2 if CFG.batch >= 8 else 2,
    #                                          shuffle=False, sampler=None if CFG.nodes else None,
    #                                          num_workers=4 if CFG.nodes else 4, pin_memory=True)

    # count_classes_in_dataset(trainset, CFG.datasetName, min_pxl=0)
    # from skimage import data, segmentation
    # for w, data in enumerate(trainset, 0):
    #     left = data['left']
    #     seg = data['seg']
    #     left = np.transpose(left.cpu().numpy(), (1,2,0))
    #     seg_wall = seg.cpu().numpy()[3,:]

    #     if np.any(seg[3,:,:].cpu().numpy()):
    #         print(seg.cpu().numpy().shape)
    #         seg_boundaries = segmentation.mark_boundaries(left, seg_wall.astype(np.int), mode='thick')
    #         plt.subplot(3,1,1), plt.imshow(seg_boundaries)
    #         plt.subplot(3,1,2), plt.imshow(seg.cpu().numpy().argmax(0))
    #         plt.subplot(3,1,3), plt.imshow(seg_wall)
    #         plt.show()

    #     # plt.imshow()
    #     print('data:', w)
    #     # input()

    # for i, data in enumerate(trainset, 0):
    #     print('data', i)
    # evaluteGenerator(trainset, CFG, n_labels, ignore_class=20)
    # from util.utilTorchAnalysis import invertDisp
    # invertDisp(trainloader)
    weight_file = os.path.join(CFG.w_savePath, 'weights')
    model_id_name = 'model_{}_i{}_{}_e{}_b{}_a{}_o{}_w{}_l{}_cr{}_aspp{}_optim{}_backbone{}_ablt{}{}_att{}_dropout{}{}_data{}'.format(
                                                                    CFG.net, CFG.crop[0], CFG.crop[1],
                                                                    CFG.epoch, CFG.batch, CFG.output_activation,
                                                                    CFG.outputType, CFG.segWeight, '_'.join(CFG.loss) if len(CFG.loss) > 1 else CFG.loss[0],
                                                                    CFG.corrType, CFG.aspp, CFG.optimType, CFG.backbone,
                                                                    '_'.join(CFG.abilation) if len(CFG.abilation) >= 1 else CFG.abilation,
                                                                    '_hanet1' if CFG.hanet else '', CFG.use_att, CFG.dropout,
                                                                    '_multaskloss'+str(CFG.multaskloss) if CFG.multaskloss else '',
                                                                    CFG.datasetName)
                                                                
    if CFG.load_weights:
        print('weights', CFG.load_weights)
    else:
        print('weights', model_id_name)
    weight_name = os.path.join(weight_file, model_id_name)

    if CFG.datasetName == 'garden':
        class_name = ['Grass', 'Ground', 'Pavement', 'Hedge', 'Topiary', 'Rose', 'Obstacle', 'Tree', 'Background']
    elif CFG.datasetName == 'roses':
        class_name = ['Background', 'Branch']
    elif CFG.datasetName in ['cityscapes', 'kitti']:
        class_name = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                      'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                      'bicycle']

    # from torchsummary import summary
    # summary(net, [(3, CFG.crop[0], CFG.crop[1]), (3, CFG.crop[0], CFG.crop[1])])
    # total_param = sum(p.numel() for p in net.parameters())
    # train_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print('total param: {}, train param: {}'.format(total_param, train_param))

    # parameter_total = []
    # for name, parameter in net.named_parameters():
    #     parameter_total.append((parameter.numel(), name))
    # sorted_param = sorted(parameter_total)
    # for vals in sorted_param:
    #     print(vals)
    if CFG.torch_amp:
        print('torch amp init')
        scaler = torch.cuda.amp.GradScaler()  
    else:
        scaler = False
    load_weights_by_name = CFG.hanet or CFG.convDeconvOut or CFG.net == 'deeplab_mod' #or not CFG.train  # or CFG.datasetName == 'garden'

    start_e, best_metric, epoch_history, \
    IoU_history_val, disp_history_val, \
    loss_history_val, IoU_history_train, \
    disp_history_train, loss_history_train = load_checkpoint_and_params(CFG, rank, load_weights_by_name, scaler,
                                                                        net, optimizer)
    if CFG.nodes:
        dist.barrier()

    # log_fh = open(f'{weight_name}.log', 'a+' if start_e else 'w+')
    log_fh = ''
    old_D_error = best_metric[0]
    old_loss = best_metric[1]
    if CFG.train:
        for epoch in range(start_e, CFG.epoch + start_e):
            t0 = time.time()
            th = 10 if epoch > 500 else 20
            if CFG.nodes:
                train_sampler.set_epoch(epoch)

            train_avgloss, train_disploss, train_segloss, \
            train_avgMAE1, train_avgMAE2, \
            train_avgpixelAcc1, train_avgpixelAcc2, train_confMatrix, \
            other_losses = train_model(trainloader, epoch, net, optimizer, CFG,
                                       rank=rank, scaler=scaler)
            # plotBatchData(trainloader, normalize_input)

            if epoch % th == 0 or epoch == CFG.epoch - 1:
                test_avgloss, test_disploss, test_segloss, \
                test_avgMAE1, test_avgMAE2, \
                test_avgpixelAcc1, test_avgpixelAcc2, test_confMatrix = test_model(testloader, epoch, net, CFG,
                                                                                   show_results=CFG.show_results,
                                                                                   rank=rank, train = CFG.train)
                # if CFG.show_results:
                if rank == 0:
                    print('evaluation finished')
                    # if epoch % 25 == 0  or epoch == CFG.epoch-1:
                    #     for name, param in net.named_parameters(): # model is the NN model, f is one set of parameters of the model
                    #         # Create a dynamic name for the histogram summary 
                    #         # Use current parameter shape to identify the variale  
                    #         writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                    # torch.save(net.state_dict(), 'weights/'+weight_name+'_epoch.pth')
                    epoch_history.append(epoch + 1)
                    IoU_history_val.append([test_avgpixelAcc1[0], test_avgpixelAcc2[0]])
                    disp_history_val.append([test_avgMAE1, test_avgMAE2])
                    loss_history_val.append([test_avgloss, test_disploss, test_segloss])
                    IoU_history_train.append([train_avgpixelAcc1[0], train_avgpixelAcc2[0]])
                    disp_history_train.append([train_avgMAE1, train_avgMAE2])
                    loss_history_train.append([train_avgloss, train_disploss, train_segloss])
                    save_dict = {'epoch': epoch + 1,
                                 'state_dict': net.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'train_cm': train_confMatrix,
                                 'test_cm': test_confMatrix,
                                 'best_metric': best_metric,
                                 'epoch_history': epoch_history,
                                 'IoU_history_val': IoU_history_val,
                                 'disp_history_val': disp_history_val,
                                 'loss_history_val': loss_history_val,
                                 'IoU_history_train': IoU_history_train,
                                 'disp_history_train': disp_history_train,
                                 'loss_history_train': loss_history_train,
                                 }
                    if CFG.f16:
                        save_dict['amp'] = apex.amp.state_dict()
                    if CFG.torch_amp:
                        save_dict['amp'] = scaler.state_dict()

                    save_checkpoint(save_dict, old_loss, test_avgpixelAcc2[0], old_D_error, test_avgMAE2, weight_name)
                    if old_loss < test_avgpixelAcc2[0]:
                        old_loss = test_avgpixelAcc2[0]
                        old_D_error = test_avgMAE2
                        # print('saving weight')
                        # torch.save(net.state_dict(), 'weights/'+weight_name+'.pth')
                else:
                    print(str(rank) + ': waiting for evaluation')
                if CFG.nodes:
                    dist.barrier()
            print('train_time: ', time.time() - t0)
        print('Training finished')
    elif rank == 0:
        epoch = 1
        # if CFG.epoch:
        #     epoch = CFG.epoch
        # plotBatchData(testloader, normalize_input)
        test_confMatrix, all_outputLosses, all_outputPixels, \
        all_outputDisp, all_outputSegm, all_outputBranch = test_model(testloader, epoch, net, CFG, show_results=CFG.show_results,
                                                                           showperStep=True,
                                                                           rank=rank)

        print('\n\n################\n Final av. std. metrics:\n################\n')
        
        printResultsMetrics(1, 1, 1, all_outputLosses, 
                            all_outputPixels, all_outputDisp, all_outputSegm, all_outputBranch, final = True, std = True)
        
        for cm_i in range(test_confMatrix.shape[0]):
            plot_confusion_matrix(test_confMatrix[cm_i], class_name, 'test_confusion_matrix ' + str(cm_i))
    if CFG.nodes and rank == 0:
        dist.destroy_process_group()


def main():
    CFG = configParser()
    os.environ['CUDA_VISIBLE_DEVICES'] = CFG.gpu_n
    if CFG.nodes:
        CFG.world_size = len(CFG.gpu_n.split(','))  # * CFG.nodes unable if using more nodes
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        mp.spawn(runNetwork, nprocs=len(CFG.gpu_n.split(',')), args=(CFG,))
    else:
        runNetwork(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), CFG)


if __name__ == "__main__":
    main()

# ================================================================================================================================
# useful garbage
# computeMeanStd(trainloader, input_size)
# plotBatchData(trainloader)
# plotBatchData(testloader)

# dd = testset[0]
# data = {'left': dd['left'].unsqueeze(0).detach().cpu(),
#         'right': dd['right'].unsqueeze(0).detach().cpu(),
#         'disp': dd['disp'].unsqueeze(0).detach().cpu(),
#         'seg': dd['seg'].unsqueeze(0).detach().cpu()}
# plt.figure()
# show_img_batch(data, torch.tensor(normalize_input))
# plt.axis('off')
# plt.ioff()
# plt.savefig('fig.png')
# plt.close()
# input('stop')
# plotBatchData(trainloader, normalize_input)


# base_lr = 0.01
# total_iter = 372//2
# epoch_total = 800

# for epoch in range(epoch_total):
#     for iter in range(total_iter):
#         T = epoch * total_iter + iter
#         N = epoch_total * total_iter
#         param = base_lr * (1 - T/float(N))
#         print(epoch, iter, param)


'''
debugging
gt_seg = seg.argmax(1)
            pred_seg = outputs[2].argmax(1)
            
            if ((gt_seg.cpu() == 16).numpy() & (pred_seg.cpu().numpy() == 15)).any():
                import matplotlib.pyplot as plt
                from util.utilTorchPlot import decode_segmap
                # decoded_output = np.transpose(decode_segmap(outputs[2].detach().cpu().argmax(1).unsqueeze(1), nc=21)[i], (1,2,0))
                # plt.imshow(np.transpose(decoded_output[1].numpy(), (1,2,0)))
                for i in range(pred_seg.shape[0]):
                    plt.subplot(5,pred_seg.shape[0],1+(i))
                    plt.imshow(np.transpose(left[i].cpu().numpy(), (1,2,0)))
                    plt.subplot(5,pred_seg.shape[0],3+(i))
                    plt.imshow(np.transpose(decode_segmap(seg.detach().cpu().argmax(1).unsqueeze(1), nc=21)[i], (1,2,0)))
                    plt.subplot(5,pred_seg.shape[0],5+(i))
                    plt.imshow(np.transpose(decode_segmap(outputs[2].detach().cpu().argmax(1).unsqueeze(1), nc=21)[i], (1,2,0)))
                    plt.subplot(5,pred_seg.shape[0],7+(i))
                    plt.imshow(gt_seg[i].cpu() == 16)
                    plt.subplot(5,pred_seg.shape[0],9+(i))
                    plt.imshow((gt_seg[i].cpu() == 16) & (pred_seg[i].cpu() == 15))
                plt.show()
            
'''

'''
net.load_state_dict(torch.load(CFG.load_weights)) 
    image_pth='/home/hanz/Documents/phd/datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/munster/munster_000173_000019_leftImg8bit.png'
    import cv2
    left = io.imread(image_pth)
    y,x,c = left.shape
    scale = 0.5
    print(left.shape)
    left = cv2.resize(left, (int(x*scale), int(y*scale)), interpolation = cv2.INTER_AREA)
    left = left.transpose(2,0,1).astype(np.float32)/255
    left = left[None,:]
    print(left.shape)
    left = torch.from_numpy(left).cuda()
    net.eval()
    with torch.no_grad():
        outputs = net(left)
    print(outputs.shape)
    plt.imshow(outputs.detach().cpu().argmax(1).numpy()[0])
    # plt.imshow(np.transpose(left.detach().cpu().numpy(), (0,2,3,1))[0])
    plt.show()
'''

# print('starting apex')
#             torch.cuda.set_device(gpu)
#             net.cuda(gpu)            
#             net = apex.parallel.convert_syncbn_model(net)
#             net, optimizer = apex.amp.initialize(net, optimizer, opt_level='O1')
#             #net = apex.parallel.convert_syncbn_model(net)
#             net = DDP(net, delay_allreduce=True, )


#             map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
#             checkpoint = torch.load(CFG.load_weights, map_location=map_location if CFG.nodes else 'cpu')
#             state_dict = checkpoint['state_dict']
#             start_e = checkpoint['epoch']
#             if rank == 0:
#                 print('loading weights')
#                 print('training will start from epoch: ', start_e)
#             optimizer_dict = checkpoint['optimizer']
#             net.load_state_dict(state_dict)
#             optimizer.load_state_dict(optimizer_dict)
