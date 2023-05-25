import torch.nn as nn
import torch, cv2
import torch.nn.functional as F
import numpy as np
from sklearn import metrics


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def get_gradient(img, diff):
    # m, n = -1, -1
    _, _, m, n = img.shape
    if diff == 'down':
        i, j = 1, 0
        # img = F.pad(img, (0, 0, 0, 1))
    if diff == 'right':
        i, j = 0, 1
        # img = F.pad(img, (0, 1, 0, 0))
    # loss = torch.abs(img[:, :, 0:m-1-i, 0:n-1-j]-img[:, :, 0+i:m-1, 0+j:n-1])
    loss = torch.abs(img[:, :, 0:m - i, 0:n - j] - img[:, :, 0 + i:m, 0 + j:n])
    if diff == 'down':
        loss = F.pad(loss, (0, 0, 0, 1))
    if diff == 'right':
        loss = F.pad(loss, (0, 1, 0, 0))
    return loss


def smoothing_gradients(left, disp, seg):
    max_disp = 128
    gaus_size = 7
    lbl_n = seg.shape[1]
    gauss_pad = int((gaus_size - 1) / 2)
    gauss_filter = torch.from_numpy(matlab_style_gauss2D(shape=(gaus_size, gaus_size), sigma=2))
    gauss_filter = gauss_filter.view(1, 1, gaus_size, gaus_size).repeat(1, 1, 1, 1).type(torch.float32).cuda()
    Clinear = 0.2126 * left[:, 0] + 0.7152 * left[:, 1] + 0.0722 * left[:, 2]
    Clinear = Clinear.unsqueeze_(1)
    smooth_Clinear = F.conv2d(F.pad(Clinear, (gauss_pad, gauss_pad, gauss_pad, gauss_pad)), gauss_filter, groups=1)
    Clinear = smooth_Clinear
    conv_area = 3
    total_area = conv_area * conv_area
    pad = int((conv_area - 1) / 2)
    weights = torch.ones((lbl_n, 1, conv_area, conv_area)).cuda()
    conv_seg = F.conv2d(F.pad(seg, (pad, pad, pad, pad)), weights, groups=lbl_n)
    mask = conv_seg == total_area
    D_I_down = get_gradient(Clinear, 'down') * seg
    D_I_right = get_gradient(Clinear, 'right') * seg
    D_d_down = get_gradient(disp / max_disp, 'down') * seg * mask
    D_d_right = get_gradient(disp / max_disp, 'right') * seg * mask
    reg_down = D_d_down * torch.exp(1.0 - D_I_down)
    reg_right = D_d_right * torch.exp(1.0 - D_I_right)
    reg = torch.mean(torch.sum(reg_down, dim=1) + torch.sum(reg_right, dim=1)) * 0.7  # * 0.5

    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    # fig,ax = plt.subplots(3,2)
    # for i in range(left.shape[0]):
    #     x, y = 362, 186
    #     # plt.subplot(2,1,1), plt.imshow(Clinear.cpu()[i].numpy().squeeze())
    #     # plt.subplot(2,1,2), plt.imshow(smooth_Clinear.cpu()[i].numpy().squeeze())
    #     # plt.show()
    #     rc1 = patches.Rectangle((x-2,y-2),5,5,linewidth=1,edgecolor='r',facecolor='none')
    #     rc2 = patches.Rectangle((x-2,y-2),5,5,linewidth=1,edgecolor='r',facecolor='none')
    #     rc3 = patches.Rectangle((x-2,y-2),5,5,linewidth=1,edgecolor='r',facecolor='none')
    #     rc4 = patches.Rectangle((x-2,y-2),5,5,linewidth=1,edgecolor='r',facecolor='none')
    #     rc5 = patches.Rectangle((x-2,y-2),5,5,linewidth=1,edgecolor='r',facecolor='none')
    #     rc6 = patches.Rectangle((x-2,y-2),5,5,linewidth=1,edgecolor='r',facecolor='none')
    #     rc = patches.Rectangle((x-2,y-2),5,5,linewidth=1,edgecolor='r',facecolor='none')
    #     rect = [rc1, rc2, rc3, rc4, rc5, rc6]
    #     print('mask', mask[i].sum(dim=0)[y, x])
    #     print('grad down disp', D_d_down[i].sum(dim=0)[y, x])
    #     print('grad down img', D_I_down[i].sum(dim=0)[y, x])
    #     print('reg down', reg_down[i].sum(dim=0)[y, x])
    #     print('Clinear', Clinear[i,0,y, x])

    #     ax[0,0].imshow(Clinear.cpu()[i].numpy().squeeze())
    #     ax[0,0].add_patch(rect[0])
    #     ax[0,1].imshow(D_I_down.cpu()[i].sum(dim=0).numpy().squeeze())
    #     ax[0,1].add_patch(rect[1])
    #     ax[1,0].imshow((disp/max_disp).cpu()[i].numpy().squeeze())
    #     ax[1,0].add_patch(rect[2])
    #     ax[1,1].imshow(D_d_down.cpu()[i].sum(dim=0).numpy().squeeze())
    #     ax[1,1].add_patch(rect[3])
    #     ax[2,0].imshow(reg_down.cpu()[i].sum(dim=0).numpy().squeeze())
    #     ax[2,0].add_patch(rect[4])
    #     ax[2,1].imshow(mask.cpu()[i].sum(dim=0).numpy().squeeze())        
    #     ax[2,1].add_patch(rect[5])
    #     plt.show()        
    return reg


def fmetricProcess(outputs, right, disp):
    # disp_pred = torch.sigmoid(outputs[:,0,:,:].unsqueeze(1))
    disp_pred = outputs[:, 0, :, :].unsqueeze(1)
    right_pred = outputs[:, 1:, :, :]
    loss = nn.MSELoss()(right_pred, right) + loss_gradient(disp, disp_pred)
    error = unnormalizedError(disp_pred, disp)
    return loss, error


def huberloss_with_th(y_, y, thr):
    abs_val = torch.abs(y_ - y)
    mask = (abs_val < 1) * 1.0
    cond1 = 0.5 * ((y_ - y) ** 2) * mask
    cond2 = torch.abs(y_ - y) * (1 - mask)
    return torch.mean((cond1 + cond2) * (10.0 * (y > thr) + 1.0 * (y <= thr)))


def dispProcess(criterion, outputs, disp):
    disp_pred = outputs[:, 0, :, :].unsqueeze(1)
    zeros = 1.0  # (disp != 0) * 1.0
    loss = criterion(disp_pred, disp)  # + loss_gradient(disp, disp_pred)
    # thr = 25.0
    # loss = huberloss_with_th(disp_pred, disp, thr)
    # loss = torch.mean(torch.abs(torch.exp(disp_pred) - torch.exp(disp)) * (10.0*(disp > thr) + 1.0*(disp <= thr)))
    error = unnormalizedError(disp_pred, disp)
    return loss, error


def LossWarp(output, left, right, trainTime=False):
    right_ = warpDisp(output, left, trainTime)
    return nn.MSELoss()(right_, right)


def warpDisp(disp_image, left_image, trainTime=False):
    # translated_disp = torch.zeros(disp_image.shape).to(device)
    translated_rgb = torch.zeros(left_image.shape, requires_grad=True).to(device)

    i_range = np.array(range(disp_image.shape[-2]))
    j_range = np.array(range(disp_image.shape[-1]))
    index = np.array(np.meshgrid(i_range, j_range)).T.reshape(-1, 2).T
    unnormDisp = max_disp * disp_image[:, :, index[0], index[1]]

    d = (index[1] - unnormDisp.detach().cpu().numpy())
    new_left = left_image[:, :, index[0], index[1]].unsqueeze(dim=2).unsqueeze(dim=2)
    translated_rgb[:, :, index[0], d] = torch.where(torch.from_numpy(d).float().to(device) > 0, new_left,
                                                    torch.tensor(0.0).to(device))  # new_left
    return translated_rgb


def cross_entropy_one_hot(output, target):
    _, labels = target.max(dim=1)
    # w = torch.tensor([1.0, 9.0, 10.0]).to(device)
    # return nn.CrossEntropyLoss(weight=w)(output, labels)
    return nn.NLLLoss()(output, labels)


def extractDispSegPrediction(pred):
    disp_pred = pred[:, 0, :, :].unsqueeze(1)
    if warpedOutput:
        seg_pred = pred[:, 1:-3, :, :]
        translated_rgb = pred[:, -3:, :, :]
        return disp_pred, seg_pred, translated_rgb
    else:
        seg_pred = pred[:, 1:, :, :]
        return disp_pred, seg_pred, seg_pred


def LossSegDisp(disp_pred, seg_pred, disp_target, seg_target, right_target, right_pred):
    # disp_loss = nn.L1Loss()(disp_pred, disp_target)
    # disp_loss = nn.L1Loss()(torch.exp(disp_pred), torch.exp(disp_target))
    print('here')
    thr = 45
    disp_loss = torch.mean(torch.abs(disp_pred - disp_target) * (10 * (disp_target > thr) + 1 * (disp_target <= thr)))
    seg_loss = cross_entropy_one_hot(seg_pred, seg_target)
    if warpedOutput:
        fmetric_loss = nn.MSELoss()(right_pred, right_target)
    return seg_loss + 2 * disp_loss  # disp_loss + seg_loss


def LossYoloLike(disp_pred, seg_pred, disp_target, seg_target, right_target, right_pred):
    _, labelsTarget = seg_target.max(dim=1)
    _, labelspred = seg_pred.max(dim=1)
    isObject = (labelsTarget == labelspred).unsqueeze(1)

    # disp_loss = nn.L1Loss()(isObject*torch.exp(disp_pred), isObject*torch.exp(disp_target))
    thr = 25
    disp_loss = torch.mean(torch.abs(torch.exp(disp_pred) - torch.exp(disp_target)) * (
                10 * (disp_target > thr / max_disp) + 1 * (disp_target <= thr / max_disp)))
    seg_loss = cross_entropy_one_hot(seg_pred, seg_target)
    if warpedOutput:
        fmetric_loss = nn.MSELoss()(isObject * right_pred, isObject * right_target)
    return seg_loss + disp_loss  # + loss_gradient(disp_pred, disp_target)#disp_loss + seg_loss


def SegAccuracy(outputs, gt, labels, showperStep=False):
    import numpy as np
    import time
    gt_seg = gt.argmax(dim=1)
    pred_seg = outputs.argmax(dim=1)
    mask = (gt_seg != labels)  # * 1.0 #because the class 20 is 19 if count from
    acc = (pred_seg == gt_seg)
    acc = acc[mask]
    acc = acc.double().mean().item()
    # acc_per_class = torch.sum(outputs == gt, dim=(2,3)) / (outputs.shape[2]*outputs.shape[3])
    acc_per_class = 0
    lbl_count = labels * gt_seg[mask] + pred_seg[mask]
    count = np.bincount(lbl_count.cpu().numpy(), minlength=labels ** 2)
    conf_matrix = count.reshape(labels, labels)
    if showperStep:
        max_out = outputs.max(dim=1, keepdim=True)
        y_ = (outputs == max_out[0]) * 1.0
        acc_per_class = torch.sum(y_ * gt, dim=(2, 3)).float()
        acc_per_class = acc_per_class / (torch.sum(gt, dim=(2, 3)) + 1e-8)
    return acc, conf_matrix, acc_per_class
    # return (pred_seg == gt_seg).sum().item() #


def SegAccuracyNp(outputs, gt, labels, showperStep=False):
    import numpy as np
    import time
    gt_seg = gt.argmax(1)
    pred_seg = outputs.argmax(1)
    mask = (gt_seg != labels)  # * 1.0 #because the class 20 is 19 if count from
    acc = (pred_seg == gt_seg)
    acc = acc[mask]
    acc = acc.mean()
    # acc_per_class = torch.sum(outputs == gt, dim=(2,3)) / (outputs.shape[2]*outputs.shape[3])
    acc_per_class = 0
    lbl_count = labels * gt_seg[mask] + pred_seg[mask]
    count = np.bincount(lbl_count, minlength=labels ** 2)
    conf_matrix = count.reshape(labels, labels)

    return acc, conf_matrix, acc_per_class
    # return (pred_seg == gt_seg).sum().item() #




def confMat(y_pred, y_test):
    TP = np.sum(np.logical_and(y_pred == 1, y_test == 1))
    TN = np.sum(np.logical_and(y_pred == 0, y_test == 0))
    FP = np.sum(np.logical_and(y_pred == 1, y_test == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_test == 1))

    return TP, FP, TN, FN


def GetSegMetricsNp(outputs, gt, labels, showperStep=False, num_image=0):


    # Background
    backMask = (gt[0][0] == 1.0)
    # Branch predicted or existent
    branchMask = np.logical_or((gt[0][1] == 1.0), (outputs[0][1] == 1.0) )


    # print(gt.shape)
    # cv2.imwrite("image3.jpg", gt[0][1] * 256)
        
    gt_img = gt[0][1]
    pred_img = outputs[0][1] # TODO check batch size 1st dimension 
    pred_img[pred_img > 0] = 1 # TODO check threshold
    pred_img[pred_img < 0] = 0
    cv2.imwrite("testResults/segPred_" + str(num_image) + ".jpg", np.array(pred_img * 256))
    cv2.imwrite("testResults/segGT_" + str(num_image)[:-5] + ".jpg", np.array(gt_img * 256))
    # print("minmax", np.amax(gt_img), np.amin(gt_img), np.median(gt_img), np.mean(gt_img))

    # gt_imgBack = gt[0][0]
    # pred_imgBack = outputs[0][0] 
    # # print("minmax", np.amax(pred_imgBack), np.amin(pred_imgBack), np.median(pred_imgBack), np.mean(pred_imgBack))
    # pred_imgBack[pred_imgBack > 0] = 1
    # pred_imgBack[pred_imgBack <= 0] = 0
    # cv2.imwrite("testResults/predBackground_" + str(num_image) + ".jpg", np.array(pred_imgBack * 256))
    # cv2.imwrite("testResults/gtBackground_" + str(num_image) + ".jpg", np.array(gt_imgBack * 256))


    # if ( gt.argmax(1).shape[0] != 1):
    #     print(gt.argmax(1).shape[0])
    #     print("ERROR: Segmentation dimension")
    

    ## sklearn
    # gt_img = gt.argmax(1)[0] 
    # pred_img = outputs.argmax(1)[0]
    # accuracy  = 1 - metrics.accuracy_score(gt_img, pred_img, normalize=True)
    precision = metrics.precision_score(gt_img, pred_img, average="micro")
    recall    = metrics.recall_score(gt_img, pred_img, average="micro")
    f1        = metrics.f1_score(gt_img, pred_img, average="micro")
    # Branch
    Bf1       = metrics.f1_score(gt_img[branchMask], pred_img[branchMask], average="micro")

    ## Manually
    # TP, FP, TN, FN = confMat(gt_img, pred_img)
    # print("TP, FP, TN, FN", TP, FP, TN, FN)
    # precision = TP / (TP + FP)
    # recall    = TP / (TP + FN)
    # f1        = 2 * (precision * recall) / (precision + recall)
    # # Branch
    # TP, FP, TN, FN = confMat(gt_img[branchMask], pred_img[branchMask])
    # precision = TP / (TP + FP)
    # recall    = TP / (TP + FN)
    # Bf1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1, Bf1 

# Sq-Rel (Squared Relative difference) not found in sklearn
def GetSqRel(y_test, y_pred):
    return np.mean(((y_test - y_pred)**2) / y_test)


def GetDispMetricsNp(outputs, gt, seg_full, num_image):

    # Background
    branchBack = (seg_full[0][0] == 1.0)
    # Branch
    branchMask = (seg_full[0][1] == 1.0)

    # if ( gt.shape[0] != 1 and gt.shape[1] != 1):
    #     print("ERROR: Dispersion dimension")

        
    gt_img = gt[0][0]
    pred_img = outputs[0][0]

    # print("minmax", np.amax(gt_img), np.amin(gt_img), np.median(gt_img), np.mean(gt_img))
    # print("minmax", np.amax(pred_img), np.amin(pred_img), np.median(pred_img), np.mean(pred_img))

    cv2.imwrite("testResults/dispGT_" + str(num_image) + ".jpg", np.array( ( gt_img - np.amin(gt_img) ) / ( np.amax(gt_img) - np.amin(gt_img) ) * 200) )
    cv2.imwrite("testResults/dispPred_" + str(num_image) + ".jpg", np.array( ( pred_img - np.amin(gt_img) ) / ( np.amax(gt_img) - np.amin(gt_img) ) * 200)  )
    # maskGt = seg_full >= 0
    
    # dispRMSE  = metrics.mean_squared_error(gt_img, pred_img)
    dispRMSE = (gt_img - pred_img) ** 2
    dispRMSE = np.sqrt(dispRMSE.mean())
    dispSqRel = GetSqRel(gt_img, pred_img)
    BdispRMSE = (gt_img[branchMask] - pred_img[branchMask]) ** 2
    BdispRMSE = np.sqrt(dispRMSE.mean())
    BdispSqRel = GetSqRel(gt_img[branchMask], pred_img[branchMask])

    return dispRMSE, dispSqRel, BdispRMSE, BdispSqRel
 


def unnormalizedError(y_, y, max_disp):
    MAEerror = nn.L1Loss()
    # max_disp = 25.0
    # max_disp = 70.0
    # y_index = (y != 0).nonzero().detach().cpu()
    # a,b,c,d = y_index[:,0], y_index[:,1], y_index[:,2], y_index[:,3]
    # e = (max_disp*torch.abs(y_[a,b,c,d] - y[a,b,c,d])).mean()
    th = (y > 0) * 1.0
    e = torch.abs(y_ * max_disp - y * max_disp) * th
    e = torch.sum((e > 3.0) * 1.0)
    valid_pixels = torch.sum(th)
    # e = torch.mean((e > 3.0)*1.0)
    # e = (torch.abs(y_  - y ) < 3.0/max_disp).double().mean()
    return e.item(), valid_pixels.item()


def unnormalizedErrorNP(y_, y, max_disp):
    th = (y > 0) * 1.0
    e = np.abs(y_ * max_disp - y * max_disp) * th
    e = np.sum((e > 3.0) * 1.0)
    valid_pixels = np.sum(th)
    # e = torch.mean((e > 3.0)*1.0)
    # e = (torch.abs(y_  - y ) < 3.0/max_disp).double().mean()
    return e, valid_pixels


def categoricalCrossEntropy(y, gt, weight=[]):
    'expects that y is a F.log_softmax(logits, 1)'
    if len(weight) != 0:
        return torch.mean(torch.sum(- gt * y * weight, 1))
    else:
        return torch.mean(torch.sum(- gt * y, 1))


def binaryCE(y, gt, weight=[]):
    criterion = nn.BCELoss()
    if len(weight) != 0:
        w = weight.squeeze()
    else:
        w = torch.ones(y.shape[1])
    loss = 0
    for i in range(y.shape[1]):
        loss = loss + w[i] * criterion(y[:, i, :, :], gt[:, i, :, :])
    return loss


def categoricalNlll(y, gt, weight=[]):
    'expects that y is a F.log_softmax(logits, 1) NOT WORKING'
    if len(weight) != 0:
        criterion = nn.NLLLoss(ignore_index=19, weight=(weight.squeeze()).float(),
                               size_average=False)  # weight.squeeze().float())
    else:
        criterion = nn.NLLLoss(ignore_index=19)
    loss = criterion(y, gt.argmax(1))
    # loss = loss * weight
    loss = loss.mean()

    return loss  # , weight=weight if weight else None)


def tversky_loss2(y, gt, weights):
    smooth = 1e-6
    gamma = 1
    beta = 1
    alpha = 0.7
    true_pos = torch.sum(gt * y, (2, 3))
    false_neg = torch.sum(gt * (1 - y), (2, 3))
    false_pos = torch.sum((1 - gt) * y, (2, 3))
    # false_pos = torch.sum((~gt) * y, (2,3))
    result = (true_pos) / (true_pos + beta * false_neg + (1 - alpha) * false_pos + smooth)
    loss = torch.pow(1 - result, 1 / gamma)
    loss = loss.mean(0)
    # print('gt', torch.sum(gt, (0,2,3)))
    # print('loss2', loss)
    if len(weights) != 0:
        # print(weights.shape, loss.shape)
        loss = loss * weights.squeeze()
    # print('loss2', loss)
    loss = loss.mean()
    return loss


def dice_loss(y, gt):
    y = torch.softmax(y, dim=1)
    numerator = 2 * torch.sum(y * gt, (2, 3))
    denominator = torch.sum(y, dim=(2, 3)) + torch.sum(gt, dim=(2, 3)) + 1
    thr = (gt.sum(dim=(2, 3)) > 1) * 1.0
    loss = thr - numerator / denominator
    loss = torch.mean(loss)
    return loss


def diceEntropy(y, gt):
    # dice
    sm_y = torch.softmax(y, dim=1)
    numerator = 2 * torch.sum(sm_y * gt, (2, 3), keepdim=True)
    denominator = torch.sum(sm_y, dim=(2, 3), keepdim=True) + torch.sum(gt, dim=(2, 3), keepdim=True) + 1
    thr = (gt.sum(dim=(2, 3), keepdim=True) > 1) * 1.0
    dice = 10 * (thr - numerator / denominator)
    # cross entropy
    ce = categoricalCrossEntropy(y, gt, weight=dice)
    return ce


def Pixel_Accuracy(confusion_matrix):
    return np.diag(confusion_matrix).sum() / confusion_matrix.sum()


def Pixel_Accuracy_Class(confusion_matrix):
    Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    Acc = np.nanmean(Acc)
    return Acc


def Mean_Intersection_over_Union(confusion_matrix):
    IoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    MIoU = np.nanmean(IoU)
    return MIoU, IoU


def area_hinge_loss(pred_seg, gt_seg):
    conv_area = 5
    lbl_n = gt_seg.shape[1]
    weights = torch.ones((lbl_n, 1, conv_area, conv_area)).cuda()
    total_area = conv_area * conv_area
    conv_seg = F.conv2d(gt_seg, weights, groups=lbl_n)
    conv_pred_seg = F.conv2d(F.softmax(pred_seg, dim=1), weights, groups=lbl_n)
    mask = (conv_seg == total_area) * 1
    conv_seg = conv_seg * mask / total_area
    conv_pred_seg = conv_pred_seg * mask / total_area
    dif = (conv_seg - conv_pred_seg) ** 2
    del weights
    return torch.mean(torch.sum(dif, 1))


def area_ce_loss(pred_seg, gt_seg, area_dim=5):
    conv_area = area_dim
    lbl_n = gt_seg.shape[1]
    weights = torch.ones((lbl_n, 1, conv_area, conv_area)).cuda()
    total_area = conv_area * conv_area
    conv_seg = F.conv2d(gt_seg, weights, groups=lbl_n)
    conv_pred_seg = F.conv2d(pred_seg, weights, groups=lbl_n)
    mask = (conv_seg == total_area) * 1  # .sum(dim=1).unsqueeze(dim=1)
    # print(conv_seg.shape, conv_pred_seg.shape, mask.shape)
    # import matplotlib.pyplot as plt
    # plt.subplot(4,1,1); plt.imshow(-conv_pred_seg[0,13].cpu().numpy())
    # plt.subplot(4,1,2); plt.imshow(-conv_pred_seg[0,1].cpu().numpy())
    # plt.subplot(4,1,3); plt.imshow(F.softmax(conv_pred_seg, dim=1).argmax(1)[0].cpu().numpy())
    conv_seg = conv_seg * mask / total_area
    conv_pred_seg = conv_pred_seg * mask / total_area
    # plt.subplot(4,1,4); plt.imshow(-conv_pred_seg[0,1].cpu().numpy())

    # #plt.subplot(3,1,3); plt.imshow(mask.max(dim=1)[0].squeeze().cpu().numpy())
    # plt.show()
    dif = categoricalCrossEntropy(conv_pred_seg, conv_seg)
    del weights
    return dif

    # classes = [0, 3, 4, 13]
    # plots = 5
    # class_n = len(classes)
    # for k in range(gt_seg.shape[0]):
    #     for c in range(class_n):
    #         import matplotlib.pyplot as plt
    #         plt.subplot(class_n,plots,1+(c*plots)), plt.imshow(gt_seg[k].argmax(0).cpu().numpy(), vmin=0, vmax=19)
    #         plt.subplot(class_n,plots,2+(c*plots)), plt.imshow(conv_seg[k,classes[c],:].cpu().numpy())#, cmap=plt.cm.binary, vmin=0, vmax=9)
    #         plt.subplot(class_n,plots,3+(c*plots)), plt.imshow(conv_pred_seg[k,classes[c]].cpu().numpy())#, cmap=plt.cm.binary, vmin=0, vmax=13)
    #         plt.subplot(class_n,plots,4+(c*plots)), plt.imshow(F.softmax(pred_seg, dim=1)[k,classes[c]].cpu().numpy(), cmap=plt.cm.binary, vmin=0, vmax=1)
    #         plt.subplot(class_n,plots,5+(c*plots)), plt.imshow(dif[k,classes[c]].cpu().numpy())
    #     plt.show()


class multiTask_loss(nn.Module):
    def __init__(self, three_out=1):
        # three_out
        # 1: 3 outputs (1 disp, 2 seg)
        # 2: 2 outputs (1 disp, 1 seg)
        super(multiTask_loss, self).__init__()
        self.three_out = three_out
        self.log_var_disp = nn.Parameter(torch.zeros(1,))
        self.log_var_seg1 = nn.Parameter(torch.zeros(1,))
        if self.three_out == 1:
            self.log_var_seg2 = nn.Parameter(torch.zeros(1,))

    def forward(self, disp, disp_gt, seg1, seg2, seg_gt):
        loss_disp = torch.exp(-self.log_var_disp) * F.l1_loss(disp, disp_gt, reduction='none') + self.log_var_disp
        loss_seg1 = torch.exp(-self.log_var_seg1) * F.cross_entropy(seg1, seg_gt, ignore_index=19, reduction='none') + self.log_var_seg1
        if self.three_out == 1:
            loss_seg2 = torch.exp(-self.log_var_seg2) * F.cross_entropy(seg2, seg_gt, ignore_index=19, reduction='none') + self.log_var_seg2
        else:
            loss_seg2 = torch.zeros((1,)).cuda()
        return loss_disp, loss_seg1, loss_seg2
