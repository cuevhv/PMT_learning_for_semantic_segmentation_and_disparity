from util.lovasz_losses import lovasz_softmax
from TverskyLoss.multitverskyloss import MultiTverskyLoss
from util.utilTorchLoss import *
from util import utilTorchGate
from torch import nn


def lossSeg_fn(lossType, seg_full, seg_pred, CFG, num_image):
    init_pred_np = seg_pred.detach().cpu().numpy()

    if CFG.datasetName == 'garden':
        seg = seg_full.cuda()
        ignore_pxl = None

    elif CFG.datasetName == 'roses':
        seg = seg_full.cuda()
        ignore_pxl = None

    else:
        seg = seg_full[:, :seg_full.shape[1] - 1].cuda()
        ignore_pxl = 19

    labels = seg.shape[1]
    dice_val = 0
    loss_seg = 0
    if 'binary_ce' in lossType:
        seg_pred = F.sigmoid(seg_pred)
    else:

        if 'dual_edge_reg' in lossType:
            dual = utilTorchGate.DualTaskLoss()
            dual_loss = dual(seg_pred, seg_full.cuda(), ignore_pixel=19)
            dice_val = dual_loss.item()
            loss_seg = loss_seg + dual_loss

        if 'ohm_loss' in lossType:
            from losses.ohm_loss import OhemCrossEntropy2d
            ohm_CE2d = OhemCrossEntropy2d(ignore_index=19)
            ohm_loss = ohm_CE2d(seg_pred, seg_full.argmax(1).cuda())
            loss_seg = loss_seg + 1.5 * ohm_loss

        seg_pred = F.log_softmax(seg_pred, dim=1)
        # loss_seg = edge_seg_loss(seg_pred, seg)
    if CFG.segWeight:
        if CFG.datasetName in ['cityscapes', 'kitti']:
            weights = np.array([5.90603017, 6.01238231, 5.90603017, 8.30641645, 7.77132999,
                                5.89333853, 7.25674024, 6.0150282, 5.94274377, 7.26202977,
                                6.12480687, 6.45807453, 8.21414722, 5.99393149, 9.55426071,
                                9.760075, 10.09886577, 9.2037169, 7.2726336], dtype=np.float32)
            # weights = np.array([ 1.90603017,  1.01238231,  1.90603017,  8.30641645,  7.77132999,
            #             2.89333853,  2.25674024,  2.0150282 ,  2.94274377,  3.26202977,
            #             2.12480687,  2.45807453,  3.21414722,  1.99393149,  9.55426071,
            #             9.760075  , 10.09886577,  9.2037169 ,  7.2726336 ], dtype=np.float32)
        elif CFG.datasetName == 'garden':
            weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        elif CFG.datasetName == 'roses':
            weights = np.array([1, 1], dtype=np.float32)
        weights = np.reshape(weights, (1, len(weights), 1, 1))
        weights = torch.from_numpy(weights).cuda()
    else:
        weights = []
    # zeros = (disp != 0) * 1.0

    if 'cross_entropy' in lossType:
        w1 = 1
        if len(lossType) > 2:
            w1 = 0.5
        loss_seg = loss_seg + w1 * categoricalCrossEntropy(seg_pred, seg, weights)

        if 'lovasz_loss' in lossType:
            lovasz_loss = lovasz_softmax(F.softmax(seg_pred, dim=1), seg_full.argmax(1).cuda(), ignore=ignore_pxl)
            loss_seg = loss_seg + w1 * lovasz_loss
            # smooth_val = loss_seg.item()

        if 'area_ce' in lossType:
            loss_ce = area_ce_loss(seg_pred, seg, area_dim=7)
            # loss_ce = area_ce_loss(seg2, seg, area_dim=7)
            loss_seg = loss_seg + loss_ce

    elif 'lovasz_loss' in lossType:
        loss_seg = loss_seg + lovasz_softmax(F.softmax(seg_pred, dim=1), seg_full.argmax(1).cuda(), ignore=ignore_pxl)
        dice_val = loss_seg.item()

    if 'tversky_loss2' in lossType:
        tversky_l = tversky_loss2(F.softmax(seg_pred, 1), seg, weights)
        loss_seg = loss_seg + 1.5 * tversky_l
        # print('tvers', tvers_l   )

    if 'tversky_loss' in lossType:
        tversky_func = MultiTverskyLoss(alpha=0.7, beta=0.3, gamma=3 / 4)
        tversky_l = tversky_func(F.softmax(seg_pred, 1), seg_full.argmax(1).cuda())
        # print(tversky_l)
        loss_seg = loss_seg + 1.5 * tversky_l

    if 'binary_ce' in lossType:
        loss_seg = loss_seg + binaryCE(seg_pred, seg, weights)

    if 'area_ce' in lossType:
        loss_seg = loss_seg + area_ce_loss(seg_pred, seg)

    if 'categoricalNlll' in lossType:
        loss_seg = loss_seg + categoricalNlll(seg_pred, seg_full.cuda(), weights)

    if 'area_hinge' in lossType:
        loss_seg = loss_seg + area_hinge_loss(seg_pred, seg)

    if 'dice_loss' in lossType:
        dice = dice_loss(seg_pred, seg)
        loss_seg = loss_seg + dice
        dice_val = dice.item()

    elif 'diceEntropy' in lossType:
        dice = diceEntropy(seg_pred, seg)
        loss_seg = loss_seg + dice
        dice_val = dice.item()
    else:
        dice_val = 0

    seg_pred_np = seg_pred.detach().cpu().numpy()
    seg_full_np = seg_full.detach().cpu().numpy()

    # pixelAcc, conf_matrix, _ = SegAccuracy(seg_pred, seg_full.cuda(), labels)
    pixelAcc, conf_matrix, _ = SegAccuracyNp(seg_pred_np, seg_full_np, labels)

    pixelPrec, pixelRecall, pixelF1, pixelBF1 = GetSegMetricsNp(init_pred_np, seg_full_np, labels, num_image=num_image)
    
    del weights
    return pixelAcc, conf_matrix, loss_seg, dice_val, pixelPrec, pixelRecall, pixelF1, pixelBF1


def lossDisp_fn(lossType, left, seg_full, disp, disp_pred, max_disp, CFG, num_image):
    smooth_val = 0
    loss_disp = 0
    if CFG.datasetName == 'garden':
        zeros = 1.0
    elif CFG.datasetName == 'roses':
        zeros = 1.0
    else:
        zeros = (disp > 0) * 1.0
    if CFG.outputType != 'multitask':
        loss_disp = loss_disp + nn.L1Loss()(disp_pred * zeros, disp * zeros)

    if 'smooth_grad' in lossType:
        reg_d_smooth = smoothing_gradients(left, disp_pred, seg_full.cuda())
        loss_disp = loss_disp + reg_d_smooth
        smooth_val = reg_d_smooth.item()

    zeros_np = zeros if CFG.datasetName == 'garden' or CFG.datasetName == 'roses' else zeros.detach().cpu().numpy()
    disp_pred_np = disp_pred.detach().cpu().numpy() * zeros_np
    disp_np = disp.detach().cpu().numpy() * zeros_np
    # err, val_pxl = unnormalizedError(disp_pred * zeros, disp  * zeros, max_disp)
    err, val_pxl = unnormalizedErrorNP(disp_pred_np, disp_np, max_disp)

    dispRMSE, dispSqRel, BdispRMSE, BdispSqRel = GetDispMetricsNp(disp_pred_np, disp_np, seg_full, num_image=num_image)
 

    return err, val_pxl, loss_disp, smooth_val, dispRMSE, dispSqRel, BdispRMSE, BdispSqRel


def lossImg_fn(outputType, left, warped_right):
    if outputType == 'photo_loss':
        photo_consistency_loss = nn.MSELoss()(warped_right, left)
    return photo_consistency_loss


def lossEdge_fn(edges, edge_pred):
    pos_index = (edges == 1)
    neg_index = (edges == 0)

    weight = torch.Tensor(edges.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num
    weight = torch.from_numpy(weight)
    weight = weight.cuda()

    loss = F.binary_cross_entropy_with_logits(edge_pred, edges.float().cuda(), weight, size_average=True)

    return loss


# class lossEdge_cls(nn.Module):
#     def __init__(self):
#         super(lossEdge_cls, self).__init__()
#
#     def forward(self, edges, edge_pred):
#         pos_index = (edges == 1)
#         neg_index = (edges == 0)
#
#         weight = torch.Tensor(edges.size()).fill_(0)
#         weight = weight.numpy()
#         pos_num = pos_index.sum()
#         neg_num = neg_index.sum()
#         sum_num = pos_num + neg_num
#         weight[pos_index] = neg_num * 1.0 / sum_num
#         weight[neg_index] = pos_num * 1.0 / sum_num
#         weight = torch.from_numpy(weight)
#         weight = weight.cuda()
#
#         loss = F.binary_cross_entropy_with_logits(edge_pred, edges.float().cuda(), weight, size_average=True)
#
#         return loss
