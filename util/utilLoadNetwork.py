from models import torch_dsnet, dsnet_t2, dsnet_t2_warp, dsnet_t2_ext_small
from models_deeplab.net import EncoderDecoderNet, SPPNet
from models_deeplab_mod.net import SPPNet as SPPNetDS
from models_psmnet import stackhourglass


def getNetwork(CFG):
    FUNCTION_MAP = {'sdnet': dsnet_t2.dsnet,
                    'sdnetv2': dsnet_t2.dsnetv2,
                    'sdnet_mini': dsnet_t2.minidsnet,
                    'sdnet_mini_ext': dsnet_t2.minidsnetExt,
                    'sdnet_mini_ext_dlab': dsnet_t2.minidsnetExt_deeplab,
                    'sdnet_mini_ext_v2': dsnet_t2.minidsnetExt2,
                    'sdnet_mini_ext_piramid': dsnet_t2.minidsnetExtPiramid,
                    'sdnet_mini_ext_piramid_res': dsnet_t2.minidsnetExtPiramidRes,
                    'sdnet_mini_ext_small': dsnet_t2_ext_small.Ext_smallv0,
                    'sdnet_mini_ext_small_edge': dsnet_t2_ext_small.Ext_small,
                    'sdnet_mini_ext_small_edgev2': dsnet_t2_ext_small.Ext_smallv2,
                    'sdnet_seg': dsnet_t2.seg_dsnet,
                    'dsnet_warp': dsnet_t2_warp.minidsnetDivide,
                    'dsnet_warp_soft': dsnet_t2_warp.minidsnetDivideSoftmax,
                    'dsnet_warp_disp': dsnet_t2_warp.minidsnetDivideDisp,
                    'dsnet_warp_disp_consist': dsnet_t2_warp.minidsnetDivideDisp2,
                    'deeplab': SPPNet,  # SPPNet
                    'deeplab_mod': SPPNetDS,
                    'pspnet': stackhourglass}
    net_arch = FUNCTION_MAP[CFG.net]
    if CFG.net == 'sdnet_mini':
        CFG.outputType = 'smallOutPair'
    if 'sdnet_mini_ext' in CFG.net:
        CFG.outputType = 'smallOutSeg'
    if CFG.net == 'sdnet_seg':
        CFG.outputType = 'smallOutWarp'
    if CFG.net == 'dsnet_warp' or CFG.net == 'dsnet_warp_soft':
        CFG.outputType = 'ThreeOutPuts'
    if CFG.net == 'dsnet_warp_disp':
        CFG.outputType = 'ThreeOutPutsDisp'
    if CFG.net == 'dsnet_warp_disp_consist':
        CFG.outputType = 'ThreeOutPutsDispConsist'
    if 'edge' in CFG.net:
        CFG.outputType = 'edgeOut'
    if CFG.hanet:
        CFG.outputType = 'hanet'
    if CFG.multaskloss:
        CFG.outputType = 'multitask'

    if 'deeplab' in CFG.net:
        CFG.outputType = CFG.net
        net = net_arch()
        net.update_bn_eps()
        # net = net.cuda()
    elif CFG.net == 'pspnet':
        CFG.outputType = 'pspnet'
        net = net_arch(192)
    else:
        net = net_arch(CFG, labels=CFG.n_labels, pretrained=True, patch_type=CFG.corrType, include_edges=CFG.edges,
                       backbone=CFG.backbone)

    return net
