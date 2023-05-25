import os
# from util import utilLoad
# import torch
import argparse
def configParser():
    parser = argparse.ArgumentParser(description='Config parser')
    parser.add_argument('-gpu_n', type=str, help='number of GPU')
    parser.add_argument('-corrType', type=str, help='correlation type, None, 1dcorr, 2dcorr')
    parser.add_argument('-datasetName', type=str, help='name of dataset [city, sceneflow, garden]')
    parser.add_argument('-load_weights',  type=str, default='', help='load saved weights')
    parser.add_argument('-optimType',  type=str, default='adam', help='choose optimizer')
    parser.add_argument('-backbone',  type=str, default='densenet', help='choose backbone')
    parser.add_argument('-net', type=str, help='name of net [disp, segdisp]')
    parser.add_argument('-n_data', type=int, help='number of data used')
    parser.add_argument('-output_type', type=str, help='type of output [all, disp_only, seg_only]')
    parser.add_argument('-train', type=int, help='train [0=false, 1=True]')
    parser.add_argument('-output_activation', default='sigmoid', type=str, help='output activation sigmoid, relu, linear')
    parser.add_argument('-b', default=8, type=int, dest='batch', help='Batch size')
    parser.add_argument('-e', default=10, type=int, dest='epoch', help='Batch size')
    parser.add_argument('-page', default=600, type=int)
    parser.add_argument('-crop', default=256, nargs='+', type=int)
    parser.add_argument('-w_savePath', type=str, default='', help='path to folder where to save the weights')
    parser.add_argument('-trainCompressed', type=str, default='', help='path to train compressed file')
    parser.add_argument('-testCompressed', type=str, default='', help='path to train compressed file')
    parser.add_argument('-colorL', type=str, default='', help='txt file of color Left GT')
    parser.add_argument('-colorR', type=str, default='', help='txt file of color Right GT')
    parser.add_argument('-seg', type=str, default='', help='txt file of segmentation GT')
    parser.add_argument('-inst', type=str, default='', help='txt file of instance GT')
    parser.add_argument('-disp', type=str, default='', help='txt file of disparity GT')
    parser.add_argument('-colorL_test', type=str, default='', help='txt file of color Left test set')
    parser.add_argument('-colorR_test', type=str, default='', help='txt file of color Right test set')
    parser.add_argument('-seg_test', type=str, default='', help='txt file of segmentation test set')
    parser.add_argument('-inst_test', type=str, default='', help='txt file of instance test set')
    parser.add_argument('-disp_test', type=str, default='', help='txt file of disparity test set')
    parser.add_argument('-save_img', type=int, help='save results in test time')
    parser.add_argument('-copy_remote', type=int, default=0, help='save results in test time')
    parser.add_argument('-segWeight', type=int, default=0, help='whether or not weight the segmentation output for loss')
    parser.add_argument('-show_results', type=int, default=1, help='plot a sample output of the network at test time')
    parser.add_argument('-loss',  nargs='+', help='type of extra losses')
    parser.add_argument('-edges', type=int,default=0, help='include edges to the input')
    parser.add_argument('-aspp', type=int,default=0, help='use aspp layer or not')
    parser.add_argument('-only_test', type=int,default=0, help='include edges to the input')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('-abilation', nargs='+', default='', help='type of abilation, None, no_dec1, no_dec2, no_dec3')
    parser.add_argument('-freeze_bn', type=int, default=0, help='freeze bn')
    parser.add_argument('-f16', type=int, default=0, help='use mixed half precision apex')
    parser.add_argument('-torch_amp', type=int, default=0, help='use mixed half precision torch.amp')
    parser.add_argument('-acmt_grad', type=int, default=1, help='accumulate gradients')
    parser.add_argument('-use_att', type=int, default=1, help='use attention layer')
    parser.add_argument('-hanet', type=int, default=0, help='use hanet layer (only valid in sdnet_mini_ext)')
    parser.add_argument('-multaskloss', type=int, default=0,
                        help='use multaskloss loss [0: no, 1: use as loss, 2: use their net decoder] (only valid in '
                             'sdnet_mini_ext)')
    parser.add_argument('-convDeconvOut', type=int, default=0,
                        help='1: use conv as output, 2: use conv and deconv as output (only valid in sdnet_mini_ext)')
    parser.add_argument('-dropout', type=float, default=0.0, help='dropout rate')
    return parser.parse_args()

#Network config 
# train = 1
# b = 4
# input_size = [256, 256]
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# old_config = False
# outputType = 'two_out'#['disp', 'seg', 'both', 'fmetric', 'two_out]
# datasetName = 'kitti' #['garden', 'kitti']
# output_activation = 'linear' #['sigmoid', 'tanh', 'linear']
# PATH = datasetName+'_'+outputType
# show_results = True
# load_weights = False 
# warpedOutput = True
# workers = 8
# # [1,     1 /   375] loss: 24.496, error: 8.534, PixelAcc: 0.017
# # [1,     6 /   375] loss: 15.249, error: 4.315, PixelAcc: 0.361
# # [1,    11 /   375] loss: 18.918, error: 6.959, PixelAcc: 0.216
# if datasetName == 'garden':
#     datasetFolder = "/home/hanz/Documents/phd/datasets/garden2018"
#     n_labels = 8 + 1
#     max_disp = 100.0#25.0
# else:
#     datasetFolder = "/home/hanz/Documents/phd/datasets/KITTI_2015"
#     n_labels = 19#8
#     max_disp = 100.0

# if disp_normalize == 'linear':
#     max_disp = 1
    
# colorL = utilLoad.GetDirFromText(os.path.join(datasetFolder, 'left_all.txt'))
# colorR = utilLoad.GetDirFromText(os.path.join(datasetFolder, 'right_all.txt'))
# disp = utilLoad.GetDirFromText(os.path.join(datasetFolder, 'disp_all.txt'))
# seg = utilLoad.GetDirFromText(os.path.join(datasetFolder, 'seg_all.txt'))

# colorL_test = utilLoad.GetDirFromText(os.path.join(datasetFolder, 'left_all_test.txt'))
# colorR_test = utilLoad.GetDirFromText(os.path.join(datasetFolder, 'right_all_test.txt'))
# disp_test = utilLoad.GetDirFromText(os.path.join(datasetFolder, 'disp_all_test.txt'))
# seg_test = utilLoad.GetDirFromText(os.path.join(datasetFolder, 'seg_all_test.txt'))
