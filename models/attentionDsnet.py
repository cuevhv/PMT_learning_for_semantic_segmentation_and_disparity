import torch.nn as nn
import torch.nn.functional as F
from models import resnet
from models.torch_model import conv2dSame, ConvTranspose2dSame
from models.densenet import densenet121
from models.resnet_deeplab import ResNet50, ResNet101
from spatial_correlation_sampler import SpatialCorrelationSampler
from efficientnet_pytorch import EfficientNet
from models.torch_dsnet import apply_disparity
import torch
import math
from models.aspp import build_aspp

from models.mobilenetv3 import mobilenetv3_large

class piramidNet2(nn.Module):
    def __init__(self, pretrained=False, backbone='densenet'):
        super(piramidNet2, self).__init__()
        #self.resnet_features = resnet.resnet50(pretrained=pretrained)
        self.backbone = backbone
        if backbone == 'densenet':
            self.resnet_features = densenet121(pretrained=pretrained)
            in_plane = [64, 128, 256]
            # densenet
            # (None, 64, 128, 128)
            # (None, 128, 64, 64)
            # (None, 256, 32, 32)
            # (None, 512, 16, 16)
            # (None, 1024, 8, 8)
        if backbone == 'mobilenet':
            self.resnet_features = mobilenetv3_large()
            in_plane = [16, 24, 40]
            # mobilenet
            # 1 torch.Size([None, 16, 128, 128])
            # 3 torch.Size([None, 24, 64, 64])
            # 6 torch.Size([None, 40, 32, 32])
            # 12 torch.Size([None, 112, 16, 16])
            # 15 torch.Size([None, 160, 8, 8])
        if backbone == 'resnet50':
            self.resnet_features = ResNet50(BatchNorm=nn.BatchNorm2d, pretrained=pretrained, output_stride=16)
            in_plane = [64, 256, 512]

        if backbone == 'resnet101':
            self.resnet_features = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=pretrained, output_stride=16)
            in_plane = [64, 256, 512]
            # torch.Size([1, 64, 128, 128])
            # torch.Size([1, 256, 64, 64])
            # torch.Size([1, 512, 32, 32])
            # torch.Size([1, 1024, 16, 16])
            # torch.Size([1, 2048, 16, 16])
            
        if backbone == 'efficientnet-b5':
            self.resnet_features = EfficientNet.from_pretrained(backbone)
            in_plane = [24, 40, 64]
            # reduction_1 torch.Size([1, 24, 128, 256]) 
            # reduction_2 torch.Size([1, 40, 64, 128]) 
            # reduction_3 torch.Size([1, 64, 32, 64]) 
            # reduction_4 torch.Size([1, 176, 16, 32]) 
            # reduction_5 torch.Size([1, 2048, 8, 16]) 

        if backbone == 'efficientnet-b3':
            self.resnet_features = EfficientNet.from_pretrained(backbone)
            in_plane = [24, 32, 48]
            # reduction_1 torch.Size([2, 24, 128, 256])
            # reduction_2 torch.Size([2, 32, 64, 128])
            # reduction_3 torch.Size([2, 48, 32, 64])
            # reduction_4 torch.Size([2, 136, 16, 32])
            # reduction_5 torch.Size([2, 1536, 8, 16])

        if backbone == 'efficientnet-b2':
            self.resnet_features = EfficientNet.from_pretrained(backbone)
            in_plane = [16, 24, 48]
            # reduction_1 torch.Size([2, 16, 128, 256])
            # reduction_2 torch.Size([2, 24, 64, 128])
            # reduction_3 torch.Size([2, 48, 32, 64])
            # reduction_4 torch.Size([2, 120, 16, 32])
            # reduction_5 torch.Size([2, 1408, 8, 16])


        pool_val = [128, 64, 32, 16, 8]
        self.branch0_0 = nn.Sequential(nn.AvgPool2d(pool_val[0], pool_val[0]),
                                     convbn(in_plane[0], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch0_1 = nn.Sequential(nn.AvgPool2d(pool_val[1], pool_val[1]),
                                     convbn(in_plane[0], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch0_2 = nn.Sequential(nn.AvgPool2d(pool_val[2], pool_val[2]),
                                     convbn(in_plane[0], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch0_3 = nn.Sequential(nn.AvgPool2d(pool_val[3], pool_val[3]),
                                     convbn(in_plane[0], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch0_4 = nn.Sequential(nn.AvgPool2d(pool_val[4], pool_val[4]),
                                     convbn(in_plane[0], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))

        self.branch1_0 = nn.Sequential(nn.AvgPool2d(pool_val[1], pool_val[1]),
                                     convbn(in_plane[1], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch1_1 = nn.Sequential(nn.AvgPool2d(pool_val[2], pool_val[2]),
                                     convbn(in_plane[1], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch1_2 = nn.Sequential(nn.AvgPool2d(pool_val[3], pool_val[3]),
                                     convbn(in_plane[1], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch1_3 = nn.Sequential(nn.AvgPool2d(pool_val[4], pool_val[4]),
                                     convbn(in_plane[1], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))

        self.branch2_0 = nn.Sequential(nn.AvgPool2d(pool_val[2], pool_val[2]),
                                     convbn(in_plane[2], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch2_1 = nn.Sequential(nn.AvgPool2d(pool_val[3], pool_val[3]),
                                     convbn(in_plane[2], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch2_2 = nn.Sequential(nn.AvgPool2d(pool_val[4], pool_val[4]),
                                     convbn(in_plane[2], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))



    def forward(self, x):
        if self.backbone in ['efficientnet-b5', 'efficientnet-b3', 'efficientnet-b2']:
            out = self.resnet_features.extract_endpoints(x) #[(64,128), (256,64), (512,32), (1024, 16), (2048, 8)]
            out_0 = out['reduction_1']
            out_1 = out['reduction_2']
            out_2 = out['reduction_3']
            out_3 = out['reduction_4']
            out_4 = out['reduction_5']
        else:
            out_0, out_1, out_2, out_3, out_4 = self.resnet_features(x) #[(64,128), (256,64), (512,32), (1024, 16), (2048, 8)]
        # densenet
        # (None, 64, 128, 128)
        # (None, 128, 64, 64)
        # (None, 256, 32, 32)
        # (None, 512, 16, 16)
        # (None, 1024, 8, 8)


        mode = 'bilinear'#'nearest'
        b0_0 = self.branch0_0(out_0)
        b0_0 = F.interpolate(b0_0, (out_0.size()[2], out_0.size()[3]), mode=mode)
        
        b0_1 = self.branch0_1(out_0)
        b0_1 = F.interpolate(b0_1, (out_0.size()[2], out_0.size()[3]), mode=mode)
        
        b0_2 = self.branch0_2(out_0)
        b0_2 = F.interpolate(b0_2, (out_0.size()[2], out_0.size()[3]), mode=mode)
        
        b0_3 = self.branch0_3(out_0)
        b0_3 = F.interpolate(b0_3, (out_0.size()[2], out_0.size()[3]), mode=mode)
        
        b0_4 = self.branch0_4(out_0)
        b0_4 = F.interpolate(b0_4, (out_0.size()[2], out_0.size()[3]), mode=mode)      
        b0 = torch.cat((out_0, b0_0, b0_1, b0_2, b0_3, b0_4), axis=1)

        # print('out1: ',  out_0.shape, out_1.shape, out_2.shape, out_3.shape, out_4.shape)
        b1_0 = self.branch1_0(out_1)
        b1_0 = F.interpolate(b1_0, (out_1.size()[2], out_1.size()[3]), mode=mode)

        b1_1 = self.branch1_1(out_1)
        b1_1 = F.interpolate(b1_1, (out_1.size()[2], out_1.size()[3]), mode=mode)
        
        b1_2 = self.branch1_2(out_1)
        b1_2 = F.interpolate(b1_2, (out_1.size()[2], out_1.size()[3]), mode=mode)
        
        b1_3 = self.branch1_3(out_1)
        b1_3 = F.interpolate(b1_3, (out_1.size()[2], out_1.size()[3]), mode=mode)
        b1 = torch.cat((out_1, b1_0, b1_1, b1_2, b1_3), axis=1)
        
        b2_0 = self.branch2_0(out_2)
        b2_0 = F.interpolate(b2_0, (out_2.size()[2], out_2.size()[3]), mode=mode)
        
        b2_1 = self.branch2_1(out_2)
        b2_1 = F.interpolate(b2_1, (out_2.size()[2], out_2.size()[3]), mode=mode)
        
        b2_2 = self.branch2_2(out_2)
        b2_2 = F.interpolate(b2_2, (out_2.size()[2], out_2.size()[3]), mode=mode)
        b2 = torch.cat((out_2, b2_0, b2_1, b2_2), axis=1)
        
        return out_0, out_1, out_2, out_3, out_4, b2, b1, b0


class attentionDsnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21, labels=8, pretrained=False, patch_type='', aspp_mod=False, include_edges=False, backbone='mobilenet', abilation=[]):
        super(attentionDsnet, self).__init__()
        self.aspp_mod = aspp_mod
        max_disp = 8
        feature_channel = 1
        self.backbone = backbone
        self.abilation = abilation
        if backbone == 'densenet':
            segnet_input = 1024*2
            if self.aspp_mod == 1:
                self.aspp = build_aspp('densenet_a1', 32)
                inplane_seg2 = 256
            elif self.aspp_mod == 2:
                self.aspp = build_aspp('densenet_a3', 32)
                inplane_seg2 = 273#256*2
                feature_channel = 64
            else:
                inplane_seg2 = 512

        if backbone == 'mobilenet':
            segnet_input = 160 * 2
            if self.aspp_mod == 1:
                self.aspp = build_aspp('mobilenet_a1', 32)
                inplane_seg2 = 256
            elif self.aspp_mod == 2:
                self.aspp = build_aspp('mobilenet_a3', 32)
                inplane_seg2 = 273#256*2
                feature_channel = 16
            else:
                inplane_seg2 = 304

        if backbone == 'resnet50' or backbone == 'resnet101':
            segnet_input = 2048 * 2
            if self.aspp_mod == 0:
                segnet_input = 512
                self.aspp_4 = build_aspp('resnet_a4', 16)
                inplane_seg2 = 256

            if self.aspp_mod == 1:
                self.aspp = build_aspp('resnet50_a1', 16)
                inplane_seg2 = 256
            elif self.aspp_mod == 2:
                self.aspp = build_aspp('resnet50_a3', 16)
                inplane_seg2 = 273
                feature_channel = 64
            else:
                inplane_seg2 = 768
        
        if backbone == 'efficientnet-b5':
            segnet_input = 2048 * 2
            inplane_seg2 = 512
            # if self.aspp_mod == 1:
                
            # elif self.aspp_mod == 2:
            # else:

        if backbone == 'efficientnet-b3':
            segnet_input = 1536 * 2
            inplane_seg2 = 320
            # if self.aspp_mod == 1:
                
            # elif self.aspp_mod == 2:
            # else:

        if backbone == 'efficientnet-b2':
            segnet_input = 1408 * 2
            inplane_seg2 = 304
            # if self.aspp_mod == 1:
                
            # elif self.aspp_mod == 2:
            # else:

        self.patch_type = patch_type
        self.include_edges = include_edges
        
        
        self.resnet_features = piramidNet2(pretrained, backbone)
        
        if self.include_edges:
            aux_img_channel = 4
        else:
            aux_img_channel = 3

        

        self.conv2d_ba0 = nn.Sequential(convbn(aux_img_channel, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True))
        self.conv2d_ba1 = nn.Sequential(convbn(aux_img_channel, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True)) 
        self.conv2d_ba2 = nn.Sequential(convbn(aux_img_channel, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True)) 
        self.conv2d_ba3 = nn.Sequential(convbn(aux_img_channel, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True)) 
        if self.patch_type == '1dcorr':
            patch_corr = (1, max_disp*2 + 1)
            out_plane_corr = 17
        else:
            patch_corr = (max_disp*2 + 1, max_disp*2 + 1)
            out_plane_corr = 289
        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1,
                                                            patch_size=patch_corr,
                                                            stride=1,
                                                            padding=0,#max_disp*2 + 1,
                                                            dilation_patch=1)
        self.s2_corr_sampler = SpatialCorrelationSampler(kernel_size=1,
                                                            patch_size=patch_corr,
                                                            stride=1,
                                                            padding=0,#max_disp*2 + 1,
                                                            dilation_patch=1)
        self.corrConv2d = nn.Sequential(conv2dSame(out_plane_corr, 128, 1, padding='same'), nn.ReLU(inplace=True))
        # self.corrConv2d = nn.Sequential(conv2dSame(512, 128, 1, padding='same'), nn.ReLU(inplace=True))
        if 'no_dec1' in self.abilation:
            self.Conv2DownUp3 = Conv2DownUp(352, 128, 3)
        else:
            self.Conv2DownUp3 = Conv2DownUp(32, 128, 3)
        self.Conv2DownUp4 = Conv2DownUp(128*2, 64, 3)
        self.segNet = segNet(segnet_input, 1, labels)
        self.conv1d_2 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = Conv2DownUp(64, 64, 5, lastLayer=False)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.conv1d_4 = nn.Sequential(conv2dSame(inplane_seg2,128,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp6 = Conv2DownUp(128, 64, 3)
        self.Conv2DownUp7 = Conv2DownUp(128, 64, 3)
        self.Conv2DownUp8 = Conv2DownUp(32, 64, 3)
        self.Conv2DownUp9 = Conv2DownUp(128, 64, 3)
        self.conv1d_at_d = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid())
        self.conv1d_at_s = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid())
        if 'no_dec3' in self.abilation:
            self.Conv2DownUp10 = Conv2DownUp(64, 64, 3)
        else:
            self.Conv2DownUp10 = Conv2DownUp(128, 64, 3)
        self.conv1d_5 = nn.Sequential(conv2dSame(64+feature_channel,32,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp11 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False), ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
        # self.Conv2DownUp11 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False))
        # self.convOutput =  ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False)

        # self.conv1d_4 = nn.Sequential(conv2dSame(192,64,1, padding='same'), nn.ReLU(inplace=True))
        # # deconvbn(in_channel, out_channel, kernel_size, stride, pad, dilation, batchnorm=True)
        # self.conv2DT_BA1 = nn.Sequential(deconvbn(64, 32, 3, 2, 'same', 1), nn.ReLU(inplace=True))
        # self.conv1d_5 = nn.Sequential(conv2dSame(96,32,1, padding='same'), nn.ReLU(inplace=True))
        # self.conv2DT_BA2 = nn.Sequential(deconvbn(32, 32, 3, 2, 'same', 1), nn.ReLU(inplace=True))
        # self.conv1d_6 = nn.Sequential(conv2dSame(33,32,1, padding='same'), nn.ReLU(inplace=True))
        # self.Conv2DownUp7 =Conv2DownUp(32, 32, 5, lastLayer=False)
        # self.branchConv = ConvTranspose2dSame(32, labels, 5, padding='same', init_he=False)
        # self.conv1d_9 = nn.Sequential(conv2dSame(448, 128, 1, padding='same'), nn.ReLU(inplace=True))
        # #self.conv1d_9 = nn.Sequential(conv2dSame(128, 128, 1, padding='same'), nn.ReLU(inplace=True))
        # self.conv1d_7 = nn.Sequential(conv2dSame(128*2,128,1, padding='same'), nn.ReLU(inplace=True))
        # self.Conv2DownUp8 = Conv2DownUp(32, 64, 3)
        # self.Conv2DownUp9 = Conv2DownUp(256, 64, 3)
        # self.conv1d_8 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        # self.Conv2DownUp10 = nn.Sequential(Conv2DownUp(64, 64, 5, lastLayer=False), ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False))
    def forward(self, input_a, input_b):
        left = input_a
        right = input_b
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_2, a_pyramidB_1, a_pyramidB_0 = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_2, b_pyramidB_1, b_pyramidB_0 = self.resnet_features(right)