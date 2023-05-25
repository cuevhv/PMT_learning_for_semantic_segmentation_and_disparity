import torch.nn as nn
import torch.nn.functional as F
from models import resnet
from models.torch_model import conv2dSame, ConvTranspose2dSame
from models.densenet import densenet121
from models.resnet_deeplab import ResNet50, ResNet101
from spatial_correlation_sampler import SpatialCorrelationSampler
from models.torch_dsnet import apply_disparity
import torch
import math
from models.aspp import build_aspp
from models.mobilenetv3 import mobilenetv3_large
# convbn(64, 32, 3, 1, 0, 1)
class convbn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, pad, dilation, batchnorm=True):
        super(convbn, self).__init__()
    
        self.layers = [conv2dSame(in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                dilation = dilation,
                bias=False if batchnorm == True else True)]
        if batchnorm:
            self.layers.append(nn.BatchNorm2d(out_channel))
        self.layers = nn.Sequential(*self.layers) 
        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        return self.layers(x)

class deconvbn(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size, stride, pad, dilation, batchnorm=True):
        super(deconvbn, self).__init__()
        self.layers = [ConvTranspose2dSame(in_channel,
                                        out_channel,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=pad,
                                        dilation = dilation,
                                        bias=False if batchnorm == True else True)]
        if batchnorm:
            self.layers.append(nn.BatchNorm2d(out_channel))
        self.layers = nn.Sequential(*self.layers) 
        
        # for m in self.modules():
        #     if isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)


class Conv2DownUp(nn.Module):
    def __init__(self, in_channels, out_channels=3, kernel_size=3, lastLayer=True):
        super(Conv2DownUp, self).__init__()
        padding = 'same'#int((kernel_size-1)/2)
        self.lastLayer = lastLayer
        self.c1 = nn.Sequential(convbn(in_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(convbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True))

        self.c3 = nn.Sequential(convbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True)) 
                                
        self.d3 = nn.Sequential(deconvbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True))

        self.d4 = nn.Sequential(deconvbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True))
        self.d5 = nn.Sequential(deconvbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        #print('conv2downup')
        x1 = self.c1(x)
        #print(x1.shape)
        x2 = self.c2(x1)
        #print(x2.shape)
        x = self.c3(x2)
        #print(x.shape)
        x = self.d3(x)
        #print(x.shape)
        x = x2 + x
        #print(x.shape)
        x = self.d4(x)
        #print(x.shape)
        x = x1 + x
        #print(x.shape)
        if not self.lastLayer:
        #    print('-'*10)
            return x
        x = self.d5(x)
        #print(x.shape)
        #print('-'*10)
        return x
        
    
class segNet(nn.Module):
    def __init__(self, in_channels, labels=8, pretrained=False):
        super(segNet, self).__init__()
        self.conv1d_1 =nn.Sequential(conv2dSame(in_channels, 64, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp1 = Conv2DownUp(64, 32, 3)
        self.conv1d_2 =nn.Sequential(conv2dSame(33, 32, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp2 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False), ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
        
        
    def forward(self, x, input_a, xleft):
        #xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1d_1(x) #self.conv1d_1 = nn.Conv2d(1024*2, 64, 1)
        x = self.Conv2DownUp1(x)
        x1 = F.interpolate(x, scale_factor=2, mode='nearest')

        x1_1 = F.interpolate(x, size=(xleft.size()[2], xleft.size()[3]), mode='nearest')
        x1_1 = torch.cat((x1_1, xleft), dim=1)
        x1_1 = self.conv1d_2(x1_1)
        seg_branch = self.Conv2DownUp2(x1_1)
        seg_branch = F.interpolate(seg_branch, size=(input_a.size()[2], input_a.size()[3]), mode='nearest')
        seg_branch = F.log_softmax(seg_branch, dim=1)
        return x, x1, seg_branch


class SmallsegNet(nn.Module):
    def __init__(self, in_channels, feature_channel, labels=8, pretrained=False):
        super(SmallsegNet, self).__init__()
        self.conv1d_1 =nn.Sequential(conv2dSame(in_channels, 64, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp1 = Conv2DownUp(64, 32, 3)
        self.conv1d_2 =nn.Sequential(conv2dSame(32+feature_channel, 32, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp2 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False), ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
        
        
    def forward(self, x, input_a, xleft):
        #xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1d_1(x) #self.conv1d_1 = nn.Conv2d(1024*2, 64, 1)
        x = self.Conv2DownUp1(x)
        # x1 = F.interpolate(x, scale_factor=2, mode='nearest')

        x1_1 = F.interpolate(x, size=(xleft.size()[2], xleft.size()[3]), mode='nearest')
        x1_1 = torch.cat((x1_1, xleft), dim=1)
        x1_1 = self.conv1d_2(x1_1)
        seg_branch = self.Conv2DownUp2(x1_1)
        seg_branch = F.interpolate(seg_branch, size=(input_a.size()[2], input_a.size()[3]), mode='nearest')
        #seg_branch = F.log_softmax(seg_branch, dim=1)
        return x, x1_1, seg_branch

class minidsnetDivideSoftmax(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False, backbone='densenet', ):
        super(minidsnetDivideSoftmax, self).__init__()
        self.patch_type = patch_type
        self.include_edges = include_edges
        self.aspp_mod = CFG.aspp
        max_disp = 8
        self.resnet_features = piramidNet2(pretrained=pretrained)
        if self.include_edges:
            aux_img_channel = 4
        else:
            aux_img_channel = 3
        if self.aspp_mod:
            self.aspp = build_aspp('densenet_a1', 32)
            inplane_seg2 = 256
        else:
            inplane_seg2 = 256

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
        self.corrConv2d = nn.Sequential(conv2dSame(out_plane_corr, 128, 1, padding='same'), nn.ReLU(inplace=True))
        # self.corrConv2d = nn.Sequential(conv2dSame(512, 128, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp3 = Conv2DownUp(32, 128, 3)
        self.Conv2DownUp4 = Conv2DownUp(128*2, 64, 3)
        self.segNet = SmallsegNet(576, 224, labels) #256
        self.conv1d_2 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = Conv2DownUp(64, 64, 5, lastLayer=False)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.segNetB2 = segNetB2(inplane_seg2, labels)
        
        #self.Conv2DownUp7 = Conv2DownUp(96, 64, 3)
        #self.conv1d_at_d = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid())
        self.Conv2DownUp7 = nn.Sequential(Conv2DownUp(96, 64, 3, lastLayer=False), ConvTranspose2dSame(64, labels, 3, 1, padding='same', init_he=False))
       
    def forward(self, input_a, input_b):
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
        else:
            left = input_a
            right = input_b
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_0, a_pyramidB_1, a_pyramidB_2, a_pyramidB_3  = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_0, b_pyramidB_1, b_pyramidB_2, b_pyramidB_3 = self.resnet_features(right)
        
        # densenet
        # 0 (None, 64, 128, 128)
        # 1 (None, 128, 64, 64)
        # 2 (None, 256, 32, 32)
        # 3 (None, 512, 16, 16)
        # 4 (None, 1024, 8, 8)
        
        # xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        # xright2 = self.conv2d_ba1(input_b)

        # xleft1 = self.conv2d_ba2(input_a)
        xleft0 = self.conv2d_ba0(input_a)
        xright0 = self.conv2d_ba0(input_b)

        x, x1_1, seg_branch = self.segNet(a_pyramidB_3, input_a, a_pyramidB_0)
        _, _, seg_branch_right = self.segNet(b_pyramidB_3, input_b, a_pyramidB_0)

        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), axis=1)
        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        if self.patch_type == '1dcorr':
            y = torch.squeeze(y, axis=1)   
        else:
            b, ph, pw, h, w = y.size()
            y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        
        y = self.corrConv2d(y)
        y1 = self.Conv2DownUp3(x)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        y = torch.cat((y1, y), axis=1)
        y = self.Conv2DownUp4(y)
        
        y2 = F.interpolate(y, scale_factor=8)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        disp_out = torch.cat((y2, xleft2), axis=1)

        disp_out = self.conv1d_2(disp_out)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = self.dispoutConv(disp_out)
        disp_out = F.interpolate(disp_out, size=(left.shape[2], left.shape[3]), mode='bilinear')

        # if self.aspp_mod:
        #     s2 = self.aspp(a_1)
        # else:
        #     s2 = a_pyramidB_1
        
        # seg_branch2, s2 = self.segNetB2(s2, x1, xleft1)
        y3 = F.interpolate(y, size=(x1_1.shape[2], x1_1.shape[3]))
        s2_d = torch.cat((x1_1, y3), axis=1)
        at_d = self.Conv2DownUp7(s2_d)
        # s2_d = self.Conv2DownUp7(s2_d)
        # at_d = self.conv1d_at_d(s2_d)
        at_d = F.interpolate(at_d, size=(seg_branch.shape[2], seg_branch.shape[3]), mode='nearest')
        at_d = F.softmax(at_d, dim=1)

        # import matplotlib.pyplot as plt
        
        # plt.figure(1)
        # for i in range(at_d.shape[1]):
        #     plt.subplot(4,5,i+1)
        #     plt.imshow(1-at_d[0,i,:,:].cpu().numpy())
        # plt.figure(2)
        # for i in range(at_d.shape[1]):
        #     plt.subplot(4,5,i+1)
        #     print(seg_branch[0,i,:,:].max())
        #     plt.imshow(seg_branch[0,i,:,:].cpu().numpy())

        seg_branch_right = apply_disparity(seg_branch_right, -disp_out)
        seg_branch_both = (1 - at_d) * seg_branch + at_d * seg_branch_right

        # from util.utilTorchPlot import decode_segmap
        # import numpy as np
        # plt.figure(3)
        # plt.subplot(3,1,1)
        # plt.imshow(np.transpose(decode_segmap(seg_branch.detach().cpu().argmax(1).unsqueeze(1), nc=21)[0].numpy(), (1,2,0)))
        # plt.subplot(3,1,2)
        # plt.imshow(np.transpose(decode_segmap(seg_branch_right.detach().cpu().argmax(1).unsqueeze(1), nc=21)[0].numpy(), (1,2,0)))
        # plt.subplot(3,1,3)
        # plt.imshow(np.transpose(decode_segmap(seg_branch_both.detach().cpu().argmax(1).unsqueeze(1), nc=21)[0].numpy(), (1,2,0)))
        # plt.show()

        return seg_branch, disp_out, seg_branch_both, disp_out, seg_branch_right, at_d#, seg_branch_both, at_d#out1, out3

class segNetB2(nn.Module):
    def __init__(self, inplane_seg2, labels):
        super(segNetB2, self).__init__()
        self.conv1d_1 = nn.Sequential(conv2dSame(inplane_seg2,128,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp1 = Conv2DownUp(128, 64, 3)
        self.Conv2DownUp2 = Conv2DownUp(32, 64, 3)
        self.Conv2DownUp3 = Conv2DownUp(128, 64, 3)

        #self.Conv2DownUp4 = Conv2DownUp(64, 64, 3)
        self.conv1d_2 = nn.Sequential(conv2dSame(65,32,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False), ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
    def forward(self, in_image, in_feature, xleft):
        s2 = self.conv1d_1(in_image)
        s2 = self.Conv2DownUp1(s2)

        x3 = self.Conv2DownUp2(in_feature)
        x3 = F.interpolate(x3, size=(s2.shape[2], s2.shape[3]))

        s2_s = torch.cat((s2, x3), axis=1)
        s2_s = self.Conv2DownUp3(s2_s)
        #s2_s = self.Conv2DownUp4(s2_s)
        s2_s = F.interpolate(s2_s, size=(xleft.shape[2], xleft.shape[3]))
        s2_s = torch.cat((s2_s, xleft), axis=1)
        
        seg_branch2 = self.conv1d_2(s2_s)
        seg_branch2 = self.Conv2DownUp5(seg_branch2)
        seg_branch2 = F.log_softmax(seg_branch2, dim=1)
        return seg_branch2, s2

class piramidNet2(nn.Module):
    def __init__(self, pretrained=False, backbone='densenet'):
        super(piramidNet2, self).__init__()
        #self.resnet_features = resnet.resnet50(pretrained=pretrained)
        if backbone == 'densenet':
            self.resnet_features = densenet121(pretrained=pretrained)
            in_plane = [64, 128, 256, 512]
            # densenet
            # (None, 64, 128, 128)
            # (None, 128, 64, 64)
            # (None, 256, 32, 32)
            # (None, 512, 16, 16)
            # (None, 1024, 8, 8)
        if backbone == 'mobilenet':
            self.resnet_features = mobilenetv3_large()
            in_plane = [16, 24, 40, 112]
            # mobilenet
            # 1 torch.Size([None, 16, 128, 128])
            # 3 torch.Size([None, 24, 64, 64])
            # 6 torch.Size([None, 40, 32, 32])
            # 12 torch.Size([None, 112, 16, 16])
            # 15 torch.Size([None, 160, 8, 8])
        if backbone == 'resnet50':
            self.resnet_features = ResNet50(BatchNorm=nn.BatchNorm2d, pretrained=pretrained, output_stride=16)
            in_plane = [64, 256, 512, 1024]

        if backbone == 'resnet101':
            self.resnet_features = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=pretrained, output_stride=16)
            in_plane = [64, 256, 512, 1024]
            # torch.Size([1, 64, 128, 128])
            # torch.Size([1, 256, 64, 64])
            # torch.Size([1, 512, 32, 32])
            # torch.Size([1, 1024, 16, 16])
            # torch.Size([1, 2048, 16, 16])
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

        self.branch3_0 = nn.Sequential(nn.AvgPool2d(pool_val[3], pool_val[3]),
                                     convbn(in_plane[3], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch3_1 = nn.Sequential(nn.AvgPool2d(pool_val[4], pool_val[4]),
                                     convbn(in_plane[3], 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))


    def forward(self, x):
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

        b3_0 = self.branch3_0(out_3)
        b3_0 = F.interpolate(b3_0, (out_3.size()[2], out_3.size()[3]), mode=mode)
        
        b3_1 = self.branch3_1(out_3)
        b3_1 = F.interpolate(b2_1, (out_3.size()[2], out_3.size()[3]), mode=mode)
        
        b3 = torch.cat((out_3, b3_0, b3_1), axis=1)
        
        return out_0, out_1, out_2, out_3, out_4, b0, b1, b2, b3



class seg_dsnet(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False):
        super(seg_dsnet, self).__init__()
        self.patch_type = patch_type
        self.include_edges = include_edges
        max_disp = 8
        self.resnet_features = piramidNet2(pretrained=pretrained)
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
        self.corrConv2d = nn.Sequential(conv2dSame(out_plane_corr, 128, 1, padding='same'), nn.ReLU(inplace=True))
        # self.corrConv2d = nn.Sequential(conv2dSame(512, 128, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp3 = Conv2DownUp(32, 128, 3)
        self.Conv2DownUp4 = Conv2DownUp(128*2, 64, 3)
        self.segNet = segNet(1024, labels)
        self.conv1d_2 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = Conv2DownUp(64, 64, 5, lastLayer=False)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))

    def forward(self, input_a, input_b):
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
        else:
            left = input_a
            right = input_b
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_0, a_pyramidB_1, a_pyramidB_2, a_pyramidB_3  = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_0, b_pyramidB_1, b_pyramidB_2, b_pyramidB_3 = self.resnet_features(right)

        xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        xleft1 = self.conv2d_ba2(input_a)
        xleft0 = self.conv2d_ba0(input_a)
        xright0 = self.conv2d_ba0(input_b)

        x, x1, seg_branch = self.segNet(a_4, input_a, xleft0)
        _, _, seg_branch_right = self.segNet(b_4, input_b, xright0)

        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        if self.patch_type == '1dcorr':
            y = torch.squeeze(y, axis=1)   
        else:
            b, ph, pw, h, w = y.size()
            y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        
        # y = torch.cat((a_2, b_2), axis=1)
        #print(y.shape)
        y = self.corrConv2d(y)
        #print(y.shape)
        y1 = self.Conv2DownUp3(x1)
        #print(y1.shape)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        #print(y1.shape)
        y = torch.cat((y1, y), axis=1)
        #print(y.shape)
        y = self.Conv2DownUp4(y)
        #print(y.shape)
        
        y2 = F.interpolate(y, scale_factor=8)
        #print(y2.shape)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), axis=1)
        #print(disp_out.shape)
        disp_out = self.conv1d_2(disp_out)
        #print(disp_out.shape)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = self.dispoutConv(disp_out)
        #print(disp_out.shape)
        disp_out = F.interpolate(disp_out, size=(left.shape[2], left.shape[3]), mode='bilinear')
        seg_branch_right = apply_disparity(seg_branch_right, -disp_out)
        
        return seg_branch, disp_out, seg_branch_right, disp_out#out1, out3


class minidsnetDivide(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False, backbone='densenet', ):
        super(minidsnetDivide, self).__init__()
        self.patch_type = patch_type
        self.include_edges = include_edges
        self.aspp_mod = CFG.aspp
        max_disp = 8
        self.resnet_features = piramidNet2(pretrained=pretrained, backbone=backbone)
        if self.include_edges:
            aux_img_channel = 4
        else:
            aux_img_channel = 3
        if self.aspp_mod:
            self.aspp = build_aspp('densenet', 32)
            inplane_seg2 = 256
        else:
            inplane_seg2 = 256

        if backbone == 'densenet':
            segnet_input = 576
            segnet_feature_ch = 256
        if backbone == 'resnet50' or backbone == 'resnet101':
            segnet_input = 1088
            segnet_feature_ch = 384
        if backbone == 'mobilenet':
            segnet_input = 176
            segnet_feature_ch = 152

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
        self.corrConv2d = nn.Sequential(conv2dSame(out_plane_corr, 128, 1, padding='same'), nn.ReLU(inplace=True))
        # self.corrConv2d = nn.Sequential(conv2dSame(512, 128, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp3 = Conv2DownUp(32, 128, 3)
        self.Conv2DownUp4 = Conv2DownUp(128*2, 64, 3)
        self.segNet = SmallsegNet(segnet_input, segnet_feature_ch,labels)
        self.conv1d_2 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = Conv2DownUp(64, 64, 5, lastLayer=False)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.segNetB2 = segNetB2(inplane_seg2, labels)
        
        self.Conv2DownUp7 = Conv2DownUp(96, 64, 3)
        self.conv1d_at_d = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid())
       
       
    def forward(self, input_a, input_b):
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
        else:
            left = input_a
            right = input_b
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_0, a_pyramidB_1, a_pyramidB_2, a_pyramidB_3  = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_0, b_pyramidB_1, b_pyramidB_2, b_pyramidB_3 = self.resnet_features(right)
        
        # densenet
        # 0 (None, 64, 128, 128)
        # 1 (None, 128, 64, 64)
        # 2 (None, 256, 32, 32)
        # 3 (None, 512, 16, 16)
        # 4 (None, 1024, 8, 8)
        
        # xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        # xright2 = self.conv2d_ba1(input_b)

        # xleft1 = self.conv2d_ba2(input_a)
        xleft0 = self.conv2d_ba0(input_a)
        xright0 = self.conv2d_ba0(input_b)

        x, x1_1, seg_branch = self.segNet(a_pyramidB_3, input_a, a_pyramidB_1)
        _, _, seg_branch_right = self.segNet(b_pyramidB_3, input_b, b_pyramidB_1)

        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), axis=1)
        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        if self.patch_type == '1dcorr':
            y = torch.squeeze(y, axis=1)   
        else:
            b, ph, pw, h, w = y.size()
            y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        
        y = self.corrConv2d(y)
        y1 = self.Conv2DownUp3(x)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        y = torch.cat((y1, y), axis=1)
        y = self.Conv2DownUp4(y)
        
        y2 = F.interpolate(y, scale_factor=8)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        disp_out = torch.cat((y2, xleft2), axis=1)

        disp_out = self.conv1d_2(disp_out)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = self.dispoutConv(disp_out)
        disp_out = F.interpolate(disp_out, size=(left.shape[2], left.shape[3]), mode='bilinear')

        # if self.aspp_mod:
        #     s2 = self.aspp(a_1)
        # else:
        #     s2 = a_pyramidB_1
        
        # seg_branch2, s2 = self.segNetB2(s2, x1, xleft1)
        y3 = F.interpolate(y, size=(x1_1.shape[2], x1_1.shape[3]))
        s2_d = torch.cat((x1_1, y3), axis=1)
        s2_d = self.Conv2DownUp7(s2_d)
        at_d = self.conv1d_at_d(s2_d)
        at_d = F.interpolate(at_d, size=(seg_branch.shape[2], seg_branch.shape[3]), mode='nearest')
        seg_branch_right = apply_disparity(seg_branch_right, -disp_out)
        seg_branch_both = (1 - at_d) * seg_branch + at_d * seg_branch_right

        return seg_branch_both, disp_out, seg_branch, disp_out, seg_branch_right, at_d#, seg_branch_both, at_d#out1, out3



class minidsnetDivideDisp(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False, backbone='densenet', ):
        super(minidsnetDivideDisp, self).__init__()
        self.patch_type = patch_type
        self.include_edges = include_edges
        self.aspp_mod = CFG.aspp
        max_disp = 8
        self.resnet_features = piramidNet2(pretrained=pretrained, backbone=backbone)
        if self.include_edges:
            aux_img_channel = 4
        else:
            aux_img_channel = 3
        if self.aspp_mod:
            self.aspp = build_aspp('densenet', 32)
            inplane_seg2 = 256
        else:
            inplane_seg2 = 256

        if backbone == 'densenet':
            segnet_input = 576
            segnet_feature_ch = 256
        if backbone == 'resnet50' or backbone == 'resnet101':
            segnet_input = 1088
            segnet_feature_ch = 384
        if backbone == 'mobilenet':
            segnet_input = 176
            segnet_feature_ch = 152

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
        self.corrConv2d = nn.Sequential(conv2dSame(out_plane_corr, 128, 1, padding='same'), nn.ReLU(inplace=True))
        # self.corrConv2d = nn.Sequential(conv2dSame(512, 128, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp3 = Conv2DownUp(32, 128, 3)
        self.Conv2DownUp4 = Conv2DownUp(128*2, 64, 3)
        self.segNet = SmallsegNet(segnet_input, segnet_feature_ch,labels)
        self.conv1d_2 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = Conv2DownUp(64, 64, 5, lastLayer=False)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.segNetB2 = segNetB2(inplane_seg2, labels)
        
        self.Conv2DownUp7 = Conv2DownUp(128, 64, 3)
        self.conv1d_at_d = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid())
       
       
    def forward(self, input_a, input_b, disp):
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
        else:
            left = input_a
            right = input_b
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_0, a_pyramidB_1, a_pyramidB_2, a_pyramidB_3  = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        _, _, _, _, _, _, _, b_pyramidB_2, _ = self.resnet_features(right)
        # densenet
        # 0 (None, 64, 128, 128)
        # 1 (None, 128, 64, 64)
        # 2 (None, 256, 32, 32)
        # 3 (None, 512, 16, 16)
        # 4 (None, 1024, 8, 8)
        
        # xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        # xright2 = self.conv2d_ba1(input_b)

        # xleft1 = self.conv2d_ba2(input_a)
        xleft0 = self.conv2d_ba0(input_a)
        xright0 = self.conv2d_ba0(input_b)

        x, x1_1, seg_branch = self.segNet(a_pyramidB_3, input_a, a_pyramidB_1)

        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), axis=1)
        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        if self.patch_type == '1dcorr':
            y = torch.squeeze(y, axis=1)   
        else:
            b, ph, pw, h, w = y.size()
            y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        
        y = self.corrConv2d(y)
        y1 = self.Conv2DownUp3(x)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        y = torch.cat((y1, y), axis=1)
        y = self.Conv2DownUp4(y)
        
        y2 = F.interpolate(y, scale_factor=8)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        disp_out = torch.cat((y2, xleft2), axis=1)

        disp_out = self.conv1d_2(disp_out)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = self.dispoutConv(disp_out)
        disp_out = F.interpolate(disp_out, size=(left.shape[2], left.shape[3]), mode='bilinear')

        b_0, b_1, b_2, b_3, b_4, b_pyramidB_0, b_pyramidB_1, _, b_pyramidB_3 = self.resnet_features(apply_disparity(right, -disp)* (disp > 0))
        # import matplotlib.pyplot as plt
        # import numpy as np
        # warp_left = apply_disparity(right, -disp) * (disp > 0)
        # plt.subplot(2,1,1)
        # plt.imshow(np.transpose(warp_left.cpu().numpy(), (0,2,3,1))[0])
        # plt.subplot(2,1,2)
        # plt.imshow(np.transpose(left.cpu().numpy(), (0,2,3,1))[0])
        # plt.show()
        _, x2_1, seg_branch_right = self.segNet(b_pyramidB_3, input_b, b_pyramidB_1)
        # if self.aspp_mod:
        #     s2 = self.aspp(a_1)
        # else:
        #     s2 = a_pyramidB_1
        
        # seg_branch2, s2 = self.segNetB2(s2, x1, xleft1)
        y3 = F.interpolate(y, size=(x1_1.shape[2], x1_1.shape[3]))
        s2_d = torch.cat((x1_1, x2_1, y3), axis=1)

        s2_d = self.Conv2DownUp7(s2_d)
        at_d = self.conv1d_at_d(s2_d)
        at_d = F.interpolate(at_d, size=(seg_branch.shape[2], seg_branch.shape[3]), mode='nearest')
        #seg_branch_right = apply_disparity(seg_branch_right, -disp_out)
        seg_branch_both = (1 - at_d) * seg_branch + at_d * seg_branch_right

        return seg_branch_both, disp_out, seg_branch, disp_out, seg_branch_right, at_d#, seg_branch_both, at_d#out1, out3


class minidsnetDivideDisp2(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False, backbone='densenet', ):
        super(minidsnetDivideDisp2, self).__init__()
        self.patch_type = patch_type
        self.include_edges = include_edges
        self.aspp_mod = CFG.aspp
        max_disp = 8
        self.resnet_features = piramidNet2(pretrained=pretrained, backbone=backbone)
        if self.include_edges:
            aux_img_channel = 4
        else:
            aux_img_channel = 3
        if self.aspp_mod:
            self.aspp = build_aspp('densenet', 32)
            inplane_seg2 = 256
        else:
            inplane_seg2 = 256

        if backbone == 'densenet':
            segnet_input = 576
            segnet_feature_ch = 256
        if backbone == 'resnet50' or backbone == 'resnet101':
            segnet_input = 1088
            segnet_feature_ch = 384
        if backbone == 'mobilenet':
            segnet_input = 176
            segnet_feature_ch = 152

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
        self.corrConv2d = nn.Sequential(conv2dSame(out_plane_corr, 128, 1, padding='same'), nn.ReLU(inplace=True))
        # self.corrConv2d = nn.Sequential(conv2dSame(512, 128, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp3 = Conv2DownUp(32, 128, 3)
        self.Conv2DownUp4 = Conv2DownUp(128*2, 64, 3)
        self.segNet = SmallsegNet(segnet_input, segnet_feature_ch,labels)
        self.conv1d_2 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = Conv2DownUp(64, 64, 5, lastLayer=False)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.segNetB2 = segNetB2(inplane_seg2, labels)
        
        self.Conv2DownUp7 = Conv2DownUp(128, 64, 3)
        self.conv1d_at_d = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid())
       
       
    def forward(self, input_a, input_b):
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
        else:
            left = input_a
            right = input_b
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_0, a_pyramidB_1, a_pyramidB_2, a_pyramidB_3  = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        _, _, _, _, _, _, _, b_pyramidB_2, _ = self.resnet_features(right)
        # densenet
        # 0 (None, 64, 128, 128)
        # 1 (None, 128, 64, 64)
        # 2 (None, 256, 32, 32)
        # 3 (None, 512, 16, 16)
        # 4 (None, 1024, 8, 8)
        
        # xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        # xright2 = self.conv2d_ba1(input_b)

        # xleft1 = self.conv2d_ba2(input_a)
        xleft0 = self.conv2d_ba0(input_a)
        xright0 = self.conv2d_ba0(input_b)

        x, x1_1, seg_branch = self.segNet(a_pyramidB_3, input_a, a_pyramidB_1)

        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), axis=1)
        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        if self.patch_type == '1dcorr':
            y = torch.squeeze(y, axis=1)   
        else:
            b, ph, pw, h, w = y.size()
            y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        
        y = self.corrConv2d(y)
        y1 = self.Conv2DownUp3(x)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        y = torch.cat((y1, y), axis=1)
        y = self.Conv2DownUp4(y)
        
        y2 = F.interpolate(y, scale_factor=8)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        disp_out = torch.cat((y2, xleft2), axis=1)

        disp_out = self.conv1d_2(disp_out)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = self.dispoutConv(disp_out)
        disp_out = F.interpolate(disp_out, size=(left.shape[2], left.shape[3]), mode='bilinear')

        warped_right = apply_disparity(right, -disp_out)
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_0, b_pyramidB_1, _, b_pyramidB_3 = self.resnet_features(warped_right)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # warp_left = apply_disparity(right, -disp) * (disp > 0)
        # plt.subplot(2,1,1)
        # plt.imshow(np.transpose(warp_left.cpu().numpy(), (0,2,3,1))[0])
        # plt.subplot(2,1,2)
        # plt.imshow(np.transpose(left.cpu().numpy(), (0,2,3,1))[0])
        # plt.show()
        _, x2_1, seg_branch_right = self.segNet(b_pyramidB_3, input_b, b_pyramidB_1)
        # if self.aspp_mod:
        #     s2 = self.aspp(a_1)
        # else:
        #     s2 = a_pyramidB_1
        
        # seg_branch2, s2 = self.segNetB2(s2, x1, xleft1)
        y3 = F.interpolate(y, size=(x1_1.shape[2], x1_1.shape[3]))
        s2_d = torch.cat((x1_1, x2_1, y3), axis=1)

        s2_d = self.Conv2DownUp7(s2_d)
        at_d = self.conv1d_at_d(s2_d)
        at_d = F.interpolate(at_d, size=(seg_branch.shape[2], seg_branch.shape[3]), mode='nearest')
        #seg_branch_right = apply_disparity(seg_branch_right, -disp_out)
        seg_branch_both = (1 - at_d) * seg_branch + at_d * seg_branch_right
        
        return seg_branch_both, disp_out, seg_branch, disp_out, seg_branch_right, warped_right#, seg_branch_both, at_d#out1, out3