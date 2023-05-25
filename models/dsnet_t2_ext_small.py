import torch #at the beggining so spatialCorrelationSampler does not show any error
import torch.nn as nn
import torch.nn.functional as F
from models import resnet
from models.torch_model import conv2dSame, ConvTranspose2dSame
from models.densenet import densenet121
from models.resnet_deeplab import ResNet50, ResNet101
from spatial_correlation_sampler import SpatialCorrelationSampler
from efficientnet_pytorch import EfficientNet
from models.torch_dsnet import apply_disparity
import math
from models.aspp import build_aspp

from models.mobilenetv3 import mobilenetv3_large
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
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x = self.c3(x2)
        x = self.d3(x)
        x = x2 + x
        x = self.d4(x)
        x = x1 + x
        if not self.lastLayer:
            return x
        x = self.d5(x)
        return x

class RCU(nn.Module):
    def __init__(self, in_channels, out_channels=3, kernel_size=3, lastLayer=True, use_deconv=True):
        super(RCU, self).__init__()
        padding = 'same'#int((kernel_size-1)/2)
        self.lastLayer = lastLayer
        self.use_deconv = use_deconv
        self.c1 = nn.Sequential(convbn(in_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(convbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True))
        if self.use_deconv:
            self.d3 = nn.Sequential(deconvbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True))
        else:
            self.c3 = nn.Sequential(convbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True)) 
    def forward(self, x):
        x = self.c1(x)
        x1 = self.c2(x)
        if self.use_deconv:
            x1 = self.d3(x1)
        else:
            x1 = self.c3(x1)    
            x1 = F.interpolate(x1, size=(x.shape[2], x.shape[3]), mode='bilinear')
        x1 = x1 + x
        return x1

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


class Ext_small(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False, backbone='mobilenet', ):
        super(Ext_small, self).__init__()
        self.aspp_mod = CFG.aspp
        max_disp = 8
        feature_channel = 1
        self.backbone = backbone
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
            inplane_seg2 = 336
            # if self.aspp_mod == 1:
                
            # elif self.aspp_mod == 2:
            # else:
        if backbone == 'efficientnet-b4':
            segnet_input = 1792 * 2
            inplane_seg2 = 320
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
        self.Conv2DownUp3 = RCU(32, 64, 3, use_deconv=False)
        self.Conv2DownUp4 = RCU(128+64, 64, 3, use_deconv=False)
        self.segNet = segNet(segnet_input, 1, 1, RCU_deconv=False)
        self.conv1d_2 = nn.Sequential(conv2dSame(64+1,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = RCU(64, 64, 5, lastLayer=False, use_deconv=False)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.conv1d_4 = nn.Sequential(conv2dSame(inplane_seg2//2,128,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp6 = RCU(128, 64, 3, use_deconv=False)
        self.Conv2DownUp7 = RCU(128, 64, 3, use_deconv=False)
        self.Conv2DownUp8 = RCU(32, 64, 3, use_deconv=False)
        self.Conv2DownUp9 = RCU(128, 64, 3, use_deconv=False)
        self.conv1d_at = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid())
        
        self.Conv2DownUp10 = RCU(128, 64, 3, use_deconv=False)
        self.conv1d_5 = nn.Sequential(conv2dSame(64+feature_channel,32,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp11 = nn.Sequential(RCU(32, 32, 3, lastLayer=False, use_deconv=False), conv2dSame(32, labels, 3, 1, padding='same'))
       
    def forward(self, input_a, input_b, left_e):
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
            #left_e = input_a[:,3:, :]#.unsqueeze(dim=1)
        else:
            left = input_a
            right = input_b
        
        
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_2, a_pyramidB_1, a_pyramidB_0 = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_2, b_pyramidB_1, b_pyramidB_0 = self.resnet_features(right)
        
        # densenet
        # (None, 64, 128, 128)
        # (None, 128, 64, 64)
        # (None, 256, 32, 32)
        # (None, 512, 16, 16)
        # (None, 1024, 8, 8)
        # left_e = utilTorchGate.compute_grad_mag(left, True, False)
        edge_1 = F.interpolate(left_e, size=(left_e.shape[2]//2, left_e.shape[3]//2), mode='bilinear')
        edge_2 = F.interpolate(left_e, size=(left_e.shape[2]//2, left_e.shape[3]//2), mode='bilinear')

        xleft2 = self.conv2d_ba1(edge_2)
        xleft1 = self.conv2d_ba2(left_e)
        xleft0 = self.conv2d_ba0(edge_1)
        if self.backbone == 'resnet50' or self.backbone == 'resnet101':
            a_4 = self.aspp_4(a_4)
            b_4 = self.aspp_4(b_4)

        x = torch.cat((a_4, b_4), axis=1)
        x, x1, seg_branch = self.segNet(x, input_a, input_b, xleft0)
        
        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), axis=1)
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
        #print(disp_out.shape)
        #input()

        if self.aspp_mod == 1:
            s2 = self.aspp(a_1)
        
        elif self.aspp_mod == 2:
            s2_1 = self.aspp(a_3)
            s2_2 = self.aspp(b_3)
            #s2 = torch.cat((s2_1, s2_2), axis=1)
            s2_corr = self.s2_corr_sampler(s2_1, s2_2)
            s2_corr = torch.squeeze(s2_corr, axis=1) 
            s2 = torch.cat((s2_corr, s2_1), axis=1)
        else:
            #s2 = torch.cat((a_pyramidB_1, b_pyramidB_1), axis=1)
            s2 = b_pyramidB_1

        s2 = self.conv1d_4(s2)
        s2 = self.Conv2DownUp6(s2)      

        y3 = F.interpolate(y, size=(s2.shape[2], s2.shape[3]))

        s2_d = torch.cat((s2, y3), axis=1)
        s2_d = self.Conv2DownUp7(s2_d)
        #at_d = self.conv1d_at_d(s2_d)

        x3 = self.Conv2DownUp8(x1)
        x3 = F.interpolate(x3, size=(s2.shape[2], s2.shape[3]))

        s2_s = torch.cat((s2, x3), axis=1)
        s2_s = self.Conv2DownUp9(s2_s)
        #at_s = self.conv1d_at_s(s2_s)

        s2_at = self.conv1d_at(s2)
        s2 = torch.cat((s2_d*s2_at, s2_s*(1-s2_at)), axis=1)
        s2 = self.Conv2DownUp10(s2)
        
        if self.aspp_mod == 2:
            s2 = F.interpolate(s2, size=(a_0.shape[2], a_0.shape[3]))
            s2 = torch.cat((s2, a_0), axis=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]), mode='nearest')
        else:
            s2 = F.interpolate(s2, size=(xleft1.shape[2], xleft1.shape[3]))
            s2 = torch.cat((s2, xleft1), axis=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)

        #seg_branch2 = F.log_softmax(seg_branch2, dim=1)

        return seg_branch, disp_out, seg_branch2, disp_out#out1, out3



class Ext_smallv2(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False, backbone='mobilenet', ):
        super(Ext_smallv2, self).__init__()
        self.aspp_mod = CFG.aspp
        max_disp = 8
        feature_channel = 1
        spp_3_size = 224
        self.backbone = backbone
        if backbone == 'densenet':
            segnet_input = 1024*2
            spp_3_size = 224
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
            spp_3_size = 224
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
            spp_3_size = 224
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
            spp_3_size = 224
            segnet_input = 2048 * 2
            inplane_seg2 = 336
            # if self.aspp_mod == 1:
                
            # elif self.aspp_mod == 2:
            # else:
        if backbone == 'efficientnet-b4':
            spp_3_size = 184
            segnet_input = 1792 * 2
            inplane_seg2 = 320
            # if self.aspp_mod == 1:
                
            # elif self.aspp_mod == 2:
            # else:

        if backbone == 'efficientnet-b3':
            spp_3_size = 184
            segnet_input = 1536 * 2
            inplane_seg2 = 320
            # if self.aspp_mod == 1:
                
            # elif self.aspp_mod == 2:
            # else:

        if backbone == 'efficientnet-b2':
            spp_3_size = 176
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
        self.Conv2DownUp3 = RCU(32, 64, 3)
        self.Conv2DownUp4 = RCU(128+64, 64, 3)
        self.segNet = segNet(segnet_input, 64, 1)
        self.conv1d_2 = nn.Sequential(conv2dSame(64+64,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = RCU(64, 64, 5, lastLayer=False)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.conv1d_4 = nn.Sequential(conv2dSame(inplane_seg2//2,128,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp6 = RCU(128, 64, 3)
        self.Conv2DownUp7 = RCU(128, 64, 3)
        self.Conv2DownUp8 = RCU(32, 64, 3)
        self.Conv2DownUp9 = RCU(128, 64, 3)
        self.conv1d_at = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid())
        
        self.Conv2DownUp10 = RCU(128, 64, 3)
        self.conv1d_5 = nn.Sequential(conv2dSame(64+spp_3_size,32,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp11 = nn.Sequential(RCU(32, 32, 3, lastLayer=False), conv2dSame(32, labels, 3, 1, padding='same'))
       
    def forward(self, input_a, input_b, left_e):
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
            #left_e = input_a[:,3:, :]#.unsqueeze(dim=1)
        else:
            left = input_a
            right = input_b
        
        
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_2, a_pyramidB_1, a_pyramidB_0 = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_2, b_pyramidB_1, b_pyramidB_0 = self.resnet_features(right)
        # densenet
        # (None, 64, 128, 128)
        # (None, 128, 64, 64)
        # (None, 256, 32, 32)
        # (None, 512, 16, 16)
        # (None, 1024, 8, 8)
        # left_e = utilTorchGate.compute_grad_mag(left, True, False)
        edge_1 = F.interpolate(left_e, size=(left_e.shape[2]//2, left_e.shape[3]//2), mode='bilinear')
        edge_2 = F.interpolate(left_e, size=(left_e.shape[2]//2, left_e.shape[3]//2), mode='bilinear')

        #xleft2 = self.conv2d_ba1(edge_2)
        #xleft1 = self.conv2d_ba2(left_e)
        #xleft0 = self.conv2d_ba0(edge_1)
        if self.backbone == 'resnet50' or self.backbone == 'resnet101':
            a_4 = self.aspp_4(a_4)
            b_4 = self.aspp_4(b_4)

        x = torch.cat((a_4, b_4), axis=1)
        x, x1, seg_branch = self.segNet(x, input_a, input_b, a_0)
        
        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), axis=1)
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
        xleft2 = F.interpolate(a_0, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), axis=1)
        disp_out = self.conv1d_2(disp_out)
        #print(disp_out.shape)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = self.dispoutConv(disp_out)
        #print(disp_out.shape)
        disp_out = F.interpolate(disp_out, size=(left.shape[2], left.shape[3]), mode='bilinear')
        #print(disp_out.shape)
        #input()

        if self.aspp_mod == 1:
            s2 = self.aspp(a_1)
        
        elif self.aspp_mod == 2:
            s2_1 = self.aspp(a_3)
            s2_2 = self.aspp(b_3)
            #s2 = torch.cat((s2_1, s2_2), axis=1)
            s2_corr = self.s2_corr_sampler(s2_1, s2_2)
            s2_corr = torch.squeeze(s2_corr, axis=1) 
            s2 = torch.cat((s2_corr, s2_1), axis=1)
        else:
            #s2 = torch.cat((a_pyramidB_1, b_pyramidB_1), axis=1)
            s2 = b_pyramidB_1

        s2 = self.conv1d_4(s2)
        s2 = self.Conv2DownUp6(s2)      

        y3 = F.interpolate(y, size=(s2.shape[2], s2.shape[3]))

        s2_d = torch.cat((s2, y3), axis=1)
        s2_d = self.Conv2DownUp7(s2_d)
        #at_d = self.conv1d_at_d(s2_d)

        x3 = self.Conv2DownUp8(x1)
        x3 = F.interpolate(x3, size=(s2.shape[2], s2.shape[3]))

        s2_s = torch.cat((s2, x3), axis=1)
        s2_s = self.Conv2DownUp9(s2_s)
        #at_s = self.conv1d_at_s(s2_s)

        s2_at = self.conv1d_at(s2)
        s2 = torch.cat((s2_d*s2_at, s2_s*(1-s2_at)), axis=1)
        s2 = self.Conv2DownUp10(s2)
        
        if self.aspp_mod == 2:
            s2 = F.interpolate(s2, size=(a_0.shape[2], a_0.shape[3]))
            s2 = torch.cat((s2, a_0), axis=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]), mode='nearest')
        else:
            s2 = F.interpolate(s2, size=(a_pyramidB_0.shape[2], a_pyramidB_0.shape[3]))
            s2 = torch.cat((s2, a_pyramidB_0), axis=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]), mode='nearest')
        #seg_branch2 = F.log_softmax(seg_branch2, dim=1)

        return seg_branch, disp_out, seg_branch2, disp_out#out1, out3


class Ext_smallv0(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False, backbone='mobilenet', ):
        super(Ext_smallv0, self).__init__()
        self.aspp_mod = CFG.aspp
        max_disp = 8
        feature_channel = 1
        spp_3_size = 224
        self.backbone = backbone
        if backbone == 'densenet':
            segnet_input = 1024*2
            spp_3_size = 224
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
            spp_3_size = 224
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
            spp_3_size = 224
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
            spp_3_size = 224
            segnet_input = 2048 * 2
            inplane_seg2 = 336
            # if self.aspp_mod == 1:
                
            # elif self.aspp_mod == 2:
            # else:
        if backbone == 'efficientnet-b4':
            spp_3_size = 184
            segnet_input = 1792 * 2
            inplane_seg2 = 320
            # if self.aspp_mod == 1:
                
            # elif self.aspp_mod == 2:
            # else:

        if backbone == 'efficientnet-b3':
            spp_3_size = 184
            segnet_input = 1536 * 2
            inplane_seg2 = 320
            # if self.aspp_mod == 1:
                
            # elif self.aspp_mod == 2:
            # else:

        if backbone == 'efficientnet-b2':
            spp_3_size = 176
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
        self.Conv2DownUp3 = RCU(32, 64, 3)
        self.Conv2DownUp4 = RCU(128+64, 64, 3)
        self.segNet = segNet(segnet_input, 64, labels)
        self.conv1d_2 = nn.Sequential(conv2dSame(64+64,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = RCU(64, 64, 5, lastLayer=False)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.conv1d_4 = nn.Sequential(conv2dSame(inplane_seg2//2,128,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp6 = RCU(128, 64, 3)
        self.Conv2DownUp7 = RCU(128, 64, 3)
        self.Conv2DownUp8 = RCU(32, 64, 3)
        self.Conv2DownUp9 = RCU(128, 64, 3)
        self.conv1d_at = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid())
        
        self.Conv2DownUp10 = RCU(128, 64, 3)
        self.conv1d_5 = nn.Sequential(conv2dSame(64+spp_3_size,32,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp11 = nn.Sequential(RCU(32, 32, 3, lastLayer=False), conv2dSame(32, labels, 3, 1, padding='same'))
       
    def forward(self, input_a, input_b):
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
            #left_e = input_a[:,3:, :]#.unsqueeze(dim=1)
        else:
            left = input_a
            right = input_b
        
        
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_2, a_pyramidB_1, a_pyramidB_0 = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_2, b_pyramidB_1, b_pyramidB_0 = self.resnet_features(right)
        # densenet
        # (None, 64, 128, 128)
        # (None, 128, 64, 64)
        # (None, 256, 32, 32)
        # (None, 512, 16, 16)
        # (None, 1024, 8, 8)
        # left_e = utilTorchGate.compute_grad_mag(left, True, False)

        #xleft2 = self.conv2d_ba1(edge_2)
        #xleft1 = self.conv2d_ba2(left_e)
        #xleft0 = self.conv2d_ba0(edge_1)
        if self.backbone == 'resnet50' or self.backbone == 'resnet101':
            a_4 = self.aspp_4(a_4)
            b_4 = self.aspp_4(b_4)

        x = torch.cat((a_4, b_4), axis=1)
        x, x1, seg_branch = self.segNet(x, input_a, input_b, a_0)
        
        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), axis=1)
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
        xleft2 = F.interpolate(a_0, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), axis=1)
        #print(disp_out.shape)
        disp_out = self.conv1d_2(disp_out)
        #print(disp_out.shape)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = self.dispoutConv(disp_out)
        #print(disp_out.shape)
        disp_out = F.interpolate(disp_out, size=(left.shape[2], left.shape[3]), mode='bilinear')
        #print(disp_out.shape)
        #input()

        if self.aspp_mod == 1:
            s2 = self.aspp(a_1)
        
        elif self.aspp_mod == 2:
            s2_1 = self.aspp(a_3)
            s2_2 = self.aspp(b_3)
            #s2 = torch.cat((s2_1, s2_2), axis=1)
            s2_corr = self.s2_corr_sampler(s2_1, s2_2)
            s2_corr = torch.squeeze(s2_corr, axis=1) 
            s2 = torch.cat((s2_corr, s2_1), axis=1)
        else:
            #s2 = torch.cat((a_pyramidB_1, b_pyramidB_1), axis=1)
            s2 = b_pyramidB_1

        s2 = self.conv1d_4(s2)
        s2 = self.Conv2DownUp6(s2)      

        y3 = F.interpolate(y, size=(s2.shape[2], s2.shape[3]))

        s2_d = torch.cat((s2, y3), axis=1)
        s2_d = self.Conv2DownUp7(s2_d)
        #at_d = self.conv1d_at_d(s2_d)

        x3 = self.Conv2DownUp8(x1)
        x3 = F.interpolate(x3, size=(s2.shape[2], s2.shape[3]))

        s2_s = torch.cat((s2, x3), axis=1)
        s2_s = self.Conv2DownUp9(s2_s)
        #at_s = self.conv1d_at_s(s2_s)

        s2_at = self.conv1d_at(s2)
        s2 = torch.cat((s2_d*s2_at, s2_s*(1-s2_at)), axis=1)
        s2 = self.Conv2DownUp10(s2)
        
        if self.aspp_mod == 2:
            s2 = F.interpolate(s2, size=(a_0.shape[2], a_0.shape[3]))
            s2 = torch.cat((s2, a_0), axis=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]), mode='nearest')
        else:
            s2 = F.interpolate(s2, size=(a_pyramidB_0.shape[2], a_pyramidB_0.shape[3]))
            s2 = torch.cat((s2, a_pyramidB_0), axis=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]), mode='nearest')
        #seg_branch2 = F.log_softmax(seg_branch2, dim=1)

        return seg_branch, disp_out, seg_branch2, disp_out#out1, out3


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

        if backbone == 'efficientnet-b4':
            self.resnet_features = EfficientNet.from_pretrained(backbone)
            in_plane = [24, 32, 56]
            # reduction_1 torch.Size([2, 24, 128, 256])
            # reduction_2 torch.Size([2, 32, 64, 128])
            # reduction_3 torch.Size([2, 56, 32, 64])
            # reduction_4 torch.Size([2, 160, 16, 32])
            # reduction_5 torch.Size([2, 1792, 8, 16])

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
        if self.backbone in ['efficientnet-b5', 'efficientnet-b4', 'efficientnet-b3', 'efficientnet-b2']:
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


class segNet(nn.Module):
    def __init__(self, in_channels, feature_channel, labels=8, pretrained=False, RCU_deconv=True):
        super(segNet, self).__init__()
        self.conv1d_1 =nn.Sequential(conv2dSame(in_channels, 64, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp1 = RCU(64, 32, 3, use_deconv=RCU_deconv)
        self.conv1d_2 =nn.Sequential(conv2dSame(32+feature_channel, 32, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp2 = nn.Sequential(RCU(32, 32, 3, lastLayer=False, use_deconv=RCU_deconv), conv2dSame(32, labels, 3, 1, padding='same'))
        
        
    def forward(self, x, input_a, input_b, xleft):
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
        #seg_branch = F.log_softmax(seg_branch, dim=1)
        return x, x1, seg_branch
