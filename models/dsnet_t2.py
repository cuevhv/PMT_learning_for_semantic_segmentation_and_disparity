import torch #at the beggining so spatialCorrelationSampler does not show any error
import torch.nn as nn
import torch.nn.functional as F
from models import resnet
from models.torch_model import conv2dSame, ConvTranspose2dSame
from models.densenet import densenet121, densenet169, densenet201, densenet161
from models.resnet_deeplab import ResNet50, ResNet101
from spatial_correlation_sampler import SpatialCorrelationSampler
from efficientnet_pytorch import EfficientNet
from models.torch_dsnet import apply_disparity
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
    def __init__(self, in_channels, out_channels=3, kernel_size=3, lastLayer=True, dropout=0):
        super(Conv2DownUp, self).__init__()
        padding = 'same'#int((kernel_size-1)/2)
        self.lastLayer = lastLayer
        self.c1 = nn.Sequential(convbn(in_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True), nn.Dropout(p=dropout))
        self.c2 = nn.Sequential(convbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True), nn.Dropout(p=dropout))

        self.c3 = nn.Sequential(convbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True), nn.Dropout(p=dropout)) 
                                
        self.d3 = nn.Sequential(deconvbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True), nn.Dropout(p=dropout))
        
        self.d4 = nn.Sequential(deconvbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True), nn.Dropout(p=dropout))
        self.d5 = nn.Sequential(deconvbn(out_channels, out_channels, kernel_size, 1, padding, 1), nn.ReLU(inplace=True), nn.Dropout(p=dropout))

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

class dsnet(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, backbone='densenet'):
        super(dsnet, self).__init__()
        
        max_disp = 8
        self.resnet_features = piramidNet(pretrained=pretrained)
        self.conv2d_ba1 = nn.Sequential(convbn(3, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True)) 
        self.conv2d_ba2 = nn.Sequential(convbn(3, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True)) 
        self.conv2d_ba3 = nn.Sequential(convbn(3, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True)) 

        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1,
                                                            patch_size=(max_disp*2 + 1, max_disp*2 + 1),
                                                            stride=1,
                                                            padding=0,#max_disp*2 + 1,
                                                            dilation_patch=1)
        self.corrConv2d = nn.Sequential(conv2dSame(289, 128, 1, padding='same'), nn.ReLU(inplace=True))
        # self.corrConv2d = nn.Sequential(conv2dSame(512, 128, 1, padding='same'), nn.ReLU(inplace=True))
        self.conv1d_1 =nn.Sequential(conv2dSame(1024*2, 64, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp1 = Conv2DownUp(64, 32, 3)
        self.Conv2DownUp2 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False), ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
        self.Conv2DownUp3 = Conv2DownUp(32, 128, 3)
        self.Conv2DownUp4 = Conv2DownUp(128*2, 64, 3)
        
        self.conv1d_2 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = Conv2DownUp(64, 64, 5, lastLayer=False)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp6 = Conv2DownUp(64, 64, 5)
        self.conv1d_4 = nn.Sequential(conv2dSame(192,64,1, padding='same'), nn.ReLU(inplace=True))
        # deconvbn(in_channel, out_channel, kernel_size, stride, pad, dilation, batchnorm=True)
        self.conv2DT_BA1 = nn.Sequential(deconvbn(64, 32, 3, 2, 'same', 1), nn.ReLU(inplace=True))
        self.conv1d_5 = nn.Sequential(conv2dSame(96,32,1, padding='same'), nn.ReLU(inplace=True))
        self.conv2DT_BA2 = nn.Sequential(deconvbn(32, 32, 3, 2, 'same', 1), nn.ReLU(inplace=True))
        self.conv1d_6 = nn.Sequential(conv2dSame(33,32,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp7 =Conv2DownUp(32, 32, 5, lastLayer=False)
        self.branchConv = ConvTranspose2dSame(32, labels, 5, padding='same', init_he=False)
        self.conv1d_9 = nn.Sequential(conv2dSame(448, 128, 1, padding='same'), nn.ReLU(inplace=True))
        #self.conv1d_9 = nn.Sequential(conv2dSame(128, 128, 1, padding='same'), nn.ReLU(inplace=True))
        self.conv1d_7 = nn.Sequential(conv2dSame(128*2,128,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp8 = Conv2DownUp(32, 64, 3)
        self.Conv2DownUp9 = Conv2DownUp(256, 64, 3)
        self.conv1d_8 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp10 = nn.Sequential(Conv2DownUp(64, 64, 5, lastLayer=False), ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)
        # for m in self.modules():
        #     print(m)
        #     print('x'*100)
        #     if isinstance(m, nn.Conv2d):
        #         print(m)
        #     #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     #     m.weight.data.normal_(0, math.sqrt(2. / n))
        #     # elif isinstance(m, nn.Conv3d):
        #     #     n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
        #     #     m.weight.data.normal_(0, math.sqrt(2. / n))
        #     # elif isinstance(m, nn.BatchNorm2d):
        #     #     m.weight.data.fill_(1)
        #     #     m.bias.data.zero_()
        #     # elif isinstance(m, nn.BatchNorm3d):
        #     #     m.weight.data.fill_(1)
        #     #     m.bias.data.zero_()
        #     # elif isinstance(m, nn.Linear):
        #     #     m.bias.data.zero_()
    


    def forward(self, input_a, input_b):
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_2, a_pyramidB_0 = self.resnet_features(input_a) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_2, b_pyramidB_0 = self.resnet_features(input_b)
        
        # densenet
        # (None, 64, 128, 128)
        # (None, 128, 64, 64)
        # (None, 256, 32, 32)
        # (None, 512, 16, 16)
        # (None, 1024, 8, 8)
        
        xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        xleft1 = self.conv2d_ba2(input_a)

        x = torch.cat((a_4, b_4), dim=1)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1d_1(x) #self.conv1d_1 = nn.Conv2d(1024*2, 64, 1)
        x = self.Conv2DownUp1(x)
        x1 = F.interpolate(x, scale_factor=2, mode='nearest')
        seg_branch = self.Conv2DownUp2(x1)
        seg_branch = F.interpolate(seg_branch, scale_factor=8, mode='nearest')
        seg_branch = F.interpolate(seg_branch, size=(input_a.size()[2], input_a.size()[3]), mode='bilinear')
        seg_branch = F.log_softmax(seg_branch, dim=1)
        # epsilon = 1e-9
        # seg_branch11 = F.softmax(seg_branch, dim=1)
        # seg_branch = torch.clamp(seg_branch11, epsilon, 1-epsilon)
        # seg_branch = torch.log(seg_branch)
        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), dim=1)
        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        b, ph, pw, h, w = y.size()
        y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        # y = torch.cat((a_2, b_2), dim=1)
        #print(y.shape)
        y = self.corrConv2d(y)
        #print(y.shape)
        y1 = self.Conv2DownUp3(x1)
        #print(y1.shape)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        #print(y1.shape)
        y = torch.cat((y1, y), dim=1)
        #print(y.shape)
        y = self.Conv2DownUp4(y)
        #print(y.shape)
        
        y2 = F.interpolate(y, scale_factor=8)
        #print(y2.shape)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), dim=1)
        #print(disp_out.shape)
        disp_out = self.conv1d_2(disp_out)
        #print(disp_out.shape)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = self.dispoutConv(disp_out)
        #print(disp_out.shape)
        disp_out = F.interpolate(disp_out, size=(input_a.shape[2], input_a.shape[3]), mode='bilinear')
        #print(disp_out.shape)
        #input()

        x = F.interpolate(x, scale_factor=4)
        y3 = F.interpolate(y, scale_factor=2)
        x = F.interpolate(x, (y3.shape[2], y3.shape[3]), mode='bilinear')

        x = torch.cat((x, y3), dim=1)
        x = self.conv1d_3(x)
        x = self.Conv2DownUp6(x)
        x = F.interpolate(x, (a_1.shape[2], a_1.shape[3]), mode='bilinear')
        x = torch.cat((x, a_1), dim=1)
        x = self.conv1d_4(x)
        x = self.conv2DT_BA1(x)
        x3 = x
        
        x = F.interpolate(x, (a_0.shape[2], a_0.shape[3]), mode='bilinear')
        
        x = torch.cat((x, a_0), dim=1)
        x = self.conv1d_5(x)
        x = self.conv2DT_BA2(x)

        xleft1 = F.interpolate(xleft1, (x.shape[2], x.shape[3]), mode='bilinear')
        x = torch.cat((x, xleft1), dim=1)
        x = self.conv1d_6(x)
        seg_branch2 = self.Conv2DownUp7(x)
        seg_branch2 = self.branchConv(seg_branch2)
        seg_branch2 = F.log_softmax(seg_branch2, dim=1)
        # seg_branch2 = F.softmax(seg_branch2, dim=1)
        seg_branch2 = F.interpolate(seg_branch2, (input_a.shape[2], input_a.shape[3]), mode='bilinear')
        seg_branch2 = 0.9*seg_branch2 + 0.1*seg_branch
        # seg_branch2 = torch.clamp(seg_branch2, epsilon, 1-epsilon)
        # seg_branch2 = torch.log(seg_branch2)

        y4 = torch.cat((a_pyramidB_0, b_pyramidB_0), dim=1)
        # y4 = torch.cat((a_0, b_0), dim=1)
        y4 = self.conv1d_9(y4)
        y = F.interpolate(y, scale_factor=4)

        y = F.interpolate(y, (y4.shape[2], y4.shape[3]), mode='bilinear')
        y = torch.cat((y4, y), dim=1)
    
        y5 = self.Conv2DownUp8(x3)
        y = F.interpolate(y, (y5.shape[2], y5.shape[3]), mode='bilinear')

        y = torch.cat((y5, y), dim=1)
        y = self.Conv2DownUp9(y)
        y = F.interpolate(y, scale_factor=2)

        xleft3 = F.interpolate(xleft3, (y.shape[2], y.shape[3]), mode='bilinear')
        disp_out2 = torch.cat((y, xleft3), dim=1)
        
        disp_out2 = self.conv1d_8(disp_out2)
        disp_out2 = self.Conv2DownUp10(disp_out2)
        disp_out2 = F.interpolate(disp_out2, (input_a.shape[2], input_a.shape[3]), mode='bilinear')
        disp_out2 = 0.8*disp_out2 + 0.2*disp_out
                
        # # print(x3.shape)
        # print(disp_out2.shape)
        # input()

        # out1 = torch.cat((a_pyramidB_0, b_pyramidB_0), dim=1)

        # out1 = self.conv2d_out1(corr1)
        # out1 = self.tanh(self.deconv2d_out1(out1))
        
        # out1 = F.interpolate(out1, (input_a.size()[2], input_a.size()[3]), mode='bilinear')
        

        # out3 = self.conv2d_out3(a_4)
        # out3 = self.tanh(self.deconv2d_out3(out3))
        # out3 = F.interpolate(out3, (input_a.size()[2], input_a.size()[3]), mode='bilinear')
        return seg_branch, disp_out, seg_branch2, disp_out2#out1, out3


class piramidNet(nn.Module):
    def __init__(self, pretrained=False):
        super(piramidNet, self).__init__()
        #self.resnet_features = resnet.resnet50(pretrained=pretrained)
        self.resnet_features = densenet121(pretrained=pretrained)
        pool_val = [128, 64, 32, 16, 8]
        self.branch0_0 = nn.Sequential(nn.AvgPool2d(pool_val[0], pool_val[0]),
                                     convbn(64, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch0_1 = nn.Sequential(nn.AvgPool2d(pool_val[1], pool_val[1]),
                                     convbn(64, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch0_2 = nn.Sequential(nn.AvgPool2d(pool_val[2], pool_val[2]),
                                     convbn(64, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch0_3 = nn.Sequential(nn.AvgPool2d(pool_val[3], pool_val[3]),
                                     convbn(64, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch0_4 = nn.Sequential(nn.AvgPool2d(pool_val[4], pool_val[4]),
                                     convbn(64, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))


        self.branch1_0 = nn.Sequential(nn.AvgPool2d(pool_val[2], pool_val[2]),
                                     convbn(256, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch1_1 = nn.Sequential(nn.AvgPool2d(pool_val[3], pool_val[3]),
                                     convbn(256, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch1_2 = nn.Sequential(nn.AvgPool2d(pool_val[4], pool_val[4]),
                                     convbn(256, 32, 3, 1, 'same', 1),
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
        b0 = torch.cat((out_0, b0_0, b0_1, b0_2, b0_3, b0_4), dim=1)

        # print('out1: ',  out_0.shape, out_1.shape, out_2.shape, out_3.shape, out_4.shape)
        b2_0 = self.branch1_0(out_2)
        b2_0 = F.interpolate(b2_0, (out_2.size()[2], out_2.size()[3]), mode=mode)
        
        b2_1 = self.branch1_1(out_2)
        b2_1 = F.interpolate(b2_1, (out_2.size()[2], out_2.size()[3]), mode=mode)
        
        b2_2 = self.branch1_2(out_2)
        b2_2 = F.interpolate(b2_2, (out_2.size()[2], out_2.size()[3]), mode=mode)
        b2 = torch.cat((out_2, b2_0, b2_1, b2_2), dim=1)
        
        return out_0, out_1, out_2, out_3, out_4, b2, b0
        
        
    
        
class dsnetv2(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False):
        super(dsnetv2, self).__init__()
        self.patch_type = patch_type
        self.include_edges = include_edges
        max_disp = 8
        self.resnet_features = piramidNet(pretrained=pretrained)
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
        self.segNet = segNet(1024*2, 1, labels)
        self.Conv2DownUp3 = Conv2DownUp(32, 128, 3)
        self.Conv2DownUp4 = Conv2DownUp(128*2, 64, 3)
        
        self.conv1d_2 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = Conv2DownUp(64, 64, 5, lastLayer=False)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp6 = Conv2DownUp(64, 64, 5)
        self.conv1d_4 = nn.Sequential(conv2dSame(192,64,1, padding='same'), nn.ReLU(inplace=True))
        # deconvbn(in_channel, out_channel, kernel_size, stride, pad, dilation, batchnorm=True)
        self.conv2DT_BA1 = nn.Sequential(deconvbn(64, 32, 3, 2, 'same', 1), nn.ReLU(inplace=True))
        self.conv1d_5 = nn.Sequential(conv2dSame(96,32,1, padding='same'), nn.ReLU(inplace=True))
        self.conv2DT_BA2 = nn.Sequential(deconvbn(32, 32, 3, 2, 'same', 1), nn.ReLU(inplace=True))
        self.conv1d_6 = nn.Sequential(conv2dSame(33,32,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp7 =Conv2DownUp(32, 32, 5, lastLayer=False)
        self.branchConv = ConvTranspose2dSame(32, labels, 5, padding='same', init_he=False)
        self.conv1d_9 = nn.Sequential(conv2dSame(448, 128, 1, padding='same'), nn.ReLU(inplace=True))
        #self.conv1d_9 = nn.Sequential(conv2dSame(128, 128, 1, padding='same'), nn.ReLU(inplace=True))
        self.conv1d_7 = nn.Sequential(conv2dSame(128*2,128,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp8 = Conv2DownUp(32, 64, 3)
        self.Conv2DownUp9 = Conv2DownUp(256, 64, 3)
        self.conv1d_8 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp10 = nn.Sequential(Conv2DownUp(64, 64, 5, lastLayer=False), ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)
        # for m in self.modules():
        #     print(m)
        #     print('x'*100)
        #     if isinstance(m, nn.Conv2d):
        #         print(m)
        #     #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     #     m.weight.data.normal_(0, math.sqrt(2. / n))
        #     # elif isinstance(m, nn.Conv3d):
        #     #     n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
        #     #     m.weight.data.normal_(0, math.sqrt(2. / n))
        #     # elif isinstance(m, nn.BatchNorm2d):
        #     #     m.weight.data.fill_(1)
        #     #     m.bias.data.zero_()
        #     # elif isinstance(m, nn.BatchNorm3d):
        #     #     m.weight.data.fill_(1)
        #     #     m.bias.data.zero_()
        #     # elif isinstance(m, nn.Linear):
        #     #     m.bias.data.zero_()
    


    def forward(self, input_a, input_b):

        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
        else:
            left = input_a
            right = input_b
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_2, a_pyramidB_0 = self.resnet_features(input_a) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_2, b_pyramidB_0 = self.resnet_features(input_b)
        
        # densenet
        # (None, 64, 128, 128)
        # (None, 128, 64, 64)
        # (None, 256, 32, 32)
        # (None, 512, 16, 16)
        # (None, 1024, 8, 8)
        
        xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        xleft1 = self.conv2d_ba2(input_a)
        xleft0 = self.conv2d_ba0(input_a)

        x = torch.cat((a_4, b_4), dim=1)
        x, x1, seg_branch = self.segNet(x, input_a, input_b, xleft0)
        # epsilon = 1e-9
        # seg_branch11 = F.softmax(seg_branch, dim=1)
        # seg_branch = torch.clamp(seg_branch11, epsilon, 1-epsilon)
        # seg_branch = torch.log(seg_branch)
        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), dim=1)
        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        b, ph, pw, h, w = y.size()
        y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        # y = torch.cat((a_2, b_2), dim=1)
        #print(y.shape)
        y = self.corrConv2d(y)
        #print(y.shape)
        y1 = self.Conv2DownUp3(x1)
        #print(y1.shape)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        #print(y1.shape)
        y = torch.cat((y1, y), dim=1)
        #print(y.shape)
        y = self.Conv2DownUp4(y)
        #print(y.shape)
        
        y2 = F.interpolate(y, scale_factor=8)
        #print(y2.shape)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), dim=1)
        #print(disp_out.shape)
        disp_out = self.conv1d_2(disp_out)
        #print(disp_out.shape)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = self.dispoutConv(disp_out)
        #print(disp_out.shape)
        disp_out = F.interpolate(disp_out, size=(input_a.shape[2], input_a.shape[3]), mode='bilinear')
        #print(disp_out.shape)
        #input()

        x = F.interpolate(x, scale_factor=4)
        y3 = F.interpolate(y, scale_factor=2)
        x = F.interpolate(x, (y3.shape[2], y3.shape[3]), mode='bilinear')

        x = torch.cat((x, y3), dim=1)
        x = self.conv1d_3(x)
        x = self.Conv2DownUp6(x)
        x = F.interpolate(x, (a_1.shape[2], a_1.shape[3]), mode='bilinear')
        x = torch.cat((x, a_1), dim=1)
        x = self.conv1d_4(x)
        x = self.conv2DT_BA1(x)
        x3 = x
        
        x = F.interpolate(x, (a_0.shape[2], a_0.shape[3]), mode='bilinear')
        
        x = torch.cat((x, a_0), dim=1)
        x = self.conv1d_5(x)
        x = self.conv2DT_BA2(x)

        xleft1 = F.interpolate(xleft1, (x.shape[2], x.shape[3]), mode='bilinear')
        x = torch.cat((x, xleft1), dim=1)
        x = self.conv1d_6(x)
        seg_branch2 = self.Conv2DownUp7(x)
        seg_branch2 = self.branchConv(seg_branch2)
        seg_branch2 = F.log_softmax(seg_branch2, dim=1)
        # seg_branch2 = F.softmax(seg_branch2, dim=1)
        seg_branch2 = F.interpolate(seg_branch2, (input_a.shape[2], input_a.shape[3]), mode='bilinear')
        seg_branch2 = 0.9*seg_branch2 + 0.1*seg_branch
        # seg_branch2 = torch.clamp(seg_branch2, epsilon, 1-epsilon)
        # seg_branch2 = torch.log(seg_branch2)

        y4 = torch.cat((a_pyramidB_0, b_pyramidB_0), dim=1)
        # y4 = torch.cat((a_0, b_0), dim=1)
        y4 = self.conv1d_9(y4)
        y = F.interpolate(y, scale_factor=4)

        y = F.interpolate(y, (y4.shape[2], y4.shape[3]), mode='bilinear')
        y = torch.cat((y4, y), dim=1)
    
        y5 = self.Conv2DownUp8(x3)
        y = F.interpolate(y, (y5.shape[2], y5.shape[3]), mode='bilinear')

        y = torch.cat((y5, y), dim=1)
        y = self.Conv2DownUp9(y)
        y = F.interpolate(y, scale_factor=2)

        xleft3 = F.interpolate(xleft3, (y.shape[2], y.shape[3]), mode='bilinear')
        disp_out2 = torch.cat((y, xleft3), dim=1)
        
        disp_out2 = self.conv1d_8(disp_out2)
        disp_out2 = self.Conv2DownUp10(disp_out2)
        disp_out2 = F.interpolate(disp_out2, (input_a.shape[2], input_a.shape[3]), mode='bilinear')
        disp_out2 = 0.8*disp_out2 + 0.2*disp_out
                
        # # print(x3.shape)
        # print(disp_out2.shape)
        # input()

        # out1 = torch.cat((a_pyramidB_0, b_pyramidB_0), dim=1)

        # out1 = self.conv2d_out1(corr1)
        # out1 = self.tanh(self.deconv2d_out1(out1))
        
        # out1 = F.interpolate(out1, (input_a.size()[2], input_a.size()[3]), mode='bilinear')
        

        # out3 = self.conv2d_out3(a_4)
        # out3 = self.tanh(self.deconv2d_out3(out3))
        # out3 = F.interpolate(out3, (input_a.size()[2], input_a.size()[3]), mode='bilinear')
        return seg_branch, disp_out, seg_branch2, disp_out2#out1, out3        
        


class dsnetnoCorr(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False):
        super(dsnetnoCorr, self).__init__()
        
        max_disp = 8
        self.resnet_features = piramidNet(pretrained=pretrained)
        self.conv2d_ba1 = nn.Sequential(convbn(3, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True)) 
        self.conv2d_ba2 = nn.Sequential(convbn(3, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True)) 
        self.conv2d_ba3 = nn.Sequential(convbn(3, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True)) 

        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1,
                                                            patch_size=(max_disp*2 + 1, max_disp*2 + 1),
                                                            stride=1,
                                                            padding=0,#max_disp*2 + 1,
                                                            dilation_patch=1)
        self.corrConv2d = nn.Sequential(conv2dSame(512, 128, 1, padding='same'), nn.ReLU(inplace=True))
        # self.corrConv2d = nn.Sequential(conv2dSame(512, 128, 1, padding='same'), nn.ReLU(inplace=True))
        self.conv1d_1 =nn.Sequential(conv2dSame(1024*2, 64, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp1 = Conv2DownUp(64, 32, 3)
        self.Conv2DownUp2 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False), ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
        self.Conv2DownUp3 = Conv2DownUp(32, 128, 3)
        self.Conv2DownUp4 = Conv2DownUp(128*2, 64, 3)
        
        self.conv1d_2 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = Conv2DownUp(64, 64, 5, lastLayer=False)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp6 = Conv2DownUp(64, 64, 5)
        self.conv1d_4 = nn.Sequential(conv2dSame(192,64,1, padding='same'), nn.ReLU(inplace=True))
        # deconvbn(in_channel, out_channel, kernel_size, stride, pad, dilation, batchnorm=True)
        self.conv2DT_BA1 = nn.Sequential(deconvbn(64, 32, 3, 2, 'same', 1), nn.ReLU(inplace=True))
        self.conv1d_5 = nn.Sequential(conv2dSame(96,32,1, padding='same'), nn.ReLU(inplace=True))
        self.conv2DT_BA2 = nn.Sequential(deconvbn(32, 32, 3, 2, 'same', 1), nn.ReLU(inplace=True))
        self.conv1d_6 = nn.Sequential(conv2dSame(33,32,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp7 =Conv2DownUp(32, 32, 5, lastLayer=False)
        self.branchConv = ConvTranspose2dSame(32, labels, 5, padding='same', init_he=False)
        self.conv1d_9 = nn.Sequential(conv2dSame(448, 128, 1, padding='same'), nn.ReLU(inplace=True))
        #self.conv1d_9 = nn.Sequential(conv2dSame(128, 128, 1, padding='same'), nn.ReLU(inplace=True))
        self.conv1d_7 = nn.Sequential(conv2dSame(128*2,128,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp8 = Conv2DownUp(32, 64, 3)
        self.Conv2DownUp9 = Conv2DownUp(256, 64, 3)
        self.conv1d_8 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp10 = nn.Sequential(Conv2DownUp(64, 64, 5, lastLayer=False), ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)
        # for m in self.modules():
        #     print(m)
        #     print('x'*100)
        #     if isinstance(m, nn.Conv2d):
        #         print(m)
        #     #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     #     m.weight.data.normal_(0, math.sqrt(2. / n))
        #     # elif isinstance(m, nn.Conv3d):
        #     #     n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
        #     #     m.weight.data.normal_(0, math.sqrt(2. / n))
        #     # elif isinstance(m, nn.BatchNorm2d):
        #     #     m.weight.data.fill_(1)
        #     #     m.bias.data.zero_()
        #     # elif isinstance(m, nn.BatchNorm3d):
        #     #     m.weight.data.fill_(1)
        #     #     m.bias.data.zero_()
        #     # elif isinstance(m, nn.Linear):
        #     #     m.bias.data.zero_()
    


    def forward(self, input_a, input_b):
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_2, a_pyramidB_0 = self.resnet_features(input_a) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_2, b_pyramidB_0 = self.resnet_features(input_b)
        
        # densenet
        # (None, 64, 128, 128)
        # (None, 128, 64, 64)
        # (None, 256, 32, 32)
        # (None, 512, 16, 16)
        # (None, 1024, 8, 8)
        
        xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        xleft1 = self.conv2d_ba2(input_a)

        x = torch.cat((a_4, b_4), dim=1)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1d_1(x) #self.conv1d_1 = nn.Conv2d(1024*2, 64, 1)
        x = self.Conv2DownUp1(x)
        x1 = F.interpolate(x, scale_factor=2, mode='nearest')
        seg_branch = self.Conv2DownUp2(x1)
        seg_branch = F.interpolate(seg_branch, scale_factor=8, mode='nearest')
        seg_branch = F.interpolate(seg_branch, size=(input_a.size()[2], input_a.size()[3]), mode='bilinear')
        seg_branch = F.log_softmax(seg_branch, dim=1)
        # epsilon = 1e-9
        # seg_branch11 = F.softmax(seg_branch, dim=1)
        # seg_branch = torch.clamp(seg_branch11, epsilon, 1-epsilon)
        # seg_branch = torch.log(seg_branch)
        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), dim=1)
        #y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        #b, ph, pw, h, w = y.size()
        #y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        y = torch.cat((a_2, b_2), dim=1)
        #print(y.shape)
        y = self.corrConv2d(y)
        #print(y.shape)
        y1 = self.Conv2DownUp3(x1)
        #print(y1.shape)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        #print(y1.shape)
        y = torch.cat((y1, y), dim=1)
        #print(y.shape)
        y = self.Conv2DownUp4(y)
        #print(y.shape)
        
        y2 = F.interpolate(y, scale_factor=8)
        #print(y2.shape)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), dim=1)
        #print(disp_out.shape)
        disp_out = self.conv1d_2(disp_out)
        #print(disp_out.shape)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = self.dispoutConv(disp_out)
        #print(disp_out.shape)
        disp_out = F.interpolate(disp_out, size=(input_a.shape[2], input_a.shape[3]), mode='bilinear')
        #print(disp_out.shape)
        #input()

        x = F.interpolate(x, scale_factor=4)
        y3 = F.interpolate(y, scale_factor=2)
        x = F.interpolate(x, (y3.shape[2], y3.shape[3]), mode='bilinear')

        x = torch.cat((x, y3), dim=1)
        x = self.conv1d_3(x)
        x = self.Conv2DownUp6(x)
        x = F.interpolate(x, (a_1.shape[2], a_1.shape[3]), mode='bilinear')
        x = torch.cat((x, a_1), dim=1)
        x = self.conv1d_4(x)
        x = self.conv2DT_BA1(x)
        x3 = x
        
        x = F.interpolate(x, (a_0.shape[2], a_0.shape[3]), mode='bilinear')
        
        x = torch.cat((x, a_0), dim=1)
        x = self.conv1d_5(x)
        x = self.conv2DT_BA2(x)

        xleft1 = F.interpolate(xleft1, (x.shape[2], x.shape[3]), mode='bilinear')
        x = torch.cat((x, xleft1), dim=1)
        x = self.conv1d_6(x)
        seg_branch2 = self.Conv2DownUp7(x)
        seg_branch2 = self.branchConv(seg_branch2)
        seg_branch2 = F.log_softmax(seg_branch2, dim=1)
        # seg_branch2 = F.softmax(seg_branch2, dim=1)
        seg_branch2 = F.interpolate(seg_branch2, (input_a.shape[2], input_a.shape[3]), mode='bilinear')
        seg_branch2 = 0.9*seg_branch2 + 0.1*seg_branch
        # seg_branch2 = torch.clamp(seg_branch2, epsilon, 1-epsilon)
        # seg_branch2 = torch.log(seg_branch2)

        y4 = torch.cat((a_pyramidB_0, b_pyramidB_0), dim=1)
        # y4 = torch.cat((a_0, b_0), dim=1)
        y4 = self.conv1d_9(y4)
        y = F.interpolate(y, scale_factor=4)

        y = F.interpolate(y, (y4.shape[2], y4.shape[3]), mode='bilinear')
        y = torch.cat((y4, y), dim=1)
    
        y5 = self.Conv2DownUp8(x3)
        y = F.interpolate(y, (y5.shape[2], y5.shape[3]), mode='bilinear')

        y = torch.cat((y5, y), dim=1)
        y = self.Conv2DownUp9(y)
        y = F.interpolate(y, scale_factor=2)

        xleft3 = F.interpolate(xleft3, (y.shape[2], y.shape[3]), mode='bilinear')
        disp_out2 = torch.cat((y, xleft3), dim=1)
        
        disp_out2 = self.conv1d_8(disp_out2)
        disp_out2 = self.Conv2DownUp10(disp_out2)
        disp_out2 = F.interpolate(disp_out2, (input_a.shape[2], input_a.shape[3]), mode='bilinear')
        disp_out2 = 0.8*disp_out2 + 0.2*disp_out
                
        # # print(x3.shape)
        # print(disp_out2.shape)
        # input()

        # out1 = torch.cat((a_pyramidB_0, b_pyramidB_0), dim=1)

        # out1 = self.conv2d_out1(corr1)
        # out1 = self.tanh(self.deconv2d_out1(out1))
        
        # out1 = F.interpolate(out1, (input_a.size()[2], input_a.size()[3]), mode='bilinear')
        

        # out3 = self.conv2d_out3(a_4)
        # out3 = self.tanh(self.deconv2d_out3(out3))
        # out3 = F.interpolate(out3, (input_a.size()[2], input_a.size()[3]), mode='bilinear')
        return seg_branch, disp_out, seg_branch2, disp_out2#out1, out3


class minidsnet(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False):
        super(minidsnet, self).__init__()
        self.patch_type = patch_type
        self.include_edges = include_edges
        max_disp = 8
        self.resnet_features = piramidNet(pretrained=pretrained)
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
        self.segNet = segNet(1024*2, 1, labels)
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
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_2, a_pyramidB_0 = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_2, b_pyramidB_0 = self.resnet_features(right)

        xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        xleft1 = self.conv2d_ba2(input_a)
        xleft0 = self.conv2d_ba0(input_a)
        x = torch.cat((a_4, b_4), dim=1)
        x, x1, seg_branch = self.segNet(x, input_a, input_b, xleft0)

        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        if self.patch_type == '1dcorr':
            y = torch.squeeze(y, dim=1)   
        else:
            b, ph, pw, h, w = y.size()
            y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        
        # y = torch.cat((a_2, b_2), dim=1)
        #print(y.shape)
        y = self.corrConv2d(y)
        #print(y.shape)
        y1 = self.Conv2DownUp3(x1)
        #print(y1.shape)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        #print(y1.shape)
        y = torch.cat((y1, y), dim=1)
        #print(y.shape)
        y = self.Conv2DownUp4(y)
        #print(y.shape)
        
        y2 = F.interpolate(y, scale_factor=8)
        #print(y2.shape)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), dim=1)
        #print(disp_out.shape)
        disp_out = self.conv1d_2(disp_out)
        #print(disp_out.shape)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = self.dispoutConv(disp_out)
        #print(disp_out.shape)
        disp_out = F.interpolate(disp_out, size=(left.shape[2], left.shape[3]), mode='bilinear')

        return seg_branch, disp_out, seg_branch, disp_out#out1, out3

    
class segNet(nn.Module):
    def __init__(self, in_channels, feature_channel, labels=8, pretrained=False, dropout=0):
        super(segNet, self).__init__()
        self.conv1d_1 =nn.Sequential(conv2dSame(in_channels, 64, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp1 = Conv2DownUp(64, 32, 3, dropout=dropout)
        self.conv1d_2 =nn.Sequential(conv2dSame(32+feature_channel, 32, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp2 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False, dropout=dropout), ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
        
        
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


class minidsnetExt(nn.Module):
    def __init__(self,  CFG, labels=8, pretrained=False, patch_type='', include_edges=False, backbone='mobilenet'):
        super(minidsnetExt, self).__init__()
        dropout = CFG.dropout
        self.multiTaskLoss = CFG.multaskloss
        self.aspp_mod = CFG.aspp
        self.use_att = CFG.use_att
        max_disp = 8
        self.hanet = CFG.hanet
        self.convDeconvOut = CFG.convDeconvOut
        feature_channel = 1
        self.backbone = backbone
        self.abilation = CFG.abilation
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

        if backbone == 'dn169':
            segnet_input = 1664*2
            if self.aspp_mod == 1:
                self.aspp = build_aspp('densenet_a1', 32)
                inplane_seg2 = 256
            elif self.aspp_mod == 2:
                self.aspp = build_aspp('densenet_a3', 32)
                inplane_seg2 = 273#256*2
                feature_channel = 64
            else:
                inplane_seg2 = 512

        if backbone == 'dn201':
            segnet_input = 1920*2
            if self.aspp_mod == 1:
                self.aspp = build_aspp('densenet_a1', 32)
                inplane_seg2 = 256
            elif self.aspp_mod == 2:
                self.aspp = build_aspp('densenet_a3', 32)
                inplane_seg2 = 273#256*2
                feature_channel = 64
            else:
                inplane_seg2 = 512

        if backbone == 'dn161':
            segnet_input = 2208*2
            if self.aspp_mod == 1:
                self.aspp = build_aspp('densenet_a1', 32)
                inplane_seg2 = 256
            elif self.aspp_mod == 2:
                self.aspp = build_aspp('densenet_a3', 32)
                inplane_seg2 = 273#256*2
                feature_channel = 64
            else:
                inplane_seg2 = 640

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
            self.Conv2DownUp3 = Conv2DownUp(352, 128, 3, dropout=dropout)
        else:
            self.Conv2DownUp3 = Conv2DownUp(32, 128, 3, dropout=dropout)
        self.Conv2DownUp4 = Conv2DownUp(128*2, 64, 3, dropout=dropout)
        self.segNet = segNet(segnet_input, 1, labels, dropout=dropout)
        self.conv1d_2 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = Conv2DownUp(64, 64, 5, lastLayer=False, dropout=dropout)
        self.dispoutConv = ConvTranspose2dSame(64, 1, 5, padding='same', init_he=False)
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.conv1d_4 = nn.Sequential(conv2dSame(inplane_seg2,128,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp6 = Conv2DownUp(128, 64, 3, dropout=dropout)
        self.Conv2DownUp7 = Conv2DownUp(128, 64, 3, dropout=dropout)
        self.Conv2DownUp8 = Conv2DownUp(32, 64, 3, dropout=dropout)
        self.Conv2DownUp9 = Conv2DownUp(128, 64, 3, dropout=dropout)
        self.conv1d_at_d = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid(), nn.Dropout(p=dropout))
        self.conv1d_at_s = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid(), nn.Dropout(p=dropout))
        if 'no_dec3' in self.abilation:
            self.Conv2DownUp10 = Conv2DownUp(64, 64, 3, dropout=dropout)
        else:
            if self.use_att:
                self.Conv2DownUp10 = Conv2DownUp(128, 64, 3, dropout=dropout)
            else:
                self.Conv2DownUp10 = Conv2DownUp(192, 64, 3, dropout=dropout)
        self.conv1d_5 = nn.Sequential(conv2dSame(64+feature_channel,32,1, padding='same'), nn.ReLU(inplace=True))

        if self.convDeconvOut:
            self.Conv2DownUp11 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False))
            #self.Conv2DownUp12 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False))#, ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
            self.convOutput2 = conv2dSame(32, labels, 3, 1, padding='same')
            if self.convDeconvOut == 2:
               #, ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
                self.convOutput = ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False)
        else:    
            self.Conv2DownUp11 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False, dropout=dropout), ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
        
        if self.multiTaskLoss:
            from util.utilTorchLoss import multiTask_loss
            self.mtloss = multiTask_loss(self.multiTaskLoss)
            if self.multiTaskLoss == 2:
                self.mt_convDisp = nn.Sequential(convbn(1024, 256, 1, 1, 'same', 1), nn.ReLU(inplace=True),
                                                 conv2dSame(256, 1, 3, 1, padding='same'))
                self.mt_convSeg = nn.Sequential(convbn(1024, 256, 1, 1, 'same', 1), nn.ReLU(inplace=True),
                                                conv2dSame(256, labels, 3, 1, padding='same'))

        if self.hanet:
            from models_hanet.HANet import HANet_Conv
            self.hanet_last = HANet_Conv(64, labels, pooling='max', pos_rfactor=2, dropout_prob=0.1)
            for module in self.hanet_last.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or \
                    isinstance(module, nn.GroupNorm) or isinstance(module, nn.SyncBatchNorm):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, input_a, input_b, pos=None, disp_gt=None, seg_gt=None):
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
        else:
            left = input_a
            right = input_b
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_2, a_pyramidB_1, a_pyramidB_0 = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_2, b_pyramidB_1, b_pyramidB_0 = self.resnet_features(right)

        if self.multiTaskLoss == 2:
            disp_out = self.mt_convDisp(a_4)
            seg_branch = self.mt_convSeg(a_4)
            disp_out = F.interpolate(disp_out, size=(left.shape[2], left.shape[3]), mode='bilinear')
            seg_branch = F.interpolate(seg_branch, size=(left.shape[2], left.shape[3]), mode='nearest')
            loss_disp, loss_seg1, loss_seg2 = self.mtloss(disp_out, disp_gt, seg_branch, seg_branch, seg_gt)
            return seg_branch, disp_out, seg_branch, disp_out, loss_disp, loss_seg1, loss_seg2
        # densenet
        # (None, 64, 128, 128)
        # (None, 128, 64, 64)
        # (None, 256, 32, 32)
        # (None, 512, 16, 16)
        # (None, 1024, 8, 8)
        
        xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        xleft1 = self.conv2d_ba2(input_a)
        xleft0 = self.conv2d_ba0(input_a)
        if self.backbone == 'resnet50' or self.backbone == 'resnet101':
            a_4 = self.aspp_4(a_4)
            b_4 = self.aspp_4(b_4)

        x = torch.cat([a_4, b_4], dim=1)
        x, x1, seg_branch = self.segNet(x, input_a, input_b, xleft0)
        
        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), dim=1)
        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        if self.patch_type == '1dcorr':
            y = torch.squeeze(y, dim=1)   
        else:
            b, ph, pw, h, w = y.size()
            y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        
        # y = torch.cat((a_2, b_2), dim=1)
        #print(y.shape)
        y = self.corrConv2d(y)
        #print(y.shape)
        if 'no_dec1' in self.abilation:
            y1 = self.Conv2DownUp3(a_pyramidB_2)
        else:
            y1 = self.Conv2DownUp3(x1)
        #print(y1.shape)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        #print(y1.shape)
        y = torch.cat((y1, y), dim=1)
        #print(y.shape)
        y = self.Conv2DownUp4(y)
        #print(y.shape)
        
        y2 = F.interpolate(y, scale_factor=8)
        #print(y2.shape)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), dim=1)
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
            #s2 = torch.cat((s2_1, s2_2), dim=1)
            s2_corr = self.s2_corr_sampler(s2_1, s2_2)
            s2_corr = torch.squeeze(s2_corr, dim=1) 
            s2 = torch.cat((s2_corr, s2_1), dim=1)
        else:
            s2 = torch.cat((a_pyramidB_1, b_pyramidB_1), dim=1)

        
        s2 = self.conv1d_4(s2)
        s2 = self.Conv2DownUp6(s2)      

        y3 = F.interpolate(y, size=(s2.shape[2], s2.shape[3]))

        if not 'no_dec3' in self.abilation:
            if self.use_att:
                s2_d = torch.cat((s2, y3), dim=1)
                s2_d = self.Conv2DownUp7(s2_d)
                at_d = self.conv1d_at_d(s2_d)

                x3 = self.Conv2DownUp8(x1)
                x3 = F.interpolate(x3, size=(s2.shape[2], s2.shape[3]))

                s2_s = torch.cat((s2, x3), dim=1)
                s2_s = self.Conv2DownUp9(s2_s)
                at_s = self.conv1d_at_s(s2_s)

                s2 = torch.cat((s2_d*at_s, s2_s*at_d), dim=1)
            else:
                x3 = self.Conv2DownUp8(x1)
                x3 = F.interpolate(x3, size=(s2.shape[2], s2.shape[3]))
                s2 = torch.cat((s2, x3, y3), dim=1)
        s2 = self.Conv2DownUp10(s2)
        
        if self.aspp_mod == 2:
            s2 = F.interpolate(s2, size=(a_0.shape[2], a_0.shape[3]))
            s2 = torch.cat((s2, a_0), dim=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]), mode='nearest')
        else:
            s2 = F.interpolate(s2, size=(xleft1.shape[2], xleft1.shape[3]))
            s2 = torch.cat([s2, xleft1], dim=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            
            if self.convDeconvOut:
                seg_branch2_2 = self.convOutput2(seg_branch2)
                if self.convDeconvOut == 2:
                    seg_branch2_1 = self.convOutput(seg_branch2)
                else:
                    seg_branch2_1 = 0
                    
                seg_branch2 = seg_branch2_1 + seg_branch2_2
            

            ##hanet
            if self.hanet:
                seg_branch2, attention = self.hanet_last(a_0, seg_branch2, pos, return_attention=False, return_posmap=False, attention_loss=True)
            # import matplotlib.pyplot as plt
            # plt.imshow(attention.detach().cpu().numpy()[0].transpose())
            # plt.show()
            ###end hanet

        if self.multiTaskLoss:
            loss_disp, loss_seg1, loss_seg2 = self.mtloss(disp_out, disp_gt, seg_branch, seg_branch2, seg_gt)
            return seg_branch, disp_out, seg_branch2, disp_out, loss_disp, loss_seg1, loss_seg2
        else:
            return seg_branch, disp_out, seg_branch2, disp_out#out1, out3



class minidsnetExtPiramid(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False, backbone='mobilenet', ):
        super(minidsnetExtPiramid, self).__init__()
        self.aspp_mod = CFG.aspp
        max_disp = 8
        feature_channel = 1
        self.backbone = backbone
        if backbone == 'densenet':
            segnet_input = 1024*2
            feature_channel = 224
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
            feature_channel = 176

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
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
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
        
        xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        xleft1 = self.conv2d_ba2(input_a)
        xleft0 = self.conv2d_ba0(input_a)
        if self.backbone == 'resnet50' or self.backbone == 'resnet101':
            a_4 = self.aspp_4(a_4)
            b_4 = self.aspp_4(b_4)

        x = torch.cat((a_4, b_4), dim=1)
        x, x1, seg_branch = self.segNet(x, input_a, input_b, xleft0)
        
        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), dim=1)
        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        if self.patch_type == '1dcorr':
            y = torch.squeeze(y, dim=1)   
        else:
            b, ph, pw, h, w = y.size()
            y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        
        # y = torch.cat((a_2, b_2), dim=1)
        #print(y.shape)
        y = self.corrConv2d(y)
        #print(y.shape)
        y1 = self.Conv2DownUp3(x1)
        #print(y1.shape)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        #print(y1.shape)
        y = torch.cat((y1, y), dim=1)
        #print(y.shape)
        y = self.Conv2DownUp4(y)
        #print(y.shape)
        
        y2 = F.interpolate(y, scale_factor=8)
        #print(y2.shape)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), dim=1)
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
            #s2 = torch.cat((s2_1, s2_2), dim=1)
            s2_corr = self.s2_corr_sampler(s2_1, s2_2)
            s2_corr = torch.squeeze(s2_corr, dim=1) 
            s2 = torch.cat((s2_corr, s2_1), dim=1)
        else:
            s2 = torch.cat((a_pyramidB_1, b_pyramidB_1), dim=1)

        
        s2 = self.conv1d_4(s2)
        s2 = self.Conv2DownUp6(s2)      

        y3 = F.interpolate(y, size=(s2.shape[2], s2.shape[3]))

        s2_d = torch.cat((s2, y3), dim=1)
        s2_d = self.Conv2DownUp7(s2_d)
        at_d = self.conv1d_at_d(s2_d)

        x3 = self.Conv2DownUp8(x1)
        x3 = F.interpolate(x3, size=(s2.shape[2], s2.shape[3]))

        s2_s = torch.cat((s2, x3), dim=1)
        s2_s = self.Conv2DownUp9(s2_s)
        at_s = self.conv1d_at_s(s2_s)

        s2 = torch.cat((s2_d*at_s, s2_s*at_d), dim=1)
        s2 = self.Conv2DownUp10(s2)
        
        if self.aspp_mod == 2:
            s2 = F.interpolate(s2, size=(a_0.shape[2], a_0.shape[3]))
            s2 = torch.cat((s2, a_0), dim=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]), mode='nearest')
        else:
            s2 = F.interpolate(s2, size=(a_pyramidB_0.shape[2], a_pyramidB_0.shape[3]))
            s2 = torch.cat((s2, a_pyramidB_0), dim=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]), mode='nearest')

        #seg_branch2 = F.log_softmax(seg_branch2, dim=1)
        # x = F.interpolate(x, scale_factor=4)
        # y3 = F.interpolate(y, scale_factor=2)
        # x = F.interpolate(x, (y3.shape[2], y3.shape[3]), mode='bilinear')

        # x = torch.cat((x, y3), dim=1)
        # x = self.conv1d_3(x)
        # x = self.Conv2DownUp6(x)
        # x = F.interpolate(x, (a_1.shape[2], a_1.shape[3]), mode='bilinear')
        # x = torch.cat((x, a_1), dim=1)
        # x = self.conv1d_4(x)
        # x = self.conv2DT_BA1(x)
        # x3 = x
        
        # x = F.interpolate(x, (a_0.shape[2], a_0.shape[3]), mode='bilinear')
        
        # x = torch.cat((x, a_0), dim=1)
        # x = self.conv1d_5(x)
        # x = self.conv2DT_BA2(x)

        # xleft1 = F.interpolate(xleft1, (x.shape[2], x.shape[3]), mode='bilinear')
        # x = torch.cat((x, xleft1), dim=1)
        # x = self.conv1d_6(x)
        # seg_branch2 = self.Conv2DownUp7(x)
        # seg_branch2 = self.branchConv(seg_branch2)
        # seg_branch2 = F.log_softmax(seg_branch2, dim=1)
        # # seg_branch2 = F.softmax(seg_branch2, dim=1)
        # seg_branch2 = F.interpolate(seg_branch2, (input_a.shape[2], input_a.shape[3]), mode='bilinear')
        # seg_branch2 = 0.9*seg_branch2 + 0.1*seg_branch
        # # seg_branch2 = torch.clamp(seg_branch2, epsilon, 1-epsilon)
        # # seg_branch2 = torch.log(seg_branch2)

        # y4 = torch.cat((a_pyramidB_0, b_pyramidB_0), dim=1)
        # # y4 = torch.cat((a_0, b_0), dim=1)
        # y4 = self.conv1d_9(y4)
        # y = F.interpolate(y, scale_factor=4)

        # y = F.interpolate(y, (y4.shape[2], y4.shape[3]), mode='bilinear')
        # y = torch.cat((y4, y), dim=1)
    
        # y5 = self.Conv2DownUp8(x3)
        # y = F.interpolate(y, (y5.shape[2], y5.shape[3]), mode='bilinear')

        # y = torch.cat((y5, y), dim=1)
        # y = self.Conv2DownUp9(y)
        # y = F.interpolate(y, scale_factor=2)

        # xleft3 = F.interpolate(xleft3, (y.shape[2], y.shape[3]), mode='bilinear')
        # disp_out2 = torch.cat((y, xleft3), dim=1)
        
        # disp_out2 = self.conv1d_8(disp_out2)
        # disp_out2 = self.Conv2DownUp10(disp_out2)
        # disp_out2 = F.interpolate(disp_out2, (input_a.shape[2], input_a.shape[3]), mode='bilinear')
        # disp_out2 = 0.8*disp_out2 + 0.2*disp_out
                
        # # print(x3.shape)
        # print(disp_out2.shape)
        # input()

        # out1 = torch.cat((a_pyramidB_0, b_pyramidB_0), dim=1)

        # out1 = self.conv2d_out1(corr1)
        # out1 = self.tanh(self.deconv2d_out1(out1))
        
        # out1 = F.interpolate(out1, (input_a.size()[2], input_a.size()[3]), mode='bilinear')
        

        # out3 = self.conv2d_out3(a_4)
        # out3 = self.tanh(self.deconv2d_out3(out3))
        # out3 = F.interpolate(out3, (input_a.size()[2], input_a.size()[3]), mode='bilinear')
        return seg_branch, disp_out, seg_branch2, disp_out#out1, out3


class minidsnetExt2(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False, backbone='mobilenet', ):
        super(minidsnetExt2, self).__init__()
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
        self.conv1d_at = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid())
        
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
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
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
        
        xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        xleft1 = self.conv2d_ba2(input_a)
        xleft0 = self.conv2d_ba0(input_a)
        if self.backbone == 'resnet50' or self.backbone == 'resnet101':
            a_4 = self.aspp_4(a_4)
            b_4 = self.aspp_4(b_4)

        x = torch.cat((a_4, b_4), dim=1)
        x, x1, seg_branch = self.segNet(x, input_a, input_b, xleft0)
        
        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), dim=1)
        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        if self.patch_type == '1dcorr':
            y = torch.squeeze(y, dim=1)   
        else:
            b, ph, pw, h, w = y.size()
            y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        
        # y = torch.cat((a_2, b_2), dim=1)
        #print(y.shape)
        y = self.corrConv2d(y)
        #print(y.shape)
        y1 = self.Conv2DownUp3(x1)
        #print(y1.shape)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        #print(y1.shape)
        y = torch.cat((y1, y), dim=1)
        #print(y.shape)
        y = self.Conv2DownUp4(y)
        #print(y.shape)
        
        y2 = F.interpolate(y, scale_factor=8)
        #print(y2.shape)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), dim=1)
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
            #s2 = torch.cat((s2_1, s2_2), dim=1)
            s2_corr = self.s2_corr_sampler(s2_1, s2_2)
            s2_corr = torch.squeeze(s2_corr, dim=1) 
            s2 = torch.cat((s2_corr, s2_1), dim=1)
        else:
            s2 = torch.cat((a_pyramidB_1, b_pyramidB_1), dim=1)

        
        s2 = self.conv1d_4(s2)
        s2 = self.Conv2DownUp6(s2)      

        y3 = F.interpolate(y, size=(s2.shape[2], s2.shape[3]))

        s2_d = torch.cat((s2, y3), dim=1)
        s2_d = self.Conv2DownUp7(s2_d)
        #at_d = self.conv1d_at_d(s2_d)

        x3 = self.Conv2DownUp8(x1)
        x3 = F.interpolate(x3, size=(s2.shape[2], s2.shape[3]))

        s2_s = torch.cat((s2, x3), dim=1)
        s2_s = self.Conv2DownUp9(s2_s)
        #at_s = self.conv1d_at_s(s2_s)

        s2_at = self.conv1d_at(s2)
        s2 = torch.cat((s2_d*s2_at, s2_s*(1-s2_at)), dim=1)
        s2 = self.Conv2DownUp10(s2)
        
        if self.aspp_mod == 2:
            s2 = F.interpolate(s2, size=(a_0.shape[2], a_0.shape[3]))
            s2 = torch.cat((s2, a_0), dim=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]), mode='nearest')
        else:
            s2 = F.interpolate(s2, size=(xleft1.shape[2], xleft1.shape[3]))
            s2 = torch.cat((s2, xleft1), dim=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)

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
        if backbone == 'dn169':
            self.resnet_features = densenet169(pretrained=pretrained)
            in_plane = [64, 128, 256]
            # torch.Size([2, 64, 128, 128])
            # torch.Size([2, 128, 64, 64])
            # torch.Size([2, 256, 32, 32])
            # torch.Size([2, 640, 16, 16])
            # torch.Size([2, 1664, 8, 8])

        if backbone == 'dn201':
            self.resnet_features = densenet201(pretrained=pretrained)
            in_plane = [64, 128, 256]
            # torch.Size([2, 64, 128, 256])
            # torch.Size([2, 128, 64, 128])
            # torch.Size([2, 256, 32, 64])
            # torch.Size([2, 896, 16, 32])
            # torch.Size([2, 1920, 8, 16])

        if backbone == 'dn161':
            self.resnet_features = densenet161(pretrained=pretrained)
            in_plane = [96, 192, 384]
            # torch.Size([2, 96, 128, 256])
            # torch.Size([2, 192, 64, 128])
            # torch.Size([2, 384, 32, 64])
            # torch.Size([2, 1056, 16, 32])
            # torch.Size([2, 2208, 8, 16])

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
        b0 = torch.cat([out_0, b0_0, b0_1, b0_2, b0_3, b0_4], dim=1)

        # print('out1: ',  out_0.shape, out_1.shape, out_2.shape, out_3.shape, out_4.shape)
        b1_0 = self.branch1_0(out_1)
        b1_0 = F.interpolate(b1_0, (out_1.size()[2], out_1.size()[3]), mode=mode)

        b1_1 = self.branch1_1(out_1)
        b1_1 = F.interpolate(b1_1, (out_1.size()[2], out_1.size()[3]), mode=mode)
        
        b1_2 = self.branch1_2(out_1)
        b1_2 = F.interpolate(b1_2, (out_1.size()[2], out_1.size()[3]), mode=mode)
        
        b1_3 = self.branch1_3(out_1)
        b1_3 = F.interpolate(b1_3, (out_1.size()[2], out_1.size()[3]), mode=mode)
        b1 = torch.cat((out_1, b1_0, b1_1, b1_2, b1_3), dim=1)
        
        b2_0 = self.branch2_0(out_2)
        b2_0 = F.interpolate(b2_0, (out_2.size()[2], out_2.size()[3]), mode=mode)
        
        b2_1 = self.branch2_1(out_2)
        b2_1 = F.interpolate(b2_1, (out_2.size()[2], out_2.size()[3]), mode=mode)
        
        b2_2 = self.branch2_2(out_2)
        b2_2 = F.interpolate(b2_2, (out_2.size()[2], out_2.size()[3]), mode=mode)
        b2 = torch.cat((out_2, b2_0, b2_1, b2_2), dim=1)
        
        return out_0, out_1, out_2, out_3, out_4, b2, b1, b0



class seg_dsnet(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False):
        super(seg_dsnet, self).__init__()
        self.patch_type = patch_type
        self.include_edges = include_edges
        max_disp = 8
        self.resnet_features = piramidNet(pretrained=pretrained)
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
        self.segNet = segNet(1024, 1, labels)
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
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_2, a_pyramidB_0 = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_2, b_pyramidB_0 = self.resnet_features(right)

        xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        xleft1 = self.conv2d_ba2(input_a)
        xleft0 = self.conv2d_ba0(input_a)
        xright0 = self.conv2d_ba0(input_b)

        x, x1, seg_branch = self.segNet(a_4, input_a, input_b, xleft0)
        _, _, seg_branch_right = self.segNet(b_4, input_a, input_b, xright0)

        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        if self.patch_type == '1dcorr':
            y = torch.squeeze(y, dim=1)   
        else:
            b, ph, pw, h, w = y.size()
            y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        
        # y = torch.cat((a_2, b_2), dim=1)
        #print(y.shape)
        y = self.corrConv2d(y)
        #print(y.shape)
        y1 = self.Conv2DownUp3(x1)
        #print(y1.shape)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        #print(y1.shape)
        y = torch.cat((y1, y), dim=1)
        #print(y.shape)
        y = self.Conv2DownUp4(y)
        #print(y.shape)
        
        y2 = F.interpolate(y, scale_factor=8)
        #print(y2.shape)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), dim=1)
        #print(disp_out.shape)
        disp_out = self.conv1d_2(disp_out)
        #print(disp_out.shape)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = self.dispoutConv(disp_out)
        #print(disp_out.shape)
        disp_out = F.interpolate(disp_out, size=(left.shape[2], left.shape[3]), mode='bilinear')
        seg_branch_right = apply_disparity(seg_branch_right, -disp_out)
        
        return seg_branch, disp_out, seg_branch_right, disp_out#out1, out3


class minidsnetExtPiramidRes(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False, backbone='mobilenet', ):
        super(minidsnetExtPiramidRes, self).__init__()
        self.aspp_mod = CFG.aspp
        max_disp = 8
        feature_channel = 1
        self.backbone = backbone
        if backbone == 'densenet':
            segnet_input = 1024*2
            feature_channel = 224
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
            feature_channel = 176

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
        self.corrConv2d = nn.Sequential(conv2dSame(out_plane_corr, 352, 1, padding='same'), nn.ReLU(inplace=True))
        # self.corrConv2d = nn.Sequential(conv2dSame(512, 128, 1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp3 = Conv2DownUp(32, 352, 3)
        self.Conv2DownUp4 = Conv2DownUp(352, 64, 3)
        self.segNet = segNet(segnet_input, 1, labels)
        self.conv1d_2 = nn.Sequential(conv2dSame(65,64,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp5 = Conv2DownUp(64, 64, 5, lastLayer=False)
        self.dispoutConv = conv2dSame(64, 1, 5, padding='same')
        self.conv1d_3 = nn.Sequential(conv2dSame(96,64,1, padding='same'), nn.ReLU(inplace=True))
        self.conv1d_4 = nn.Sequential(conv2dSame(inplane_seg2,128,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp6 = Conv2DownUp(128, 64, 3)
        self.Conv2DownUp7 = Conv2DownUp(128, 64, 3)
        self.Conv2DownUp8 = Conv2DownUp(32, 64, 3)
        self.Conv2DownUp9 = Conv2DownUp(128, 64, 3)
        self.conv1d_at_d = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid())
        self.conv1d_at_s = nn.Sequential(conv2dSame(64,1,1, padding='same'), nn.Sigmoid())
        self.Conv2DownUp10 = Conv2DownUp(64, 64, 3)
        self.conv1d_5 = nn.Sequential(conv2dSame(64+feature_channel,32,1, padding='same'), nn.ReLU(inplace=True))
        self.Conv2DownUp11 = Conv2DownUp(32, 64, 3, lastLayer=False)
        self.convSegOut = conv2dSame(64, labels, 3, 1, padding='same')
       
    def forward(self, input_a, input_b):
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
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
        
        xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        xleft1 = self.conv2d_ba2(input_a)
        xleft0 = self.conv2d_ba0(input_a)
        if self.backbone == 'resnet50' or self.backbone == 'resnet101':
            a_4 = self.aspp_4(a_4)
            b_4 = self.aspp_4(b_4)

        x = torch.cat((a_4, b_4), dim=1)
        x, x1, seg_branch = self.segNet(x, input_a, input_b, xleft0)
        
        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), dim=1)
        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        if self.patch_type == '1dcorr':
            y = torch.squeeze(y, dim=1)   
        else:
            b, ph, pw, h, w = y.size()
            y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        
        # y = torch.cat((a_2, b_2), dim=1)
        y = self.corrConv2d(y)
        y = a_pyramidB_2 + y
        #print(y.shape)
        y1 = self.Conv2DownUp3(x1)
        #print(y1.shape)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        #print(y1.shape)
        y = y + y1#torch.cat((y1, y), dim=1)
        #print(y.shape)
        y = self.Conv2DownUp4(y)
        #print(y.shape)
        
        y2 = F.interpolate(y, scale_factor=8)
        #print(y2.shape)
        xleft2 = F.interpolate(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), dim=1)
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
            #s2 = torch.cat((s2_1, s2_2), dim=1)
            s2_corr = self.s2_corr_sampler(s2_1, s2_2)
            s2_corr = torch.squeeze(s2_corr, dim=1) 
            s2 = torch.cat((s2_corr, s2_1), dim=1)
        else:
            s2 = torch.cat((a_pyramidB_1, b_pyramidB_1), dim=1)

        
        s2 = self.conv1d_4(s2)
        s2 = self.Conv2DownUp6(s2)      

        y3 = F.interpolate(y, size=(s2.shape[2], s2.shape[3]))

        s2_d = torch.cat((s2, y3), dim=1)
        s2_d = self.Conv2DownUp7(s2_d)
        at_d = self.conv1d_at_d(s2_d)

        x3 = self.Conv2DownUp8(x1)
        x3 = F.interpolate(x3, size=(s2.shape[2], s2.shape[3]))

        s2_s = torch.cat((s2, x3), dim=1)
        s2_s = self.Conv2DownUp9(s2_s)
        at_s = self.conv1d_at_s(s2_s)

        # print((x3*at_s).shape, (y3*at_d).shape, s2.shape)
        disp_seg_f = x3*at_s + y3*at_d #torch.cat((x3*at_s, y3*at_d), dim=1)
        s2 = s2 + disp_seg_f
        s2 = self.Conv2DownUp10(s2)
        
        if self.aspp_mod == 2:
            s2 = F.interpolate(s2, size=(a_0.shape[2], a_0.shape[3]))
            s2 = torch.cat((s2, a_0), dim=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = self.convSegOut(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]), mode='nearest')
        else:
            s2 = F.interpolate(s2, size=(a_pyramidB_0.shape[2], a_pyramidB_0.shape[3]))
            s2 = torch.cat((s2, a_pyramidB_0), dim=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = self.convSegOut(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]), mode='nearest')

        return seg_branch, disp_out, seg_branch2, disp_out#out1, out3



class minidsnetExt_deeplab(nn.Module):
    def __init__(self, CFG, labels=8, pretrained=False, patch_type='', include_edges=False, backbone='deeplab'):
        super(minidsnetExt_deeplab, self).__init__()
        from models_hanet.resnet_pytorch import deeplabV3plus
        from models_hanet.mynn import initialize_weights
        self.aspp_mod = CFG.aspp
        max_disp = 8
        self.hanet = CFG.hanet
        self.convDeconvOut = CFG.convDeconvOut
        feature_channel = 1
        self.backbone = backbone
        self.abilation = CFG.abilation
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


        if backbone == 'efficientnet-b3':
            segnet_input = 1536 * 2
            inplane_seg2 = 320


        if backbone == 'efficientnet-b2':
            segnet_input = 1408 * 2
            inplane_seg2 = 304


        self.patch_type = patch_type
        self.include_edges = include_edges
        
        
        #self.resnet_features = piramidNet2(pretrained, backbone)
        self.resnet_features = deeplabV3plus(labels, return_layers=True)
        
        if self.include_edges:
            aux_img_channel = 4
        else:
            aux_img_channel = 3

        

        # self.conv2d_ba0 = nn.Sequential(convbn(aux_img_channel, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True))
        # self.conv2d_ba1 = nn.Sequential(convbn(aux_img_channel, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True)) 
        # self.conv2d_ba2 = nn.Sequential(convbn(aux_img_channel, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True)) 
        # self.conv2d_ba3 = nn.Sequential(convbn(aux_img_channel, 1, 5, 1, 'same', 2), nn.ReLU(inplace=True)) 
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
        segnet_input = 256
        self.segNet = segNet(segnet_input, 48, labels)
        self.conv1d_2 = nn.Sequential(conv2dSame(112,64,1, padding='same'), nn.ReLU(inplace=True))
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
        self.conv1d_5 = nn.Sequential(conv2dSame(64+48,32,1, padding='same'), nn.ReLU(inplace=True))

        if self.convDeconvOut:
            self.Conv2DownUp11 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False))
            #self.Conv2DownUp12 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False))#, ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
            self.convOutput2 =  conv2dSame(32, labels, 3, 1, padding='same')
            if self.convDeconvOut == 2:
               #, ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
                self.convOutput =  ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False)
        else:    
            self.Conv2DownUp11 = nn.Sequential(Conv2DownUp(32, 32, 3, lastLayer=False), ConvTranspose2dSame(32, labels, 3, 1, padding='same', init_he=False))
        

        if self.hanet:
            from models_hanet.HANet import HANet_Conv
            self.hanet_last = HANet_Conv(64, labels, pooling='max', pos_rfactor=2, dropout_prob=0.1)
            for module in self.hanet_last.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or \
                    isinstance(module, nn.GroupNorm) or isinstance(module, nn.SyncBatchNorm):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
       
        initialize_weights(self.Conv2DownUp3)
        initialize_weights(self.Conv2DownUp4)
        initialize_weights(self.segNet)
        initialize_weights(self.Conv2DownUp5)
        initialize_weights(self.conv1d_2)
        initialize_weights(self.dispoutConv)
        initialize_weights(self.conv1d_3)
        initialize_weights(self.conv1d_4)
        initialize_weights(self.Conv2DownUp6)
        initialize_weights(self.Conv2DownUp7)
        initialize_weights(self.Conv2DownUp8)
        initialize_weights(self.Conv2DownUp9)
        initialize_weights(self.Conv2DownUp10)
        initialize_weights(self.Conv2DownUp11)
        initialize_weights(self.conv1d_5)
        initialize_weights(self.conv1d_at_d)
        initialize_weights(self.conv1d_at_s)

    def forward(self, input_a, input_b, pos=None):
        if self.include_edges:
            left = input_a[:,:3, :]
            right = input_b[:,:3, :]
        else:
            left = input_a
            right = input_b
        # a_0, a_1, a_2, a_3, a_4, a_pyramidB_2, a_pyramidB_1, a_pyramidB_0 = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        # b_0, b_1, b_2, b_3, b_4, b_pyramidB_2, b_pyramidB_1, b_pyramidB_0 = self.resnet_features(right)
        a_4, a_pyramidB_2, a_pyramidB_1, a_pyramidB_0 = self.resnet_features(left) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        _, b_pyramidB_2, b_pyramidB_1, b_pyramidB_0 = self.resnet_features(right)
        # densenet
        # (None, 64, 128, 128)
        # (None, 128, 64, 64)
        # (None, 256, 32, 32)
        # (None, 512, 16, 16)
        # (None, 1024, 8, 8)
        
        # xleft3 = self.conv2d_ba3(input_a)
        # xleft2 = self.conv2d_ba1(input_a)
        # xleft1 = self.conv2d_ba2(input_a)
        # xleft0 = self.conv2d_ba0(input_a)
        if self.backbone == 'resnet50' or self.backbone == 'resnet101':
            a_4 = self.aspp_4(a_4)
            b_4 = self.aspp_4(b_4)

        #x = torch.cat((a_4, b_4), dim=1)
        x = a_4
        x, x1, seg_branch = self.segNet(x, input_a, input_b, a_pyramidB_0)
        
        #y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), dim=1)
        y = self.correlation_sampler(a_pyramidB_2, b_pyramidB_2)
        if self.patch_type == '1dcorr':
            y = torch.squeeze(y, dim=1)   
        else:
            b, ph, pw, h, w = y.size()
            y = y.view(b, ph * pw, h, w)/a_pyramidB_2.size(1)
        
        # y = torch.cat((a_2, b_2), dim=1)
        #print(y.shape)
        y = self.corrConv2d(y)
        #print(y.shape)
        if 'no_dec1' in self.abilation:
            y1 = self.Conv2DownUp3(a_pyramidB_2)
        else:
            y1 = self.Conv2DownUp3(x1)
        #print(y1.shape)
        y1 = F.interpolate(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')
        #print(y1.shape)
        y = torch.cat((y1, y), dim=1)
        #print(y.shape)
        y = self.Conv2DownUp4(y)
        #print(y.shape)
        
        y2 = F.interpolate(y, scale_factor=8)
        #print(y2.shape)
        xleft2 = F.interpolate(a_pyramidB_0, size=(y2.shape[2], y2.shape[3]), mode='bilinear')
        #print(xleft2.shape)
        disp_out = torch.cat((y2, xleft2), dim=1)
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
            #s2 = torch.cat((s2_1, s2_2), dim=1)
            s2_corr = self.s2_corr_sampler(s2_1, s2_2)
            s2_corr = torch.squeeze(s2_corr, dim=1) 
            s2 = torch.cat((s2_corr, s2_1), dim=1)
        else:
            # s2 = torch.cat((a_pyramidB_1, b_pyramidB_1), dim=1)
            s2 = a_pyramidB_1
        
        s2 = self.conv1d_4(s2)
        s2 = self.Conv2DownUp6(s2)      

        y3 = F.interpolate(y, size=(s2.shape[2], s2.shape[3]))

        if not 'no_dec3' in self.abilation:
            s2_d = torch.cat((s2, y3), dim=1)
            s2_d = self.Conv2DownUp7(s2_d)
            at_d = self.conv1d_at_d(s2_d)

            x3 = self.Conv2DownUp8(x1)
            x3 = F.interpolate(x3, size=(s2.shape[2], s2.shape[3]))

            s2_s = torch.cat((s2, x3), dim=1)
            s2_s = self.Conv2DownUp9(s2_s)
            at_s = self.conv1d_at_s(s2_s)

            s2 = torch.cat((s2_d*at_s, s2_s*at_d), dim=1)
        s2 = self.Conv2DownUp10(s2)
        
        if self.aspp_mod == 2:
            s2 = F.interpolate(s2, size=(a_0.shape[2], a_0.shape[3]))
            s2 = torch.cat((s2, a_0), dim=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]), mode='nearest')
        else:
            s2 = F.interpolate(s2, size=(a_pyramidB_0.shape[2], a_pyramidB_0.shape[3]))
            s2 = torch.cat((s2, a_pyramidB_0), dim=1)
            seg_branch2 = self.conv1d_5(s2)
            seg_branch2 = self.Conv2DownUp11(seg_branch2)
            seg_branch2 = F.interpolate(seg_branch2, size=(input_a.shape[2], input_a.shape[3]))
            if self.convDeconvOut:
                seg_branch2_2 = self.convOutput2(seg_branch2)
                if self.convDeconvOut == 2:
                    seg_branch2_1 = self.convOutput(seg_branch2)
                else:
                    seg_branch2_1 = 0
                    
                seg_branch2 = seg_branch2_1 + seg_branch2_2
            

            ##hanet
            if self.hanet:
                seg_branch2, attention = self.hanet_last(a_0, seg_branch2, pos, return_attention=False, return_posmap=False, attention_loss=True)
            
        return seg_branch, disp_out, seg_branch2, disp_out#out1, out3
