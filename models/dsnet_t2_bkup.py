import torch.nn as nn
import torch.nn.functional as F
from models import resnet
from models.densenet import densenet121
from spatial_correlation_sampler import SpatialCorrelationSampler
import torch

def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation, batchnorm=True):
    layers = [nn.Conv2d(in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation = dilation,
            bias=False if batchnorm == True else True)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channel))
    return nn.Sequential(*layers) 

def deconvbn(in_channel, out_channel, kernel_size, stride, pad, dilation, batchnorm=True):
    layers = [nn.ConvTranspose2d(in_channel,
                                    out_channel,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=dilation if dilation > 1 else pad,
                                    dilation = dilation,
                                    bias=False if batchnorm == True else True)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channel))
    return nn.Sequential(*layers)

class Conv2DownUp(nn.Module):
    def __init__(self, in_channels, out_channels=3, kernel_size=3, lastLayer=True):
        super(Conv2DownUp, self).__init__()
        self.lastLayer = lastLayer
        self.c1 = nn.Sequential(convbn(in_channels, out_channels, kernel_size, 1, 0, 1), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(convbn(out_channels, out_channels, kernel_size, 1, 0, 1), nn.ReLU(inplace=True))

        self.cd3 = nn.Sequential(convbn(out_channels, out_channels, kernel_size, 1, 0, 1), nn.ReLU(inplace=True), 
                                deconvbn(out_channels, out_channels, kernel_size, 1, 0, 1), nn.ReLU(inplace=True))

        self.d4 = nn.Sequential(deconvbn(out_channels, out_channels, kernel_size, 1, 0, 1), nn.ReLU(inplace=True))
        self.d5 = nn.Sequential(deconvbn(out_channels, out_channels, kernel_size, 1, 0, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x = self.cd3(x2)
        x = x2 + x
        x = self.d4(x)
        x = x1 + x
        if not self.lastLayer:
            return x
        x = self.d5(x)
        
        return x



class dsnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21, labels=8, pretrained=False):
        super(dsnet, self).__init__()
        
        max_disp = 8
        self.resnet_features = piramidNet(pretrained=pretrained)
        self.conv2d_ba1 = nn.Sequential(convbn(3, 1, 5, 1, 0, 2), nn.ReLU(inplace=True)) 
        self.conv2d_ba2 = nn.Sequential(convbn(3, 1, 5, 1, 0, 2), nn.ReLU(inplace=True)) 
        self.conv2d_ba3 = nn.Sequential(convbn(3, 1, 5, 1, 0, 2), nn.ReLU(inplace=True)) 

        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1,
                                                            patch_size=(1, max_disp*2 + 1),
                                                            stride=1,
                                                            padding=0,#max_disp*2 + 1,
                                                            dilation_patch=1)
        self.corrConv2d = nn.Conv2d(max_disp*2 + 1, 128,1)
        self.conv1d_1 = nn.Conv2d(2048*2,64,1)
        self.Conv2DownUp1 = Conv2DownUp(64, 32, 3)
        self.Conv2DownUp2 = nn.Sequential(Conv2DownUp(32, 32, 3, False), nn.ConvTranspose2d(32, labels, 3))
        self.Conv2DownUp3 = Conv2DownUp(32, 128, 3)
        self.Conv2DownUp4 = Conv2DownUp(128*2, 64, 3)
        
        self.conv1d_2 = nn.Conv2d(65,64,1)
        self.Conv2DownUp5 = nn.Sequential(Conv2DownUp(64, 64, 5, lastLayer=False), nn.ConvTranspose2d(64, 1, 5))
        self.conv1d_3 = nn.Conv2d(96,64,1)
        self.Conv2DownUp6 = Conv2DownUp(64, 64, 5)
        self.conv1d_4 = nn.Conv2d(320,64,1)
        self.conv2DT_BA1 = nn.Sequential(deconvbn(64, 32, 3, 2, 0, 1), nn.ReLU(inplace=True))
        self.conv1d_5 = nn.Conv2d(96,32,1)
        self.conv2DT_BA2 = nn.Sequential(deconvbn(32, 32, 3, 2, 0, 1), nn.ReLU(inplace=True))
        self.conv1d_6 = nn.Conv2d(33,32,1)
        self.Conv2DownUp7 =nn.Sequential(Conv2DownUp(32, 32, 5, lastLayer=False), nn.ConvTranspose2d(32, labels, 5))
        self.conv1d_9 = nn.Conv2d(256, 128, 1)
        self.conv1d_7 = nn.Conv2d(128*2,128,1)
        self.Conv2DownUp8 = Conv2DownUp(32, 64, 3)
        self.Conv2DownUp9 = Conv2DownUp(256, 64, 3)
        self.conv1d_8 = nn.Conv2d(65,64,1)
        self.Conv2DownUp10 = nn.Sequential(Conv2DownUp(64, 64, 5, lastLayer=False), nn.ConvTranspose2d(64, 1, 5))

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
        a_0, a_1, a_2, a_3, a_4, a_pyramidB_0, a_pyramidB_2 = self.resnet_features(input_a) #[(64,64), (256,32), (512,16), (1024, 8), (2048, 4)]
        b_0, b_1, b_2, b_3, b_4, b_pyramidB_0, b_pyramidB_2 = self.resnet_features(input_b)

        xleft3 = self.conv2d_ba3(input_a)
        xleft2 = self.conv2d_ba1(input_a)
        xleft1 = self.conv2d_ba2(input_a)

        x = torch.cat((a_4, b_4), axis=1)
        x = self.conv1d_1(x)
        x = F.upsample(x, scale_factor=2)
        x = self.Conv2DownUp1(x)
        x1 = F.upsample(x, scale_factor=2)
        seg_branch = self.Conv2DownUp2(x1)
        seg_branch = F.upsample(seg_branch, (input_a.size()[2], input_a.size()[3]))

        y = torch.squeeze(self.correlation_sampler(a_pyramidB_2, b_pyramidB_2), axis=1)
        y = self.corrConv2d(y)
        y1 = self.Conv2DownUp3(x1)
        y1 = F.upsample(y1, size=(y.shape[2], y.shape[3]), mode='bilinear')

        y = torch.cat((y1, y), axis=1)
        y = self.Conv2DownUp4(y)
        y2 = F.upsample(y, scale_factor=8)

        xleft2 = F.upsample(xleft2, size=(y2.shape[2], y2.shape[3]), mode='bilinear')

        disp_out = torch.cat((y2, xleft2), axis=1)
        disp_out = self.conv1d_2(disp_out)
        disp_out = self.Conv2DownUp5(disp_out)
        disp_out = F.upsample(disp_out, size=(input_a.shape[2], input_a.shape[3]), mode='bilinear')

        x = F.upsample(x, scale_factor=(4))
        y3 = F.upsample(y, scale_factor=(2))
        x = F.upsample(x, (y3.shape[2], y3.shape[3]), mode='bilinear')

        x = torch.cat((x, y3), axis=1)
        x = self.conv1d_3(x)
        x = self.Conv2DownUp6(x)
        x = F.upsample(x, (a_1.shape[2], a_1.shape[3]), mode='bilinear')
        x = torch.cat((x, a_1), axis=1)
        x = self.conv1d_4(x)
        x = self.conv2DT_BA1(x)
        x3 = x
        
        x = F.upsample(x, (a_0.shape[2], a_0.shape[3]), mode='bilinear')
        
        x = torch.cat((x, a_0), axis=1)
        x = self.conv1d_5(x)
        x = self.conv2DT_BA2(x)

        xleft1 = F.upsample(xleft1, (x.shape[2], x.shape[3]), mode='bilinear')
        x = torch.cat((x, xleft1), axis=1)
        x = self.conv1d_6(x)
        seg_branch2 = self.Conv2DownUp7(x)
        seg_branch2 = F.upsample(seg_branch2, (input_a.shape[2], input_a.shape[3]), mode='bilinear')

        y4 = torch.cat((a_pyramidB_0, b_pyramidB_0), axis=1)
        y4 = self.conv1d_9(y4)
        y = F.upsample(y, scale_factor=4)

        y = F.upsample(y, (y4.shape[2], y4.shape[3]), mode='bilinear')
        y = torch.cat((y4, y), axis=1)
    
        y5 = self.Conv2DownUp8(x3)
        y = F.upsample(y, (y5.shape[2], y5.shape[3]), mode='bilinear')
        y = torch.cat((y5, y), axis=1)
        y = self.Conv2DownUp9(y)
        y = F.upsample(y, scale_factor=2)

        xleft3 = F.upsample(xleft3, (y.shape[2], y.shape[3]), mode='bilinear')
        disp_out2 = torch.cat((y, xleft3), axis=1)
        disp_out2 = self.conv1d_8(disp_out2)
        disp_out2 = self.Conv2DownUp10(disp_out2)
        disp_out2 = F.upsample(disp_out2, (input_a.shape[2], input_a.shape[3]), mode='bilinear')
        disp_out2 = 0.8*disp_out2 + 0.2*disp_out
                
        # # print(x3.shape)
        # print(disp_out2.shape)
        # input()

        # out1 = torch.cat((a_pyramidB_0, b_pyramidB_0), axis=1)

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
        self.branch0_0 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))
        self.branch0_1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))
        self.branch0_2 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))
        self.branch0_3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))


        self.branch1_0 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(512, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))
        self.branch1_1 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(512, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))
        self.branch1_2 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(512, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))



    def forward(self, x):
        out_0, out_1, out_2, out_3, out_4 = self.resnet_features(x) #[(64,128), (256,64), (512,32), (1024, 16), (2048, 8)]
        # densenet
        # (None, 128, 128, 64)
        # (None, 64, 64, 128)
        # (None, 32, 32, 256)
        # (None, 16, 16, 512)
        # (None, 8, 8, 1024)
        b0_0 = F.interpolate(self.branch0_0(out_0), (out_0.size()[2], out_0.size()[3]), mode='bilinear')
        b0_1 = F.interpolate(self.branch0_1(out_0), (out_0.size()[2], out_0.size()[3]), mode='bilinear')
        b0_2 = F.interpolate(self.branch0_2(out_0), (out_0.size()[2], out_0.size()[3]), mode='bilinear')
        b0_3 = F.interpolate(self.branch0_3(out_0), (out_0.size()[2], out_0.size()[3]), mode='bilinear')      
        b0 = torch.cat((b0_0, b0_1, b0_2, b0_3), axis=1)

        # print('out1: ',  out_0.shape, out_1.shape, out_2.shape, out_3.shape, out_4.shape)
        b1_0 = F.interpolate(self.branch1_0(out_2), (out_2.size()[2], out_2.size()[3]), mode='bilinear')
        b1_1 = F.interpolate(self.branch1_1(out_2), (out_2.size()[2], out_2.size()[3]), mode='bilinear')
        b1_2 = F.interpolate(self.branch1_2(out_2), (out_2.size()[2], out_2.size()[3]), mode='bilinear')
        b1 = torch.cat((b1_0, b1_1, b1_2), axis=1)
        
        return out_0, out_1, out_2, out_3, out_4, b0, b1
        
        
    
        
        
        
