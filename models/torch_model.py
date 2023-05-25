import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from torchvision import datasets, models, transforms
from models.densenet import densenet121
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

    # return nn.Sequential(nn.Conv2d(in_channel,
    #                                 out_channel,
    #                                 kernel_size=kernel_size,
    #                                 stride=stride,
    #                                 padding=dilation if dilation > 1 else pad,
    #                                 dilation = dilation,
    #                                 bias=False if batchnorm == True else True),
    #                      nn.BatchNorm2d(out_channel))

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
    # return nn.Sequential(nn.ConvTranspose2d(in_channel,
    #                                 out_channel,
    #                                 kernel_size=kernel_size,
    #                                 stride=stride,
    #                                 padding=dilation if dilation > 1 else pad,
    #                                 dilation = dilation,
    #                                 bias=False if batchnorm == True else True),
    #                      nn.BatchNorm2d(out_channel))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class featureExtractor(nn.Module):
    def __init__(self, dropout, batchnorm=True):
        super(featureExtractor, self).__init__()
        #3,1,1
        pad = 1
        self.p = dropout
        self.in_channel = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 7, stride=2, pad=3 if pad == 1 else 0, dilation=1, batchnorm=batchnorm),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3,  stride=1, pad=1 if pad == 1 else 0, dilation=1, batchnorm=batchnorm),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3,  stride=1, pad=1 if pad == 1 else 0, dilation=1, batchnorm=batchnorm),
                                       nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, blocks=3, stride=1,pad=1 if pad == 1 else 0,dilation=1, batchnorm=batchnorm)
        self.layer2 = self._make_layer(BasicBlock, 64, blocks=16, stride=2,pad=1 if pad == 1 else 0,dilation=1, batchnorm=batchnorm) 
        self.layer3 = self._make_layer(BasicBlock, 128, blocks=3, stride=1,pad=1 if pad == 1 else 0,dilation=1, batchnorm=batchnorm)
        self.layer4 = self._make_layer(BasicBlock, 128, blocks=3, stride=1,pad=1 if pad == 1 else 0,dilation=2, batchnorm=batchnorm)
        #self.layer5 = self._make_layer(BasicBlock, 128, blocks=3, stride=1,pad=1 if pad == 1 else 0,dilation=2)
        self.branches = []
        for i in range(1, 5):
            self.branches.append(nn.Sequential(nn.Conv2d(128,
                                32,
                                kernel_size=5,
                                stride=1,
                                padding=0,
                                dilation = i*2), nn.ReLU(), nn.Dropout(p=self.p, inplace=True)))
        self.branches = nn.ModuleList(self.branches)
        
   
    def forward(self, x):
        output0 = self.firstconv(x)
        output1 = self.layer1(output0)
        output_raw  = self.layer2(output1)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branches = []
        for i in range(4):
            branch = self.branches[i](output_skip)
            output_branches.append(F.upsample(branch, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear'))
        
        output_feature = torch.cat((output_raw, output_skip,
                                    output_branches[3],output_branches[2], output_branches[1], output_branches[0]), 1)
        
        return output_feature, output1


    def _make_layer(self, block, out_channel, blocks, stride, pad, dilation, batchnorm=True):
        downsample = None
        if stride != 1 or self.in_channel != out_channel:
            layers = [nn.Conv2d(self.in_channel, out_channel,
                                        kernel_size=1, stride=stride, bias=False if batchnorm == True else True)]
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_channel))
            downsample = nn.Sequential(*layers)
            # downsample = nn.Sequential(nn.Conv2d(self.in_channel, out_channel,
            #                             kernel_size=1, stride=stride, bias=False if batchnorm == True else True),
            #                                 nn.BatchNorm2d(out_channel),)

        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample, pad, dilation, batchnorm=batchnorm))
        self.in_channel = out_channel
        for i in range(1, blocks):
            layers.append(block(self.in_channel, out_channel,1,None,pad,dilation, batchnorm=batchnorm))

        return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride, downsample, pad, dilation, batchnorm=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(in_channel, out_channel, 3, stride, pad, dilation, batchnorm),
                                   nn.ReLU(inplace=True))
        self.conv2 = convbn(out_channel, out_channel, 3, 1, pad, dilation, batchnorm)

        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
            
        out += x

        return out

    















class mainNN(nn.Module):
    def __init__(self):
        super(mainNN, self).__init__()
        model = resnetBlocks(block, layers, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
            model.load_state_dict(state_dict)
        #self.blocks = self.extract_blocks('resnet50')
    
    def forward(self, x):
        y1 = self.blocks[0](x)
        #y2 = self.blocks[1](x)
        #y3 = self.blocks[2](x)
        #y4 = self.blocks[4](x)

        return y1#, y2, y3, y4
        

    def extract_blocks(self, modelType):
        blocks = []
        if 'densenet' in modelType:
            model_ft = models.densenet161(pretrained=True).features
            block_name = 'denseblock'
        
        if 'resnet' in modelType:
            model_ft = models.resnet50(pretrained=True)
            block_name = 'layer'
        
        keys = model_ft._modules.keys() 
        for i, k in enumerate(keys):
            if block_name in k:
                print(k)
                blocks.append(nn.Sequential(*list(model_ft.children())[:i+1]))
        print(keys)
        #model_ft.features._modules.get('denseblock2')
        return blocks
        
# class resnetBlocks(torch.models.ResNet):
#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None):
#         super(resnetBlocks, self).__init__(block, layers, num_classes, zero_init_residual,
#                                              groups, width_per_group, replace_stride_with_dilation,
#                                              norm_layer)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)

#         x = self.avgpool(x4)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x1, x2, x3, x4


class conv2dSame(torch.nn.Module):
    def __init__(self, in_channel, 
                        out_channel,
                        kernel_size,
                        stride=1,
                        padding='valid',
                        dilation = 1,
                        bias=False):

        super(conv2dSame, self).__init__()
        self.padding = padding
        self.c2d = torch.nn.Conv2d(in_channel,
                                   out_channel,
                                   kernel_size = kernel_size,
                                   stride      = stride,
                                   dilation    = dilation,
                                   bias        = bias)
        # self.c2d.weight = torch.nn.Parameter(torch.ones_like(self.c2d.weight))
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
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
        if self.padding == 'same':
            pad_top, pad_bottom = self.conv2dpad(x.shape[2], x.shape[2], self.c2d.stride[0], self.c2d.kernel_size[0], self.c2d.dilation[0])
            pad_left, pad_right = self.conv2dpad(x.shape[3], x.shape[3], self.c2d.stride[0], self.c2d.kernel_size[0], self.c2d.dilation[0])
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        x = self.c2d(x)
        return x
            
    def conv2dpad(self, input_size, output_size, stride, kernel, dilation):
        output_size = np.ceil(input_size/float(stride))
        total_pad = max((output_size-1)*stride - input_size + dilation*(kernel-1) + 1, 0)
        pad_top = total_pad // 2
        pad_bottom = total_pad - pad_top
        return int(pad_top), int(pad_bottom)


class ConvTranspose2dSame(torch.nn.Module):
    def __init__(self, in_channel, 
                        out_channel,
                        kernel_size,
                        stride=1,
                        padding='valid',
                        dilation = 1,
                        bias=False,
                        init_he=True):

        super(ConvTranspose2dSame, self).__init__()
        self.padding = padding
        self.ct2d = torch.nn.ConvTranspose2d(in_channel,
                                            out_channel,
                                            kernel_size = kernel_size,
                                            stride      = stride,
                                            dilation    = dilation,
                                            bias        = bias)
        # self.ct2d.weight = torch.nn.Parameter(torch.ones_like(self.ct2d.weight))
        # for m in self.modules():
        #     if isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        if init_he:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    pass
                #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #     m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


    def forward(self, x):       
        if self.padding == 'same':
            # pad_top, pad_bottom, height_size = self.deconv2dpad(x.shape[2], x.shape[2], self.ct2d.stride[0], self.ct2d.kernel_size[0], self.ct2d.dilation[0])
            # pad_left, pad_right, width_size = self.deconv2dpad(x.shape[3], x.shape[3], self.ct2d.stride[0], self.ct2d.kernel_size[0], self.ct2d.dilation[0])
            # x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
            _, _, height_size, width_size = x.shape
            # x = self.ct2d(x)
            # # print(x)
            # _, _, h, w = x.shape
            # h, w = int(h//2), int(w//2)
            # out_h = height_size//2
            # out_w = width_size//2
            # #print(h, w, out_h, w)
            # x = x[:,:, h-out_h:h+height_size-out_h, w-out_w:w+width_size-out_w]
            height_size *= self.ct2d.stride[0]
            width_size *= self.ct2d.stride[1]
            x = self.ct2d(x)
            _, _, h, w = x.shape
            h, w = int(h//2), int(w//2)
            out_h = height_size//2 if h - height_size//2 >= 0 else h
            out_w = width_size//2 if w - width_size//2 >= 0 else w
            
            # print(height_size, width_size)
            # print(out_h, out_w)
            # print(h, w)
            # #print(h, w, out_h, w)
            x = x[:,:, h-out_h:h+height_size-out_h, w-out_w:w+width_size-out_w]
        else:
            x = self.ct2d(x)
        return x

    def deconv2dpad(self, input_size, output_size, stride, kernel, dilation, out_pad=0):
        output_size = input_size * stride
        kernel = dilation*(kernel-1) + 1
        total_pad = max((input_size-1)*stride - output_size + kernel + out_pad, 0)  
        pad_top = total_pad // 2
        pad_bottom = total_pad - pad_top
        return int(pad_top), int(pad_bottom), output_size      
    

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
        b0 = torch.cat((out_0, b0_0, b0_1, b0_2, b0_3, b0_4), axis=1)

        # print('out1: ',  out_0.shape, out_1.shape, out_2.shape, out_3.shape, out_4.shape)
        b2_0 = self.branch1_0(out_2)
        b2_0 = F.interpolate(b2_0, (out_2.size()[2], out_2.size()[3]), mode=mode)
        
        b2_1 = self.branch1_1(out_2)
        b2_1 = F.interpolate(b2_1, (out_2.size()[2], out_2.size()[3]), mode=mode)
        
        b2_2 = self.branch1_2(out_2)
        b2_2 = F.interpolate(b2_2, (out_2.size()[2], out_2.size()[3]), mode=mode)
        b2 = torch.cat((out_2, b2_0, b2_1, b2_2), axis=1)
        
        return out_0, out_1, out_2, out_3, out_4, b2, b0
        
        

class piramidNet2(nn.Module):
    def __init__(self, pretrained=False):
        super(piramidNet2, self).__init__()
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

        self.branch1_0 = nn.Sequential(nn.AvgPool2d(pool_val[1], pool_val[1]),
                                     convbn(128, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch1_1 = nn.Sequential(nn.AvgPool2d(pool_val[2], pool_val[2]),
                                     convbn(128, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch1_2 = nn.Sequential(nn.AvgPool2d(pool_val[3], pool_val[3]),
                                     convbn(128, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch1_3 = nn.Sequential(nn.AvgPool2d(pool_val[4], pool_val[4]),
                                     convbn(128, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))

        self.branch2_0 = nn.Sequential(nn.AvgPool2d(pool_val[2], pool_val[2]),
                                     convbn(256, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch2_1 = nn.Sequential(nn.AvgPool2d(pool_val[3], pool_val[3]),
                                     convbn(256, 32, 3, 1, 'same', 1),
                                     nn.ReLU(inplace=True))
        self.branch2_2 = nn.Sequential(nn.AvgPool2d(pool_val[4], pool_val[4]),
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


