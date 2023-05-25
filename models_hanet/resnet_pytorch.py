import torch
import torch.nn as nn
import torchvision.models as models
from models_hanet import Resnet
from models_hanet.mynn import Norm2d, Upsample, initialize_weights

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

class deeplabV3plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-101', criterion=None, criterion_aux=None,
                variant='D', skip='m1', skip_num=48, args=None, return_layers=False):
        super(deeplabV3plus, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.args = args
        self.num_attention_layer = 0
        self.trunk = trunk
        self.return_layers = return_layers

        channel_1st = 3
        channel_2nd = 64
        channel_3rd = 256
        channel_4th = 512
        prev_final_channel = 1024
        final_channel = 2048

        if trunk == 'resnet-50':
            resnet = Resnet.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101': # three 3 X 3
            resnet = Resnet.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                        resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        # elif trunk == 'resnet-152':
        #     resnet = Resnet.resnet152()
        #     resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-50':
            resnet = models.resnext50_32x4d(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-101':
            resnet = models.resnext101_32x8d(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'wide_resnet-50':
            resnet = models.wide_resnet50_2(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'wide_resnet-101':
            resnet = models.wide_resnet101_2(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4


        

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D4':
            for n, m in self.layer2.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (8, 8), (8, 8), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            # raise 'unknown deepv3 variant: {}'.format(self.variant)
            print("Not using Dilation ")

        if self.variant == 'D':
            os = 8
        elif self.variant == 'D4':
            os = 4
        elif self.variant == 'D16':
            os = 16
        else:
            os = 32

        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1_1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))
            
        self.final1_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1_1)
        initialize_weights(self.final1_2)
        initialize_weights(self.final2)

    def forward(self, x, gts=None, aux_gts=None, pos=None, attention_map=False, attention_loss=False):
        x_size = x.size()  # 800

        x = self.layer0(x)  # 400
        x = self.layer1(x)  # 400
        low_level = x
        x = self.layer2(x)  # 100
        middle_level = x
        x = self.layer3(x)  # 100
        high_level = x
        x = self.layer4(x)  # 100
        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(low_level)
        if self.return_layers:
            return dec0_up, high_level, middle_level, dec0_fine
        else:
            dec0_up = Upsample(dec0_up, low_level.size()[2:])
            dec0 = torch.cat([dec0_fine, dec0_up], 1)
            dec1 = self.final1_1(dec0)
            dec1 = self.final1_2(dec1)
            dec2 = self.final2(dec1)
            main_out = Upsample(dec2, x_size[2:])

        return main_out