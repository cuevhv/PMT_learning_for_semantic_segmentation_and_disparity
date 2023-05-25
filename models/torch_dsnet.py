import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torchvision import datasets, models, transforms
from spatial_correlation_sampler import SpatialCorrelationSampler
from models.torch_model import *
from models import Resnet
from torch.nn.functional import pad
def apply_disparity(input_images, x_offset, wrap_mode='edge', tensor_type = 'torch.cuda.FloatTensor'):
    num_batch, num_channels, height, width = input_images.size()

    # Handle both texture border types
    edge_size = 0
    if wrap_mode == 'border':
        edge_size = 1
        # Pad last and second-to-last dimensions by 1 from both sides
        input_images = pad(input_images, (1, 1, 1, 1))
    elif wrap_mode == 'edge':
        edge_size = 0
    else:
        return None

    # Put channels to slowest dimension and flatten batch with respect to others
    input_images = input_images.permute(1, 0, 2, 3).contiguous()
    im_flat = input_images.view(num_channels, -1)

    # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated
    # meshgrid function)
    x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type)#.to(opt.gpu_ids)
    # print(x.shape, 'shape')
    y = torch.linspace(0, height - 1, height).repeat(width, 1).transpose(0, 1).type(tensor_type)#.to(opt.gpu_ids)
    # Take padding into account
    x = x + edge_size
    y = y + edge_size
    # Flatten and repeat for each image in the batch
    x = x.reshape(-1).repeat(1, num_batch)
    y = y.reshape(-1).repeat(1, num_batch)

    # Now we want to sample pixels with indicies shifted by disparity in X direction
    # For that we convert disparity from % to pixels and add to X indicies
    x = x + x_offset.contiguous().view(-1)# * width
    # Make sure we don't go outside of image
    x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
    # Round disparity to sample from integer-valued pixel grid
    y0 = torch.floor(y)
    # In X direction round both down and up to apply linear interpolation
    # between them later
    x0 = torch.floor(x)
    x1 = x0 + 1
    # After rounding up we might go outside the image boundaries again
    x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

    # Calculate indices to draw from flattened version of image batch
    dim2 = (width + 2 * edge_size)
    dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
    # print('dims', dim1, width,height)
    # Set offsets for each image in the batch
    base = dim1 * torch.arange(num_batch).type(tensor_type)#.to(opt.gpu_ids)
    # print('base batch ', base.shape, base)
    # print('base reshaped ',base.view(-1, 1).shape)
    base = base.view(-1, 1).repeat(1, height * width).view(-1)
    # print('base repeat ',base.shape)
    # One pixel shift in Y  direction equals dim2 shift in flattened array
    base_y0 = base + y0 * dim2
    # print('base y0 ',base_y0.shape, base_y0)
    # print('x0', x0)
    # Add two versions of shifts in X direction separately
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1
    # print('idx_l ', idx_l)
    # input()
    # Sample pixels from images
    pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
    pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())
    # print('pix_l', pix_l)

    # Apply linear interpolation to account for fractional offsets
    weight_l = x1 - x
    weight_r = x - x0
    output = weight_l * pix_l + weight_r * pix_r

    # Reshape back into image batch and permute back to (N,C,H,W) shape
    output = output.view(num_channels, num_batch, height, width).permute(1,0,2,3)

    return output

class testWarp(nn.Module):
    def __init__(self):
        super(testWarp, self).__init__()
    def forward(self, input, disp):
        return apply_disparity(input, disp)



Backward_tensorGrid = {}

def Backward(tensorInput, tensorDisp):
    minVal = torch.min(tensorInput)
    if str(tensorDisp.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorDisp.size(3)).view(1, 1, 1, tensorDisp.size(3)).expand(tensorDisp.size(0), -1, tensorDisp.size(2), -1)
        tensorVertical = torch.zeros_like(tensorHorizontal)
        # tensorVertical = torch.linspace(-1.0, 1.0, tensorDisp.size(2)).view(1, 1, tensorDisp.size(2), 1).expand(tensorDisp.size(0), -1, -1, tensorDisp.size(3))
        # tensorHorizontal = torch.zeros_like(tensorVertical)
        Backward_tensorGrid[str(tensorDisp.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
    # end
    print(Backward_tensorGrid[str(tensorDisp.size())] + tensorDisp)
    tensorDisp = torch.cat([ tensorDisp[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorDisp[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
    wrapped = torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorDisp.size())] + tensorDisp).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)
    print(wrapped[:,1,:,:])
    print(tensorInput[:,1,:,:])
    # print(tensorInput[:,0,:,:])
    # print(wrapped[:,0,:,:])
    # print(tensorDisp)
    # print(torch.all(wrapped == tensorInput))
    input()
    return wrapped
# end

class DSnet(nn.Module):
    def __init__(self, outputType, warpedOutput, label=9, disp_th=25.0, disp_normalize='sigmoid', old_config='False'):
        super(DSnet, self).__init__()
        max_disp = 21#21
        batchnorm = True
        dropout = 0.2
        self.lablel = label
        self.disp_th = disp_th
        self.old_config = old_config
        self.outputType = outputType
        self.disp_normalize = disp_normalize
        self.warpedOutput = warpedOutput
        self.feature_extraction = featureExtractor(dropout=dropout, batchnorm=batchnorm)
        self.correlation_sampler = SpatialCorrelationSampler(
                                                            kernel_size=1,
                                                            patch_size=(1, max_disp),
                                                            stride=1,
                                                            padding=0,
                                                            dilation_patch=4)
        
        self.gate = nn.Sequential(convbn(in_channel=320, out_channel=16, kernel_size=1, 
                                    stride=1, pad=0, dilation=1, batchnorm=batchnorm), nn.ReLU(inplace=True))
        
        self.compressGate = nn.Sequential(convbn(in_channel=16+max_disp, out_channel=1, kernel_size=1, 
                                    stride=1, pad=0, dilation=1, batchnorm=batchnorm), nn.Sigmoid())

        self.conv_redir = nn.Sequential(convbn(in_channel=320, out_channel=128, kernel_size=3, 
                                    stride=1, pad=1, dilation=1, batchnorm=batchnorm), nn.ReLU(inplace=True))
    
        # self.res1 = Resnet.BasicBlock(128,128, stride=1, dropout=0.2, downsample=None)
        self.res1 = Resnet.BasicBlock(128,128, stride=1, dropout=dropout, downsample=None, batchnorm=batchnorm)
        self.res2 = Resnet.BasicBlock(128+1,128+1, stride=1, dropout=dropout, downsample=None, batchnorm=batchnorm)
    
        self.compress = nn.Sequential(convbn(in_channel=128*2, out_channel=128, kernel_size=1, 
                                    stride=1, pad=0, dilation=1, batchnorm=batchnorm), nn.ReLU(inplace=True))
        
        self.compress2 = nn.Sequential(convbn(in_channel=128+32, out_channel=128, kernel_size=1, 
                                    stride=1, pad=0, dilation=1, batchnorm=batchnorm), nn.ReLU(inplace=True))

        self.deconv = nn.Sequential(deconvbn(in_channel=128+1, out_channel=128+1, kernel_size=5, 
                                    stride=2, pad=1, dilation=1, batchnorm=batchnorm), nn.ReLU(inplace=True))

        self.xin = nn.Sequential(convbn(in_channel=3, out_channel=1, kernel_size=9, 
                                    stride=1, pad=0, dilation=1, batchnorm=batchnorm), nn.ReLU(inplace=True))
        
        if disp_normalize == 'sigmoid':
            self.activationDisp = nn.Sigmoid()
        if disp_normalize == 'tanh':
            self.activationDisp = nn.Tanh()

        if self.outputType == 'disp':
            self.output = nn.Sequential(convbn(in_channel=128+1, out_channel=1, kernel_size=5, 
                                    stride=1, pad=0, dilation=1, batchnorm=batchnorm))#, nn.Sigmoid())
        if self.outputType == 'fmetric':
            self.output = nn.Sequential(convbn(in_channel=128+1, out_channel=1, kernel_size=5, 
                                    stride=1, pad=0, dilation=1), nn.Sigmoid())
        if self.outputType == 'seg':
            # self.output = nn.Sequential(deconvbn(in_channel=128+1, out_channel=1, kernel_size=3, 
            #                          stride=1, pad=0, dilation=1), nn.Sigmoid())
            # self.output = nn.Sequential(convbn(in_channel=128+1, out_channel=9, kernel_size=3, 
            #                             stride=1, pad=0, dilation=1))
            self.output = nn.Sequential(convbn(in_channel=128+1, out_channel=label, kernel_size=3, 
                                    stride=1, pad=0, dilation=1, batchnorm=batchnorm))
        if self.outputType == 'both':
            self.output = nn.Sequential(convbn(in_channel=128+1, out_channel=label+1, kernel_size=3, 
                                    stride=1, pad=0, dilation=1, batchnorm=batchnorm))
                                    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):

        featureLeft, block1 = self.feature_extraction(left)
        featureRight, _ = self.feature_extraction(right)
        corr_layer = self.correlation_sampler(featureLeft, featureRight)
        corr_layer = torch.squeeze(corr_layer, 1)

        alpha = self.gate(featureLeft)
        alpha = self.compressGate(torch.cat((corr_layer, alpha), 1))
        cr = self.conv_redir(featureLeft)
        cr_main = torch.mul(cr, alpha)
        for i in range(3):
            cr = self.res1(cr_main)
            cr_main = torch.cat((cr_main, cr), 1)
            cr_main = self.compress(cr_main)
            
        cr_main = F.upsample(cr_main, (block1.size()[2], block1.size()[3]), mode='bilinear')
        cr_main = self.compress2(torch.cat((cr_main, block1), 1))
        cr_main = self.res1(cr_main)

        xin = self.xin(left)
        cr_main = F.upsample(cr_main, (xin.size()[2], xin.size()[3]), mode='bilinear')
        
        
        cr_main = torch.cat((cr_main, xin), 1)
        #cr_main = self.deconv(cr_main)
        cr_main = self.res2(cr_main)
        cr_main = self.output(cr_main)
        ##cr_main = F.upsample(cr_main, (left.size()[2], left.size()[3]), mode='bilinear')
        if self.outputType == 'disp':
            if self.disp_normalize != 'linear':
                cr_main = self.activationDisp(cr_main)

        if self.outputType == 'seg':
            cr_main = F.upsample(cr_main, (left.size()[2], left.size()[3]), mode='nearest')
            return cr_main
        elif self.outputType == 'both':
            cr_main = F.upsample(cr_main, (left.size()[2], left.size()[3]), mode='nearest')
        else:
            cr_main = F.interpolate(cr_main, (left.size()[2], left.size()[3]), mode='bilinear')

        #if self.outputType == 'fmetric':
        if self.outputType == 'both':
            if not self.old_config:
                if self.disp_normalize != 'linear':
                    cr_main[:,0,:,:] = self.activationDisp(cr_main[:,0,:,:])
            warpedOutput = apply_disparity(left, self.disp_th*cr_main[:,0,:,:].unsqueeze(1)/100.0)
        else:
            if self.disp_normalize == 'tanh':
                warpedOutput = apply_disparity(left, ((cr_main + 1)*self.disp_th*0.5)/100.0)#Backward(left, torch.zeros_like(cr_main))                
            else:
                warpedOutput = apply_disparity(left, self.disp_th*cr_main)#Backward(left, torch.zeros_like(cr_main))

        
        if self.warpedOutput:
            return torch.cat((cr_main, warpedOutput), 1)
        else:
            return cr_main

