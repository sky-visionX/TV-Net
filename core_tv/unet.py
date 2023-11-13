import torch.nn as nn
import torch
from torch import autograd
from myutils import get_pa
from core_tv.attention import attention_mask_tv


class ConvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet_tv(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super(Unet_tv, self).__init__()
        self.Convolution1 = ConvolutionBlock(in_ch, 64)  
        self.maxpooling1 = nn.MaxPool2d(2)
        self.Convolution2 = ConvolutionBlock(64, 128)
        self.maxpooling2 = nn.MaxPool2d(2)
        self.Convolution3 = ConvolutionBlock(128, 256)
        self.maxpooling3 = nn.MaxPool2d(2)
        self.Convolution4 = ConvolutionBlock(256, 512)
        self.maxpooling4 = nn.MaxPool2d(2)
        self.Convolution5 = ConvolutionBlock(512, 1024)

        self.maxpooling6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.Convolution6 = ConvolutionBlock(1024, 512)
        self.maxpooling7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.Convolution7 = ConvolutionBlock(512, 256)
        self.maxpooling8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.Convolution8 = ConvolutionBlock(256, 128)
        self.maxpooling9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.Convolution9 = ConvolutionBlock(128, 64)
        self.Convolution10 = nn.Conv2d(64, out_ch, 1)
        self.aten_mask_tv= attention_mask_tv(out_ch)
        
    def forward(self, x,tv_img):
        conv1 = self.Convolution1(x)
        pool1 = self.maxpooling1(conv1)
        conv2 = self.Convolution2(pool1)
        pool2 = self.maxpooling2(conv2)
        conv3 = self.Convolution3(pool2)
        pool3 = self.maxpooling3(conv3)
        conv4 = self.Convolution4(pool3)
        pool4 = self.maxpooling4(conv4)
        conv5 = self.Convolution5(pool4)
        up_6 = self.maxpooling6(conv5)
        merge6 = torch.cat([up_6, conv4], dim=1)
        conv6 = self.Convolution6(merge6)
        up_7 = self.maxpooling7(conv6)
        merge7 = torch.cat([up_7, conv3], dim=1)
        conv7 = self.Convolution7(merge7)
        up_8 = self.maxpooling8(conv7)
        merge8 = torch.cat([up_8, conv2], dim=1)
        conv8 = self.Convolution8(merge8)
        up_9 = self.maxpooling9(conv8)
        merge9 = torch.cat([up_9, conv1], dim=1)
        conv9 = self.Convolution9(merge9)
        conv10 = self.Convolution10(conv9)
        #out = nn.Softmax2d()(conv10)
        amt= self.aten_mask_tv(conv10,tv_img)
        out =amt
        out1 = nn.Softmax2d()(out)

        return out1