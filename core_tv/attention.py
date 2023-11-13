import numpy
import torch.nn as nn
import torch
from torch import autograd
from myutils import get_pa


class Cat_tensorvoting(nn.Module):#这里就先实验tv图与特征图在C处，cat，通道融合.(这里代码应该不和unet一样，图片尺寸在卷积时没有改变)
    def __init__(self,channel):
        super(Cat_tensorvoting, self).__init__()
        self.layer1=nn.Conv2d(channel,channel//2,1,1)
        self.layer2=nn.Conv2d(1,channel//2,1,1)
    def forward(self,x,tv_img):
        out=self.layer1(x)
        tv_map=self.layer2(tv_img)
        #print(out.shape,tv_map.shape)
        return torch.cat((out,tv_map),dim=1)

class attention_mask_tv1(nn.Module):
    def __init__(self,channel):
        super (attention_mask_tv1, self).__init__()
        self.layer=nn.Conv2d(channel,channel ,1,1)
    def forward(self, x, tv_img):
        mask= tv_img.bool()
        out =self.layer(x)
        out.masked_fill_(~mask, 0)
        return out 

class attention_mask_tv(nn.Module):
    def __init__(self,channel):
        super (attention_mask_tv, self).__init__()
        self.layer=nn.Softmax2d()
        self.layer1=nn.Softmax2d()
    def forward(self, x, tv_img):
        #print("x.shape",x.shape)
        #print("tv_img",tv_img.shape)
        out = x
        x_out0 = out[:,0,:,:]
        x_out1 = out[:,1,:,:]
        #out= self.layer(x)
        x_out = torch.argmax(out, 1).float()
        #tv_img1= tv_img.mean(dim=0).view(1,64,64)
        tv_img = tv_img.mean(dim=1)

        tv_img_mask = torch.zeros(tv_img.shape)
        tv_img_mask.copy_(tv_img)
        tv_img_mask[tv_img_mask > 0] = 1
        tv_img_mask[tv_img_mask < 0] = 0
        w = torch.zeros(x_out.shape).cuda()

        x_out_c = torch.zeros(x_out.shape)
        x_out_c.copy_(x_out)
        if get_pa(x_out_c.cpu(),tv_img_mask.cpu())>0.9:
            #print("yes^^^^^^^^^^^^^^^^^^^^^^^^^getpa>0.9!")
            #print(w.device,x_out.device,tv_img.device)
            w[x_out1 <= tv_img] = x_out1[x_out1<= tv_img] / tv_img[x_out1 <= tv_img]
            w[x_out1 >  tv_img] = (1-x_out1[x_out1 > tv_img]) /(1- tv_img[x_out1 > tv_img])
            ad = torch.logical_and(tv_img == 0, x_out1 == 0)

            w[ad] = 0
            result = w * x_out1 +(1-w)* tv_img
            result = torch.unsqueeze(result,dim=1)
            result = torch.cat((1-result,result),dim=1)
            #mask= self.layer1(tv_img)
            #result= x*mask
            #result = nn.Sigmoid()(result)
            return result
        return out



if __name__ =="__main__":
    x_out1 = numpy.array([[0.9,0.8,0.1],[0,0.1,0.2],[0.7,0.5,0]])
    x_out2 = 1-x_out1
    tv_img = numpy.array([[1.0,0.2,0],[0,0.8,0],[0,0,0.1]])
    tv_img2 = 1-tv_img

    w = numpy.zeros((3,3))
    w[x_out1 <= tv_img] = x_out1[x_out1 <= tv_img] / tv_img[x_out1 <= tv_img]
    w[x_out1 > tv_img] = (1 - x_out1[x_out1 > tv_img]) / (1 - tv_img[x_out1 > tv_img])
    #print("w",w)
    #print("tv",tv_img)
    ad = numpy.logical_and(tv_img==0,x_out1==0)
    w[ad]=0
    result = w * x_out1 + (1 - w) * tv_img

    w2 = numpy.zeros((3, 3))
    w2[x_out2 <= tv_img2] = x_out2[x_out2 <= tv_img2] / tv_img2[x_out2 <= tv_img2]
    w2[x_out2 > tv_img2] = (1 - x_out2[x_out2 > tv_img2]) / (1 - tv_img2[x_out2> tv_img2])
    print("w", w2)
    ad = numpy.logical_and(tv_img2 == 0, x_out2 == 0)
    w2[ad] = 0
    print("w", w2)
    result2 = w2 * x_out2 + (1 - w2) * tv_img2

    print(result)
    print(result2)
