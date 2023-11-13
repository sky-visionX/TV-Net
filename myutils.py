<<<<<<< HEAD

from statistics import mode
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(), 
    transforms.Normalize([0.5],[0.5])  # transforms.Normalize(mean=[.5,.5,0.5],std=[.5,.5,0.5])
])



def cal_iou(model, dataset ,model_name):
    pa=mpa=iou=fwiou=miou=0.
    pre=rec=f1=acc=0.
    dice = 0.
    for img, mask,tv_img in dataset:
        mask = mask.cuda()
        img=img.unsqueeze(0).cuda()
        tv_img = tv_img.cuda()
        tv_img=tv_img.unsqueeze(0).cuda()


        with torch.no_grad():
            if 'tv' in model_name:
                pred = model(img,tv_img)
            else:
                pred = model(img)

            #pred1 = pred[:,1,:,:]
            pred = torch.argmax(pred,1).float()#pred = torch.argmax(pred, 1).float()



            """tv_img = tv_img.mean(dim=1)
            tv_img1 = torch.zeros(tv_img.shape)
            tv_img1.copy_(tv_img)
            tv_img1[tv_img1 <0] = 0.0
            tv_img1[tv_img1 >0] = 1.0
            if get_pa(pred.cpu(), tv_img1.cpu()) > 0.9:
                w = torch.zeros(pred1.shape).cuda()
                #print(pred.shape, tv_img.shape,w.shape)
                w[pred1 < tv_img] = pred1[pred1 < tv_img] / tv_img[pred1 < tv_img]
                w[pred1 > tv_img] = (1 - pred1[pred1 > tv_img]) / (1 - tv_img[pred1 > tv_img])
                pred1 = w * pred1 + (1 - w) * tv_img
                pred1[pred1 >= 0.5] = 1
                pred1[pred1 < 0.5] = 0
            pred = pred1"""





            pred = np.array(pred.cpu())
            mask = np.array(mask.cpu())
            pa += get_pa(pred, mask)
            mpa += get_mpa(pred, mask)
            iou += get_iou(pred, mask)
            fwiou += get_fwiou(pred, mask)
            miou += get_miou(pred,mask)


            pre+= get_precision(pred, mask)
            rec+= get_recall(pred, mask)

            acc+= get_accuracy(pred, mask)
            dice+= get_dice(pred,mask)
            """pa += get_pa(pred, mask)
            mpa += get_mpa(pred, mask)
            miou += get_miou(pred, mask)
            fwiou += get_fwiou(pred, mask)"""
    lenth = len(dataset)
    pa /= lenth
    mpa /= lenth
    iou /= lenth
    fwiou /= lenth
    miou /=lenth

    pre /= lenth
    rec /= lenth

    acc /= lenth
    dice /= lenth

    if pre+rec ==0:
        f1 = 0
        print("pre,:", pre, "!!!!!!!!!!", "rec", rec, "!!!!!!!!!!", "f1score,", f1, "!!!!!!!!!!")

        return pa, mpa, iou, fwiou, miou, pre, rec, f1, acc, dice

    else:
        f1 = (2*pre*rec)/(pre+rec)
        print("pre,:", pre, "!!!!!!!!!!", "rec", rec, "!!!!!!!!!!", "f1score,", f1, "!!!!!!!!!!")

        return pa.item(), mpa.item(), iou.item(), fwiou.item(), miou.item(), pre.item(), rec.item(), f1.item(), acc.item(),dice.item()




def confusion_matrix(pred, mask):
    mat1 = np.zeros(pred.shape)
    w = np.logical_and((pred ==1), (mask==1))
    mat1[w] = 1
    tp = mat1.sum()

    mat1 = np.zeros(pred.shape)
    w = np.logical_and((pred == 0) , (mask == 0))
    mat1[w] = 1
    tn = mat1.sum()

    mat1 = np.zeros(pred.shape)
    w = np.logical_and((pred == 1) , (mask == 0))
    mat1[w] = 1
    fp = mat1.sum()

    mat1 = np.zeros(pred.shape)
    w = np.logical_and((pred == 0) , (mask == 1))
    mat1[w] = 1
    fn = mat1.sum()


    return tp,fp,fn,tn

def get_precision(pred,mask):
    tp,fp,fn,tn=confusion_matrix(pred, mask)

    if tp+fp == 0:
        return 0
        #print("tp,", tp, "fp,", fp, "fn,", fn, "tn,", tn)
    else:
        return tp/(tp+fp)



def get_recall(pred,mask):
    tp,fp,fn,tn=confusion_matrix(pred, mask)
    #print("tp,",tp,"fp,",fp,"fn,",fn,"tn,",tn)
    if tp+fn == 0:
        return 0
    else:
        return tp/(tp+fn)

def get_f1(pred,mask):##budui
    pre=get_precision(pred,mask)
    rec=get_recall(pred,mask)
    if pre +rec == 0:
        return 0
    else:
        f1=(2*pre*rec)/(pre+rec)
        return f1

def get_accuracy(pred,mask):
    tp,fp,fn,tn=confusion_matrix(pred, mask)
    return (tp+tn)/(tp+fp+tn+fn)

    

def get_pa(pred, mask):#(acc准确率)
    return (pred==mask).sum()/(64*64)


def get_mpa(pred, mask):
    pred_crack = pred
    pred_fine = 1-pred
    mask_crack = mask
    mask_fine = 1-mask
    return (pred_crack*mask_crack).sum()/\
            (mask_crack.sum())/2 +\
            (pred_fine*mask_fine).sum()/\
            (mask_fine.sum())/2


def get_miou(pred, mask):

    """pred_crack = pred
    pred_fine = 1-pred
    mask_crack = mask
    mask_fine = 1-mask
    return (pred_crack*mask_crack).sum()/\
            ((mask_crack+pred_crack)!=0).sum()/2+\
            (pred_fine*mask_fine).sum()/\
            ((mask_fine+pred_fine)!=0).sum()/2"""

    tp1, fp1, fn1, tn1 = confusion_matrix(pred, mask)
    tp2, fp2, fn2, tn2 = confusion_matrix((1-pred), (1-mask))
    miou = (tp1/(fp1+tp1+fn1) +  tp2/(fp2+tp2+fn2))/2
    return  miou

def get_iou(pred,mask):
    tp,fp,fn,tn=confusion_matrix(pred, mask)
    return tp/(fp+tp+fn)

"""def get_iou(pred, mask):

    pred_crack = pred
   
    mask_crack = mask

    return (pred_crack*mask_crack).sum().float()/((mask_crack+pred_crack)!=0).sum()
"""
def get_fwiou(pred, mask):
    pred_crack = pred
    pred_fine = 1-pred
    mask_crack = mask
    mask_fine = 1-mask
    return  mask_crack.sum()*(pred_crack*mask_crack).sum()/\
            ((mask_crack+pred_crack)!=0).sum()/(512*512)+\
            mask_fine.sum()*(pred_fine*mask_fine).sum()/\
            ((mask_fine+pred_fine)!=0).sum()/(512*512)

def get_dice(pred, mask):
    tp, fp, fn, tn = confusion_matrix(pred, mask)
    return (2*tp)/(fn+tp+tp+fp)



def onehot(masks):
    masks_t = torch.zeros(masks.size(0), 2, 
                    masks.size(2), masks.size(3)).cuda()
    masks_t[:,0,:,:][masks[:,0,:,:]==0] = 1
    masks_t[:,1,:,:][masks[:,0,:,:]==1] = 1   
    return masks_t


if __name__=='__main__':
    mask=np.array(
    [[1,1,1,1],
    [1,1,1,1,],
    [0,1,1,1],
    [1,1,0,0],
    [0,0,0,0]])
    pred=np.array(
    [[1,1,1,1],
    [1,1,1,1],
    [1,0,0,0],
    [0,0,1,1],
    [0,0,0,0]])
    tp,fp,fn,tn= confusion_matrix(pred ,mask)
=======

from statistics import mode
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(), 
    transforms.Normalize([0.5],[0.5])  # transforms.Normalize(mean=[.5,.5,0.5],std=[.5,.5,0.5])
])



def cal_iou(model, dataset ,model_name):
    pa=mpa=iou=fwiou=miou=0.
    pre=rec=f1=acc=0.
    dice = 0.
    for img, mask,tv_img in dataset:
        mask = mask.cuda()
        img=img.unsqueeze(0).cuda()
        tv_img = tv_img.cuda()
        tv_img=tv_img.unsqueeze(0).cuda()


        with torch.no_grad():
            if 'tv' in model_name:
                pred = model(img,tv_img)
            else:
                pred = model(img)

            #pred1 = pred[:,1,:,:]
            pred = torch.argmax(pred,1).float()#pred = torch.argmax(pred, 1).float()



            """tv_img = tv_img.mean(dim=1)
            tv_img1 = torch.zeros(tv_img.shape)
            tv_img1.copy_(tv_img)
            tv_img1[tv_img1 <0] = 0.0
            tv_img1[tv_img1 >0] = 1.0
            if get_pa(pred.cpu(), tv_img1.cpu()) > 0.9:
                w = torch.zeros(pred1.shape).cuda()
                #print(pred.shape, tv_img.shape,w.shape)
                w[pred1 < tv_img] = pred1[pred1 < tv_img] / tv_img[pred1 < tv_img]
                w[pred1 > tv_img] = (1 - pred1[pred1 > tv_img]) / (1 - tv_img[pred1 > tv_img])
                pred1 = w * pred1 + (1 - w) * tv_img
                pred1[pred1 >= 0.5] = 1
                pred1[pred1 < 0.5] = 0
            pred = pred1"""





            pred = np.array(pred.cpu())
            mask = np.array(mask.cpu())
            pa += get_pa(pred, mask)
            mpa += get_mpa(pred, mask)
            iou += get_iou(pred, mask)
            fwiou += get_fwiou(pred, mask)
            miou += get_miou(pred,mask)


            pre+= get_precision(pred, mask)
            rec+= get_recall(pred, mask)

            acc+= get_accuracy(pred, mask)
            dice+= get_dice(pred,mask)
            """pa += get_pa(pred, mask)
            mpa += get_mpa(pred, mask)
            miou += get_miou(pred, mask)
            fwiou += get_fwiou(pred, mask)"""
    lenth = len(dataset)
    pa /= lenth
    mpa /= lenth
    iou /= lenth
    fwiou /= lenth
    miou /=lenth

    pre /= lenth
    rec /= lenth

    acc /= lenth
    dice /= lenth

    if pre+rec ==0:
        f1 = 0
        print("pre,:", pre, "!!!!!!!!!!", "rec", rec, "!!!!!!!!!!", "f1score,", f1, "!!!!!!!!!!")

        return pa, mpa, iou, fwiou, miou, pre, rec, f1, acc, dice

    else:
        f1 = (2*pre*rec)/(pre+rec)
        print("pre,:", pre, "!!!!!!!!!!", "rec", rec, "!!!!!!!!!!", "f1score,", f1, "!!!!!!!!!!")

        return pa.item(), mpa.item(), iou.item(), fwiou.item(), miou.item(), pre.item(), rec.item(), f1.item(), acc.item(),dice.item()




def confusion_matrix(pred, mask):
    mat1 = np.zeros(pred.shape)
    w = np.logical_and((pred ==1), (mask==1))
    mat1[w] = 1
    tp = mat1.sum()

    mat1 = np.zeros(pred.shape)
    w = np.logical_and((pred == 0) , (mask == 0))
    mat1[w] = 1
    tn = mat1.sum()

    mat1 = np.zeros(pred.shape)
    w = np.logical_and((pred == 1) , (mask == 0))
    mat1[w] = 1
    fp = mat1.sum()

    mat1 = np.zeros(pred.shape)
    w = np.logical_and((pred == 0) , (mask == 1))
    mat1[w] = 1
    fn = mat1.sum()


    return tp,fp,fn,tn

def get_precision(pred,mask):
    tp,fp,fn,tn=confusion_matrix(pred, mask)

    if tp+fp == 0:
        return 0
        #print("tp,", tp, "fp,", fp, "fn,", fn, "tn,", tn)
    else:
        return tp/(tp+fp)



def get_recall(pred,mask):
    tp,fp,fn,tn=confusion_matrix(pred, mask)
    #print("tp,",tp,"fp,",fp,"fn,",fn,"tn,",tn)
    if tp+fn == 0:
        return 0
    else:
        return tp/(tp+fn)

def get_f1(pred,mask):##budui
    pre=get_precision(pred,mask)
    rec=get_recall(pred,mask)
    if pre +rec == 0:
        return 0
    else:
        f1=(2*pre*rec)/(pre+rec)
        return f1

def get_accuracy(pred,mask):
    tp,fp,fn,tn=confusion_matrix(pred, mask)
    return (tp+tn)/(tp+fp+tn+fn)

    

def get_pa(pred, mask):#(acc准确率)
    return (pred==mask).sum()/(64*64)


def get_mpa(pred, mask):
    pred_crack = pred
    pred_fine = 1-pred
    mask_crack = mask
    mask_fine = 1-mask
    return (pred_crack*mask_crack).sum()/\
            (mask_crack.sum())/2 +\
            (pred_fine*mask_fine).sum()/\
            (mask_fine.sum())/2


def get_miou(pred, mask):

    """pred_crack = pred
    pred_fine = 1-pred
    mask_crack = mask
    mask_fine = 1-mask
    return (pred_crack*mask_crack).sum()/\
            ((mask_crack+pred_crack)!=0).sum()/2+\
            (pred_fine*mask_fine).sum()/\
            ((mask_fine+pred_fine)!=0).sum()/2"""

    tp1, fp1, fn1, tn1 = confusion_matrix(pred, mask)
    tp2, fp2, fn2, tn2 = confusion_matrix((1-pred), (1-mask))
    miou = (tp1/(fp1+tp1+fn1) +  tp2/(fp2+tp2+fn2))/2
    return  miou

def get_iou(pred,mask):
    tp,fp,fn,tn=confusion_matrix(pred, mask)
    return tp/(fp+tp+fn)

"""def get_iou(pred, mask):

    pred_crack = pred
   
    mask_crack = mask

    return (pred_crack*mask_crack).sum().float()/((mask_crack+pred_crack)!=0).sum()
"""
def get_fwiou(pred, mask):
    pred_crack = pred
    pred_fine = 1-pred
    mask_crack = mask
    mask_fine = 1-mask
    return  mask_crack.sum()*(pred_crack*mask_crack).sum()/\
            ((mask_crack+pred_crack)!=0).sum()/(512*512)+\
            mask_fine.sum()*(pred_fine*mask_fine).sum()/\
            ((mask_fine+pred_fine)!=0).sum()/(512*512)

def get_dice(pred, mask):
    tp, fp, fn, tn = confusion_matrix(pred, mask)
    return (2*tp)/(fn+tp+tp+fp)



def onehot(masks):
    masks_t = torch.zeros(masks.size(0), 2, 
                    masks.size(2), masks.size(3)).cuda()
    masks_t[:,0,:,:][masks[:,0,:,:]==0] = 1
    masks_t[:,1,:,:][masks[:,0,:,:]==1] = 1   
    return masks_t


if __name__=='__main__':
    mask=np.array(
    [[1,1,1,1],
    [1,1,1,1,],
    [0,1,1,1],
    [1,1,0,0],
    [0,0,0,0]])
    pred=np.array(
    [[1,1,1,1],
    [1,1,1,1],
    [1,0,0,0],
    [0,0,1,1],
    [0,0,0,0]])
    tp,fp,fn,tn= confusion_matrix(pred ,mask)
>>>>>>> 93fbe5c (first commit)
    print(tp,fp,fn,tn)