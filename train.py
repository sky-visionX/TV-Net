<<<<<<< HEAD
import logging
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from myutils import transform, cal_iou, onehot




from core_tv.res_unet import ResUnet as ResUnet_tv
from core_tv.unetsmall import UNetSmall as UNetSmall_tv
from core_tv.res_unet_plus import ResUnetPlusPlus as ResUnetPlusPlus_tv
from core_tv.unet import Unet_tv as UNet_tv
from core.unet import Unet as UNet
from core.res_unet import ResUnet as ResUnet
from core.unetsmall import UNetSmall as UNetSmall
from core.res_unet_plus import ResUnetPlusPlus as ResUnetPlusPlus

from compare_models.core.models.deeplabv3 import DeepLabV3
from compare_models.core.models.pspnet import PSPNet

from dataset import myDataset
import lovasz_losses as L
from Dice_coeff_loss import dice_loss
from focalloss import FocalLoss
from boundaryloss import BoundaryLoss
from torch.utils.tensorboard import SummaryWriter
from myutils import get_pa


import os

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'





def main(ver,train_data_file,val_data_file,is_loss= False):

    model_name_list = ["UNet_tv"]
    for model_name in model_name_list:
        torch.backends.cudnn.enabled=False
        version = ver
        print("write_begin!!!!!!!!!!!!",model_name)
        writer = SummaryWriter('logs_compare_model/logs'+version+'_'+model_name)
        batch_size = 2
        num_epochs = [200,200,200,200,200]
        num_workers = 1
        lr = 0.001

        losslist = ['focal']
        optimlist = ['sgd']
        iflog = True
        img_size = 64
        train_dataset = myDataset(train_data_file, transform=transform,img_size =img_size)
        val_dataset = myDataset(val_data_file, transform=transform, img_size =img_size)
        train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)
        
        criterion =  nn.BCEWithLogitsLoss()
        focallos = FocalLoss(gamma=2)

        epoidx = -1
        torch.cuda.empty_cache()
        for los in losslist:
            for opt in optimlist:
                start =  time.time()
                print(los, opt)
                torch.manual_seed(77)
                torch.cuda.manual_seed(77)

                if model_name == "deeplabv3":
                    model = DeepLabV3(nclass=2).cuda()
                if model_name == "pspnet":
                    model = PSPNet(nclass=2).cuda()

                if model_name=="UNet":
                    model = UNet().cuda()
                elif model_name=="ResUnet":
                    model = ResUnet(3).cuda()
                elif model_name=="ResUnetPlusPlus":
                    print("model is resunetplusplus")
                    model = ResUnetPlusPlus(3).cuda()
                elif model_name=="UNetSmall":
                    model = UNetSmall().cuda() 

                elif model_name=="UNet_tv":
                    model = UNet_tv().cuda()
                elif model_name=="ResUnet_tv":
                    model = ResUnet_tv(3).cuda()
                elif model_name=="ResUnetPlusPlus_tv":
                    model = ResUnetPlusPlus_tv(3).cuda()
                elif model_name=="UNetSmall_tv":
                    model = UNetSmall_tv().cuda()



                history = []
                if 'adam' in opt :
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                elif 'sgd' in opt:
                    optimizer = torch.optim.SGD(model.parameters(), lr=10*lr, momentum=0.9)

                #logging.basicConfig(filename='./logs/logger_unet.log', level=logging.INFO)

                total_step = len(train_loader)
                epoidx += 1
                max_valiou=0
                model.train()
                for epoch in range(num_epochs[epoidx]):
                    model.train()
                    if epoch%10==0:
                        print(opt,"_",los,"_",epoch)
                        #lr=lr*0.5
                        #optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    trainloss=0
                    valloss=0
                    totalloss = 0
                    for i, (images, masks, tv_images) in enumerate(train_loader):
                        images = images.cuda()
                        masks = masks.cuda()
                        tv_images= tv_images.cuda()


                        if 'tv' in model_name:
                            outputs = model(images ,tv_images)
                        else:
                            outputs = model(images)


                        

                        if 'bce' in los :
                            masks = onehot(masks)  
                            loss1 = criterion(outputs, masks)
                        elif 'dice' in los :
                            masks = onehot(masks)              
                            loss1 = dice_loss(outputs, masks)
                        elif 'lovasz' in los :
                            masks = onehot(masks) 
                            loss1= L.lovasz_hinge(outputs, masks)
                            
                        elif 'focal' in los :
                            loss1 = focallos(outputs, masks.long())

                        if 'tv' in model_name:
                            loss2=halfBCELossWithSigmod(outputs,tv_img)
                            loss2 = BoundaryLoss(outputs, tv_img)
                            loss = (loss1+loss2)/2
                        else:
                            loss = loss1

                        if is_loss:
                            b_loss = BoundaryLoss(outputs,tv_images)
                            loss = loss+ 0.1*b_loss

                        totalloss += loss*images.size(0)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        if i+1 == total_step:
                            model.eval()
                            train_pa, train_mpa, train_iou, train_fwiou,train_miou ,train_pre, train_rec, train_f1,train_acc = \
                                                cal_iou(model,train_dataset ,model_name)
                            val_pa, val_mpa, val_iou, val_fwiou, val_miou, val_pre, val_rec, val_f1, val_acc= \
                                                cal_iou(model,val_dataset, model_name)

                            val_loss= validation(val_dataset, model ,model_name, is_loss)

                            history.append([totalloss.item()/len(train_dataset), 
                                            train_pa, train_mpa, train_iou, train_fwiou,train_miou,train_pre, train_rec, train_f1,train_acc,val_loss ,
                                            val_pre, val_rec, val_f1, val_acc, val_pa, val_mpa, val_iou, val_fwiou,val_miou])

                            #print(totalloss.item()/len(train_dataset), train_pa, train_mpa, train_miou, train_fwiou, val_pa, val_mpa, val_miou, val_fwiou)
                            trainloss= totalloss.item()/len(train_dataset)
                            valloss=val_loss
                            print("trainLOSS:",trainloss)
                            print("validLOSS:",valloss)
                            
                            
                            if val_miou > max_valiou:
                                max_valiou= val_miou
                                print("save model,val_miou:",val_miou)
                                torch.save(model.state_dict(), './trained_models/unet3_'+model_name+'_'+opt+'_'+los+version+'.pkl')
                        


                    writer.add_scalar("valloss",valloss, epoch)
                    writer.add_scalar("val_iou",val_iou, epoch) 
                    writer.add_scalar("val_miou",val_miou, epoch) 
                    writer.add_scalar("val_rec",val_rec, epoch) 
                    writer.add_scalar("val_pre",val_pre, epoch)
                    writer.add_scalar("val_f1-score",val_f1,epoch)
                    writer.add_scalar("val_acc",val_acc, epoch)

                    writer.add_scalar("trainloss",trainloss, epoch)
                    writer.add_scalar("train_iou",train_iou, epoch) 
                    writer.add_scalar("train_miou",train_miou, epoch) 
                    writer.add_scalar("train_rec",train_rec, epoch) 
                    writer.add_scalar("train_pre",train_pre, epoch)
                    writer.add_scalar("train_f1-score", train_f1, epoch)
                    writer.add_scalar("train_acc",train_acc, epoch)   

                    history_np = np.array(history)
                    #np.save('./logs/unet2_'+model_name+'_'+opt+'_'+los+'.npy',history_np)


                writer.close()
                end = time.time()
                print((end-start)/60)
    return 0


def validation(val_dataset, model ,model_name,is_loss = False):
    criterion =nn.BCEWithLogitsLoss()
    valid_loader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                shuffle=False)

    # switch to evaluate mode
    model.eval()
    totalloss = 0

    # Iterate over data.
    for i, (images, masks, tv_images) in enumerate(valid_loader):
        # get the inputs and wrap in Variable
        images = images.cuda()
        masks = masks.cuda()
        tv_images= tv_images.cuda()

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        if 'tv' in model_name:
            outputs = model(images ,tv_images)
        else:
            outputs = model(images)

        loss = criterion(outputs, masks.long())
        if is_loss:
            loss = loss + BoundaryLoss(outputs, tv_images)
        totalloss += loss*images.size(0)

    model.train()
    return totalloss.item()/len(val_dataset)


if __name__ =='__main__':

    train_data_file = "./data/crack500/train/"
    val_data_file = "/data/crack500/val/"

    ver = "IS_BLOSS"
    #a = main(ver,train_data_file,val_data_file,is_loss=False)
    b = main(ver,train_data_file,val_data_file,is_loss= True)


            
=======
import logging
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from myutils import transform, cal_iou, onehot




from core_tv.res_unet import ResUnet as ResUnet_tv
from core_tv.unetsmall import UNetSmall as UNetSmall_tv
from core_tv.res_unet_plus import ResUnetPlusPlus as ResUnetPlusPlus_tv
from core_tv.unet import Unet_tv as UNet_tv
from core.unet import Unet as UNet
from core.res_unet import ResUnet as ResUnet
from core.unetsmall import UNetSmall as UNetSmall
from core.res_unet_plus import ResUnetPlusPlus as ResUnetPlusPlus

from compare_models.core.models.deeplabv3 import DeepLabV3
from compare_models.core.models.pspnet import PSPNet

from dataset import myDataset
import lovasz_losses as L
from Dice_coeff_loss import dice_loss
from focalloss import FocalLoss
from boundaryloss import BoundaryLoss
from torch.utils.tensorboard import SummaryWriter
from myutils import get_pa


import os

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'





def main(ver,train_data_file,val_data_file,is_loss= False):

    model_name_list = ["UNet_tv"]
    for model_name in model_name_list:
        torch.backends.cudnn.enabled=False
        version = ver
        print("write_begin!!!!!!!!!!!!",model_name)
        writer = SummaryWriter('logs_compare_model/logs'+version+'_'+model_name)
        batch_size = 2
        num_epochs = [200,200,200,200,200]
        num_workers = 1
        lr = 0.001

        losslist = ['focal']
        optimlist = ['sgd']
        iflog = True
        img_size = 64
        train_dataset = myDataset(train_data_file, transform=transform,img_size =img_size)
        val_dataset = myDataset(val_data_file, transform=transform, img_size =img_size)
        train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)
        
        criterion =  nn.BCEWithLogitsLoss()
        focallos = FocalLoss(gamma=2)

        epoidx = -1
        torch.cuda.empty_cache()
        for los in losslist:
            for opt in optimlist:
                start =  time.time()
                print(los, opt)
                torch.manual_seed(77)
                torch.cuda.manual_seed(77)

                if model_name == "deeplabv3":
                    model = DeepLabV3(nclass=2).cuda()
                if model_name == "pspnet":
                    model = PSPNet(nclass=2).cuda()

                if model_name=="UNet":
                    model = UNet().cuda()
                elif model_name=="ResUnet":
                    model = ResUnet(3).cuda()
                elif model_name=="ResUnetPlusPlus":
                    print("model is resunetplusplus")
                    model = ResUnetPlusPlus(3).cuda()
                elif model_name=="UNetSmall":
                    model = UNetSmall().cuda() 

                elif model_name=="UNet_tv":
                    model = UNet_tv().cuda()
                elif model_name=="ResUnet_tv":
                    model = ResUnet_tv(3).cuda()
                elif model_name=="ResUnetPlusPlus_tv":
                    model = ResUnetPlusPlus_tv(3).cuda()
                elif model_name=="UNetSmall_tv":
                    model = UNetSmall_tv().cuda()



                history = []
                if 'adam' in opt :
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                elif 'sgd' in opt:
                    optimizer = torch.optim.SGD(model.parameters(), lr=10*lr, momentum=0.9)

                #logging.basicConfig(filename='./logs/logger_unet.log', level=logging.INFO)

                total_step = len(train_loader)
                epoidx += 1
                max_valiou=0
                model.train()
                for epoch in range(num_epochs[epoidx]):
                    model.train()
                    if epoch%10==0:
                        print(opt,"_",los,"_",epoch)
                        #lr=lr*0.5
                        #optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    trainloss=0
                    valloss=0
                    totalloss = 0
                    for i, (images, masks, tv_images) in enumerate(train_loader):
                        images = images.cuda()
                        masks = masks.cuda()
                        tv_images= tv_images.cuda()


                        if 'tv' in model_name:
                            outputs = model(images ,tv_images)
                        else:
                            outputs = model(images)


                        

                        if 'bce' in los :
                            masks = onehot(masks)  
                            loss1 = criterion(outputs, masks)
                        elif 'dice' in los :
                            masks = onehot(masks)              
                            loss1 = dice_loss(outputs, masks)
                        elif 'lovasz' in los :
                            masks = onehot(masks) 
                            loss1= L.lovasz_hinge(outputs, masks)
                            
                        elif 'focal' in los :
                            loss1 = focallos(outputs, masks.long())

                        if 'tv' in model_name:
                            loss2=halfBCELossWithSigmod(outputs,tv_img)
                            loss2 = BoundaryLoss(outputs, tv_img)
                            loss = (loss1+loss2)/2
                        else:
                            loss = loss1

                        if is_loss:
                            b_loss = BoundaryLoss(outputs,tv_images)
                            loss = loss+ 0.1*b_loss

                        totalloss += loss*images.size(0)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        if i+1 == total_step:
                            model.eval()
                            train_pa, train_mpa, train_iou, train_fwiou,train_miou ,train_pre, train_rec, train_f1,train_acc = \
                                                cal_iou(model,train_dataset ,model_name)
                            val_pa, val_mpa, val_iou, val_fwiou, val_miou, val_pre, val_rec, val_f1, val_acc= \
                                                cal_iou(model,val_dataset, model_name)

                            val_loss= validation(val_dataset, model ,model_name, is_loss)

                            history.append([totalloss.item()/len(train_dataset), 
                                            train_pa, train_mpa, train_iou, train_fwiou,train_miou,train_pre, train_rec, train_f1,train_acc,val_loss ,
                                            val_pre, val_rec, val_f1, val_acc, val_pa, val_mpa, val_iou, val_fwiou,val_miou])

                            #print(totalloss.item()/len(train_dataset), train_pa, train_mpa, train_miou, train_fwiou, val_pa, val_mpa, val_miou, val_fwiou)
                            trainloss= totalloss.item()/len(train_dataset)
                            valloss=val_loss
                            print("trainLOSS:",trainloss)
                            print("validLOSS:",valloss)
                            
                            
                            if val_miou > max_valiou:
                                max_valiou= val_miou
                                print("save model,val_miou:",val_miou)
                                torch.save(model.state_dict(), './trained_models/unet3_'+model_name+'_'+opt+'_'+los+version+'.pkl')
                        


                    writer.add_scalar("valloss",valloss, epoch)
                    writer.add_scalar("val_iou",val_iou, epoch) 
                    writer.add_scalar("val_miou",val_miou, epoch) 
                    writer.add_scalar("val_rec",val_rec, epoch) 
                    writer.add_scalar("val_pre",val_pre, epoch)
                    writer.add_scalar("val_f1-score",val_f1,epoch)
                    writer.add_scalar("val_acc",val_acc, epoch)

                    writer.add_scalar("trainloss",trainloss, epoch)
                    writer.add_scalar("train_iou",train_iou, epoch) 
                    writer.add_scalar("train_miou",train_miou, epoch) 
                    writer.add_scalar("train_rec",train_rec, epoch) 
                    writer.add_scalar("train_pre",train_pre, epoch)
                    writer.add_scalar("train_f1-score", train_f1, epoch)
                    writer.add_scalar("train_acc",train_acc, epoch)   

                    history_np = np.array(history)
                    #np.save('./logs/unet2_'+model_name+'_'+opt+'_'+los+'.npy',history_np)


                writer.close()
                end = time.time()
                print((end-start)/60)
    return 0


def validation(val_dataset, model ,model_name,is_loss = False):
    criterion =nn.BCEWithLogitsLoss()
    valid_loader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                shuffle=False)

    # switch to evaluate mode
    model.eval()
    totalloss = 0

    # Iterate over data.
    for i, (images, masks, tv_images) in enumerate(valid_loader):
        # get the inputs and wrap in Variable
        images = images.cuda()
        masks = masks.cuda()
        tv_images= tv_images.cuda()

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        if 'tv' in model_name:
            outputs = model(images ,tv_images)
        else:
            outputs = model(images)

        loss = criterion(outputs, masks.long())
        if is_loss:
            loss = loss + BoundaryLoss(outputs, tv_images)
        totalloss += loss*images.size(0)

    model.train()
    return totalloss.item()/len(val_dataset)


if __name__ =='__main__':

    train_data_file = "./data/crack500/train/"
    val_data_file = "/data/crack500/val/"

    ver = "IS_BLOSS"
    #a = main(ver,train_data_file,val_data_file,is_loss=False)
    b = main(ver,train_data_file,val_data_file,is_loss= True)


            
>>>>>>> 93fbe5c (first commit)
            