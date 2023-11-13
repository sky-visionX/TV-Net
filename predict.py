<<<<<<< HEAD
import os
from pyexpat import model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np



from core_tv.res_unet import ResUnet as ResUnet_tv
from core_tv.unetsmall import UNetSmall as UNetSmall_tv
from core_tv.unet import Unet_tv as UNet_tv
from core.unet import Unet as UNet
from core.res_unet import ResUnet as ResUnet
from core.unetsmall import UNetSmall as UNetSmall
from dataset import myDataset
from myutils import transform, cal_iou, onehot,get_pa

from compare_models.core.models.deeplabv3 import DeepLabV3
from compare_models.core.models.pspnet import PSPNet

torch.cuda.manual_seed(777)
torch.manual_seed(777)
#pathall= #GAPS384 #crackforest_data #crack500
path ='./data/crack500/test/'
img_folder ='./data/crack500/test/images/'
img_dir = os.listdir(img_folder)
img_list = [img_folder+k for k in img_dir]
img_list.sort()


transforms_tv = transforms.Compose([
   transforms.Grayscale(num_output_channels=1),
   transforms.ToTensor(),])

model_name_list = ["deeplabv3","pspnet","UNet"]

for model_name in model_name_list:

    if model_name == "deeplabv3":
        model = DeepLabV3(nclass=2).cuda()
    if model_name == "pspnet":
        model = PSPNet(nclass=2).cuda()
    if model_name=="UNet":
        model = UNet()
    elif model_name=="ResUnet":
        model = ResUnet(3)
    elif model_name=="UNetSmall":
        model = UNetSmall()
    elif model_name=="UNet_tv":
        model = UNet_tv()
    elif model_name=="ResUnet_tv":
        model = ResUnet_tv(3)
    elif model_name=="UNetSmall_tv":
        model = UNetSmall_tv()

    


    #model = EfficientNet.from_pretrained(‘efficientnet-b0’)
    if model_name == "deeplabv3":
        model.load_state_dict(torch.load('trained_models/x.pkl'))

    if model_name == "pspnet":
        model.load_state_dict(torch.load('trained_models/x.pkl'))

    if model_name=="UNet":
        model.load_state_dict(torch.load('trained_models/x.pkl'))
    elif model_name=="ResUnet":
        model.load_state_dict(torch.load('trained_models/x.pkl'))
    elif model_name=="UNetSmall":
        model.load_state_dict(torch.load('trained_models/x.pkl'))
    elif model_name=="UNet_tv":
        model.load_state_dict(torch.load('trained_models/x.pkl'))
    elif model_name=="ResUnet_tv":
        model.load_state_dict(torch.load('trained_models/x.pkl'))
    elif model_name=="UNetSmall_tv":
        model.load_state_dict(torch.load('trained_models/x.pkl'))
    
    
    model=model.cuda()
    model.eval()
    test_dataset = myDataset(path ,transform=transform,img_size=128)

    test_pa, test_mpa, test_iou, test_fwiou,test_miou ,test_pre, test_rec, test_f1, test_acc,test_dice= cal_iou(model ,test_dataset ,model_name)
    print("path:",path,"model_name:",model_name)
    print( "test_miou:",test_iou,"test_dice:",test_dice)
    print("test_pre:",test_pre, "test_rec:",test_rec, "test_f1:",test_f1)


    for file in img_list:
        tv_img_name=str.split(file,'/')[-1]
        tvimage_path=os.path.join(path,'tv_images',tv_img_name)
        tv_img= Image.open(tvimage_path).resize([256,256])
        tv_img= transforms_tv(tv_img).unsqueeze(0).cuda()

        img = Image.open(file).resize([256,256])
        img = transform(img).unsqueeze(0).cuda()


        model.eval()
        with torch.no_grad():
            if 'tv' in model_name:

                pred = model(img,tv_img)
            else:
                pred = model(img)
        pred = torch.argmax(pred,1)
        pred = pred.squeeze().cpu().numpy()
        pred = np.uint8(pred*255)
        pred_img = Image.fromarray(pred)
        pred_img = pred_img.resize([640,360])
        img_name = str.split(file, '/')[-1]
        #img_name = model_name+ img_name
        img_path = path +'pred/'+model_name+'/'
        img_name =  img_path + img_name
        #print(img_name)
        pred_img.save(img_name)
=======
import os
from pyexpat import model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np



from core_tv.res_unet import ResUnet as ResUnet_tv
from core_tv.unetsmall import UNetSmall as UNetSmall_tv
from core_tv.unet import Unet_tv as UNet_tv
from core.unet import Unet as UNet
from core.res_unet import ResUnet as ResUnet
from core.unetsmall import UNetSmall as UNetSmall
from dataset import myDataset
from myutils import transform, cal_iou, onehot,get_pa

from compare_models.core.models.deeplabv3 import DeepLabV3
from compare_models.core.models.pspnet import PSPNet

torch.cuda.manual_seed(777)
torch.manual_seed(777)
#pathall= #GAPS384 #crackforest_data #crack500
path ='./data/crack500/test/'
img_folder ='./data/crack500/test/images/'
img_dir = os.listdir(img_folder)
img_list = [img_folder+k for k in img_dir]
img_list.sort()


transforms_tv = transforms.Compose([
   transforms.Grayscale(num_output_channels=1),
   transforms.ToTensor(),])

model_name_list = ["deeplabv3","pspnet","UNet"]

for model_name in model_name_list:

    if model_name == "deeplabv3":
        model = DeepLabV3(nclass=2).cuda()
    if model_name == "pspnet":
        model = PSPNet(nclass=2).cuda()
    if model_name=="UNet":
        model = UNet()
    elif model_name=="ResUnet":
        model = ResUnet(3)
    elif model_name=="UNetSmall":
        model = UNetSmall()
    elif model_name=="UNet_tv":
        model = UNet_tv()
    elif model_name=="ResUnet_tv":
        model = ResUnet_tv(3)
    elif model_name=="UNetSmall_tv":
        model = UNetSmall_tv()

    


    #model = EfficientNet.from_pretrained(‘efficientnet-b0’)
    if model_name == "deeplabv3":
        model.load_state_dict(torch.load('trained_models/x.pkl'))

    if model_name == "pspnet":
        model.load_state_dict(torch.load('trained_models/x.pkl'))

    if model_name=="UNet":
        model.load_state_dict(torch.load('trained_models/x.pkl'))
    elif model_name=="ResUnet":
        model.load_state_dict(torch.load('trained_models/x.pkl'))
    elif model_name=="UNetSmall":
        model.load_state_dict(torch.load('trained_models/x.pkl'))
    elif model_name=="UNet_tv":
        model.load_state_dict(torch.load('trained_models/x.pkl'))
    elif model_name=="ResUnet_tv":
        model.load_state_dict(torch.load('trained_models/x.pkl'))
    elif model_name=="UNetSmall_tv":
        model.load_state_dict(torch.load('trained_models/x.pkl'))
    
    
    model=model.cuda()
    model.eval()
    test_dataset = myDataset(path ,transform=transform,img_size=128)

    test_pa, test_mpa, test_iou, test_fwiou,test_miou ,test_pre, test_rec, test_f1, test_acc,test_dice= cal_iou(model ,test_dataset ,model_name)
    print("path:",path,"model_name:",model_name)
    print( "test_miou:",test_iou,"test_dice:",test_dice)
    print("test_pre:",test_pre, "test_rec:",test_rec, "test_f1:",test_f1)


    for file in img_list:
        tv_img_name=str.split(file,'/')[-1]
        tvimage_path=os.path.join(path,'tv_images',tv_img_name)
        tv_img= Image.open(tvimage_path).resize([256,256])
        tv_img= transforms_tv(tv_img).unsqueeze(0).cuda()

        img = Image.open(file).resize([256,256])
        img = transform(img).unsqueeze(0).cuda()


        model.eval()
        with torch.no_grad():
            if 'tv' in model_name:

                pred = model(img,tv_img)
            else:
                pred = model(img)
        pred = torch.argmax(pred,1)
        pred = pred.squeeze().cpu().numpy()
        pred = np.uint8(pred*255)
        pred_img = Image.fromarray(pred)
        pred_img = pred_img.resize([640,360])
        img_name = str.split(file, '/')[-1]
        #img_name = model_name+ img_name
        img_path = path +'pred/'+model_name+'/'
        img_name =  img_path + img_name
        #print(img_name)
        pred_img.save(img_name)
>>>>>>> 93fbe5c (first commit)
