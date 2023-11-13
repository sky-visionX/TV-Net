<<<<<<< HEAD
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class myDataset(Dataset):
    def __init__(self, root ,train=True, transform = None, img_size = None):
        Dataset.__init__(self)
        images_dir = os.path.join(root,'images')
        images = os.listdir(images_dir)
        self.images = [os.path.join(images_dir, k) for k in images]
        self.images.sort()
        tv_images_dir=os.path.join(root,'tv_images')
        tv_images=os.listdir(tv_images_dir)
        self.tv_images=[ os.path.join(tv_images_dir,k) for k in tv_images]
        self.tv_images.sort()
        self.img_size = img_size

        if train:
            masks_dir = os.path.join(root,'masks')
            masks = os.listdir(masks_dir)
            self.masks = [os.path.join(masks_dir, k) for k in masks]
            self.masks.sort()
            

        self.transforms = transform
        #self.transforms_tv = transform
        self.transforms_tv = transforms.Compose([
   transforms.Grayscale(num_output_channels=1)
   transforms.ToTensor(),])
        self.train = train

    def add_t(img,tv_img):
        channel=3
        c2=nn.Conv2d(1,channel,1,1)
        tv_img=c2(tv_img)
        return torch.cat((img,tv_img),dim=1)
        

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).resize([self.img_size,self.img_size])
        tv_image_path =self.tv_images[index]
        tv_image = Image.open(tv_image_path).resize([self.img_size,self.img_size])


        if self.transforms is not None:
            image = self.transforms(image)
            tv_image= self.transforms_tv(tv_image)
        image = image
        tv_image= tv_image



        if self.train :
            mask_path = self.masks[index]
            mask = Image.open(mask_path).resize([self.img_size,self.img_size])
            if self.transforms is not None:
                mask = self.transforms(mask)
                #tv_image= tv_image.mean(dim=0).view(1,512,512)
                mask = mask.mean(dim=0).view(1,self.img_size,self.img_size)
                mask[mask>0] = 1
                mask[mask<=0] = 0

            return image, mask, tv_image
        return image ,tv_image
    
    def __len__(self):
        return len(self.images)
=======
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class myDataset(Dataset):
    def __init__(self, root ,train=True, transform = None, img_size = None):
        Dataset.__init__(self)
        images_dir = os.path.join(root,'images')
        images = os.listdir(images_dir)
        self.images = [os.path.join(images_dir, k) for k in images]
        self.images.sort()
        tv_images_dir=os.path.join(root,'tv_images')
        tv_images=os.listdir(tv_images_dir)
        self.tv_images=[ os.path.join(tv_images_dir,k) for k in tv_images]
        self.tv_images.sort()
        self.img_size = img_size

        if train:
            masks_dir = os.path.join(root,'masks')
            masks = os.listdir(masks_dir)
            self.masks = [os.path.join(masks_dir, k) for k in masks]
            self.masks.sort()
            

        self.transforms = transform
        #self.transforms_tv = transform
        self.transforms_tv = transforms.Compose([
   transforms.Grayscale(num_output_channels=1)
   transforms.ToTensor(),])
        self.train = train

    def add_t(img,tv_img):
        channel=3
        c2=nn.Conv2d(1,channel,1,1)
        tv_img=c2(tv_img)
        return torch.cat((img,tv_img),dim=1)
        

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).resize([self.img_size,self.img_size])
        tv_image_path =self.tv_images[index]
        tv_image = Image.open(tv_image_path).resize([self.img_size,self.img_size])


        if self.transforms is not None:
            image = self.transforms(image)
            tv_image= self.transforms_tv(tv_image)
        image = image
        tv_image= tv_image



        if self.train :
            mask_path = self.masks[index]
            mask = Image.open(mask_path).resize([self.img_size,self.img_size])
            if self.transforms is not None:
                mask = self.transforms(mask)
                #tv_image= tv_image.mean(dim=0).view(1,512,512)
                mask = mask.mean(dim=0).view(1,self.img_size,self.img_size)
                mask[mask>0] = 1
                mask[mask<=0] = 0

            return image, mask, tv_image
        return image ,tv_image
    
    def __len__(self):
        return len(self.images)
>>>>>>> 93fbe5c (first commit)
    