import glob
import os

import cv2
import numpy as np
import torch

from net import *
from utils import *
from data import *
from torchvision.utils import save_image
from PIL import Image
from unet_1 import UNet1
# net=UNet(2).cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet1(n_classes=2).to(device)

weight_path = 'params/unet.pth'
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('successful load weight！')
else:
    print('not successful load weight')



img_list=glob.glob("data/JPEGImages/*.jpg")
img_list=["/home/boer/xugeofei-project/ketu_symmetry/pics/baie_10.jpg"]
for img_path in img_list:
    img = keep_image_size_open_rgb(img_path, size=(512, 512))
    img_data=transform(img).cuda()
    img_data=torch.unsqueeze(img_data,dim=0)
    # net.eval()
    # with torch.no_grad():
    #     out=net(img_data)
    out=net(img_data)
    out=torch.argmax(out[0], dim=0).unsqueeze(0) * 255
    out=(out).permute((1,2,0)).cpu().detach().numpy()
    # cv2.imshow('out',np.uint8(out))
    # cv2.waitKey(0)


    img_shape = cv2.imread(img_path).shape
    A = cv2.resize(out, (max(img_shape), max(img_shape)), interpolation=cv2.INTER_NEAREST)
    # A=cv2.resize(out, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_path=img_path.split("/")[-1].replace(".jpg",'.png')
    cv2.imwrite(mask_path, A[:img_shape[0],:])

    # cv2.imwrite(mask_path,out*255)
    # cv2.imshow('out',A)
    # cv2.waitKey(0)

