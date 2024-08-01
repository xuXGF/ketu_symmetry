
import torch
import cv2
import numpy as np
from skimage import transform

from utils.iou import *
from utils.Select_largest_contour import Select_largest_contour
from utils.calculate_box import bounding_box
from utils.hsah_adp import *
from utils.split_Mirror import split_image_horizontally

from PIL import Image
import glob
from torch.autograd import Variable

from U2_Netp import U2NETP # small version u2net 4.7 MB

model_dir='U2_Netp/saved_models/u2netp/u2netp_bce_itr_16500_train_0.019957_tar_0.000009.pth'
model_dir='U2_Netp/saved_models/u2netp/u2netp_bce_itr_19000_train_0.019562_tar_0.000004.pth'


net = U2NETP(3,1)
cuda=True
if cuda:
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
# net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
net.eval()


yes_list = []
no_list = []
y,n=0,0


# img_list=glob.glob("img/*/*.jpg")
img_list=glob.glob('data/imgs/*.jpg')

for img_path in img_list:
    mask_label_path = img_path.replace('imgs', 'masks').replace('.jpg', '.png')
    # mask_label_path=img_path.replace('img','masks').replace('.jpg','.png')
    img=cv2.imread(img_path)
    output_size=320
    image = transform.resize(img, (output_size, output_size), mode='constant')

    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    image = image / np.max(image)
    if image.shape[2] == 1:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
    else:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
    tmpImg = tmpImg.transpose((2, 0, 1))
    img_data=torch.from_numpy(tmpImg)
    del tmpImg
    img_data = img_data.type(torch.FloatTensor)
    if cuda:
        img_data = Variable(img_data.cuda())
    else:
        img_data = Variable(img_data)
    # img_data = Variable(img_data)

    img_data=torch.unsqueeze(img_data,dim=0)
    # net.eval()
    with torch.no_grad():
        d1,d2,d3,d4,d5,d6,d7=net(img_data)
    pred = d1[:, 0, :, :]
    pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))

    out=(pred).permute((1,2,0)).cpu().detach().numpy()
    img_shape = cv2.imread(img_path).shape
    A = cv2.resize(out, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
    A=Select_largest_contour(A)
    # cv2.imwrite('1.png', A)

    mask_label=cv2.imread(mask_label_path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('new')
    plt.imshow(cv2.cvtColor(np.uint8(A), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('old')
    plt.imshow(cv2.cvtColor(mask_label, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    similarity_index=calculate_iou(A,mask_label)
    print(similarity_index)


