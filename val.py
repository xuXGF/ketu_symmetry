import glob
import os

import torch
from PIL import Image

from utils.iou import *
from utils.Select_largest_contour import Select_largest_contour
from unet_1 import UNet1
from utils.calculate_box import bounding_box
from utils.hsah_adp import *
from utils.split_Mirror import split_image_horizontally


def keep_image_size_open_rgb(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor()
])


# net=UNet(2).cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet1(n_classes=2).to(device)
# net = UNet1(n_classes=2)

weight_path = 'pytorch-UNet/params/unet.pth'
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('successful load weight！')
else:
    print('not successful load weight')



yes_list = []
no_list = []
y,n=0,0

# img_list=glob.glob("img/*/*.jpg")
img_list=glob.glob("data/imgs/*.jpg")

for img_path in img_list:
    mask_label_path=img_path.replace('imgs','masks').replace('.jpg','_mask.png')
    # mask_label_path=img_path.replace('img','masks').replace('.jpg','_mask.png')
    img = keep_image_size_open_rgb(img_path, size=(512, 512))
    img_data=transform(img).cuda()
    # img_data = transform(img)
    img_data=torch.unsqueeze(img_data,dim=0)
    # net.eval()
    # with torch.no_grad():
    #     out=net(img_data)
    out=net(img_data)
    out=torch.argmax(out[0], dim=0).unsqueeze(0) * 255
    out=(out).permute((1,2,0)).cpu().detach().numpy()
    img_shape = cv2.imread(img_path).shape
    A = cv2.resize(out, (max(img_shape), max(img_shape)), interpolation=cv2.INTER_NEAREST)
    A=Select_largest_contour(A[:img_shape[0],:])
    # cv2.imwrite('1.png', A)

    mask_label=cv2.imread(mask_label_path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('new')
    plt.imshow(cv2.cvtColor(np.uint8(A), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('old')
    plt.imshow(cv2.cvtColor(mask_label*255, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    similarity_index=calculate_iou(A,mask_label)
    print(similarity_index)


