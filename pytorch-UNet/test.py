import glob

import cv2

from utils.Select_largest_contour import Select_largest_contour
from data import *
from unet_1 import UNet1
# net=UNet(2).cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet1(n_classes=2).to(device)
# net = UNet1(n_classes=2)

weight_path = 'params/unet.pth'
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('successful load weight！')
else:
    print('not successful load weight')


img_list=glob.glob("../img/*/*.jpg")
# img_list=glob.glob("data/JPEGImages/*.jpg")
# img_list=["/home/boer/xugeofei-project/ketu_symmetry/pics/baie_10.jpg"]
for img_path in img_list:
    img = keep_image_size_open_rgb(img_path, size=(512, 512))
    img_data=transform(img).cuda()
    # img_data = transform(img)
    img_data=torch.unsqueeze(img_data,dim=0)
    # net.eval()
    # with torch.no_grad():
    #     out=net(img_data)
    out=net(img_data)
    out=torch.argmax(out[0], dim=0).unsqueeze(0) * 255
    # out=torch.argmax(out,dim=1)
    # out=torch.squeeze(out,dim=0)
    # out=out.unsqueeze(dim=0)
    # print(set((out).reshape(-1).tolist()))
    out=(out).permute((1,2,0)).cpu().detach().numpy()
    # mask_path=img_path.split("/")[-1].replace(".jpg",'.png')
    mask_path=img_path.replace("img","mask_result").replace("jpg","png")
    img_shape = cv2.imread(img_path).shape

    A = cv2.resize(out, (max(img_shape), max(img_shape)), interpolation=cv2.INTER_NEAREST)
    # A=cv2.resize(out, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
    A=Select_largest_contour(A[:img_shape[0],:])
    cv2.imwrite(mask_path, A)
    # cv2.imwrite(mask_path, A[:img_shape[0],:])

    # cv2.imwrite(mask_path,out*255)
    # cv2.imshow('out',A)
    # cv2.waitKey(0)

