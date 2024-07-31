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

img_list=glob.glob("img/*/*.jpg")
img_list=['data/imgs/1.jpg']
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
    out=(out).permute((1,2,0)).cpu().detach().numpy()
    img_shape = cv2.imread(img_path).shape
    A = cv2.resize(out, (max(img_shape), max(img_shape)), interpolation=cv2.INTER_NEAREST)
    A=Select_largest_contour(A[:img_shape[0],:])
    # cv2.imwrite('1.png', A)

    # cv2.imshow('out',np.uint8(A[:img_shape[0],:]))
    # cv2.waitKey(0)
    # plt.figure(figsize=(10, 5))
    # plt.title('Upper Half')
    # plt.imshow(cv2.cvtColor(np.uint8(A[:img_shape[0],:]), cv2.COLOR_BGR2RGB))
    # plt.axis('off')

    upper_half, lower_half = split_image_horizontally(A)
    upper_result_image = bounding_box(upper_half)
    lower_result_image = bounding_box(lower_half)


    image1 = np.uint8(upper_result_image)
    image2 = np.uint8(lower_result_image)
    # start1 = time.time()
    # d_dist = Hamming_distance(dHash(image1), dHash(image2))
    # end1 = time.time()
    #
    # start2 = time.time()
    # p_dist = Hamming_distance(pHash(image1), pHash(image2))
    # end2 = time.time()
    #
    # start3 = time.time()
    # a_dist = Hamming_distance(aHash(image1), aHash(image2))
    # end3 = time.time()

    # print('a_dist is ' + '%d' % a_dist + ', similarity is ' + '%f' % (1 - a_dist * 1.0 / 64) + ', time is ' + '%f' % (
    #             end3 - start3))
    # print('p_dist is ' + '%d' % p_dist + ', similarity is ' + '%f' % (1 - p_dist * 1.0 / 64) + ', time is ' + '%f' % (
    #             end2 - start2))
    # print('%f' % (1 - p_dist * 1.0 / 64))
    # a = 1 - p_dist * 1.0 / 64
    # print('d_dist is ' + '%d' % d_dist + ', similarity is ' + '%f' % (1 - d_dist * 1.0 / 64) + ', time is ' + '%f' % (
    #             end1 - start1))


    #
    # hu_moments1=calculate_hu_moments(upper_result_image)
    # hu_moments2=calculate_hu_moments(lower_result_image)
    # # distance = np.sqrt(np.sum((hu_moments1 - hu_moments2) ** 2))
    # distance = np.linalg.norm(hu_moments1 - hu_moments2)
    # print(('yes' if np.float64(distance)>))
    # print(distance)

    similarity_index=calculate_iou_center_aligned(upper_result_image,lower_result_image)
    print(similarity_index)
    similarity_index = calculate_ssim_center_aligned(upper_result_image, lower_result_image)

    # print(similarity_index)
    similarity_index = calculate_structure_center_aligned(upper_result_image, lower_result_image)
    # print(similarity_index)
    # print(f"The similarity index between the mirror images is: {similarity_index:.4f}")
    a = similarity_index
    # x = 0.8 #0.996 ssim
    x = 0.9995
    if "yes" in img_path:
        yes_list.append(a)
        if a > x:
            y += 1
        else:
            n += 1
    elif "no" in img_path:
        no_list.append(a)
        if a < x:
            y += 1
        else:
            n += 1

print('yes', yes_list)
print('no', no_list)
print(y, n)

