

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

model_dir='U2_Netp/saved_models/u2netp/u2netp_bce_itr_18700_train_0.019434_tar_0.000004.pth'
net = U2NETP(3,1)
# if torch.cuda.is_available():
#     net.load_state_dict(torch.load(model_dir))
#     net.cuda()
# else:
#     net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
net.eval()


yes_list = []
no_list = []
y,n=0,0


# img_list=glob.glob("img/*/*.jpg")
img_list=glob.glob('data/imgs/1.jpg')
# img_list=glob.glob("pics_no_yes/no/*.jpg")
for img_path in img_list:
    print(img_path)
    img_path = '/home/boer/xugeofei-project/ketu_symmetry/pics/baie_10.jpg'
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
    # if torch.cuda.is_available():
    #     inputs_test = Variable(inputs_test.cuda())
    # else:
    #     inputs_test = Variable(inputs_test)
    img_data = Variable(img_data)

    img_data=torch.unsqueeze(img_data,dim=0)
    # net.eval()
    with torch.no_grad():
        d1,d2,d3,d4,d5,d6,d7=net(img_data)
    pred = d1[:, 0, :, :]
    pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))

    out=(pred).permute((1,2,0)).cpu().detach().numpy()
    # out = (pred.squeeze(0)).permute((1,0)).cpu().detach().numpy()
    # A = np.uint8((out > 0.5).astype(float) * 255)
    img_shape = cv2.imread(img_path).shape
    A = cv2.resize(out, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
    A = (A > 0.5).astype(float)*255
    A=Select_largest_contour(A)
    cv2.imwrite('1.png', A)

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

