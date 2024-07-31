import glob

from utils.calculate_box import bounding_box
from utils.hsah_adp import *
from utils.iou import *
from utils.split_Mirror import split_image_horizontally



# image_path = '1_mask.png'
# upper_half, lower_half = split_image_horizontally(image_path)
# upper_result_image = bounding_box(upper_half)
# lower_result_image = bounding_box(lower_half)
#
# similarity_index=calculate_iou_center_aligned(upper_result_image,lower_result_image)
# print(similarity_index)
#
# similarity_index=calculate_ssim_center_aligned(upper_result_image,lower_result_image)
# print(similarity_index)
# print(f"The similarity index between the mirror images is: {similarity_index:.4f}")
#
# if similarity_index == 1:
#     print("The images are perfect mirror images of each other.")
# elif similarity_index > 0.75:
#     print("The images are highly similar mirror images.")
# elif similarity_index > 0.5:
#     print("The images are somewhat similar mirror images.")
# else:
#     print("The images are not similar mirror images.")


yes_list = []
no_list = []
y,n=0,0
mask_list=glob.glob("mask_result/*/*.png")
for image_path in mask_list:
    print(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    upper_half, lower_half = split_image_horizontally(img)
    upper_result_image = bounding_box(upper_half)
    lower_result_image = bounding_box(lower_half)


    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title('Upper Half')
    # plt.imshow(cv2.cvtColor(upper_result_image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.title('Lower Half')
    # plt.imshow(cv2.cvtColor(lower_result_image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()


    #
    # hu_moments1=calculate_hu_moments(upper_result_image)
    # hu_moments2=calculate_hu_moments(lower_result_image)
    # # distance = np.sqrt(np.sum((hu_moments1 - hu_moments2) ** 2))
    # distance = np.linalg.norm(hu_moments1 - hu_moments2)
    # print(('yes' if np.float64(distance)>))
    # print(distance)

    image1=upper_result_image
    image2=lower_result_image
    start1 = time.time()
    d_dist = Hamming_distance(dHash(image1), dHash(image2))
    end1 = time.time()

    start2 = time.time()
    p_dist = Hamming_distance(pHash(image1), pHash(image2))
    end2 = time.time()

    start3 = time.time()
    a_dist = Hamming_distance(aHash(image1), aHash(image2))
    end3 = time.time()

    # print('a_dist is ' + '%d' % a_dist + ', similarity is ' + '%f' % (1 - a_dist * 1.0 / 64) + ', time is ' + '%f' % (
    #             end3 - start3))
    # print('p_dist is ' + '%d' % p_dist + ', similarity is ' + '%f' % (1 - p_dist * 1.0 / 64) + ', time is ' + '%f' % (
    #             end2 - start2))
    print('%f' % (1 - p_dist * 1.0 / 64))
    a=1 - p_dist * 1.0 / 64
    # print('d_dist is ' + '%d' % d_dist + ', similarity is ' + '%f' % (1 - d_dist * 1.0 / 64) + ', time is ' + '%f' % (
    #             end1 - start1))


    similarity_index=calculate_iou_center_aligned(upper_result_image,lower_result_image)
    # print(similarity_index)
    # similarity_index=calculate_ssim_center_aligned(upper_result_image,lower_result_image)

    # print(similarity_index)
    # similarity_index=calculate_structure_center_aligned(upper_result_image,lower_result_image)
    # print(similarity_index)
    # print(f"The similarity index between the mirror images is: {similarity_index:.4f}")
    a = similarity_index
    x = 0.8 #0.996 ssim
    # x = 0.9995
    if "yes" in image_path:
        yes_list.append(a)
        if a > x:
            y += 1
        else:
            n += 1
    elif "no" in image_path:
        no_list.append(a)
        if a < x:
            y += 1
        else:
            n += 1

print('yes', yes_list)
print('no', no_list)
print(y, n)