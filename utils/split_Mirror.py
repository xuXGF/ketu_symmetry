import cv2


def split_image_horizontally(img):
    # 读取图片
    # img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    # 获取图片的高度和宽度
    height, width = img.shape[:2]

    # 计算中间位置
    middle = int(width // 2)

    # 水平分割图片
    upper_half = img[:, :middle]
    lower_half = img[:, middle:]

    #对图片进行镜像翻转
    upper_half = cv2.flip(upper_half, 1)

    return upper_half, lower_half


# # 示例使用
# image_path = '1_mask.png'
#
# upper_half, lower_half = split_image_horizontally(image_path)
# print(upper_half.shape)
# # 显示分割后的两张图片
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Upper Half')
# plt.imshow(cv2.cvtColor(upper_half, cv2.COLOR_BGR2RGB))
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.title('Lower Half')
# plt.imshow(cv2.cvtColor(lower_half, cv2.COLOR_BGR2RGB))
# plt.axis('off')
#
# plt.show()
#
#
# def calculate_mirror_similarity(img1, img2):
#     # 读取两张图片
#     # img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
#     # img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
#
#     # 检查两张图片是否大小相同
#     if img1.shape != img2.shape:
#         raise ValueError("Images must have the same dimensions")
#
#     # 将第一张图片水平翻转
#     flipped_img1 = cv2.flip(img1, 1)
#
#     # 计算结构相似性指数 (SSIM)
#     similarity_index, diff = ssim(flipped_img1, img2, full=True)
#
#     return similarity_index
#
#
# # # 示例使用
# # img1_path = 'path/to/your/image1.jpg'
# # img2_path = 'path/to/your/image2.jpg'
# similarity_index = calculate_mirror_similarity(upper_half, lower_half)
#
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
