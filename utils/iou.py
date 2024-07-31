import cv2
import numpy as np
from matplotlib import pyplot as plt


def calculate_iou(mask1, mask2):
    # 确保两张图片的大小和类型相同
    if mask1.shape != mask2.shape:
        raise ValueError("The size of the two masks must be the same.")

    # 计算交集和并集
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    # 计算交并比
    iou = np.sum(intersection) / np.sum(union)

    return iou

def calculate_iou_center_aligned(mask1, mask2, label_value=255):

    # 获取两张图片的大小
    h1, w1 = mask1.shape[:2]
    h2, w2 = mask2.shape[:2]

    # 计算中点
    center1 = (w1 // 2, h1 // 2)
    center2 = (w2 // 2, h2 // 2)

    # 创建空白画布，大小为两张图片尺寸之和
    canvas_height = max(h1, h2) * 2
    canvas_width = max(w1, w2) * 2
    canvas1 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    canvas2 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # 计算放置位置的偏移量
    offset1_x = (canvas_width // 2) - center1[0]
    offset1_y = (canvas_height // 2) - center1[1]
    offset2_x = (canvas_width // 2) - center2[0]
    offset2_y = (canvas_height // 2) - center2[1]

    # 在画布上放置第一张掩码图像
    canvas1[offset1_y:offset1_y + h1, offset1_x:offset1_x + w1] = mask1

    # 在画布上放置第二张掩码图像
    canvas2[offset2_y:offset2_y + h2, offset2_x:offset2_x + w2] = mask2

    # 将掩码图像转换为二值图像
    mask1_binary = (canvas1 == label_value).astype(np.uint8)
    mask2_binary = (canvas2 == label_value).astype(np.uint8)


    #
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title('Upper Half')
    # plt.imshow(cv2.cvtColor(mask1_binary*255, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.title('Lower Half')
    # plt.imshow(cv2.cvtColor(mask2_binary*255, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    # 计算交集
    intersection = np.logical_and(mask1_binary, mask2_binary).astype(np.uint8)

    # 计算并集
    union = np.logical_or(mask1_binary, mask2_binary).astype(np.uint8)

    # 计算IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou



from skimage.metrics import structural_similarity as ssim

def calculate_ssim_center_aligned(mask1, mask2, label_value=255):

    # 获取两张图片的大小
    h1, w1 = mask1.shape[:2]
    h2, w2 = mask2.shape[:2]

    # 计算中点
    center1 = (w1 // 2, h1 // 2)
    center2 = (w2 // 2, h2 // 2)

    # 创建空白画布，大小为两张图片尺寸之和
    canvas_height = max(h1, h2) * 2
    canvas_width = max(w1, w2) * 2
    canvas1 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    canvas2 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # 计算放置位置的偏移量
    offset1_x = (canvas_width // 2) - center1[0]
    offset1_y = (canvas_height // 2) - center1[1]
    offset2_x = (canvas_width // 2) - center2[0]
    offset2_y = (canvas_height // 2) - center2[1]

    # 在画布上放置第一张掩码图像
    canvas1[offset1_y:offset1_y + h1, offset1_x:offset1_x + w1] = mask1

    # 在画布上放置第二张掩码图像
    canvas2[offset2_y:offset2_y + h2, offset2_x:offset2_x + w2] = mask2

    # 将掩码图像转换为二值图像
    mask1_binary = (canvas1 == label_value).astype(np.uint8)
    mask2_binary = (canvas2 == label_value).astype(np.uint8)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title('Upper Half')
    # plt.imshow(cv2.cvtColor(mask1_binary*255, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.title('Lower Half')
    # plt.imshow(cv2.cvtColor(mask2_binary*255, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    similarity_index, diff = ssim(mask1_binary, mask2_binary, full=True)

    return similarity_index





def calculate_structure_center_aligned(mask1, mask2, label_value=255):

    # 获取两张图片的大小
    h1, w1 = mask1.shape[:2]
    h2, w2 = mask2.shape[:2]

    # 计算中点
    center1 = (w1 // 2, h1 // 2)
    center2 = (w2 // 2, h2 // 2)

    # 创建空白画布，大小为两张图片尺寸之和
    canvas_height = max(h1, h2) * 2
    canvas_width = max(w1, w2) * 2
    canvas1 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    canvas2 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # 计算放置位置的偏移量
    offset1_x = (canvas_width // 2) - center1[0]
    offset1_y = (canvas_height // 2) - center1[1]
    offset2_x = (canvas_width // 2) - center2[0]
    offset2_y = (canvas_height // 2) - center2[1]

    # 在画布上放置第一张掩码图像
    canvas1[offset1_y:offset1_y + h1, offset1_x:offset1_x + w1] = mask1

    # 在画布上放置第二张掩码图像
    canvas2[offset2_y:offset2_y + h2, offset2_x:offset2_x + w2] = mask2

    # 将掩码图像转换为二值图像
    mask1 = (canvas1 == label_value).astype(np.uint8)
    mask2 = (canvas2 == label_value).astype(np.uint8)

    # 计算图像块的均值和标准差
    mu1 = np.mean(mask1)
    mu2 = np.mean(mask2)
    sigma1 = np.std(mask1)
    sigma2 = np.std(mask2)
    sigma12 = np.mean((mask1 - mu1) * (mask2 - mu2))

    # 稳定常数
    C3 = (0.03 * 255) ** 2 / 2

    # 计算结构相似性
    structure_similarity = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    return structure_similarity
