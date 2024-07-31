import cv2
import numpy as np


def calculate_hu_moments(binary_image):
    # 计算图像的矩
    moments = cv2.moments(binary_image)

    # 计算Hu不变矩
    # hu_moments = cv2.HuMoments(moments)

    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

    return hu_moments


# 计算两个图像的Hu不变矩
# hu_moments1 = calculate_hu_moments('path_to_first_image')
# hu_moments2 = calculate_hu_moments('path_to_second_image')
#
# # 计算Hu不变矩之间的距离（例如欧氏距离）
# distance = np.sqrt(np.sum((hu_moments1 - hu_moments2) ** 2))
#
# print("Distance between Hu Moments: ", distance)
