import cv2
import numpy as np

# 读取图片并转换为灰度图像
# image = cv2.imread('/home/boer/xugeofei-project/ketu_symmetry/mask_result/yes/xusiyuan_1749.png')  # 替换为你的图片路径
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def Select_largest_contour(gray):
    # 二值化处理
    # _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, _ = cv2.findContours(np.uint8(gray), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 根据轮廓面积排序，取出最大的两块区域
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    # 创建空白图像
    result = np.zeros_like(gray)
    # 保留大于阈值的白色区域
    for contour in contours:
        # cv2.drawContours(result, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        cv2.drawContours(result, [contour], -1, 255, thickness=cv2.FILLED)

    return result


