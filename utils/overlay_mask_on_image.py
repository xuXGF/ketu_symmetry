import cv2
import numpy as np

def overlay_mask_on_image(image, mask, color=(0, 255, 0)):
    # 确保图片和mask大小相同
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("The size of the image and the mask must be the same.")

    # 创建一个三通道的mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask != 0] = color

    # 将mask叠加到原图上
    overlay = cv2.addWeighted(image, 1, colored_mask, 0.4, 0)
    return overlay

# 读取原图和mask图
# image = cv2.imread('../data/imgs/1.jpg')  # 替换为你的图片路径
# mask = cv2.imread('../U2_Netp/results/1.png', cv2.IMREAD_GRAYSCALE)  # 替换为你的mask路径

image = cv2.imread('../pytorch-UNet/data/JPEGImages/chenjia_94.jpg')  # 替换为你的图片路径
mask = cv2.imread('../pytorch-UNet/data/labels/chenjia_94.png', cv2.IMREAD_GRAYSCALE)  # 替换为你的mask路径

# 确保mask是二值化的
_, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

# 将mask叠加到原图上
overlay_image = overlay_mask_on_image(image, mask)
cv2.imwrite("../1.jpg", overlay_image)

# 显示结果
cv2.imshow('Overlay Image', overlay_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
