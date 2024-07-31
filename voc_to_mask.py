import glob
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


def voc_json_to_mask(json_path, image_shape):
    # 读取并解析JSON文件
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    # 创建空白掩码图像
    mask = np.zeros(image_shape, dtype=np.uint8)

    # 遍历标注信息并绘制掩码
    for obj in annotations['shapes']:
        points = np.array(obj['points'], dtype=np.int32)
        if obj['shape_type'] == 'polygon':
            cv2.fillPoly(mask, [points], color=1)
        elif obj['shape_type'] == 'rectangle':
            cv2.rectangle(mask, tuple(points[0]), tuple(points[1]), color=1, thickness=-1)
        elif obj['shape_type'] == 'circle':
            center = tuple(points[0])
            radius = int(np.linalg.norm(np.array(points[1]) - np.array(points[0])))
            cv2.circle(mask, center, radius, color=1, thickness=-1)

    return mask


# 示例使用
# json_path = '1.json'
# img=cv2.imread(json_path.replace("json","jpg"))
# image_shape=img.shape[:2]
# # image_shape = (height, width)  # 图像的高度和宽度
# mask = voc_json_to_mask(json_path, image_shape)
#
# cv2.imwrite(json_path.replace(".json","_mask.png"),mask*255)

json_list=glob.glob("pytorch-UNet/data/image/*.json")
for json_path in json_list:
    img = cv2.imread(json_path.replace("json", "jpg"))
    image_shape = img.shape[:2]
    mask = voc_json_to_mask(json_path, image_shape)
    cv2.imwrite(json_path.replace(".json", ".png").replace("image","labels"), mask * 255)



# # 显示掩码图像
# plt.imshow(mask, cmap='gray')
# plt.show()
