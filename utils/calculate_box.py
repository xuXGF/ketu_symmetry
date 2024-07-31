import cv2
import numpy as np

# from iou import calculate_iou_center_aligned
# from split_Mirror import split_image_horizontally


def bounding_box(mask, label_value=255):
    # 找到标签位置
    points = np.column_stack(np.where(mask == label_value))
    # 如果没有找到标签，则返回
    if points.size == 0:
        print("Label not found in the mask.")
        return
    # 计算边界框
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return mask[x_min:x_max,y_min:y_max]








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

