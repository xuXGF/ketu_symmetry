import cv2
import numpy as np
import glob

dir = r'/pytorch-UNet/data/SegmentationClass/*'

for file in glob.glob(dir):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    print(np.max(img))
    img = img*255
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.imwrite(file.replace('SegmentationClass','labels'),img)
