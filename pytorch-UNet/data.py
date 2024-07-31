import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name).replace("png","jpg")
        # image = Image.open(image_path)
        # segment_image = Image.open(segment_path)
        segment_image = keep_image_size_open(segment_path, size=(512, 512))
        image = keep_image_size_open_rgb(image_path, size=(512, 512))
        return transform(image), torch.Tensor(np.array(segment_image))
        # return transform(image), transform(segment_image)

    # cv2.imwrite("111.jpg", np.array(transform(image).permute(1, 2, 0)) * 255)


if __name__ == '__main__':
    from torch.nn.functional import one_hot
    data = MyDataset('data')
    print(data[0][0].shape)
    print(data[0][1].shape)
    out=one_hot(data[0][1].long())
    print(out.shape)
