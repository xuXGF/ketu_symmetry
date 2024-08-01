import os
from skimage import io
import torch

from torch.autograd import Variable

from torch.utils.data import  DataLoader
from torchvision import transforms#, utils

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_dir='u2netp.pth'# fixed as u2netp
    img_name_list = glob.glob("/home/boer/xugeofei-project/ketu_symmetry/pics_no_yes/yes/yes_24.jpg")

    model_dir = 'saved_models/u2netp/u2netp_bce_itr_18700_train_0.019434_tar_0.000004.pth'
    # img_name_list = glob.glob("../data/imgs/*.jpg")

    print(img_name_list)



    prediction_dir = os.path.join(os.getcwd(), 'results/') # changed to 'results' directory which is populated after the predictions
    # model_dir = os.path.join(os.getcwd(), model_name + '.pth') # path to u2netp pretrained weights



    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    net = U2NETP(3,1)    
    # if torch.cuda.is_available():
    #     net.load_state_dict(torch.load(model_dir))
    #     net.cuda()
    # else:
    #     net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        # if torch.cuda.is_available():
        #     inputs_test = Variable(inputs_test.cuda())
        # else:
        #     inputs_test = Variable(inputs_test)
        inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        
        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
