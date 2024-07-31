import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import RandomCrop
from data_loader import RescaleT
from data_loader import SalObjDataset
from data_loader import ToTensorLab
from model import U2NET
from model import U2NETP

if __name__ == '__main__':

    # ------- 1. define loss function --------

    bce_loss = nn.BCELoss(size_average=True)

    def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

        # category_mask_1 = torch.gt(labels_v, 0)  # 大于0的label
        # labels_v_1 = torch.masked_select(labels_v, category_mask_1)
        #
        # category_mask_0 = torch.lt(labels_v, 1)  # 小于1的label
        # labels_v_0 = torch.masked_select(labels_v, category_mask_0)
        # d0_1 = torch.masked_select(d0, category_mask_1)
        #
        #
        # d0_0 = torch.masked_select(d0, category_mask_0)
        #
        # d1_1 = torch.masked_select(d1, category_mask_1)
        # d1_0 = torch.masked_select(d1, category_mask_0)
        #
        # d2_1 = torch.masked_select(d2, category_mask_1)
        # d2_0 = torch.masked_select(d2, category_mask_0)
        #
        # d3_1 = torch.masked_select(d3, category_mask_1)
        # d3_0 = torch.masked_select(d3, category_mask_0)
        #
        # d4_1 = torch.masked_select(d4, category_mask_1)
        # d4_0 = torch.masked_select(d4, category_mask_0)
        #
        # d5_1 = torch.masked_select(d5, category_mask_1)
        # d5_0 = torch.masked_select(d5, category_mask_0)
        #
        # d6_1 = torch.masked_select(d6, category_mask_1)
        # d6_0 = torch.masked_select(d6, category_mask_0)

        # loss0 = torch.mean(((1 - d0_1) ** 2) * bce_loss(d0_1, labels_v_1)) + torch.mean((d0_0 ** 2) * bce_loss(d0_0, labels_v_0))
        # loss1 = torch.mean(((1 - d1_1) ** 2) * bce_loss(d1_1, labels_v_1)) + torch.mean((d1_0 ** 2) * bce_loss(d1_0, labels_v_0))
        # loss2 = torch.mean(((1 - d2_1) ** 2) * bce_loss(d2_1, labels_v_1)) + torch.mean((d2_0 ** 2) * bce_loss(d2_0, labels_v_0))
        # loss3 = torch.mean(((1 - d3_1) ** 2) * bce_loss(d3_1, labels_v_1)) + torch.mean((d3_0 ** 2) * bce_loss(d3_0, labels_v_0))
        # loss4 = torch.mean(((1 - d4_1) ** 2) * bce_loss(d4_1, labels_v_1)) + torch.mean((d4_0 ** 2) * bce_loss(d4_0, labels_v_0))
        # loss5 = torch.mean(((1 - d5_1) ** 2) * bce_loss(d5_1, labels_v_1)) + torch.mean((d5_0 ** 2) * bce_loss(d5_0, labels_v_0))
        # loss6 = torch.mean(((1 - d6_1) ** 2) * bce_loss(d6_1, labels_v_1)) + torch.mean((d6_0 ** 2) * bce_loss(d6_0, labels_v_0))

        loss0 = bce_loss(d0, labels_v)
        loss1 = bce_loss(d1, labels_v)
        loss2 = bce_loss(d2, labels_v)
        loss3 = bce_loss(d3, labels_v)
        loss4 = bce_loss(d4, labels_v)
        loss5 = bce_loss(d5, labels_v)
        loss6 = bce_loss(d6, labels_v)

        loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6)
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.detach().cpu().item(), loss1.detach().cpu().item(), loss2.detach().cpu().item(), loss3.detach().cpu().item(), loss4.detach().cpu().item(), loss5.detach().cpu().item(), loss6.detach().cpu().item()))

        return loss0, loss




    # ------- 2. set the directory of training dataset --------

    # model_name = 'u2net'  # 'u2netp'
    model_name = 'u2net'  # 'u2netp'

    data_dir = './train_data/'
    tra_image_dir = 'DUTS/im_aug/'
    tra_label_dir = 'DUTS/gt_aug/im_aug/'
    data_dir = '/home/boer/xugeofei-project/ketu_symmetry/'
    tra_image_dir = 'pytorch-UNet/data/JPEGImages/'
    tra_label_dir = 'pytorch-UNet/data/labels/'

    # tra_image_dir = '/home/boer/xugeofei-project/ketu_symmetry/pytorch-UNet/data/JPEGImages/'
    # tra_label_dir = '/home/boer/xugeofei-project/ketu_symmetry/pytorch-UNet/data/labels/'
    print(tra_image_dir)
    image_ext = '.jpg'
    label_ext = '.png'

    model_dir = './saved_models/' + model_name + '/'

    epoch_num = 100000
    batch_size_train = 1
    batch_size_val = 1
    train_num = 0
    val_num = 0

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split("/")[-1]


        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]

        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]
        print(data_dir + tra_label_dir + imidx + label_ext)
        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))


    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            # RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

    # ------- 3. define model --------
    # define the net
    if (model_name == 'u2net'):
        net = U2NET(3, 1)
        # net.load_state_dict(torch.load(r'N:\gongzuo\AI\U2Net\saved_models\u2net\u2net.pth'))
        # net.load_state_dict(torch.load(r'/home/lyg/PycharmProjects/U2Net/saved_models/u2net/u2net_jingzhui.pth'))
    elif (model_name == 'u2netp'):
        net = U2NETP(3, 1)
        # print('.........load weight..........')
        # net.load_state_dict(torch.load(r'/home/lyg/PycharmProjects/U2Net/saved_models/u2net/u2netp_jingzhui.pth'))
    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # optimizer = optim.SGD(net.parameters(), lr=0.1,  weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 2000  # save the model every 2000 iterations

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.detach().cpu().item()
            running_tar_loss += loss2.detach().cpu().item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:
                # torch.save(net.state_dict(), model_dir + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                # ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                # torch.save(net.state_dict(),r'/home/lyg/PycharmProjects/U2Net/saved_models/u2net/u2net_jingzhui.pth')
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0


