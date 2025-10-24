import torch
import torch.nn as nn
import util.utils as tools
import dataset.pascal_data as pascal_data
import time
import os
import numpy as np
from util.Diceloss import DiceCeLoss

BATCH = 4
class_num = 2


# 对整个验证集进行计算
def eval_val(net, criterion=None, show_step=True, epoch=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_val = pascal_data.PASCAL_BSD("val")
    val_data = torch.utils.data.DataLoader(data_val, batch_size=BATCH, shuffle=False)
    net = net.to(device)
    net = net.eval()

    if (criterion == None):
        # criterion = nn.CrossEntropyLoss()
        criterion = DiceCeLoss()

    loss_all = 0
    acc = 0
    acc_cls = 0
    iou = 0
    miou = 0
    f1_cls = 0
    m_f1 = 0
    kappa = 0


    hist = np.zeros((class_num, class_num))
    st_epoch = time.time()
    for step, data in enumerate(val_data):
        st_step = time.time()
        img, img_gt = data
        img = img.to(device)
        img_gt = img_gt.to(device)

        with torch.no_grad():
            output = net(img)
            # 计算各项性能指标
            acc, acc_cls, iou, miou, f1_cls, m_f1, kappa, precision ,precision_cls, recall, recall_cls, hist = tools.get_MIoU(pred=output, label=img_gt, hist=hist)

            # 计算损失值
            # output = output.unsqueeze(1)
            loss = criterion(output, img_gt.long())
            loss_all = loss_all + loss.item()

            if (show_step == True):
                print("(val)step[%d/%d]->loss:%.4f acc:%.4f miou:%.4f mF1:%.4f Kappa:%.4f time:%ds" %
                      (step + 1, len(val_data), loss.item(), acc, miou, m_f1, kappa, time.time() - st_epoch))

    epoch_loss = loss_all / len(val_data)
    epoch_acc = acc
    epoch_miou = miou
    print("val->loss:%.4f acc:%.4f miou:%.4f time:%ds" %
          (epoch_loss, epoch_acc, epoch_miou, time.time() - st_epoch))


    with open("eval.txt", "a") as f:
        f.write("epoch%d IoU->" % (epoch) + str(iou) + " " + "Acc->" + str(acc_cls) + " " + "f1->" + str(f1_cls) + " " + "Kappa->" + str(kappa) + "\n\n")

    return epoch_loss, epoch_acc, epoch_miou


