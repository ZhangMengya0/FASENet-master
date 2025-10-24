import torch
import torch.nn as nn
import util.utils as tools
import dataset.pascal_data as pascal_data
from torch.optim.lr_scheduler import CosineAnnealingLR
from util.Diceloss import DiceCeLoss
import eval
import time
import numpy as np

from .model.FASENet import FASENet

colormap = [[0, 0, 0], [255, 255, 255]]

cm = np.array(colormap).astype("uint8")

#############
# 超参数设置 #
#############
BATCH = 4
LR = 1e-4
EPOCHES = 100
class_num = 2
WEIGHT_DECAY = 1e-4
start_epoch = 0

def train(offset, model):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = FASENet()

    net = net.to(device)

    data_train = pascal_data.PASCAL_BSD("train")

    train_data = torch.utils.data.DataLoader(data_train, batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True)

    criterion = DiceCeLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-07, weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHES, last_epoch=-1)


    learning_rate = LR


    # 开始训练
    print("开始训练")
    for epoch in range(start_epoch, EPOCHES):

        loss_all = 0
        num_images = 0
        acc = 0
        iou = 0
        miou = 0
        hist = np.zeros((class_num, class_num))

        st_epoch = time.time()
        net = net.train()
        for step, data in enumerate(train_data):
            img, img_gt = data
            img = img.to(device)
            img_gt = img_gt.to(device)
            output = net(img)

            # 计算各项性能指标
            acc, acc_cls, iou, miou, f1_cls, m_f1, kappa, precision, precision_cls, recall, recall_cls, hist = tools.get_MIoU(
                pred=output, label=img_gt, hist=hist)

            loss = criterion(output, img_gt)
            loss_all = loss_all + loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_images += BATCH

            print(
                "epoch[%d/%d] step[%d/%d]->loss:%.4f acc:%.4f iou1:%.4f miou:%.4f mF1:%.4f recall:%.4f lr:%.6f time:%ds" %
                (
                epoch, EPOCHES, step + 1, len(train_data), loss.item(), acc, iou[1:], miou, m_f1, recall, learning_rate,
                time.time() - st_epoch))

        et_epoch = time.time()
        epoch_fps = num_images / (et_epoch - st_epoch)

        epoch_loss = loss_all / len(train_data)
        epoch_acc = acc
        epoch_miou = miou

        print("epoch[%d/%d]->loss:%.4f acc:%.4f miou:%.4f lr:%.6f time:%ds fps:%.2f" %
              (epoch, EPOCHES, epoch_loss, epoch_acc, epoch_miou, learning_rate, time.time() - st_epoch, epoch_fps))

        val_loss, val_acc, val_miou = eval.eval_val(net=net, criterion=criterion, epoch=epoch + offset)


        path = "./checkpoint/epoch-%03d_loss-%.4f_loss(val)-%.4f_acc-%.4f_miou-%.4f_miou(val)-%.4f.pth" % \
               (epoch + offset, epoch_loss, val_loss, epoch_acc, epoch_miou, val_miou)
        torch.save(net.state_dict(), path)
        print("成功保存模型和训练状态 %s✿✿ヽ(°▽°)ノ✿" % (path))

        with open("iou_train.txt", "a") as f:
            f.write("epoch%d->" % (epoch + offset) + str(iou) + "\n\n")

        scheduler.step()
        learning_rate = optimizer.param_groups[0]['lr']

    return 0


if __name__ == "__main__":
    offset = 0
    model = None
    train(offset=offset, model=model)