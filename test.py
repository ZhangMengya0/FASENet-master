import os
import torch
import util.utils as tools
import dataset.pascal_data as pascal_data
import time
import numpy as np

from util.Diceloss import DiceCeLoss

from .model.FASENet import FASENet



BATCH = 4
EPOCHES = 1
class_num = 2
WEIGHT_DECAY = 1e-4

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = FASENet()

    net.load_state_dict(torch.load(
        r".checkpoint\epoch-099.pth",
        map_location=device))
    net.eval().to(device)

    data_test = pascal_data.PASCAL_BSD("test")

    test_data = torch.utils.data.DataLoader(data_test, batch_size=BATCH, shuffle=False)

    criterion = DiceCeLoss()

    print("开始测试")
    for epoch in range(EPOCHES):

        loss_all = 0

        acc = 0
        iou = 0
        miou = 0
        hist = np.zeros((class_num, class_num))

        st_epoch = time.time()
        net = net.eval()
        for step, data in enumerate(test_data):
            img, img_gt = data
            img = img.to(device)
            img_gt = img_gt.to(device)

            with torch.no_grad():
                output = net(img)

                acc, acc_cls, iou, miou, f1_cls, m_f1, kappa, precision ,precision_cls, recall, recall_cls, hist = tools.get_MIoU(pred=output, label=img_gt,
                                                                                    hist=hist)
                loss = criterion(output, img_gt.long())
                loss_all = loss_all + loss.item()

        epoch_loss = loss_all / len(test_data)
        epoch_acc = acc
        epoch_miou = miou
        print("val->loss:%.4f acc:%.4f miou:%.4f time:%ds" %
              (epoch_loss, epoch_acc, epoch_miou, time.time() - st_epoch))

        if not os.path.isfile("test.txt"):
            with open("test.txt", "a") as f:
                f.write("Acc->" + str(epoch_acc) + " " +"Acc_class->" + str(acc_cls)+ "\n\n" +
                        "Pricision->" + str(precision) + " " + "Pricision_class->" + str(precision_cls) + "\n\n" +
                        "Recall->" + str(recall) + " " + "Pricision_class->" + str(recall_cls) + "\n\n" +
                        "mf1->" + str(m_f1) + " " + "f1_class->" + str(f1_cls) + "\n\n" +
                        "miou->" + str(epoch_miou) + " " + "IoU_class->" + str(iou) + "\n\n" +
                        "Kappa->" + str(kappa) + " " + "loss->" + str(epoch_loss) + "\n\n")


    return 0


if __name__ == "__main__":
    offset = 0
    test()
