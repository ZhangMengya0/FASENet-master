import torch
import numpy as np
# import matplotlib.pyplot as plt

# 超参数，类别数量
class_num = 2


####################
# 计算各种评价指标  #
####################

def fast_hist(a, b, n):
    """
    生成混淆矩阵
    a 是形状为(HxW,)的预测值
    b 是形状为(HxW,)的真实值
    n 是类别数
    """
    # 确保a和b在0~n-1的范围内，k是(HxW,)的True和False数列
    a = torch.softmax(a, dim=1)
    _, a = torch.max(a, dim=1)
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    k = (a >= 0) & (a < n)
    # 横坐标是预测的类别，纵坐标是真实的类别
    hist = np.bincount(a[k].astype(int) + n * b[k].astype(int), minlength=n ** 2).reshape(n, n)
    return hist

def per_class_iou(hist):
    """
    hist传入混淆矩阵(n, n)
    """

    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    # 把报错设回来
    # np.seterr(divide="warn", invalid="warn")
    # 如果分母为0，结果是nan，会影响后续处理，因此把nan都置为0
    iou[np.isnan(iou)] = 0.
    return iou


def per_class_acc(hist):
    """
    :param hist: 混淆矩阵
    :return: 没类的acc和平均的acc
    """
    np.seterr(divide="ignore", invalid="ignore")
    acc_cls = np.diag(hist) / hist.sum(1)
    np.seterr(divide="warn", invalid="warn")
    acc_cls[np.isnan(acc_cls)] = 0.
    return acc_cls

def per_class_f1(hist):
    """
    计算每个类别的F1指数。
    """
    f1_cls = np.zeros(hist.shape[0])
    for i in range(hist.shape[0]):
        tp = hist[i, i]
        fp = hist[:, i].sum() - tp
        fn = hist[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1_cls[i] = 2 * (precision * recall) / (precision + recall + 1e-12)
    return f1_cls

def mean_f1_score(f1_cls):
    """
    计算平均F1分数。
    """
    return np.nanmean(f1_cls)

def kappa_coefficient(hist):
    """
    计算Cohen的Kappa系数。
    """
    total = np.sum(hist)
    po = np.trace(hist) / total  # 观察到的一致性
    pe = np.sum(np.sum(hist, axis=0) * np.sum(hist, axis=1)) / (total * total)
    kappa = (po - pe) / (1 - pe + 1e-12)
    return kappa

def per_class_precision(hist):
    """
    计算每个类别的Precision。
    """
    precision_cls = np.zeros(hist.shape[0])
    for i in range(hist.shape[0]):
        tp = hist[i, i]  # 真正例
        fp = hist[:, i].sum() - tp  # 假正例
        precision_cls[i] = tp / (tp + fp + 1e-12)
    return precision_cls

def per_class_recall(hist):
    """
    计算每个类别的Recall。
    """
    recall_cls = np.zeros(hist.shape[0])
    for i in range(hist.shape[0]):
        tp = hist[i, i]  # 真正例
        fn = hist[i, :].sum() - tp  # 假负例
        recall_cls[i] = tp / (tp + fn + 1e-12)
    return recall_cls


def get_MIoU(pred, label, hist):
    hist = hist + fast_hist(pred, label, class_num)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = per_class_acc(hist)
    iou = per_class_iou(hist)
    miou = np.nanmean(iou)
    f1_cls = per_class_f1(hist)
    m_f1 = np.nanmean(f1_cls)
    kappa = kappa_coefficient(hist)
    precision_cls = per_class_precision(hist)
    recall_cls = per_class_recall(hist)
    precision = np.nanmean(precision_cls)
    recall = np.nanmean(recall_cls)
    return acc, acc_cls, iou, miou, f1_cls, m_f1, kappa, precision ,precision_cls, recall, recall_cls , hist


# 更新学习率
def getNewLR(LR, net):
    LR = LR / 2
    print("更新学习率LR=%.6f" % LR)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    return optimizer, LR


# 绘制hist矩阵的可视化图并保存
# def drawHist(hist, path):
#     # print(hist)
#     hist_ = hist[1:]
#     hist_tmp = np.zeros((class_num - 1, class_num - 1))
#
#     for i in range(len(hist_)):
#         hist_tmp[i] = hist_[i][1:]
#
#     # print(hist_tmp)
#     hist = hist_tmp
#     plt.matshow(hist)
#     plt.xlabel("Predicted label")
#     plt.ylabel("True label")
#     plt.axis("off")
#     # plt.colorbar()
#     # plt.show()
#     if (path != None):
#         plt.savefig(path)
#         print("%s保存成功✿✿ヽ(°▽°)ノ✿" % path)

#
# if __name__ == "__main__":
#     hist = np.random.randint(0, 20, size=(21, 21))
#     drawHist(hist, None)