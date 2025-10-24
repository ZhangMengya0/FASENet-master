import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weights = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        nclass = predict.shape[1]
        target = torch.nn.functional.one_hot(target.long(), nclass)#[1, 4]->[1, 4, 5]
        #target = torch.transpose(torch.transpose(target, 1, 3), 2, 3)
        target = torch.transpose(target, 1, 2)

        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weights is not None:
                    assert self.weights.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1] if self.weights is None else total_loss/(torch.sum(self.weights))


class Diceloss(nn.Module):
    def __init__(self):
        super(Diceloss, self).__init__()

    def forward(self, preds, labels):
        # assert preds.shape == labels.shape, "predict & target shape do not match"
        # 使用pytorch的交叉熵损失函数接口


        # nums = labels.shape[1]

        # 计算dice系数时去掉背景类，因为背景类不是我们预测的目标，如果是，需要加上
        # preds = preds[:, 1:]  # 去掉背景类
        # labels = labels[:, 1:]
        preds = preds[:, 1:, :, :]  # 去掉背景类
        # labels = labels[:, :, :, :]  # 裁剪标签尺寸以匹配预测尺寸
        # print(preds.shape)
        # print(labels.shape)
        # 将预测和标签转换为一维向量
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)

        # 计算交集
        intersection = torch.sum(preds * labels)
        # 计算 Dice 系数
        dice = (2. * intersection + 1e-8) / \
               (torch.sum(preds) + torch.sum(labels) + 1e-8)

        if dice >= 1:
            dice = 1
        # 这里使用的Log函数来加权
        dice_loss = -1 * torch.log(dice)
        return dice_loss


class DiceCeLoss(nn.Module):
    def __init__(self):
        super(DiceCeLoss, self).__init__()

    def forward(self, preds, labels):
        # assert preds.shape == labels.shape, "predict & target shape do not match"
        # 使用pytorch的交叉熵损失函数接口
        ce_loss = nn.CrossEntropyLoss()
        ce_total_loss = ce_loss(preds, labels)
        preds = F.softmax(preds, dim=1)

        # nums = labels.shape[1]

        # 计算dice系数时去掉背景类，因为背景类不是我们预测的目标，如果是，需要加上
        # preds = preds[:, 1:]  # 去掉背景类
        # labels = labels[:, 1:]
        preds = preds[:, 1:, :, :]  # 去掉背景类
        # labels = labels[:, :, :, :]  # 裁剪标签尺寸以匹配预测尺寸
        # print(preds.shape)
        # print(labels.shape)
        # 将预测和标签转换为一维向量
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)

        # 计算交集
        intersection = torch.sum(preds * labels)
        # 计算 Dice 系数
        dice = (2. * intersection + 1e-8) / \
               (torch.sum(preds) + torch.sum(labels) + 1e-8)

        if dice >= 1:
            dice = 1
        # 这里使用的Log函数来加权
        dice_ce_loss = -1 * torch.log(dice) + ce_total_loss
        # dice_ce_loss = -1 * torch.log(dice)
        return dice_ce_loss


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res

def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# def class2one_hot(seg: Tensor, C: int) -> Tensor:
#     if len(seg.shape) == 2:  # Only w, h, used by the dataloader
#         seg = seg.unsqueeze(dim=0)
#     # seg = seg.clamp(min=0, max=1)  # Ensures all values are 0 or 1
#
#
#         # seg = seg.long()  # 确保 seg 是整数类型
#
#     assert sset(seg, list(range(C)))
#
#     b, w, h = seg.shape  # type: Tuple[int, int, int]
#
#     res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
#     assert res.shape == (b, C, w, h)
#     assert one_hot(res)
#
#     return res

def class2one_hot(seg: torch.Tensor, C: int) -> torch.Tensor:
    # 如果 seg 是一个 [B, C, H, W] 的张量（包含每个像素的概率分布）
    if len(seg.shape) == 4:  # [B, C, H, W] (batch, channels, height, width)
        # 将每个像素点的类别索引提取出来，得到一个 [B, H, W] 的张量
        seg = torch.argmax(seg, dim=1)  # [B, H, W]

    # 确保 seg 的值在 [0, C-1] 范围内
    assert torch.all((seg >= 0) & (seg < C)), f"Invalid values in seg: {seg.unique()}"

    # 获取 batch_size, height, width
    b, w, h = seg.shape

    # 创建 one-hot 编码：为每个类别创建一个通道，结果是 [B, C, H, W]
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)

    # 确保输出形状正确
    assert res.shape == (b, C, w, h)

    return res

def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool_)  # 使用 np.bool_ 替代 np.bool


        if posmask.any():
            negmask = ~posmask
            # print('negmask:', negmask)
            # print('distance(negmask):', distance(negmask))
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            # print('res[c]', res[c])
    return res


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

    # Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.detach()).numpy())
    # return set(torch.unique(a.cpu()).numpy())



def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)



class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=40):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        img.float()

        if len(img.shape) == 4:

            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):

        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):

        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):

        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):

        return self.soft_skel(img)

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1., exclude_background=False):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self,y_pred,y_true):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_pred,y_true):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1

    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 0.001, exclude_background=True):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_pred, y_true):

        if self.exclude_background:
            # y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
            y_true = y_true.unsqueeze(1)

        y_pred = y_pred.float()
        y_true = y_true.float()
        # dice = soft_dice(y_pred, y_true)
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice
        # return (1.0-self.alpha)*dice+self.alpha*cl_dice




class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]   #这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3

    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        print('pc', pc)
        print('dc', dc)
        pc = pc.to(dc.device)
        # dc = dc.to(pc.device)
        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss





# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool_)

        if posmask.any():
            negmask = ~posmask
            # print('negmask:', negmask)
            # print('distance(negmask):', distance(negmask))
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            # print('res[c]', res[c])
    return res


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

    # Assert utils


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]  # 这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3

    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        # print('pc', pc)
        # print('dc', dc)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss



class BoundayCeLoss(nn.Module):
    def __init__(self):
        super(BoundayCeLoss, self).__init__()

    def forward(self, preds, labels):
        # assert preds.shape == labels.shape, "predict & target shape do not match"
        # 使用pytorch的交叉熵损失函数接口

        preds = F.softmax(preds, dim=1)
        dice = Diceloss()
        diceloss = dice(preds, labels)



        # preds = preds[:, 1:, :, :]  # 去掉背景类
        # preds = preds.squeeze(1)



        # labels = torch.round(labels)
        labels = class2one_hot(labels, 2)
        # print(data2)
        labels = labels.numpy()
        labels = one_hot2dist(labels)  # bcwh
        labels = torch.tensor(labels).unsqueeze(0)

        preds = class2one_hot(preds.argmax(dim=1), 2)


        Loss = SurfaceLoss()
        boundary = Loss(preds, labels, None)


        # 这里使用的Log函数来加权
        dice_boundary_loss = 0.01*boundary + diceloss
        return dice_boundary_loss
