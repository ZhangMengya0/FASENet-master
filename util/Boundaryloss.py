import torch
import torch.nn as nn
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff
import torch.nn.functional as F
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union


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
    return set(torch.unique(a.cpu()).numpy())


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
        ce_loss = nn.CrossEntropyLoss()
        ce_total_loss = ce_loss(preds, labels)
        preds = F.softmax(preds, dim=1)



        # preds = preds[:, 1:, :, :]  # 去掉背景类
        # preds = preds.squeeze(1)



        # labels = torch.round(labels)
        labels = class2one_hot(labels, 2)
        # print(data2)
        labels = labels[0].cpu().numpy()
        labels = one_hot2dist(labels)  # bcwh
        labels = torch.tensor(labels).unsqueeze(0)

        preds = class2one_hot(preds.argmax(dim=1), 2).cpu()


        Loss = SurfaceLoss()
        res = Loss(preds, labels, None)


        # 这里使用的Log函数来加权
        dice_ce_loss = 0.01*res + ce_total_loss
        return dice_ce_loss


if __name__ == "__main__":
    data = torch.round(torch.rand(1, 512, 512))

    data2 = class2one_hot(data, 2)
    # print(data2)
    data2 = data2[0].numpy()
    data3 = one_hot2dist(data2)  # bcwh

    # print(data3)
    # print("data3.shape:", data3.shape)

    logits = torch.rand(1, 2, 512, 512).argmax(dim=1)

    logits = class2one_hot(logits, 2)

    Loss = SurfaceLoss()
    data3 = torch.tensor(data3).unsqueeze(0)

    res = Loss(logits, data3, None)
    # print('loss:', res)

