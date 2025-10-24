import torch
import torchvision.transforms as tfs
import os
import scipy.io as scio
import numpy as np
from PIL import Image
import random

prefix = "D:\Ditch\Code\segmentation_ir\VOCdevkit_sd\VOC2007"
CROP = 512

class PASCAL_BSD(object):
    def __init__(self, mode="train", change=False):
        super(PASCAL_BSD, self).__init__()
        # 读取数据的模式
        self.mode = mode


        self.im_tfs = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if change:
            self.mat2png()

        self.image_name = []
        self.label_name = []
        self.readImage()
        print("%s->成功加载%d张图片" % (self.mode, len(self.image_name)))


    # 读取图像和标签信息
    def readImage(self):
        img_root = prefix + "/JPEGImages/"
        label_root = prefix + "/SegmentationClass/"
        if (self.mode == "train"):
            with open(prefix + "/ImageSets/Segmentation/train.txt", "r") as f:
                list_dir = f.readlines()
        elif (self.mode == "val"):
            with open(prefix + "/ImageSets/Segmentation/val.txt", "r") as f:
                list_dir = f.readlines()
        elif (self.mode == "test"):
            with open(prefix + "/ImageSets/Segmentation/test.txt", "r") as f:
                list_dir = f.readlines()

        # 使用全部数据
        for item in list_dir:
            self.image_name.append(img_root + item.split("\n")[0] + ".jpg")
            self.label_name.append(label_root + item.split("\n")[0] + ".png")

        # # 随机选择10%的数据
        # sample_size = int(len(list_dir) * 0.1)
        # sampled_list = random.sample(list_dir, sample_size)  # 随机选取10%数据
        #
        # for item in sampled_list:
        #     self.image_name.append(img_root + item.split("\n")[0] + ".jpg")
        #     self.label_name.append(label_root + item.split("\n")[0] + ".png")


    # 数据处理，输入Image对象，返回tensor对象
    def data_process(self, img, img_gt):
        if img.mode == 'RGBA':
            img = img.convert("RGB")
        if (self.mode == "train"):
            # 以50%的概率左右翻转
            a = random.random()
            if (a > 0.5):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img_gt = img_gt.transpose(Image.FLIP_LEFT_RIGHT)
            # 以50%的概率上下翻转
            a = random.random()
            if (a > 0.5):
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                img_gt = img_gt.transpose(Image.FLIP_TOP_BOTTOM)
            # 以50%的概率像素矩阵转置
            a = random.random()
            if (a > 0.5):
                img = img.transpose(Image.TRANSPOSE)
                img_gt = img_gt.transpose(Image.TRANSPOSE)
            # 以30%的概率随机色彩抖动
            a = random.random()
            if (a > 0.3):
                color_jitter = tfs.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                img = color_jitter(img)



        img = self.im_tfs(img)
        img_gt = np.array(img_gt, dtype=np.uint8)  # 使用 uint8 类型
        img_gt = torch.from_numpy(img_gt).long()  # 转换为 long 类型


        return img, img_gt

    def __getitem__(self, idx):
        # idx = 100
        img = Image.open(self.image_name[idx])
        img_gt = Image.open(self.label_name[idx])
        img, img_gt = self.data_process(img, img_gt)
        # img = self.add_noise(img)
        return img, img_gt

    def __len__(self):
        return len(self.image_name)

    # 将mat数据转换成png
    def mat2png(self, dataDir=None, outputDir=None):
        if (dataDir == None):
            dataroot = prefix + "cls/"
        else:
            dataroot = dataDir
        if (outputDir == None):
            outroot = prefix + "SegmentationClass/"
        else:
            outroot = outputDir
        list_dir = os.listdir(dataroot)
        for item in list_dir:
            matimg = scio.loadmat(dataroot + item)
            mattmp = matimg["GTcls"]["Segmentation"]
            # 将mat转换成png
            # print(mattmp[0][0])
            new_im = Image.fromarray(mattmp[0][0])
            print(outroot + item[:-4] + ".png")
            new_im.save(outroot + item[:-4] + ".png")



if __name__ == "__main__":
    data_train = PASCAL_BSD("train")
    data_val = PASCAL_BSD("val")
    data_test = PASCAL_BSD("test")
    train_data = torch.utils.data.DataLoader(data_train, batch_size=16, shuffle=True)
    val_data = torch.utils.data.DataLoader(data_val, batch_size=16, shuffle=False)
    test_data = torch.utils.data.DataLoader(data_test, batch_size=16, shuffle=False)
    for item in test_data:
        img, img_gt = item
        print(img.shape)
        print(img_gt.shape)