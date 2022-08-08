import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
import json
import parser_utils
import config
import argparse
from tqdm import tqdm
import transform

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_resnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/pascal/pascal_split0_resnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None):
    assert split in [0, 1, 2, 3, 10, 11, 999]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    split_data_list = data_list.split('.')[0] + '_split{}'.format(split) + '.pth'
    if os.path.isfile(split_data_list):
        image_label_list, sub_class_file_list = torch.load(split_data_list)
        # print(len(image_label_list))
        # print(len(sub_class_file_list))
        return image_label_list, sub_class_file_list

    image_label_list = []
    list_read = json.load(open(data_list))
    print("Processing data...")
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []


    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        # line = line.strip()
        # line_split = line.split(' ')
        line_split = line
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []
        for c in label_class:
            if c in sub_list:
                tmp_label = np.zeros_like(label)
                target_pix = np.where(label == c)
                tmp_label[target_pix[0], target_pix[1]] = 1
                if tmp_label.sum() >= 2 * 32 * 32:
                    new_label_class.append(c)

        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)

    print("Checking image&label pair {} list done! ".format(split))
    print("Saving processed data...")
    torch.save((image_label_list, sub_class_file_list), split_data_list)
    print("Done")
    return image_label_list, sub_class_file_list

class SemData(Dataset):
    def __init__(self, split=3, shot=1, data_root=None, data_list=None, transform=None, mode='train', use_coco=False,
                 use_split_coco=False):
        assert mode in ['train', 'val', 'test']
        # 18 classes
        # 注意220对应的是外部的轮廓颜色值
        self.classes = [128,132,14,147,19,150,33,37,38,52,57,72,75,89,94,108,112,113]
        # self.classes = [7,14,21,37,38,45,52,60,75,82,89,97,112,113,120,127,128,135]
        self.mode = mode
        self.split = split
        self.shot = shot
        self.data_root = data_root
        self.use_coco = use_coco

        if not use_coco:
            # self.class_list = self.classes[0:20]
            self.class_list = self.classes[0:19]  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3:
                self.sub_list = self.classes[0:16]  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = self.classes[16:19]  # [16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = self.classes[0:11] + self.classes[16:19]  # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = self.classes[11:16]  # [11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = self.classes[0:6] + self.classes[11:19]# [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = self.classes[6:11]  # [6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = self.classes[6:19]  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = self.classes[0:6]  # [1,2,3,4,5]
                # self.sub_list = self.classes[0:19]
                # self.sub_val_list = []

        else:
            if use_split_coco:
                print('INFO: using SPLIT COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
            else:
                print('INFO: using COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81))
                    self.sub_val_list = list(range(1, 21))

        # print('sub_list: ', self.sub_list)
        # print('sub_val_list: ', self.sub_val_list)

        if self.mode == 'train':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
        elif self.mode == 'val':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_val_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        padding_mask = np.zeros_like(label)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        new_label_class = []
        for c in label_class:
            if c in self.sub_val_list:
                if self.mode == 'val' or self.mode == 'test':
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0

        # choose
        class_chosen = label_class[random.randint(0, len(label_class)) - 1]
        # class_chosen = class_chosen
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        label[:, :] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0], target_pix[1]] = 1
        label[ignore_pix[0], ignore_pix[1]] = 255

        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1, num_file) - 1
            support_image_path = image_path
            support_label_path = label_path
            while ((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1, num_file) - 1
                support_image_path, support_label_path = file_class_chosen[support_idx]
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        support_padding_list = []
        subcls_list = []
        for k in range(self.shot):
            if self.mode == 'train':
                # print(class_chosen)
                # print(self.sub_list.index(class_chosen))
                subcls_list.append(self.sub_list.index(class_chosen))
                # subcls_list.append(class_chosen)
            else:
                subcls_list.append(self.sub_val_list.index(class_chosen))
                # subcls_list.append(class_chosen)
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:, :] = 0
            support_label[target_pix[0], target_pix[1]] = 1
            support_label[ignore_pix[0], ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError(
                    "Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))

            if not self.use_coco:
                support_padding_label = np.zeros_like(support_label)
                support_padding_label[support_label == 255] = 255
            else:
                support_padding_label = np.zeros_like(support_label)
            support_image_list.append(support_image)
            support_label_list.append(support_label)
            support_padding_list.append(support_padding_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot

        raw_label = label.copy()
        if self.transform is not None:
            # print("transform")
            image, label, padding_mask = self.transform(image, label, padding_mask)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k], support_padding_list[k] = self.transform(
                    support_image_list[k], support_label_list[k], support_padding_list[k])
        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        if support_padding_list is not None:
            s_eys = support_padding_list
            s_ey = s_eys[0].unsqueeze(0)
            for i in range(1, self.shot):
                s_ey = torch.cat([s_eys[i].unsqueeze(0), s_ey], 0)

        if self.mode == 'train':
            s_y = torch.LongTensor(s_y)
            return image, label, s_x, s_y, subcls_list
        else:
            return image, label, s_x, s_y, subcls_list

if __name__ == '__main__':
    args = get_parser()
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    train_transform = [
        transform.RandomHorizontalFlip(),
        transform.Resize(size=args.train_size),
        transform.ToTensor()]
    train_transform = transform.Compose(train_transform)
    train_data = SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                                 data_list=args.train_list, transform=train_transform, mode='train',
                                 use_coco=args.use_coco, use_split_coco=args.use_split_coco)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    for i, (query_image, query_mask, support_image, support_mask, subcls) in enumerate(train_loader):
        # 1个episode中有4个类的图片,支持集中每个类的图片有shot张,查询集中每个类有一张
        print("query_mask shape:" + str(query_mask.shape))
        print("query_mask shape:" + str(query_image.shape))
        print("support_image shape:" + str(support_image.shape))
        print("support_mask:" + str(support_mask.shape))
        break