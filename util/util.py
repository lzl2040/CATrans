import torch
import numpy as np
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Iou(object):
    def __init__(self):
        # 定义每个类别
        # self.category_miou = np.zeros(20,1)
        self.category_sum = {}
        self.category_count = {}
        self.category_val = {}

    def update(self,category_index,val,n):
        if self.category_val.get(category_index) == None:
            self.category_val[category_index] = val
        if self.category_sum.get(category_index) == None:
            self.category_sum[category_index] = 0
        self.category_sum[category_index] += val
        if self.category_count.get(category_index) == None:
            self.category_count[category_index] = 0
        self.category_count[category_index] += n
        # self.category_miou[category_index] = self.category_sum[category_index] / self.category_count[category_index]



def get_mIou(union,intersection):
    # 先计算每个类的
    iou_class = {}
    class_sum = 0
    for key in union.keys():
        u = union[key]
        i = intersection[key]
        if iou_class.get(key) == None:
            iou_class[key] = 0
        iou_class[key] = (i + 1e-7) / (u + 1e-7)
        class_sum += 1

    # 计算mIou,即每个类的iou平均
    iou_sum = 0
    for key in iou_class.keys():
        iou_sum += iou_class[key]
    return iou_sum / class_sum

# 计算IOU
def get_iou(segs, segannos, classes):
    B, Q, H, W = segs.size()
    stride = len(classes)
    iou = {}
    segs = segs.clone()
    segs[segannos == 255] = 255
    mask_pred = (segs == 1)
    # print("mask_pred shape:" + str(mask_pred.shape))
    mask_anno = (segannos == 1)
    # print("mask_anno shape:" + str(mask_anno.shape))
    intersection = (mask_pred * mask_anno).sum(dim=(2, 3))
    intersection = intersection.cpu().numpy()
    # print("intersection:" + str(intersection))
    union = mask_pred.sum(dim=(2, 3)).cpu().numpy() + mask_anno.sum(dim=(2, 3)).cpu().numpy() - intersection
    #union = union.cpu().numpy()

    # mid = ((intersection + 1e-5) / (union + 1e-5)).cpu().numpy()
    class_num = {}
    intersections = {}
    unions = {}
    for i in range(stride):
        target = classes[i]
        # if iou.get(target) == None:
        #     iou[target] = 0
        # iou[target] += mid[i]
        if class_num.get(target) == None:
            class_num[target] = 0
        class_num[target] += 1
        # 计算交
        if intersections.get(target) == None:
            intersections[target] = 0
        intersections[target] += intersection[i]
        # 计算并
        if unions.get(target) == None:
            unions[target] = 0
        unions[target] += union[i]
    # print(iou.mean())
    return {'intersection': intersections, 'union':unions,'class_num':class_num}