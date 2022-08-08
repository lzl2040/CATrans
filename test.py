import os
import os.path
import cv2
import numpy as np
import torch.nn as nn
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
from dataset import SemData
from model.CATrans import CATrans
from util import util
import logging
import os

from util.util import Iou, AverageMeter

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main(args):
    # 创建打印日志的
    global logger
    logger = get_logger()
    value_scale = 255
    # 定义转换图片的
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    validate_transform = [
        transform.RandomHorizontalFlip(),
        transform.Resize(size=args.train_size),
        transform.ToTensor()]
    validate_transform = transform.Compose(validate_transform)
    # 加载数据
    logger.info("=> loading data ...")
    validate_data = SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                         data_list=args.val_list, transform=validate_transform, mode='val',
                         use_coco=args.use_coco, use_split_coco=args.use_split_coco)
    validate_sampler = None
    validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=args.batch_size, shuffle=(validate_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=validate_sampler,
                                               drop_last=True)
    # 创建模型
    logger.info("=> creating model ...")
    model = CATrans(shot=args.shot, training=False)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 4.0]), ignore_index=255)
    # 加载权重
    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))
    logger.info('evaluating ...')
    for epoch in range(args.start_epoch,args.epochs):
        validate(validate_loader,model,epoch,criterion)

def validate(validate_loader, model, epoch,criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    model.eval()
    model = model.cuda(device)
    union_meter = Iou()
    intersection_meter = Iou()
    loss_meter = AverageMeter()
    for i, (query_image, query_mask, support_image, support_mask, subcls) in enumerate(validate_loader):
        # 1个episode中有4个类的图片,支持集中每个类的图片有shot张,查询集中每个类有一张
        # subcls = subcls[0].numpy()
        # 转到gpu上去
        query_mask = query_mask.cuda(non_blocking=True)
        query_image = query_image.cuda(non_blocking=True)
        support_image = support_image.cuda(non_blocking=True)
        support_mask = support_mask.cuda(non_blocking=True)
        subcls = subcls[0].numpy()
        # 训练
        model = model.cuda(device)
        loss,model_output = model(x=(support_image, query_image, support_mask, query_mask))
        # logger.info('output shape:{}'.format(model_output.shape))
        # 更新iou
        model_out = torch.unsqueeze(model_output, dim=1)
        out_segs = model_out.argmax(dim=2)
        annos = torch.unsqueeze(query_mask,dim=1)
        iou = util.get_iou(segs=out_segs, segannos=annos, classes=subcls)
        n = query_image.shape[0]
        intersection_value = iou['intersection']
        union_value = iou['union']
        class_num = iou['class_num']
        for j in range(n):
            category_index = subcls[j]
            num = class_num[category_index]
            intersection_meter.update(category_index, intersection_value[category_index], num)
            union_meter.update(category_index, union_value[category_index], num)

        # 显示一张图片
        output = model_output.cpu().detach().numpy()
        query = query_mask.cpu().detach().numpy()
        mask_name = str(i) + '_predict.png'
        query_name = str(i) + "_target.png"
        #print(mask_name)
        # 将query mask转换为二值图
        mid = query[0]
        mid[mid == 255] = 0
        mid[mid == 1] = 255
        # [0,1]为前景 [0,0]为背景
        cv2.imwrite('outimg/' + mask_name,output[0,1] * 255)
        cv2.imwrite('outimg/' + query_name, mid)

        loss = torch.mean(loss)
        if (i + 1) % args.print_freq == 0:
            mIou = util.get_mIou(union=union_meter.category_sum, intersection=intersection_meter.category_sum)
            logger.info('epoch:[{}/{}][{}/{}] '
                        'loss:{} '
                        'mIou:{}'.format(epoch + 1, args.epochs, i + 1, len(validate_loader), loss, mIou))


if __name__ == '__main__':
    global args
    args = get_parser()
    main(args)