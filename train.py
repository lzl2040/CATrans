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
from dataset import SemData
from model.CATrans import CATrans
from util import util
import logging
import os

from util.util import AverageMeter, Iou

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
    train_transform = [
        transform.RandomHorizontalFlip(),
        transform.Resize(size=args.train_size),
        transform.ToTensor()]
    train_transform = transform.Compose(train_transform)
    # 加载数据
    logger.info("=> loading data ...")
    train_data = SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                         data_list=args.train_list, transform=train_transform, mode='train',
                         use_coco=args.use_coco, use_split_coco=args.use_split_coco)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    # 创建模型
    logger.info("=> creating model ...")
    model = CATrans(shot=args.shot, training=True)
    # 创建优化器AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    # 查看可优化的参数有哪些
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # print(model.parameters())
    try:
        logger.info('training ...')
        for epoch in range(args.start_epoch,args.epochs):
            train(train_loader,model,optimizer,epoch)
        # 保存模型
        filename = 'final2.pth'
        logger.info('Saving checkpoint to: ' + filename)
        torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
    except Exception as err:
        filename = 'final2.pth'
        logger.info('Saving checkpoint to: ' + filename)
        torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   filename)
        print('err is:'+str(err))
def train(train_loader, model, optimizer,epoch):
    union_meter = Iou()
    intersection_meter = Iou()
    loss_meter = AverageMeter()
    for i, (query_image, query_mask, support_image, support_mask, subcls) in enumerate(train_loader):
        # 1个episode中有4个类的图片,支持集中每个类的图片有shot张,查询集中每个类有一张
        # print("query_mask shape:" + str(query_mask.shape))
        # print("query_image shape:" + str(query_image.shape))
        # print("support_image shape:" + str(support_image.shape))
        # print("support_mask:" + str(support_mask.shape))
        subcls = subcls[0].numpy()
        # print("classes:" + str(subcls))
        # 转到gpu上去
        query_mask = query_mask.cuda(non_blocking=True)
        query_image = query_image.cuda(non_blocking=True)
        support_image = support_image.cuda(non_blocking=True)
        support_mask = support_mask.cuda(non_blocking=True)
        # 训练
        model = model.cuda(device)
        model.train(True)
        loss,model_output = model(x=(support_image, query_image, support_mask, query_mask))
        model.train(False)
        # 将模型的梯度参数设置为0
        optimizer.zero_grad()
        # 损失的后向传播
        loss.backward()
        # 更新所有的参数
        optimizer.step()

        # 计算mIOU
        ## 维度:B * S * C * H * W
        model_output = torch.unsqueeze(model_output,dim=1)
        # print("output shape" + str(model_output.shape))
        out_segs = model_output.argmax(dim=2)
        # print("out segs shape" + str(out_segs.shape))
        ## 压缩维度
        query_mask = torch.unsqueeze(query_mask,dim=1)
        iou = util.get_iou(segs=out_segs,segannos=query_mask,classes=subcls)
        # 样本个数
        n = query_image.shape[0]
        intersection_value = iou['intersection']
        union_value = iou['union']
        class_num = iou['class_num']
        for j in range(n):
            category_index = subcls[j]
            num = class_num[category_index]
            intersection_meter.update(category_index,intersection_value[category_index],num)
            union_meter.update(category_index,union_value[category_index],num)
        # 更新损失
        loss_meter.update(loss,1)
        if (i + 1) % args.print_freq == 0:
            # 计算一次mIou
            # mIou = util.get_mIou(union=union_meter.category_sum,intersection=intersection_meter.category_sum)
            mIou = (intersection_meter.category_val[subcls[0]] + 1e-6) / (union_meter.category_val[subcls[0]] + 1e-6)
            logger.info('epoch:[{}/{}][{}/{}] '
                        'loss:{} '
                        'mIou:{}'.format(epoch+1,args.epochs,i+1,len(train_loader),loss,mIou))

if __name__ == '__main__':
    global args
    args = get_parser()
    main(args)