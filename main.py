import os
import preprocess
import utils
import parser_utils
import cv2
import random
import numpy as np
import encoder_models
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2

if __name__ == '__main__':
    args = parser_utils.get_parser()
    opts = args.parse_args()
    train_list,test_list = preprocess.generate_train_and_test()
    # 获得支持集、支持集掩码、查询集、查询集掩码
    support, smasks, query, qmask = utils.get_episode(opts, train_list)
    # 提取支持集图像特征
    # 支持集的image_encoder
    Fs1, Fs2, Fs3, Fs4 = encoder_models.resnet50_encoder(support[0,0])
    print("Fs1维度:" + str(Fs1.shape) + " Fs2维度:" + str(Fs2.shape))
    print("Fs3维度:" + str(Fs3.shape) + " Fs4维度:" + str(Fs4.shape))
    # 查询集的image_encoder
    Fq1, Fq2, Fq3, Fq4 = encoder_models.resnet50_encoder(query[0])
    cv2.imshow("target",query[0])
    cv2.waitKey(0)
    # 支持掩码的mask encoder
    Fm1,Fm2,Fm3,Fm4 = encoder_models.mask_encoder(smasks[0,0])
    print("Fm1维度:" + str(Fm1.shape) + " Fm2维度:" + str(Fm2.shape))
    print("Fm3维度:" + str(Fm3.shape) + " Fm4维度:" + str(Fm4.shape))
    # RCT和RCT
    rct3 = utils.RCT(Fm3,Fs3,Fq3,8)
    rat3 = utils.RAT(Fm3,Fs3,Fq3,8,out_dim=2304)
    # 看下rat3
    # out = rct3.transpose(1,2,0)
    # print(out.shape)
    # out = encoder_models.conv(out,in_channel=out.shape[2],out_channel=1,stride=1,padding=1,ks=3)
    # out = encoder_models.upsample(out,times=4)
    # #out = out.transpose(1, 2, 0)
    # print(out.shape)
    # cv2.imshow('frame',out)
    # cv2.waitKey(0)

    rct4 = utils.RCT(Fm4,Fs4,Fq4,8)
    rat4 = utils.RAT(Fm4,Fs4,Fq4,8,out_dim=576)
    # out = rct4
    # out = out.transpose(1, 2, 0)
    # out = encoder_models.conv(out, in_channel=out.shape[2], out_channel=3, stride=1, padding=1, ks=3)
    # print("维度:" + str(out.shape))
    # out = out.transpose(1, 2, 0)
    # out = encoder_models.upsample(out, times=8)
    # out = out.transpose(1, 2, 0)
    # print("维度:" + str(out.shape))
    # cv2.imshow("raw", smasks[0, 0])
    # cv2.waitKey(0)
    # cv2.imshow("extract", out)
    # cv2.waitKey(0)
    # cv2.imwrite('rct.png',out,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    # decoder部分
    # 拼接
    input1 = np.concatenate([rct4,rat4],axis=0)
    input2 = np.concatenate([rct3,rat3],axis=0)
    input3 = Fq2
    input4 = Fq1
    utils.image_decoder(input1,input2,input3,input4)



