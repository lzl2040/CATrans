# 工具类
import random
import numpy as np
import cv2
import torch
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
import encoder_models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor()]
)

# 获得episode,包含有query set和surport set,support mask
def get_episode(opt, setX):
    # 在0-len(setX)中随机取n个不重复的数,作为分类的
    # 选择类别,这里我们将一张图片看成一个类别(具体不知道怎么操作)
    # indx_c = random.sample(range(0, len(setX)), opt.nway)
    indx_c = [0,1]
    # 支持集
    support = np.zeros([opt.nway, opt.kshot, opt.img_h, opt.img_w, 3], dtype=np.float32)
    smasks = np.zeros([opt.nway, opt.kshot, opt.img_h,opt.img_w], dtype=np.float32)
    query = np.zeros([opt.nway, opt.img_h, opt.img_w, 3], dtype=np.float32)
    qmask = np.zeros([opt.nway, opt.img_h, opt.img_w], dtype=np.float32)
    for idx in range(len(indx_c)-1):
        s_img = cv2.imread(opt.data_path + setX[indx_c[idx]] + ".jpg")
        s_msk = cv2.imread(opt.mask_data_path + setX[indx_c[idx]] + ".png")
        s_img = cv2.resize(s_img, (opt.img_h, opt.img_w))
        s_msk = cv2.resize(s_msk, (opt.img_h, opt.img_w))
        s_msk = cv2.cvtColor(s_msk, cv2.COLOR_RGB2GRAY)
        s_msk = np.where(s_msk > 0, 1., 0.)
        support[idx, 0] = s_img
        smasks[idx, 0] = s_msk

    q_img = cv2.imread(opt.data_path + setX[indx_c[idx-1]] + '.jpg')
    q_msk = cv2.imread(opt.mask_data_path + setX[indx_c[idx-1]] + '.png')
    q_img = cv2.resize(q_img, (opt.img_h, opt.img_w))
    q_msk = cv2.resize(q_msk, (opt.img_h, opt.img_w))
    q_msk = cv2.cvtColor(q_msk, cv2.COLOR_RGB2GRAY)
    q_msk = np.where(q_msk > 0, 1., 0.)
    query[0] = q_img
    qmask[0] = q_msk
    # 类似于归一化
    support = support / 255.
    query = query / 255.
    return support, smasks, query, qmask

def RCT(Fm,Fs,Fq,num_heads):
    # 由3个Muti attention组成
    print("平铺前:Fm的维度:" + str(Fm.shape) + " Fs的维度:" + str(Fs.shape) + " Fq的维度:" + str(Fq.shape))
    Hl = Fs.shape[1]
    Wl = Fs.shape[2]
    Cml = Fm.shape[0]
    # 平铺
    Fm = Fm.reshape(Fm.shape[0],-1)
    Fs = Fs.reshape(Fs.shape[0],-1)
    Fq = Fq.reshape(Fq.shape[0],-1)
    print("平铺后:Fm的维度:"+ str(Fm.shape) + " Fs的维度:"+str(Fs.shape) + " Fq的维度:" + str(Fq.shape))

    # 使用全连接层转换为Q,K,V
    d_model = Fm.shape[1]
    V1 = encoder_models.linear_layer(Fm,d_model)
    K1 = encoder_models.linear_layer(Fs,d_model)
    Q1 = encoder_models.linear_layer(Fs,d_model)
    print("V1维度:" + str(V1.shape) + " K1维度:" + str(K1.shape) + " Q1维度:" + str(Q1.shape))
    css = multi_attetion_layer(V1,K1,Q1,num_heads,Fm,d_in=Fm.shape[1],d_out=Fs.shape[1])
    print("css的维度:"+str(css.shape))

    # 然后是第二个
    d_model = Fq.shape[1]
    V2 = encoder_models.linear_layer(Fq,d_model)
    K2 = encoder_models.linear_layer(Fq,d_model)
    Q2 = encoder_models.linear_layer(Fq,d_model)
    cqq = multi_attetion_layer(V2,K2,Q2,num_heads,Fq,d_in = Fq.shape[1],d_out = Fq.shape[1])
    print("cqq的维度:" + str(cqq.shape))
    # 然后是第三个
    d_model = css.shape[1]
    V3 = encoder_models.linear_layer(css,d_model)
    K3 = encoder_models.linear_layer(Fs,d_model)
    Q3 = encoder_models.linear_layer(cqq,d_model)
    csq = multi_attetion_layer(V3,K3,Q3,num_heads=num_heads,reset=css,d_in = css.shape[1],d_out = cqq.shape[1])
    print("csq的维度:" + str(csq.shape))
    # reshape
    csq = csq.reshape(Cml,Hl,Wl)
    # print("rearranged dimen:" + str(csq.shape))
    print("rct输出维度:" + str(csq.shape))
    return csq

def RAT(Fm,Fs,Fq,num_heads,out_dim):
    # 平铺
    print("平铺前:Fm的维度:" + str(Fm.shape) + " Fs的维度:" + str(Fs.shape) + " Fq的维度:" + str(Fq.shape))
    Hl = Fs.shape[1]
    Wl = Fs.shape[2]
    Fm = Fm.reshape(Fm.shape[0], -1)
    Fs = Fs.reshape(Fs.shape[0], -1)
    Fq = Fq.reshape(Fq.shape[0], -1)
    print("平铺后:Fm的维度:" + str(Fm.shape) + " Fs的维度:" + str(Fs.shape) + " Fq的维度:" + str(Fq.shape))
    # 拼接
    Fms = np.concatenate((Fm,Fs),axis=0)
    print("Fm和Fs拼接后的维度:" + str(Fms.shape))
    # 计算ass
    cml = Fm.shape[0]
    cl = Fs.shape[0]
    ass = self_affinity(Fms,Fms,cl=cl,cml=cml)
    print("ass维度" + str(ass.shape))
    # 计算asq
    print("-----计算asq-----")
    asq = self_affinity(Fs,Fq,cl=cl,cml=0)
    print("asq维度" + str(asq.shape))
    # 计算aqq
    print("-----计算aqq-----")
    aqq = self_affinity(Fq,Fq,cl=cl,cml=0)
    print("aqq维度" + str(aqq.shape))
    # 计算最后的输出
    d_model = aqq.shape[1]
    print("d_model:" + str(d_model))
    K = encoder_models.linear_layer(ass,out_channel=d_model)
    Q = encoder_models.linear_layer(asq,out_channel=d_model)
    V = encoder_models.linear_layer(asq,out_channel=d_model)
    final_output = multi_attetion_layer(V1 = V,K1 = K,Q1 = Q,num_heads = num_heads,reset = aqq,d_in=aqq.shape[0],d_out=out_dim)
    final_output = final_output.reshape(final_output.shape[0],Hl,Wl)
    print("rat输出维度:" + str(final_output.shape))
    return final_output

def self_affinity(f1,f2,cl,cml=0):
    output_dim = f1.shape[1]
    fc_value1 = encoder_models.linear_layer(data=f1, out_channel=output_dim)
    fc_value2 = encoder_models.linear_layer(data=f2, out_channel=output_dim)
    print("全连接层后的维度:fc:" + str(fc_value1.shape) + " fc:" + str(fc_value2.shape))
    fc_value1 = transform(fc_value1)
    fc_value2 = transform(fc_value2)
    fc_value2 = torch.transpose(fc_value2, 2, 1)
    outputs = torch.matmul(fc_value2, fc_value1)
    outputs /= (cl ** 0.5)
    output = torch.softmax(outputs, dim=0)
    output = torch.squeeze(output)
    output = output.numpy()
    return output

def multi_attetion_layer(V1,K1,Q1,num_heads,reset,d_out,d_in):
    # 2.将Q,K,V分为num_heads个组
    V1 = transform(V1)
    K1 = transform(K1)
    Q1 = transform(Q1)
    print("v1:" + str(V1.shape) + " k1:" + str(K1.shape) + " q1:" + str(Q1.shape))
    group_size = int(V1.shape[1] / num_heads)
    V1_ = torch.concat(torch.split(V1, group_size, dim=1), dim=0)
    group_size = int(K1.shape[1] / num_heads)
    K1_ = torch.concat(torch.split(K1, group_size, dim=1), dim=0)
    group_size = int(Q1.shape[1] / num_heads)
    Q1_ = torch.concat(torch.split(Q1, group_size, dim=1), dim=0)

    print("Q1_:" + str(Q1_.shape) + " K1_:" + str(K1_.shape))
    # 3.开始注意力机制
    d = Q1.shape[1]
    outputs = torch.matmul(Q1_, torch.transpose(K1_, 1, 2))
    print("outputs:" + str(outputs.shape))
    outputs /= (d ** 0.5)
    outputs = torch.softmax(outputs, dim=0)
    print("outputs:" + str(outputs.shape) + " V1_:" + str(V1_.shape))
    outputs = torch.matmul(outputs, V1_)
    # 还原形状
    outputs = torch.concat(torch.split(outputs, 1, dim=0), dim=1)
    # 4.加入残差损失
    outputs += reset
    # 5.进行LN操作
    outputs = encoder_models.layer_normalization(outputs)
    # 6.FFN转换
    outputs = encoder_models.ffn(outputs,d_in = d_in,d_out=d_out)
    return outputs

def image_decoder(input1,input2,input3,input4):
    print("input1维度:" + str(input1.shape) + " input2维度:" + str(input2.shape))
    print("input3维度:" + str(input3.shape) + " input4维度:" + str(input4.shape))

    input1 = input1.transpose(1,2,0)
    ## rrb
    out1 = encoder_models.rrb(input1,in_c=input1.shape[2],out_c=192,upsample=True,times=2)
    print("上采样操作后out1的维度" + str(out1.shape))
    input2 = input2.transpose(1, 2, 0)
    out2 = encoder_models.rrb(input2,in_c=input2.shape[2],out_c=192,upsample=False)
    print("卷积操作后out2的维度" + str(out2.shape))
    ## 拼接
    out2 = np.concatenate([out1,out2],axis=0)
    print("拼接后out2维度:" + str(out2.shape))
    ## 上采样
    out2 = out2.transpose(1,2,0)
    out2 = encoder_models.upsample(out2,times=2)
    print("上采样后out2维度:" + str(out2.shape))
    ## 卷积操作
    input3 = input3.transpose(1,2,0)
    out3 = encoder_models.rrb(input3,in_c = input3.shape[2],out_c=192,upsample=False)
    print("卷积操作后out3的维度" + str(out3.shape))
    ## 拼接
    out3 = np.concatenate([out2,out3],axis=0)
    print("拼接后out3维度:" + str(out3.shape))
    ## 上采样
    out3 = out3.transpose(1, 2, 0)
    out3 = encoder_models.upsample(out3, times=2)
    print("上采样后out3维度:" + str(out3.shape))
    ## 卷积操作
    input4 = input4.transpose(1, 2, 0)
    out4 = encoder_models.rrb(input4, in_c=input4.shape[2], out_c=192, upsample=False)
    print("卷积操作后out4的维度" + str(out4.shape))
    ## 拼接
    out4 = np.concatenate([out3, out4], axis=0)
    print("拼接后out4维度:" + str(out4.shape))
    ## 1*1 卷积
    out4 = out4.transpose(1, 2, 0)
    out5 = encoder_models.conv(out4,in_channel=out4.shape[2],out_channel=2,stride=1,padding=1,ks=3)
    print("1*1卷积操作后out5的维度" + str(out5.shape))
    ## 上采样
    out5 = out5.transpose(1, 2, 0)
    out5 = encoder_models.upsample(out5, times=4)
    print("上采样后out5维度:" + str(out5.shape))
    ## 得到输出
    out5 = out5.transpose(1, 2, 0)
    # predict = query[0] * out5
    # 灰度图
    # gray = cv2.cvtColor(out5 * 255, cv2.COLOR_RGB2GRAY)
    # gray[gray == 0.] = 0
    # gray[gray != 0.] = 1
    mid = out5[:,:,1]
    print(mid.shape)
    cv2.imshow("predict",mid)
    cv2.waitKey(0)
