# 各种encoder
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from PIL import Image
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor()]
)

# resbet50网络结构
class resnet50(nn.Module):
    def __init__(self,freeze_bn = False):
        super(resnet50, self).__init__()
        # 调用reset50模型
        self.net = models.resnet50(progress=True)
        self.freeze_bn = freeze_bn

    def train(self,mode = True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, input):
        # 模型的结构
        # S1
        output = self.net.conv1(input)

        output = self.net.bn1(output)
        output = self.net.relu(output)
        F1 = output
        # S2
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        F2 = output
        # S3
        output = self.net.layer2(output)
        F3 = output
        # S4
        output = self.net.layer3(output)
        F4 = output
        # 得到输出
        return F1,F2,F3,F4

class maskEncoder(nn.Module):
    def __init__(self):
        super(maskEncoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        output = self.layer1(x)
        Fm1 = output
        output = self.layer2(output)
        Fm2 = output
        output = self.layer3(output)
        Fm3 = output
        output = self.layer4(output)
        Fm4 = output
        return Fm1,Fm2,Fm3,Fm4

class linear(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(linear,self).__init__()
        self.layer = nn.Linear(in_channel,out_channel,bias=True)
    def forward(self,x):
        # x = transform(x)
        output = self.layer(x)
        return output

class LayerNormalization(nn.Module):
    def __init__(self,data):
        super(LayerNormalization, self).__init__()
        self.layer = nn.LayerNorm([data.shape[1],data.shape[2]],eps=1e-05)

    def forward(self,x):
        output = self.layer(x)
        return output

class FeedForward(nn.Module):
    def __init__(self,input_channel,d_in,d_hid,d_out,dropout=0.1):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, d_out)
        self.activate = nn.ReLU()
        self.layer_norm = nn.LayerNorm([input_channel,d_out], eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w1(x)
        x = self.activate(x)
        x = self.w2(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class Conv(nn.Module):
    def __init__(self,ks,padding,in_channel,out_channel,stride):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=ks,stride=stride,padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        output = self.conv(x)
        return output

# dpn网络中的cab块
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels,use_silu=False):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if use_silu:
            self.relu = nn.SiLU()
        else:
            self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x
        x = torch.cat([x1,x2],dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res

# dpn网络中的rrb块
class RRB(nn.Module):
    def __init__(self, in_channels, out_channels,use_silu=False):
        super(RRB, self).__init__()
        # 增加了这个减半的卷积层
        self.half1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=2,padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if use_silu:
            self.relu = nn.SiLU()
        else:
            self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.half1(x)
        x = self.conv1(x)
        res  = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.relu(x)
        return x

def resnet50_encoder(img):
    # 转换数据
    img = transform(img)
    # 使用resnet50 模型
    model = resnet50(freeze_bn=True)
    # 切换到cpu或者gpu
    model = model.to(device)
    with torch.no_grad():
        # 对数据维度进行扩充，requires_grad为False说明不用被求导
        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
        # print(x.shape)
        # 将x传到cuda上,使用gpu
        x = x.cuda()
        # 将输出结果转到CPU上
        F1,F2,F3,F4 = model(x)
        F1 = F1.cpu()
        F2 = F2.cpu()
        F3 = F3.cpu()
        F4 = F4.cpu()
        # 将维度为1的维度删掉
        F1 = torch.squeeze(F1)
        F2 = torch.squeeze(F2)
        F3 = torch.squeeze(F3)
        F4 = torch.squeeze(F4)
        # 变为numpy
        F1 = F1.data.numpy()
        F2 = F2.data.numpy()
        F3 = F3.data.numpy()
        F4 = F4.data.numpy()
    return F1,F2,F3,F4

def linear_layer(data,out_channel):
    data = transform(data)
    model = linear(data.shape[2],out_channel=out_channel)
    model = model.to(device)
    with torch.no_grad():
        # print(data.shape)
        x = Variable(data,requires_grad=False)
        # print(x.shape)
        x = x.cuda()
        y = model(x).cpu()
        y = torch.squeeze(y)
        y = y.data.numpy()
    return y

def ffn(data,d_in,d_out,d_hidden=1024):
    data = transform(data)
    model = FeedForward(input_channel=data.shape[1],d_in=d_in,d_hid=d_hidden,d_out=d_out)
    model = model.to(device)
    # 无需计算梯度,推理阶段
    with torch.no_grad():
        # requires_grad=False 没有参数更新
        x = Variable(data,requires_grad=False)
        x = x.cuda()
        y = model(x).cpu()
        y = torch.squeeze(y)
        y = y.data.numpy()
    return y

def conv(data,in_channel,out_channel,stride,padding,ks):
    data = transform(data)
    model = Conv(ks=ks,padding=padding,in_channel=in_channel,out_channel=out_channel,stride=stride)
    model = model.to(device)
    with torch.no_grad():
        x = Variable(data, requires_grad=False)
        x = x.cuda()
        y = model(x).cpu()
        y = torch.squeeze(y)
        y = y.data.numpy()
    return y

def layer_normalization(data):
    # data = transform(data)
    model = LayerNormalization(data)
    model = model.to(device)
    with torch.no_grad():
        x = Variable(data, requires_grad=False)
        x = x.cuda()
        y = model(x).cpu()
        y = torch.squeeze(y)
        y = y.data.numpy()
    return y

def resnet101_encoder():
    print("123")

def swim_encoder():
    print("123")

def mask_encoder(img):
    # 卷积神经网络
    img = transform(img)
    net = maskEncoder()
    net = net.to(device)
    print(img.shape)
    with torch.no_grad():
        x = Variable(torch.unsqueeze(img,dim=0), requires_grad=False)
        x = x.cuda()
        Fm1,Fm2,Fm3,Fm4 = net(x)
        Fm1 = Fm1.cpu()
        Fm2 = Fm2.cpu()
        Fm3 = Fm3.cpu()
        Fm4 = Fm4.cpu()
        # 压缩维度
        Fm1 = torch.squeeze(Fm1)
        Fm2 = torch.squeeze(Fm2)
        Fm3 = torch.squeeze(Fm3)
        Fm4 = torch.squeeze(Fm4)
        Fm1 = Fm1.data.numpy()
        Fm2 = Fm2.data.numpy()
        Fm3 = Fm3.data.numpy()
        Fm4 = Fm4.data.numpy()
    return Fm1,Fm2,Fm3,Fm4

def rrb(data,in_c,out_c,upsample,times=2):
    data = transform(data)
    model = RRB(in_channels = in_c,out_channels = out_c)
    model = model.to(device)
    with torch.no_grad():
        x = Variable(torch.unsqueeze(data,dim=0), requires_grad=False)
        # print(x.shape)
        x = x.cuda()
        y = model(x).cpu()
    # 进行上采样
    if upsample == True:
        target_dim = y.shape[2] * times
        y = F.interpolate(
            y,
            size=(target_dim, target_dim),
            mode='nearest',
            align_corners=None)
    y = torch.squeeze(y)
    y = y.data.numpy()
    return y

def upsample(data,times):
    data = transform(data)
    target_dim = data.shape[2] * times
    data = torch.unsqueeze(data,dim=0)
    # print(data.shape)
    data = F.interpolate(
        data,
        size=(target_dim, target_dim),
        mode='nearest',
        align_corners=None)
    data = torch.squeeze(data)
    data = data.data.numpy()
    return data

def classifier(data):
    data = transform(data)
    model = Classifier()
    with torch.no_grad():
        x = Variable(torch.unsqueeze(data, dim=0), requires_grad=False)
        x = x.cuda()
        y = model(x).cpu()
        y = torch.squeeze(y)
        y = y.data.numpy()
    return y