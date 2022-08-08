import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import time
import cv2
import encoder_models
import parser_utils

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        # 调用reset50模型
        self.net = models.resnet50(pretrained=True)
        # self.net.fc = nn.Linear(2048,2048)

    def forward(self, input):
        # 模型的结构
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        # output = self.net.fc(output)
        # 得到输出
        return output


features_dir = "../index/"
imgPath = "../datasets/PASCAL-5/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg"
#fileName = imgPath.split('/')[-1]
feature_path = os.path.join(features_dir,"2007_000027.txt")

transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor()]
)

if __name__ == '__main__':
    img = Image.open(imgPath)
    img = transform(img)
    print(img.shape)
    startTime = time.perf_counter()
    #
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    print(x.shape)
    with torch.no_grad():
        model = net()
        # if use_gpu:
        #     x = x.cuda()
        #     net = net.cuda()
        y = model(x).cpu()
        y = torch.squeeze(y)
        y = y.data.numpy()
        # print(y.shape)
    endTime = time.perf_counter()
    print("cpu time:" + str(endTime - startTime))
    # 调用model.cuda()，可以将模型加载到GPU上去
    model2 = model.cuda()
    endTime = time.perf_counter()
    with torch.no_grad():
        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
        x = x.cuda()
        y = model2(x).cpu()
        # 将维度为1的维度删掉
        y = torch.squeeze(y)
        y = y.data.numpy()
        # print(y.shape)
    endTime2 = time.perf_counter()
    print("GPU time:" + str(endTime2 - endTime))
    print(type(y))
    print(len(y))
    np.savetxt(feature_path, y, delimiter=',')