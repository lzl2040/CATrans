import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import numpy as np

# resbet50����ṹ
class Resnet50(nn.Module):
    def __init__(self,freeze_bn = False):
        super(Resnet50, self).__init__()
        # ����reset50ģ��
        self.net = models.resnet50(progress=True)
        # �Ƿ񶳽�BN��
        self.freeze_bn = freeze_bn

    def train(self,mode = True):
        super().train(mode)
        # ����BN��
        if self.freeze_bn:
            print('freeze')
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, input):
        # ģ�͵Ľṹ
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        # modify 2022.9.3
        # S1
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        F1 = output
        # S2
        output = self.net.layer2(output)
        F2 = output
        # S3
        output = self.net.layer3(output)
        F3 = output
        # S4
        output = self.net.layer4(output)
        F4 = output
        return F1,F2,F3,F4