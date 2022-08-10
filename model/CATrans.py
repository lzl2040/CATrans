import torch
from torch import nn
import torch.nn.functional as F

from model.CATrans_transformer import MaskEncoder, RCT, RAT, Decoder
from model.Resnet import Resnet50


class CATrans(nn.Module):
    def __init__(self,shot,training):
        super(CATrans, self).__init__()

        # 训练集中每个类别的图片样本个数
        self.shot = shot
        # 损失函数,这里我们使用交叉损失,前景权重为1,背景为4
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 4.0]), ignore_index=255)
        # 是否是训练
        self.training = training
        # 接下来定义网络中的各种层
        ## image_encoder
        self.image_encoder = Resnet50(freeze_bn=True)
        ## mask_encoder
        self.mask_encoder = MaskEncoder()
        ## decoder
        self.decoder = Decoder(in1=1600,in2=2816,in3=256,in4=64,in5=1024)
        ## rct
        self.rct1 = RCT(num_heads=2,out_dim=512,input_dim=512,channel=512)
        self.rct2 = RCT(num_heads=2,out_dim=1024,input_dim=1024,channel=1024)
        ## rat
        self.rat1 = RAT(num_heads=2,Fms_out=1024,Fq_out=512,Fs_out=512,out_dim=2304,final_out_dim=2304)
        self.rat2 = RAT(num_heads=2,Fms_out=2048,Fq_out=1024,Fs_out=1024,out_dim=576,final_out_dim=576)
        # 初始权值
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 初始化均匀分布
                # nn.init.xavier_uniform_(m.weight)
                nn.init.xavier_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        # 维度:support_image (N * S * C * H * W),S为shot数,N为类别数,C为通道数,H为图像的高,W为图像的宽
        # 维度:query_image (N * C * H * W)
        # 维度:support mask (N * S * H * W)
        support_image,query_image,support_mask,query_mask = x
        # 对数据维度进行处理,变为N * C * H * W
        C,H,W = support_image.shape[-3:]
        support_image = support_image.view(-1,C,H,W)

        Fq1,Fq2,Fq3,Fq4 = self.image_encoder(support_image)

        Fs1, Fs2, Fs3, Fs4 = self.image_encoder(query_image)

        support_mask = torch.unsqueeze(support_mask,dim=2)
        support_mask = support_mask.view(-1,1,H,W)
        support_mask = support_mask.float()
        Fm1,Fm2,Fm3,Fm4 = self.mask_encoder(support_mask)

        # 使用RAT和RCT
        ## 首先平铺Fm,Fs,Fq
        ## 计算level=3的情况,维度:B * C * (H*W)
        Fm3_flatten = Fm3.view(Fm3.shape[0],Fm3.shape[1],-1)
        Fs3_flatten = Fs3.view(Fs3.shape[0],Fs3.shape[1],-1)
        Fq3_flatten = Fq3.view(Fq3.shape[0], Fq3.shape[1], -1)
        ## 转换维度,变为B * (H * W) * C
        Fm3_flatten = torch.transpose(Fm3_flatten,1,2)
        Fs3_flatten = torch.transpose(Fs3_flatten,1,2)
        Fq3_flatten = torch.transpose(Fq3_flatten,1,2)
        # print("Fq3 dim:" + str(Fq3_flatten.shape))
        out1 = self.rct1(Fm=Fm3_flatten,Fq=Fq3_flatten,Fs=Fs3_flatten,out_dim=2304)
        out1 = torch.transpose(out1,1,2)
        Fms3 = torch.concat([Fm3_flatten, Fs3_flatten], dim=2)
        # print(Fms3.shape)
        out2 = self.rat1(Fq=Fq3_flatten,Fs=Fs3_flatten,out_dim = 2304,Fms = Fms3)
        out2 = torch.transpose(out2, 1, 2)

        # 计算level=4的情况
        Fm4_flatten = Fm4.view(Fm4.shape[0], Fm4.shape[1], -1)
        Fs4_flatten = Fs4.view(Fs4.shape[0], Fs4.shape[1], -1)
        Fq4_flatten = Fq4.view(Fq4.shape[0], Fq4.shape[1], -1)
        ## 转换维度,变为B * (H * W) * C
        Fm4_flatten = torch.transpose(Fm4_flatten, 1, 2)
        Fs4_flatten = torch.transpose(Fs4_flatten, 1, 2)
        Fq4_flatten = torch.transpose(Fq4_flatten, 1, 2)
        out3 = self.rct2(Fm=Fm4_flatten, Fq=Fq4_flatten, Fs=Fs4_flatten, out_dim=576)
        out3 = torch.transpose(out3, 1, 2)
        # print(out3[0])
        Fms4 = torch.concat([Fm4_flatten, Fs4_flatten], dim=2)
#         print(Fms4.shape)
        out4 = self.rat2(Fq=Fq4_flatten, Fs=Fs4_flatten, out_dim=576,Fms = Fms4)
        out4 = torch.transpose(out4, 1, 2)
        feature1 = torch.concat([out3, out4], dim=1)
        feature1 = feature1.view(feature1.shape[0],feature1.shape[1],Fm4.shape[-1],-1)
        feature2 = torch.concat([out1, out2], dim=1)
        feature2 = feature2.view(feature2.shape[0], feature2.shape[1], Fm3.shape[-1], -1)
        feature3 = Fq2
        feature4 = Fq1
#         print("feature 1 dim:" + str(feature1.shape))
#         print("feature 2 dim:" + str(feature2.shape))
#         print("feature 3 dim:" + str(feature3.shape))
#         print("feature 4 dim:" + str(feature4.shape))
        # decoder工作
        predict_label = self.decoder(feature1=feature1,feature2=feature2,feature3=feature3,feature4=feature4)
        # print(predict_label[0,0])
#         print("predict dim:" + str(predict_label.shape))
        # predict_label = predict_label * 255
        main_loss = self.criterion(predict_label, query_mask)
        if self.training:
            # print("compute loss")
            # main_loss = self.criterion(predict_label,query_mask)
            return main_loss,predict_label
        else:
            return main_loss,predict_label