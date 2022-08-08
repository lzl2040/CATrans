import torch
import torch.nn as nn
import torch.nn.functional as F

# 前馈神经网络
class FFN(nn.Module):
    def __init__(self,input_channel,d_in,d_hid,d_out,dropout=0.1):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(d_in, d_hid,bias=False)
        self.w2 = nn.Linear(d_hid, d_out,bias=False)
        self.activate = nn.ReLU(inplace=True)
        # self.layer_norm = nn.LayerNorm([input_channel,d_out], eps=1e-6)
        self.layer_norm = nn.LayerNorm(input_channel, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #
        residual = x
        x = self.w1(x)
        x = self.activate(x)
        x = self.w2(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

# dpn网络中的rrb块
class RRB(nn.Module):
    def __init__(self, in_channels, out_channels,use_silu=False):
        super(RRB, self).__init__()
        # 我增加了这个减半的卷积层
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

# context
class RCT(nn.Module):
    def __init__(self,num_heads,out_dim,input_dim,channel):
        super(RCT,self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.out_dim = out_dim
        # 三个LN和FFN
        # self.LN1 = nn.LayerNorm([channel, out_dim], eps=1e-5)
        self.LN1 = nn.LayerNorm(channel, eps=1e-5)
        self.FFN1 = FFN(input_channel=channel, d_in=out_dim, d_hid=2048, d_out=out_dim)

        # self.LN2 = nn.LayerNorm([channel, out_dim], eps=1e-5)
        self.LN2 = nn.LayerNorm(channel, eps=1e-5)
        self.FFN2 = FFN(input_channel=channel, d_in=out_dim, d_hid=2048, d_out=out_dim)

        # self.LN3 = nn.LayerNorm([channel, out_dim], eps=1e-5)
        self.LN3 = nn.LayerNorm(channel, eps=1e-5)
        self.FFN3 = FFN(input_channel=channel, d_in=out_dim, d_hid=2048, d_out=out_dim)

        self.block1 = AttentionBlock(input_dim=out_dim, output_dim=out_dim, num_heads=2,LN=self.LN1,FFN=self.FFN1)
        self.block2 = AttentionBlock(input_dim=out_dim, output_dim=out_dim, num_heads=2,LN=self.LN2,FFN=self.FFN2)
        self.block3 = AttentionBlock(input_dim = out_dim,output_dim = out_dim,num_heads = 2,LN=self.LN3,FFN=self.FFN3)

    def forward(self,Fm,Fs,Fq,out_dim):
        # print("start rct")
        # 第一个Attention Block
        css = self.block1(Q = Fs,K = Fs,V = Fm,resnet = Fm,out_dim=Fm.shape[-1])
        # 第二个Attention Block
        cqq = self.block2(Q = Fq,K = Fq,V = Fq,resnet = Fq,out_dim=Fq.shape[-1])
        # 第三个Attention Block
        csq = self.block3(Q = cqq,K = Fs,V = css,resnet = css,out_dim=out_dim)
        # print("csq dimension:" + str(csq.shape))
        return csq

# affinity
class RAT(nn.Module):
    def __init__(self,num_heads,Fms_out,Fs_out,Fq_out,out_dim,final_out_dim):
        super(RAT,self).__init__()
        self.num_heads = num_heads
        self.fc1 = nn.Linear(in_features=Fms_out,out_features=Fms_out,bias=False)
        self.fc2 = nn.Linear(in_features=Fms_out,out_features=Fms_out,bias=False)
        self.fc3 = nn.Linear(in_features=Fs_out, out_features=Fs_out, bias=False)
        self.fc4 = nn.Linear(in_features=Fq_out, out_features=Fq_out, bias=False)
        self.fc5 = nn.Linear(in_features=Fq_out, out_features=Fq_out, bias=False)
        self.fc6 = nn.Linear(in_features=Fq_out, out_features=Fq_out, bias=False)
        # self.LN = nn.LayerNorm([out_dim, out_dim], eps=1e-5)
        self.LN = nn.LayerNorm(out_dim, eps=1e-5)
        self.FFN = FFN(input_channel=out_dim, d_in=out_dim, d_hid=1024, d_out=final_out_dim)
        self.block1 = AttentionBlock(input_dim = out_dim,output_dim = out_dim,num_heads = 2,LN=self.LN,FFN=self.FFN)

    def forward(self,Fs,Fq,out_dim,Fms):
        # Fs,Fq:B * (H*W) * C
        # Fms:B * (H*W) * (Cm+Cs)
        # print("Fms shape:" + str(Fms.shape))
        # 计算ass,asq,aqq
        ass = self.cal_affinity(a = Fms,b = Fms,cl = Fms.shape[2],fc_a=self.fc1,fc_b=self.fc2)
#         print("ass维度" + str(ass.shape))
        asq = self.cal_affinity(a = Fs,b = Fq,cl = Fs.shape[2],fc_a=self.fc3,fc_b=self.fc4)
#         print("asq维度" + str(asq.shape))
        aqq = self.cal_affinity(a = Fq,b = Fq,cl = Fq.shape[2],fc_a=self.fc5,fc_b=self.fc6)
#         print("aqq维度" + str(aqq.shape))
        output = self.block1(Q = asq,K = ass,V = asq,resnet = aqq,out_dim=out_dim)
        # print("rat output dimesion:" + str(output.shape))
        return output

    def cal_affinity(self,a,b,fc_a,fc_b,cl,cml = 0):
        # 结果维度: B * (H*W) * C
        fc1_mat = fc_a(a)
        fc2_mat = fc_b(b)
        # 转置 变为B * (H*W) * C
        # fc1_mat = fc1_mat.transpose(1, 2)
        # fc2_mat = fc2_mat.transpose(1, 2)
        # 两个矩阵相乘
        ## 维度 (H*W) * (H*W)
        affinity = torch.matmul(fc1_mat,torch.transpose(fc2_mat,1,2))
        # affinity = torch.matmul(torch.transpose(fc2_mat, 1, 2),fc1_mat)
        affinity /= ((cl + cml) ** 0.5)
        # print("affinity dimesion:" + str(affinity.shape))
        # row-wise,因为维度是B * (H*W) * (H*W),则对应行应该为第一个(H * W)
        affinity = torch.softmax(affinity,dim=2)
        # print("affinity shape:" + str(affinity.shape))
        # affinity = torch.softmax(affinity, dim=1)
        return affinity

class Conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        output = self.conv(x)
        return output

# mask encoder
class MaskEncoder(nn.Module):
    def __init__(self):
        super(MaskEncoder,self).__init__()
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

# multi head attention
class AttentionBlock(nn.Module):
    def __init__(self,input_dim,output_dim,num_heads,LN,FFN):
        super(AttentionBlock,self).__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(in_features=input_dim,out_features=output_dim,bias=False)
        self.fc_k = nn.Linear(in_features=input_dim,out_features=output_dim,bias=False)
        self.fc_v = nn.Linear(in_features=input_dim, out_features=output_dim,bias=False)
        self.LN = LN
        self.FFN = FFN

    def forward(self,Q,K,V,out_dim,resnet=None):
        # 1.首先计算Q,K,V,维度:B * (H*W) * C
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)
        # 2.multi head attention
        ## 获得通道数目
        # print("Q shape:" + str(Q.shape))
        channel = Q.shape[-1]
        group_size = channel // self.num_heads
        ## 增加维度,便于multi head操作
        Q = torch.unsqueeze(Q,dim=1)
        K = torch.unsqueeze(K, dim=1)
        V = torch.unsqueeze(V, dim=1)
        Qs = torch.concat(torch.split(Q,group_size,dim=3),dim=1)
        Ks = torch.concat(torch.split(K,group_size,dim=3),dim=1)
        Vs = torch.concat(torch.split(V, group_size, dim=3), dim=1)
        # print("Qs shape" + str(Qs.shape))
        # 将维度变为 B * Head * (H * W) * C
        # Qs = torch.transpose(Qs,2,3)
        # Ks = torch.transpose(Ks,2,3)
        # Vs = torch.transpose(Vs,2,3)
        #print("Qs shape:" + str(Qs.shape))
        #print("Ks shape:" + str(Ks.shape))
        ## 进行attention操作 得到的维度:(H*W) * (H*W)
        attention = torch.matmul(Qs,torch.transpose(Ks,2,3))
        #print("mul q k:attention:" + str(attention.shape))
        attention /= (group_size ** 0.5)

        # row-wise,因为维度是B * Head * (H*W) * (H*W),对应的行应该是第一个H*W
        attention = torch.softmax(attention,dim=3)
        # attention = torch.softmax(attention, dim=1)
        ## 此时维度变为 (H*W) * (C)
        attention = torch.matmul(attention,Vs)
        #print("mul v:attetntion shape:" + str(attention.shape))
        ## 把维度转为 B * Head * C * (H*W)
        # attention = attention.transpose(2,3)
        #print("resnet shape:" + str(resnet.shape))
        ## 拼接
        attention = torch.concat(torch.split(attention,1,dim=1),dim=3)
        #print("attetntion shape:" + str(attention.shape))
        ## 加入残差损失
        ### 去掉维度为1的维度,跟之间增加维度相抵消
        attention = torch.squeeze(attention)
        # print("attention维度:" + str(attention.shape))
        # print("resnet维度:" + str(resnet.shape))
        attention += resnet
        ## 进行LN归一化操作
        attention = self.LN(attention)
        ## 进行FFN转换
        attention = self.FFN(attention)
        # print("attention block output:" + str(attention.shape))
        return attention

class Decoder(nn.Module):
    def __init__(self,in1,in2,in3,in4,in5):
        super(Decoder,self).__init__()
        self.conv1 = Conv(in_channels=in1,out_channels=256)
        self.conv2 = Conv(in_channels=in2,out_channels=256)
        self.conv3 = Conv(in_channels=in3,out_channels=256)
        self.conv4 = Conv(in_channels=in4,out_channels=256)
        # 1*1 conv 转换维度
        self.conv5 = nn.Conv2d(in_channels=in5,out_channels=2,kernel_size=1,stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,feature1,feature2,feature3,feature4):
        # print("start decoder")
        predict_label = self.conv(feature1,conv_layer=self.conv1,upsample=True)
        mid_ft = self.conv(feature2,conv_layer=self.conv2,upsample=False)
        predict_label = torch.concat([predict_label,mid_ft],dim=1)
        # 上采样
        predict_label = self.upsample(predict_label,times=2)
        mid_ft2 = self.conv(feature3,conv_layer=self.conv3,upsample=False)
        predict_label = torch.concat([predict_label,mid_ft2],dim=1)
        # 上采样
        predict_label = self.upsample(predict_label, times=2)
        mid_ft3 = self.conv(feature4,conv_layer=self.conv4,upsample=False)
        predict_label = torch.concat([predict_label,mid_ft3],dim=1)
        predict_label = self.conv5(predict_label)
        predict_label = self.upsample(predict_label,times=4,up_mode='bilinear')
        # predict_label = self.upsample(predict_label, times=4, up_mode='nearest')
        predict_label = torch.squeeze(predict_label)
        predict_label = self.relu(predict_label)
        # print("predict dim:" + str(predict_label.shape))
        return predict_label

    def conv(self,x,conv_layer,upsample=False,times=2,up_mode='nearest'):
        output = conv_layer(x)
        # 需要上采样
        if upsample == True:
            output = self.upsample(output,times=times,up_mode=up_mode)
        return output

    def upsample(self,x,times,up_mode='nearest'):
        target_dim = x.shape[-1] * times
        output = F.interpolate(
            x,
            size=(target_dim, target_dim),
            mode=up_mode,
            align_corners=None)
        return output