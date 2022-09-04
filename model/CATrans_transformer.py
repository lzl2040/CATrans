import torch
import torch.nn as nn
import torch.nn.functional as F

# 前馈神经网络
class FFN(nn.Module):
    def __init__(self,input_channel,d_in,d_hid,d_out,dropout=0.4):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(d_in, d_hid,bias=False)
        self.w2 = nn.Linear(d_hid, d_out,bias=False)
        self.activate = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(input_channel, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        #
        residual = x
        x = self.w1(x)
        x = self.activate(x)
        x = self.dropout_1(x)
        x = self.w2(x)
        x = self.dropout_2(x)
        x = self.layer_norm(x)
        x += residual
        # x = self.layer_norm(x)
        return x

# context
class RCT(nn.Module):
    def __init__(self,num_heads,channel,m_dim,nm_dim,d_hid):
        super(RCT,self).__init__()
        self.num_heads = num_heads
        # 三个LN和FFN
        self.LN1 = nn.LayerNorm(channel, eps=1e-6)
        self.FFN1 = FFN(input_channel=channel, d_in=channel, d_hid=d_hid, d_out=channel)

        self.LN2 = nn.LayerNorm(channel, eps=1e-6)
        self.FFN2 = FFN(input_channel=channel, d_in=channel, d_hid=d_hid, d_out=channel)

        self.LN3 = nn.LayerNorm(channel, eps=1e-6)
        self.FFN3 = FFN(input_channel=channel, d_in=channel, d_hid=d_hid, d_out=channel)

        self.block1 = AttentionBlock(num_heads=2,LN=self.LN1,FFN=self.FFN1,q_in=nm_dim,v_in=m_dim,k_in=nm_dim)
        self.block2 = AttentionBlock(num_heads=2,LN=self.LN2,FFN=self.FFN2,q_in=nm_dim,v_in=nm_dim,k_in=nm_dim)
        self.block3 = AttentionBlock(num_heads=2,LN=self.LN3,FFN=self.FFN3,q_in=nm_dim,v_in=m_dim,k_in=nm_dim)

    def forward(self,Fm,Fs,Fq,out_dim):
        # 第一个Attention Block
        css = self.block1(Q = Fs,K = Fs,V = Fm,resnet = Fm,out_dim=Fm.shape[-1],dropout=True)
        # 第二个Attention Block
        cqq = self.block2(Q = Fq,K = Fq,V = Fq,resnet = Fq,out_dim=Fq.shape[-1],dropout=True)
        # 第三个Attention Block
        csq = self.block3(Q = cqq,K = Fs,V = css,resnet = css,out_dim=out_dim,dropout=True)
        return csq

# affinity
class RAT(nn.Module):
    def __init__(self,num_heads,Fms_out,Fs_out,Fq_out,channel,d_hid,q_dim,k_dim,v_dim,dropout=0.1):
        super(RAT,self).__init__()
        self.num_heads = num_heads
        self.fc1 = nn.Linear(in_features=Fms_out,out_features=Fms_out,bias=False)
        self.fc2 = nn.Linear(in_features=Fms_out,out_features=Fms_out,bias=False)
        self.fc3 = nn.Linear(in_features=Fs_out, out_features=Fs_out,bias=False)
        self.fc4 = nn.Linear(in_features=Fq_out, out_features=Fq_out,bias=False)
        self.fc5 = nn.Linear(in_features=Fq_out, out_features=Fq_out,bias=False)
        self.fc6 = nn.Linear(in_features=Fq_out, out_features=Fq_out,bias=False)
        self.LN = nn.LayerNorm(channel, eps=1e-6)
        self.FFN = FFN(input_channel=channel, d_in=channel, d_hid=d_hid, d_out=channel)
        self.block1 = AttentionBlock(num_heads = 2,LN=self.LN,FFN=self.FFN,q_in=q_dim,k_in=k_dim,v_in=v_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self,Fs,Fq,out_dim,Fms):
        # Fs,Fq:B * (H*W) * C
        # Fms:B * (H*W) * (Cm+Cs)
        # 计算ass,asq,aqq
        ass = self.cal_affinity(a = Fms,b = Fms,cl = Fms.shape[2],fc_a=self.fc1,fc_b=self.fc2)
        asq = self.cal_affinity(a = Fs,b = Fq,cl = Fs.shape[2],fc_a=self.fc3,fc_b=self.fc4)
        aqq = self.cal_affinity(a = Fq,b = Fq,cl = Fq.shape[2],fc_a=self.fc5,fc_b=self.fc6)
        output = self.block1(Q = asq,K = ass,V = asq,resnet = aqq,out_dim=out_dim,dropout=True)
        return output

    def cal_affinity(self,a,b,fc_a,fc_b,cl,cml = 0):
        # 结果维度: B * (H*W) * C
        fc1_mat = fc_a(a)
        fc2_mat = fc_b(b)
        # 两个矩阵相乘
        ## 维度 (H*W) * (H*W)
        affinity = torch.matmul(fc1_mat,torch.transpose(fc2_mat,1,2))
        affinity /= ((cl + cml) ** 0.5)
        # row-wise,因为维度是B * (H*W) * (H*W),则对应行应该为第2个(H * W)
        affinity = torch.softmax(affinity,dim=-1)
        affinity = self.dropout_layer(affinity)
        return affinity

class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,ks,st,p):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=ks,stride=st,padding=p),
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
            nn.Conv2d(1,256,kernel_size=3,stride=4,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2048),
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
    def __init__(self,num_heads,LN,FFN,q_in,k_in,v_in,dropout=0.1):
        super(AttentionBlock,self).__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(in_features=q_in,out_features=q_in,bias=False)
        self.fc_k = nn.Linear(in_features=k_in,out_features=k_in,bias=False)
        self.fc_v = nn.Linear(in_features=v_in, out_features=v_in,bias=False)
        self.fc_qkv = nn.Linear(in_features=v_in, out_features=v_in,bias=False)
        self.LN = LN
        self.FFN = FFN
        self.dropout_layer = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self,Q,K,V,out_dim,resnet=None,dropout=False):
        # 1.首先计算Q,K,V,维度:B * (H*W) * C
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)
        # 2.multi head attention
        ## 获得通道数目
        B,HW,channel = Q.shape
        group_size = channel // self.num_heads
        ## 增加维度,便于multi head操作
        Qs = Q.view(B,-1,self.num_heads,group_size).transpose(1,2)
        Ks = K.view(B, -1, self.num_heads, group_size).transpose(1, 2)
        Vs = V.view(B, -1, self.num_heads, group_size).transpose(1, 2)
        ## 进行attention操作 得到的维度:(H*W) * (H*W)
        attention = torch.matmul(Qs,torch.transpose(Ks,2,3))
        attention /= (group_size ** 0.5)
        # row-wise,因为维度是B * Head * (H*W) * (H*W),对应的行应该是第二个H*W
        attention = torch.softmax(attention,dim=-1)
        if dropout != False:
            attention = self.dropout_layer(attention)
        ## 此时维度变为 (H*W) * (C)
        attention = torch.matmul(attention,Vs)
        ## 拼接
        attention = attention.transpose(1,2).contiguous().view(B,-1,self.num_heads * group_size)
        ### 再次映射
        attention = self.fc_qkv(attention)
        ### dropout
        attention = self.proj_drop(attention)
        ## 加入残差损失
        attention += resnet
        ## 进行LN归一化操作
        attention = self.LN(attention)
        ## 进行FFN转换
        attention = self.FFN(attention)
        return attention

class Decoder(nn.Module):
    def __init__(self,in1,in2,in3,in4,in5):
        super(Decoder,self).__init__()
        self.conv1 = Conv(in_channels=in1,out_channels=512,ks=1,st=1,p=0)
        self.conv2 = Conv(in_channels=in2,out_channels=512,ks=1,st=1,p=0)
        self.conv3 = Conv(in_channels=in3,out_channels=512,ks=1,st=1,p=0)
        self.conv4 = Conv(in_channels=in4,out_channels=512,ks=1,st=1,p=0)
        # 1*1 conv 转换维度
        self.conv5 = Conv(in_channels=in5,out_channels=2,ks=1,st=1,p=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,feature1,feature2,feature3,feature4):
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
        predict_label = torch.squeeze(predict_label)
        predict_label = self.relu(predict_label)
        return predict_label

    def conv(self,x,conv_layer,upsample=False,times=2,up_mode='bilinear'):
        output = conv_layer(x)
        # 需要上采样
        if upsample == True:
            output = self.upsample(output,times=times,up_mode=up_mode)
        return output

    def upsample(self,x,times,up_mode='bilinear'):
        target_dim = x.shape[-1] * times
        if up_mode == 'bilinear':
            output = F.interpolate(
                x,
                size=(target_dim, target_dim),
                mode=up_mode,
                align_corners=True)
        else:
            output = F.interpolate(
                x,
                size=(target_dim, target_dim),
                mode=up_mode,
                align_corners=None)
        return output
