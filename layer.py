import math
import torch  
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GAT_Layer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, negative_slope=0.2):
        super(GAT_Layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w = torch.nn.Parameter(torch.FloatTensor(self.dim_in, self.dim_out))
        self.a_target = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        self.a_neighbor = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        torch.nn.init.xavier_normal_(self.w, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_target, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_neighbor, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU(negative_slope)

    def forward(self, x, adj):
        x_ = torch.mm(x, self.w)
        scores_target = torch.mm(x_, self.a_target)
        scores_neighbor = torch.mm(x_, self.a_neighbor)
        scores = scores_target + torch.transpose(scores_neighbor, 0, 1)
        
        scores = torch.mul(adj, scores)
        scores = self.leakyrelu(scores)
        scores = torch.where(adj>0, scores, -9e15*torch.ones_like(scores))
        coefficients = torch.nn.functional.softmax(scores, dim=1)
        x_ = torch.nn.functional.elu(torch.mm(coefficients, x_))
        return x_
    
#动态复杂度调整与动态注意力机制
class DynamicGAT_Layer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, complexity_factor=1.0, negative_slope=0.2):
        super(DynamicGAT_Layer, self).__init__()
        
        # 输入和输出特征维度
        self.dim_in = dim_in
        # 输出维度基于复杂度因子进行调整
        self.dim_out = int(dim_out * complexity_factor)  
        
        # 定义权重矩阵，目标节点和邻居节点的注意力权重
        self.w = torch.nn.Parameter(torch.FloatTensor(self.dim_in, self.dim_out))
        self.a_target = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        self.a_neighbor = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        
        # 使用Xavier初始化权重矩阵
        torch.nn.init.xavier_normal_(self.w, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_target, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_neighbor, gain=1.414)
        
        # 使用LeakyReLU激活函数
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope)
        
        # 引入动态调整的注意力权重参数
        self.attention_weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, adj):
        # 图输入特征x与权重w相乘得到新的节点表示
        x_ = torch.mm(x, self.w)
        
        # 计算目标节点和邻居节点的注意力分数
        scores_target = torch.mm(x_, self.a_target)
        scores_neighbor = torch.mm(x_, self.a_neighbor)
        
        # 汇总目标节点与邻居节点的注意力分数
        scores = scores_target + torch.transpose(scores_neighbor, 0, 1)
        
        # 动态调整注意力分数
        scores = scores * self.attention_weight
        
        # 乘上邻接矩阵，确保仅通过有连接的节点计算注意力
        scores = torch.mul(adj, scores)
        
        # 使用LeakyReLU激活函数进行非线性变换
        scores = self.leakyrelu(scores)
        
        # 将邻接矩阵为0的部分设为极小值，以避免不稳定的数值
        scores = torch.where(adj > 0, scores, -9e15 * torch.ones_like(scores))
        
        # 对注意力分数进行softmax归一化
        coefficients = torch.nn.functional.softmax(scores, dim=1)
        
        # 最终的节点表示，通过softmax后的注意力系数与变换后的节点特征相乘
        x_ = torch.nn.functional.elu(torch.mm(coefficients, x_))
        
        return x_  # 返回新的节点表示


class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        result = torch.mean(result)
        return result

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
    
    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)