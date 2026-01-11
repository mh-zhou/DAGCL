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
    

class DynamicGAT_Layer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, complexity_factor=1.0, negative_slope=0.2):
        super(DynamicGAT_Layer, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = int(dim_out * complexity_factor)  
        
        self.w = torch.nn.Parameter(torch.FloatTensor(self.dim_in, self.dim_out))
        self.a_target = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        self.a_neighbor = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        
        torch.nn.init.xavier_normal_(self.w, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_target, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_neighbor, gain=1.414)
        

        self.leakyrelu = torch.nn.LeakyReLU(negative_slope)
        
        self.attention_weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, adj):
        x_ = torch.mm(x, self.w)
        scores_target = torch.mm(x_, self.a_target)
        scores_neighbor = torch.mm(x_, self.a_neighbor)
        scores = scores_target + torch.transpose(scores_neighbor, 0, 1)
        
        scores = scores * self.attention_weight
        
        scores = torch.mul(adj, scores)
        
        scores = self.leakyrelu(scores)
    
        scores = torch.where(adj > 0, scores, -9e15 * torch.ones_like(scores))
        
        coefficients = torch.nn.functional.softmax(scores, dim=1)
        
        x_ = torch.nn.functional.elu(torch.mm(coefficients, x_))
        
        return x_  


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
