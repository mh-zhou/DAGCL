from layer import GAT_Layer, DynamicGAT_Layer, ZINBLoss, MeanAct, DispAct
import torch
from utils import pdf_norm
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional

class AE_GAT(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder):
        super(DynamicGAT_Layer, self).__init__()
        self.dims_en = [dim_input] + dims_encoder
        self.dims_de = dims_decoder + [dim_input]

        self.num_layer = len(self.dims_en)-1
        self.Encoder = torch.nn.ModuleList()
        self.Decoder = torch.nn.ModuleList()
        for index in range(self.num_layer):
            self.Encoder.append(DynamicGAT_Layer(self.dims_en[index], self.dims_en[index+1]))
            self.Decoder.append(DynamicGAT_Layer(self.dims_de[index], self.dims_de[index+1]))

    def forward(self, x, adj):
        for index in range(self.num_layer):
            x = self.Encoder[index].forward(x, adj)
        h = x
        for index in range(self.num_layer):
            x = self.Decoder[index].forward(x, adj)
        x_hat = x      
        return h, x_hat

class AE_NN(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder):
        super(AE_NN, self).__init__()
        self.dims_en = [dim_input] + dims_encoder
        self.dims_de = dims_decoder + [dim_input]

        self.num_layer = len(self.dims_en)-1
        self.Encoder = torch.nn.ModuleList()
        self.Decoder = torch.nn.ModuleList()
        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        for index in range(self.num_layer):
            self.Encoder.append(torch.nn.Linear(self.dims_en[index], self.dims_en[index+1]))
            self.Decoder.append(torch.nn.Linear(self.dims_de[index], self.dims_de[index+1]))

    def forward(self, x, adj):
        for index in range(self.num_layer):
            x = self.Encoder[index].forward(x)
        h = x
        for index in range(self.num_layer):
            x = self.Decoder[index].forward(x)
        x_hat = x      
        return h, x_hat

class FULL(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder, num_class, pretrain_model_load_path):
        super(FULL, self).__init__()
        self.dims_encoder = dims_encoder
        self.num_class = num_class

        self.AE = AE_GAT(dim_input, dims_encoder, dims_decoder)
        self.AE.load_state_dict(torch.load(pretrain_model_load_path, map_location='cpu'))
    
    def forward(self, x, adj):
        h, x_hat = self.AE.forward(x, adj)
        self.z = torch.nn.functional.normalize(h, p=2, dim=1)
        return self.z, x_hat

    def prediction(self, kappas, centers, normalize_constants, mixture_cofficences):
        cos_similarity = torch.mul(kappas, torch.mm(self.z, centers.T))
        pdf_component = torch.mul(normalize_constants, torch.exp(cos_similarity))
        p = torch.nn.functional.normalize(torch.mul(mixture_cofficences, pdf_component), p=1, dim=1)
        return p

class FULL_NN(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder, num_class, pretrain_model_load_path):
        super(FULL_NN, self).__init__()
        self.dims_encoder = dims_encoder
        self.num_class = num_class

        self.AE = AE_NN(dim_input, dims_encoder, dims_decoder)
        self.AE.load_state_dict(torch.load(pretrain_model_load_path, map_location='cpu'))
  
    def forward(self, x, adj):
        h, x_hat = self.AE.forward(x, adj)
        self.z = torch.nn.functional.normalize(h, p=2, dim=1)
        return self.z, x_hat

    def prediction(self, kappas, centers, normalize_constants, mixture_cofficences):
        cos_similarity = torch.mul(kappas, torch.mm(self.z, centers.T))
        pdf_component = torch.mul(normalize_constants, torch.exp(cos_similarity))
        p = torch.nn.functional.normalize(torch.mul(mixture_cofficences, pdf_component), p=1, dim=1)
        return p
    
#课程学习与动态调整复杂度
class DynamicFULL_NN(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder, num_class, pretrain_model_load_path):
        super(DynamicFULL_NN, self).__init__()
        
        self.dims_encoder = dims_encoder
        self.num_class = num_class
        
        # 先加载预训练模型的权重
        pretrain_state = torch.load(pretrain_model_load_path, map_location='cpu')
        
        # 创建与预训练模型相同维度的AE
        self.AE = AE_NN(dim_input, dims_encoder, dims_decoder)
        self.AE.load_state_dict(pretrain_state)
        
        # 定义课程学习中的权重参数
        self.curriculum_weight = nn.Parameter(torch.ones(1))

    def forward(self, x, adj):
        # 通过预训练的自动编码器得到节点表示
        h, x_hat = self.AE.forward(x, adj)
        
        # 应用课程学习的权重，逐渐增加复杂度
        h = h * self.curriculum_weight  
        
        # 对得到的表示进行归一化
        self.z = torch.nn.functional.normalize(h, p=2, dim=1)
        
        return self.z, x_hat  # 返回归一化的节点表示和重构的输入

    def prediction(self, kappas, centers, normalize_constants, mixture_cofficences):
        # 计算节点与中心的余弦相似度
        cos_similarity = torch.mul(kappas, torch.mm(self.z, centers.T))
        
        # 计算PDF组件
        pdf_component = torch.mul(normalize_constants, torch.exp(cos_similarity))
        
        # 计算最终的预测结果，并进行归一化
        p = torch.nn.functional.normalize(torch.mul(mixture_cofficences, pdf_component), p=1, dim=1)
        
        return p  # 返回归一化的预测分布


#结合课程学习与聚类分配的动态优化
class ClusterAssignment(nn.Module):
    def __init__(self, cluster_number: int, embedding_dimension: int, alpha: float = 1.0,
                 cluster_centers: Optional[torch.Tensor] = None) -> None:
        super(ClusterAssignment, self).__init__()
        
        # 初始化聚类中心数目和嵌入空间的维度
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha  # 调整聚类分配的复杂度
        
        # 如果没有提供聚类中心，则初始化为零并进行Xavier均匀初始化
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        
        # 聚类中心作为参数
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # 计算每个节点与聚类中心的距离
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        
        # 计算分配的权重
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        
        # 返回归一化的分配概率
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
