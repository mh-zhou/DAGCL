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
    
class DynamicFULL_NN(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder, num_class, pretrain_model_load_path):
        super(DynamicFULL_NN, self).__init__()
        
        self.dims_encoder = dims_encoder
        self.num_class = num_class
        
        pretrain_state = torch.load(pretrain_model_load_path, map_location='cpu')
        
        self.AE = AE_NN(dim_input, dims_encoder, dims_decoder)
        self.AE.load_state_dict(pretrain_state)
        
        self.curriculum_weight = nn.Parameter(torch.ones(1))

    def forward(self, x, adj):
        h, x_hat = self.AE.forward(x, adj)
        h = h * self.curriculum_weight  

        self.z = torch.nn.functional.normalize(h, p=2, dim=1)
        
        return self.z, x_hat 

    def prediction(self, kappas, centers, normalize_constants, mixture_cofficences):

        cos_similarity = torch.mul(kappas, torch.mm(self.z, centers.T))
        
        pdf_component = torch.mul(normalize_constants, torch.exp(cos_similarity))
        
        p = torch.nn.functional.normalize(torch.mul(mixture_cofficences, pdf_component), p=1, dim=1)
        
        return p  


class ClusterAssignment(nn.Module):
    def __init__(self, cluster_number: int, embedding_dimension: int, alpha: float = 1.0,
                 cluster_centers: Optional[torch.Tensor] = None) -> None:
        super(ClusterAssignment, self).__init__()
        
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha  
        
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
