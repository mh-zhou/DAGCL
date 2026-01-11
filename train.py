# -*- encoding: utf-8 -*-

import os
import argparse
import random
from loguru import logger
import numpy as np
import pickle
import torch
from sklearn.cluster import KMeans
from torchmetrics.functional import pairwise_cosine_similarity

from model import AE_GAT, FULL, AE_NN, FULL_NN, DynamicFULL_NN, ClusterAssignment
from utils import pdf_norm, estimate_kappa, evaluation, visual, target_distribution, get_laplace_matrix
import torch.nn as nn
import warnings
import torch.nn.functional as F
import scanpy as sc
from preprocess import *
import h5py

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
from time import time

warnings.filterwarnings('ignore')

def sinkhorn(pred, lambdas, row, col):
    num_node = pred.shape[0]
    num_class = pred.shape[1]
    p = np.power(pred, lambdas)
    
    u = np.ones(num_node)
    v = np.ones(num_class)

    for index in range(1000):
        u = row * np.power(np.dot(p, v), -1)
        u[np.isinf(u)] = -9e-15
        v = col * np.power(np.dot(u, p), -1)
        v[np.isinf(v)] = -9e-15
    u = row * np.power(np.dot(p, v), -1)
    target = np.dot(np.dot(np.diag(u), p), np.diag(v))
    return target

label_dataset_1 = ['Xiaoping_mouse_bladder_cell','Junyue_worm_neuron_cell',
                   'Grace_CITE_CBMC_counts_top2000','Sonya_HumanLiver_counts_top5000']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--dataname', default='Adam', type=str)
    parser.add_argument('--dataname', default='Maayan_Mouse_Pancreas_cell_1', type=str)
    parser.add_argument('--num_class', default=10, type=int, help='number of classes')
    parser.add_argument('--gpu', default=0, type=int)

    embedding_num = 16
    parser.add_argument('--dims_encoder', default=[256, embedding_num], type=list)
    parser.add_argument('--dims_decoder', default=[embedding_num, 256], type=list)

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lambdas', default=5, type=float)
    parser.add_argument('--balancer', default=0.5, type=float)
    
    parser.add_argument('--curriculum_alpha', default=0.1, type=float)
    parser.add_argument('--complexity_warmup', default=0.3, type=float)

    parser.add_argument('--factor_ort', default=1, type=float)
    parser.add_argument('--factor_KL', default=0.5, type=float)
    parser.add_argument('--factor_corvar', default=0.05, type=float)

    parser.add_argument('--pretrain_model_save_path', default='pkl', type=str)
    parser.add_argument('--pretrain_centers_save_path', default='pkl', type=str)
    parser.add_argument('--pretrain_pseudo_labels_save_path', default='pkl', type=str)
    parser.add_argument('--pretrain_model_load_path', default='pkl', type=str)
    parser.add_argument('--pretrain_centers_load_path', default='pkl', type=str)
    parser.add_argument('--pretrain_pseudo_labels_load_path', default='pkl', type=str)

    parser.add_argument('--foldername', default='MAIN_modified', type=str)
    parser.add_argument('--noramlize_flag', default=False, type=bool)

    args = parser.parse_args()
    args.pretrain_model_save_path = 'DAGCL/result/{}/{}_model.pkl'.format(args.foldername, args.dataname)
    args.pretrain_centers_save_path = 'DAGCL/result/{}/{}_centers.pkl'.format(args.foldername, args.dataname)
    args.pretrain_pseudo_labels_save_path = 'DAGCL/result/{}/{}_pseudo_labels.pkl'.format(args.foldername, args.dataname)
    args.pretrain_model_load_path = 'DAGCL/result/{}/{}_model.pkl'.format(args.foldername, args.dataname)
    args.pretrain_centers_load_path = 'DAGCL/result/{}/{}_centers.pkl'.format(args.foldername, args.dataname)
    args.pretrain_pseudo_labels_load_path = 'DAGCL/result/{}/{}_pseudo_labels.pkl'.format(args.foldername, args.dataname)
    
    if os.path.isdir('DAGCL/result/{}/'.format(args.foldername)) == False:
        os.makedirs('DAGCL/result/{}/'.format(args.foldername))
    if os.path.isdir('DAGCL/log/{}/'.format(args.foldername)) == False:
        os.makedirs('DAGCL/log/{}/'.format(args.foldername))

    if args.dataname == 'Maayan_Mouse_Pancreas_cell_1':
        args.learning_rate = 1e-3
        args.weight_decay = 5e-3
        args.balancer = 0.55
        args.factor_ort = 0.65
        args.factor_KL = 0.12
        args.factor_corvar = 0.17
        args.factor_construct = 0.23
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Maayan_Mouse_Pancreas_cell_2':
        args.learning_rate = 1e-3
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Maayan_Human_Pancreas_cell_2':
        args.learning_rate = 5e-3
        args.weight_decay = 1e-3
        args.balancer = 1
        args.factor_ort = 0.35
        args.factor_KL = 0.02
        args.factor_corvar = 0.01
        args.factor_construct = 0.0
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Maayan_Human_Pancreas_cell_1':
        args.learning_rate = 1e-3
        args.weight_decay = 5e-4
        args.balancer = 0.55
        args.factor_ort = 0.65
        args.factor_KL = 0.17
        args.factor_corvar = 0.16
        args.factor_construct = 0.259
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Meuro_human_Pancreas_cell':
        args.learning_rate = 5e-4
        args.weight_decay = 5e-3
        args.balancer = 0.42        
        args.factor_ort = 0.87
        args.factor_KL = 0.45
        args.factor_corvar = 0.2
        args.factor_construct = 0.93
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
    if args.dataname == 'Xiaoping_mouse_bladder_cell':
        args.learning_rate = 5e-4
        args.weight_decay = 5e-4
        args.balancer = 0.4
        args.factor_ort = 0.3
        args.factor_KL = 0.4
        args.factor_corvar = 0.1
        args.factor_construct = 0.3
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Maayan_Human_Pancreas_cell_3':
        args.learning_rate = 1e-3
        args.weight_decay = 5e-3
        args.balancer = 0.55
        args.factor_ort = 0.65
        args.factor_KL = 0.12
        args.factor_corvar = 0.17
        args.factor_construct = 0.23
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Junyue_worm_neuron_cell':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-3
        args.balancer = 0.7
        args.factor_ort = 0.95
        args.factor_KL = 0.48
        args.factor_corvar = 0.12
        args.factor_construct = 0.6
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Grace_CITE_CBMC_counts_top2000':
        args.learning_rate = 3e-3 #修改前1e-3
        args.weight_decay = 5e-4
        args.balancer = 0.05
        args.factor_ort = 0.71
        args.factor_KL = 0.12
        args.factor_corvar = 0.6
        args.factor_construct = 0.06
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Sonya_HumanLiver_counts_top5000':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.21
        args.factor_ort = 0.32
        args.factor_KL = 0.32
        args.factor_corvar = 0.15
        args.factor_construct = 0.3
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Quake_Smart-seq2_Limb_Muscle':
        # 先用一组中规中矩的参数 
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Quake_Smart-seq2_Diaphragm':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Adam':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Quake_Smart-seq2_Heart':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Quake_Smart-seq2_Lung':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Quake_Smart-seq2_Trachea':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Quake_10x_Bladder':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Quake_10x_Limb_Muscle':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Quake_10x_Spleen':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Quake_10x_Trachea':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Bach':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Chen':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Klein':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Muraro':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Plasschaert':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Pollen':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Romanov':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Tosches_turtle':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Wang_Lung':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5
    if args.dataname == 'Young':
        args.learning_rate = 5e-4
        args.weight_decay = 1e-4
        args.balancer = 0.75
        args.factor_ort = 0.73
        args.factor_KL = 0.63
        args.factor_corvar = 0.01
        args.factor_construct = 0.02
        args.alpha_pre = 0.58
        args.eta = 1e-5
        args.alpha = 0.5
        args.beta = 0.5



    logger.add('DAGCL/log/{}/{}.log'.format(args.foldername, args.dataname), rotation="500 MB", level="INFO")
    logger.info(args)
    
    torch.cuda.set_device(args.gpu)
    
    datapath = os.path.join('C:/Users/mh/Desktop/code/OURS/datasets', args.dataname) 
    
    if args.dataname == 'Meuro_human_Pancreas_cell':
        x, y = prepro(datapath+'.h5')
    elif args.dataname == 'Quake_Smart-seq2_Limb_Muscle':
        # 新数据集：也使用 prepro 读取 exprs + obs["cell_type1"]
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Quake_Smart-seq2_Diaphragm':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Quake_Smart-seq2_Heart':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Quake_Smart-seq2_Lung':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Quake_Smart-seq2_Trachea':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Quake_10x_Bladder':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Quake_10x_Limb_Muscle':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Quake_10x_Spleen':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Quake_10x_Trachea':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Bach':
        x, y = prepro(datapath + '.h5')  
    elif args.dataname == 'Chen':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Klein':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Muraro':
        x, y = prepro(datapath + '.h5')     
    elif args.dataname == 'Adam':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Plasschaert':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Pollen':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Romanov':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Plasschaert':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Pollen':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Romanov':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Tosches_turtle':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Wang_Lung':
        x, y = prepro(datapath + '.h5')
    elif args.dataname == 'Young':
        x, y = prepro(datapath + '.h5')
    else:
        data = h5py.File(datapath+'.h5','r')
        x = data['X'][:]
        y = data['Y'][:]

    if args.dataname == 'Quake_Smart-seq2_Limb_Muscle':
        # y 此时还是 numpy 数组（来自 prepro）
        args.num_class = int(np.max(y) + 1)
        print("num_class for Quake_Smart-seq2_Limb_Muscle:", args.num_class)
    if args.dataname == 'Quake_Smart-seq2_Diaphragm':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Quake_Smart-seq2_Diaphragm:", args.num_class)
    if args.dataname == 'Quake_Smart-seq2_Heart':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Quake_Smart-seq2_Heart:", args.num_class)
    if args.dataname == 'Quake_Smart-seq2_Lung':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Quake_Smart-seq2_Lung:", args.num_class)
    if args.dataname == 'Quake_Smart-seq2_Trachea':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Quake_Smart-seq2_Trachea:", args.num_class)
    if args.dataname == 'Quake_10x_Bladder':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Quake_10x_Bladder:", args.num_class)
    if args.dataname == 'Quake_10x_Limb_Muscle':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Quake_10x_Limb_Muscle:", args.num_class)
    if args.dataname == 'Quake_10x_Spleen':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Quake_10x_Spleen:", args.num_class)
    if args.dataname == 'Quake_10x_Trachea':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Quake_10x_Trachea:", args.num_class)
    if args.dataname == 'Bach':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Bach:", args.num_class)
    if args.dataname == 'Chen':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Chen:", args.num_class)
    if args.dataname == 'Klein':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Klein:", args.num_class)    
    if args.dataname == 'Muraro':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Muraro:", args.num_class)
    if args.dataname == 'Adam':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Adam:", args.num_class)
    if args.dataname == 'Plasschaert':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Plasschaert:", args.num_class)
    if args.dataname == 'Pollen':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Pollen:", args.num_class)
    if args.dataname == 'Romanov':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Romanov:", args.num_class)
    if args.dataname == 'Tosches_turtle':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Tosches_turtle:", args.num_class)
    if args.dataname == 'Wang_Lung':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Wang_Lung:", args.num_class)
    if args.dataname == 'Young':
        args.num_class = int(np.max(y) + 1)
        print("num_class for Young:", args.num_class)

    if args.dataname == 'Meuro_human_Pancreas_cell':
        x = np.round(x).astype(int)
    if args.dataname in label_dataset_1:
        y = y-1
        
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    if args.dataname in ['amazon'] and args.noramlize_flag == True:
        x = torch.tensor(x, dtype=torch.float)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        print("normalize done!")
    x_ = torch.nn.functional.normalize(x, p=2, dim=1)   

    adj_self_loop = torch.mm(x_, x_.T)
    adj_f = np.abs(pairwise_cosine_similarity(x_, x_))
    adj_f = torch.mm(adj_f, adj_f.T)
    L_1 = get_laplace_matrix(adj_self_loop)
    L_2 = get_laplace_matrix(adj_f)

    #for seed in [3047]:
    for seed in [random.randint(0000, 10000) for  i in range(99)]:
    # for seed in [3047,3041,2021,2022,2050]:
        logger.info('Seed {}'.format(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        pro_start_time = time()

        ################################### PRE-TRAIN ########################################
        Model = AE_NN(dim_input=x.shape[1], dims_encoder=args.dims_encoder, dims_decoder=args.dims_decoder).cuda()
        optimizer = torch.optim.Adagrad(Model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
       
        acc_max = 0
        for epoch in range(1, args.epochs+1):
            h, x_hat = Model.forward(x.cuda(), adj_self_loop.cuda())
            z = torch.nn.functional.normalize(h, p=2, dim=0)
            adj_pred = torch.mm(z, z.T)

            loss_x = torch.nn.functional.mse_loss(x_hat, x.cuda())
            loss_corvariates = -torch.mm(torch.mm(z.T, (args.balancer * L_1.cuda() + (1-args.balancer) * L_2.cuda())),z).trace()/len(z.T)
            loss_ort = torch.nn.functional.mse_loss(torch.mm(z.T,z).view(-1).cuda(),torch.eye(len(z.T)).view(-1).cuda())
            loss = args.factor_construct * loss_x + args.factor_ort * loss_ort + args.factor_corvar * loss_corvariates

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                kmeans = KMeans(n_clusters=args.num_class, random_state=2021, n_init=20).fit(z.cpu().numpy())
                acc, nmi, ari, f1_macro = evaluation(y, kmeans.labels_)
                centers = torch.tensor(kmeans.cluster_centers_)

                logger.info('Epoch {}/{} Pre-Train ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}'.format(
                    epoch, args.epochs, acc, nmi, ari, f1_macro))
                logger.info('Epoch {}/{} | loss_corvariate: {:.6f} | loss_ort: {:.6f} | loss_total: {:.6f}'.format(
                    epoch, args.epochs, loss_corvariates.cpu().item(), loss_ort.cpu().item(), loss.cpu().item()))
                
                if acc > acc_max:
                    acc_max = acc
                    torch.save(Model.state_dict(), args.pretrain_model_save_path)
                    with open(args.pretrain_centers_save_path,'wb') as save1:
                        pickle.dump(centers, save1, protocol=pickle.HIGHEST_PROTOCOL)
                    pseudo_labels = torch.LongTensor(kmeans.labels_)
                    with open(args.pretrain_pseudo_labels_save_path,'wb') as save2:
                        pickle.dump(pseudo_labels, save2, protocol=pickle.HIGHEST_PROTOCOL)
        
        pro_time = time() - pro_start_time
        train_start_time = time()
    
        ####################################### TRAIN ########################################
        Model = DynamicFULL_NN(dim_input=x.shape[1], dims_encoder=args.dims_encoder,
                            dims_decoder=args.dims_decoder, num_class=args.num_class,
                            pretrain_model_load_path=args.pretrain_model_load_path).cuda()
        
        optimizer = torch.optim.AdamW(Model.parameters(), lr=args.learning_rate)
        
        with open(args.pretrain_centers_load_path,'rb') as load1:
            centers = pickle.load(load1).cuda()
        with open(args.pretrain_pseudo_labels_load_path,'rb') as load2:
            pseudo_labels = pickle.load(load2).cuda()

        acc_max, nmi_max, ari_max, f1_macro_max = 0, 0, 0, 0
        for epoch in range(1, args.epochs+1):
            z, x_hat = Model.forward(x.cuda(), adj_self_loop.cuda())
            z = torch.nn.functional.normalize(z, p=2, dim=0)
            centers = centers.detach()
            adj_pred = torch.mm(z, z.T)
            
            # 计算复杂度权重
            complexity_weight = min(1.0, epoch / (args.epochs * args.complexity_warmup))
            
            loss_x = torch.nn.functional.mse_loss(x_hat, x.cuda())
            loss_corvariates = -torch.mm(torch.mm(z.T, (args.balancer * L_1.cuda() + (1-args.balancer) * L_2.cuda())),z).trace()/len(z.T)
            loss_ort = torch.nn.functional.mse_loss(torch.mm(z.T,z).view(-1).cuda(),torch.eye(len(z.T)).view(-1).cuda())
            
            class_assign_model = ClusterAssignment(args.num_class, len(z.T), 1, centers).cuda()
            temp_class = class_assign_model(z.cuda())
            
            if epoch == 1 or epoch % 10 == 0:
                p_distribution = torch.tensor(sinkhorn(
                    temp_class.cpu().detach().numpy(), 
                    args.lambdas * complexity_weight,
                    torch.ones(x.shape[0]).numpy(),
                    torch.tensor([torch.sum(pseudo_labels==i) for i in range(args.num_class)]).numpy()
                )).float().cuda().detach()
                q_max, q_max_index = torch.max(p_distribution, dim=1)

            KL_loss_function = nn.KLDivLoss(reduction='sum')
            loss_KL = KL_loss_function(temp_class.cuda(), p_distribution.cuda())
            
      
            curriculum_loss = Model.curriculum_weight.pow(2).sum() * args.curriculum_alpha * (1 - complexity_weight)
            
            loss = args.factor_construct * loss_x + args.factor_ort * loss_ort + \
                   args.factor_corvar * loss_corvariates + args.factor_KL * loss_KL * complexity_weight + \
                   curriculum_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                kmeans = KMeans(n_clusters=args.num_class, random_state=2021, n_init=20).fit(z.cpu().numpy())
                acc, nmi, ari, f1_macro = evaluation(y, kmeans.labels_)
                if acc_max < acc:
                    acc_max, nmi_max, ari_max, f1_macro_max = acc, nmi, ari, f1_macro
                pseudo_labels = torch.LongTensor(kmeans.labels_)
                centers = torch.tensor(kmeans.cluster_centers_)
                
            
                #### logger
                logger.info('Epoch {}/{} | loss_corvariate: {:.6f} | loss_ort: {:.6f} | loss_KL: {:.6f} | loss_total: {:.6f}'.format(epoch, args.epochs, loss_corvariates.cpu().item(), loss_ort.cpu().item(), loss_KL.cpu().item(),  loss.cpu().item()))
                logger.info('Epoch {}/{} ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}'.format(epoch, args.epochs, acc, nmi, ari, f1_macro))
        logger.info('MAX ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}'.format(acc_max, nmi_max, ari_max, f1_macro_max))
        

        train_time = time() - train_start_time
        all_time = time() - pro_start_time
        logger.info('dataset_name:{},pro_time:{:.7f},train_time:{:.7f},all_time:{:.7f}'.format(args.dataname,pro_time,train_time,all_time))

# import pandas as pd
# cluster_labels = kmeans.labels_
# dataset_name = args.dataname
# results_df = pd.DataFrame(x.cpu().numpy())  # 每一行代表一个样本，每一列代表一个基因的表达量
# results_df['Cluster'] = cluster_labels 
# output_file_name = f'{dataset_name}_cluster_results.csv'
# output_file = os.path.join('/data/zhentianlong/OURS/datasets', output_file_name)
# header_description = """
# # Data Description:
# # Rows represent individual samples (cells).
# # Columns represent gene expression levels for different genes.
# # 'Cluster' column represents the cluster label assigned to each sample based on KMeans clustering.
# # Each number in the 'Cluster' column (e.g., 0, 1, 2, ...) represents the cluster ID.
# """
# with open(output_file, 'w') as f:
#     f.write(header_description) 
#     results_df.to_csv(f, index=False)  
# print(f"聚类结果已保存到 {output_file}")
