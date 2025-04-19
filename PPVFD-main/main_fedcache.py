import logging
import os
import sys
import numpy as np
import argparse
from sys import argv
import torch
import random

from fashionmnist_data_loader import load_partition_data_FashionMNIST
from FedMD_API import FedMD_standalone_API
from PPVFedMD_API import PPVFedMD_standalone_API
from PPVFedCache_API import PPVFedCache_standalone_API
# from data_util.FMNIST.fashionmnist_data_loader import load_partition_data_FashionMNIST
from data_util.MNIST.data_loader import load_partition_data_mnist
from data_util.SVHN.SVHN import load_partition_data_svhn
from data_util.cifar10.data_loader import load_partition_data_cifar10
from data_util.cinic10.data_loader import load_partition_data_cinic10
from resnet_client import resnet20, resnet16, resnet8
from FedCache import FedCache_standalone_API
from DSFLAPI import DSFL_standalone_API
from PPVFD import PPVFD_standalone_API
from FD import FD_standalone_API
from PPVDSFLAPI import PPVDSFL_standalone_API


def add_args(parser):
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers hetero/homo')
    parser.add_argument('--model_setting', type=str, default='hetero', metavar='N',
                        help='how to set on-device models on clients hetero/homo')
    parser.add_argument('--wd', type=float, default=5e-4, 
                        help='weight decay parameter;')
    parser.add_argument('--comm_round', type=int, default=200,
                        help='how many round of communications we shoud use (default: 1000)')
    parser.add_argument('--alpha', default=5, type=float,
                        help='Input the relative weight: default (1.5)')
    parser.add_argument('--sel', type=int, default=1, metavar='EP',
                        help='one out of every how many clients is selected to conduct testing  (default: 1)')
    parser.add_argument('--interval', type=int, default=1, metavar='EP',
                        help='how many communication round intervals to conduct testing  (default: 1)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',#0.01
                        help='learning rate (default: 0.01)')

    parser.add_argument('--class_num', type=int, default=10,
                        help='class_num')

    parser.add_argument('--dataset', type=str, default='fashionmnist', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--cache_R', type=int, default=1000,
                        help='多于Rhow many other workers are associated with each workers in fedcache')
    parser.add_argument('--N_alignment', type=str, default=0,
                        help='公共数据集的数量')
    parser.add_argument('--prime_number',type=int,default=2**25-39,
                        help='伽罗域')
    parser.add_argument('--partition_alpha', type=float, default=1.0, metavar='PA',
                        help='partition alpha (default: 1.0)')


    parser.add_argument('--batch_num', type=str, default=20,
                        help='batches selected for clients')
    parser.add_argument('--partition_rate', type=str, default=0.00,
                        help='partition rate for public data 公共数据集合 0 for default')
    parser.add_argument('--R', type=int, default=60,
                        help='how many other workers are associated with each workers')
    parser.add_argument('--mal_rate',type=float,default=0.4,
                        help='恶意客户端比例')
    parser.add_argument('--client_number', type=int, default=100, metavar='NN',
                        help='number of worke`rs in a distributed cluster')

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--T', type=float, default=1.5,
                        help='distrillation temperature (default: 1.0)')
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='distrillation temperature (default: 1.0)')
    parser.add_argument('--public_T', type=float, default=1.5,
                        help='public distrillation temperature (default: 0.1)')


    parser.add_argument('--api', type=str, default='PPVFEDCACHEAPI',
                        help='API that is utilized')
    parser.add_argument('--attack_method_id', type=str, default=0,
                        help='0-9 for none,\n'
                             '1FDLAttack,\n'
                             '2PCFDLAttack,\n'
                             '3FDPLAttack,\n'
                             '4FixedAttacl,\n'
                             '5RandomAttacl,\n'
                             '6IPMAttack,\n'
                             '7ParameterFlipAttack,\n'
                             '8LabelFlipAttack,\n'
                             '9WitchAttack')

    args = parser.parse_args()
    args.client_number_per_round=args.client_number
    args.client_num_in_total=args.client_number
    return args

# def load_data(args, dataset_name):
#     if dataset_name == "fmnist":
#         data_loader = load_partition_data_FashionMNIST
#     elif dataset_name == "mnist":
#         data_loader = load_partition_data_mnist
#     elif dataset_name == "cifar10":
#         data_loader = load_partition_data_cifar10
#     elif dataset_name == "cinic10":
#         data_loader = load_partition_data_cinic10
#     else:
#         data_loader = load_partition_data_svhn
#     train_data_num, test_data_num, train_data_global, test_data_global, \
#     train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#     class_num_train, class_num_test, public_train_data_local_dict, public_test_data_local_dict = data_loader(
#         args.dataset, args.data_dir,
#         args.partition_method,
#         args.partition_alpha, args.client_number,
#         args.batch_size)
#     dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
#                train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict,
#                class_num_train, class_num_test, public_train_data_local_dict, public_test_data_local_dict]
#     return dataset

def load_data(args, dataset_name):
    if dataset_name=="fashionmnist" or "cifar10":
        data_loader = load_partition_data_FashionMNIST
        train_data_num, test_data_num, train_data_global, test_data_global,train_data_public, \
        train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num_train, class_num_test = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_number, args.batch_size,args.partition_rate)
        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,train_data_public,
                   train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num_train, class_num_test]
    return dataset
# 使用 data_loader 函数加载数据并进行划分。这些操作涉及训练数据数量、测试数据数量、全局训练数据、
# 全局测试数据、本地训练数据数量字典、本地测试数据数量字典、本地训练数据字典、本地测试数据字典、训练集类别数和测试集类别数等信息的获取和处理。

def create_client_model(args, n_classes,index):
    if args.model_setting=='hetero':
        if index%3==0:
            return resnet8(n_classes)
        elif index%3==1:
            return resnet16(n_classes)
        else:
            return resnet20(n_classes)
    elif args.model_setting=='homo':
        return resnet20(n_classes)
    else:
        raise Exception("model setting exception")

def create_client_models(args, n_classes):
    #client_model中存有每个客户端的models
    random.seed(1)
    client_models=[]
    for _ in range(args.client_number):
        client_models.append(create_client_model(args,n_classes,_))
    return client_models

def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs
    # 设置 PyTorch 的可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)
    set_seed(123)

    #以上内容完成了client_num个客户端数据集、模型的初始化
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,train_data_public,
     train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num_train, class_num_test] = dataset
    client_models=create_client_models(args,class_num_train)

    # 下述内容调用本文模型的API
    # 训练集数据数量 测试集数据数量 全局训练集提取器 全局测试集提取器 每个clients本例训练数据数量 每个client本地测试数据数量 每个客户端训练集的dataloader 每个客户端测试机的dataloader 训练集和测试集类的数目
    if args.api=="FDAPI":
        api = FD_standalone_API(client_models, train_data_local_num_dict, test_data_local_num_dict,
                                train_data_local_dict, test_data_local_dict, args, test_data_global)
        api.do_fd_stand_alone(client_models, train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict,
                              test_data_local_dict, args)
    elif args.api=="PPVFDAPI":
        api = PPVFD_standalone_API(client_models, train_data_local_num_dict, test_data_local_num_dict,train_data_local_dict, test_data_local_dict, args, test_data_global)
        api.do_ppvfd_stand_alone(client_models, train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, args)


    elif args.api=="FMDAPI":
        api = FedMD_standalone_API(client_models, train_data_local_num_dict, test_data_local_num_dict,
                                   train_data_local_dict, test_data_local_dict, args, test_data_global,
                                   train_data_public)
        api.do_fedMD_stand_alone(client_models, train_data_local_num_dict, test_data_local_num_dict,
                                 train_data_local_dict,
                                 test_data_local_dict, args)
    elif args.api=="PPVFMDAPI":
        api = PPVFedMD_standalone_API(client_models, train_data_local_num_dict, test_data_local_num_dict,
                                   train_data_local_dict, test_data_local_dict, args, test_data_global,
                                   train_data_public)
        api.do_ppvfedMD_stand_alone(client_models, train_data_local_num_dict, test_data_local_num_dict,
                                 train_data_local_dict,
                                 test_data_local_dict, args)
    elif args.api=="FEDCACHEAPI":
        api=FedCache_standalone_API(client_models,train_data_local_num_dict,test_data_local_num_dict, train_data_local_dict, test_data_local_dict, args,test_data_global)
        api.do_fedcache_stand_alone(client_models,train_data_local_num_dict, test_data_local_num_dict,train_data_local_dict, test_data_local_dict, args)
    elif args.api=="PPVFEDCACHEAPI":
        api=PPVFedCache_standalone_API(client_models,train_data_local_num_dict,test_data_local_num_dict, train_data_local_dict, test_data_local_dict, args,test_data_global)
        api.do_ppvfedcache_stand_alone(client_models,train_data_local_num_dict, test_data_local_num_dict,train_data_local_dict, test_data_local_dict, args)
    elif args.api=="DSFLAPI":
        api = DSFL_standalone_API(client_models, train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict,test_data_local_dict, args, test_data_global,train_data_public)
        api.do_dsfl_stand_alone(client_models, train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict,test_data_local_dict,args)
    elif args.api=="PPVDSFLAPI":
        api = PPVDSFL_standalone_API(client_models, train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict,test_data_local_dict, args, test_data_global,train_data_public)
        api.do_ppvdsfl_stand_alone(client_models, train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict,test_data_local_dict,args)


