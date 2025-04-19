"""
Reference:
FedML: A Research Library and Benchmark for Federated Machine Learning
@misc{he2020fedml,
      title={FedML: A Research Library and Benchmark for Federated Machine Learning}, 
      author={Chaoyang He and Songze Li and Jinhyun So and Xiao Zeng and Mi Zhang and Hongyi Wang and Xiaoyang Wang and Praneeth Vepakomma and Abhishek Singh and Hang Qiu and Xinghua Zhu and Jianzong Wang and Li Shen and Peilin Zhao and Yan Kang and Yang Liu and Ramesh Raskar and Qiang Yang and Murali Annavaram and Salman Avestimehr},
      year={2020},
      eprint={2007.13518},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""

import json
import logging
import os

import numpy as np
import torch
from torchvision import transforms
import logging

import numpy as np
import torch.utils.data as data
from PIL import Image

import torchvision

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def load_partition_data_FashionMNIST_by_device_id(batch_size,
                                           device_id,
                                           train_path="FashionMNIST_mobile",
                                           test_path="FashionMNIST_mobile"):
    train_path += '/' + device_id + '/' + 'train'
    test_path += '/' + device_id + '/' + 'test'
    return load_partition_data_FashionMNIST(batch_size, train_path, test_path)


def partition_data_dataset(X_train,y_train, n_nets, alpha,partition_rate=0):
    min_size = 0   #每个客户端最小样本数量
    K = 10   #样本类别
    N = y_train.shape[0]       #测试集样本数
    # print("N = " + str(N))
    net_dataidx_map = {}
    indices = np.random.permutation(N)
    split_index = int(partition_rate * N)
    y_train_public_index=list(indices)[:split_index]
    # y_train_private, y_train_public = y_train[indices[:split_index]], y_train[indices[split_index:]]

    while min_size < 10:
        #print(min_size)
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):#K个类别
            idx_k = np.where(y_train == k)[0]
            # print("id_k",idx_k) #类别索引
            idx_k = np.setdiff1d(idx_k, y_train_public_index)
            np.random.seed(k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            np.random.shuffle(idx_k)
            proportions = np.array([p * (len(idx_j) < N*(1-partition_rate) / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    net_dataidx_map_train_public = y_train_public_index
    return net_dataidx_map,net_dataidx_map_train_public


#不同的划分方法将给定的数据集进行划分，并返回划分后的数据以及每个子数据集的索引映射
def partition_data(dataset, datadir, partition, n_nets, alpha, partition_rate=0):
    print("*********partition data***************")
    X_train, y_train, X_test, y_test = load_FashionMNIST_data(datadir)
    # 训练集和测试集的样本数量
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        test_total_num= n_test
        #训练集合的索引
        idxs = np.random.permutation(total_num)
        idxs_test= np.random.permutation(test_total_num)
        if partition_rate:
            batch_idxs = np.array_split(idxs[:int(total_num*(1-partition_rate))], n_nets)
        else:
            batch_idxs = np.array_split(idxs, n_nets)
        batch_idxs_test = np.array_split(idxs_test, n_nets)
        net_dataidx_map_train = {i: batch_idxs[i] for i in range(n_nets)}
        net_dataidx_map_train_public=idxs[int(total_num*(1 - partition_rate)):]
        net_dataidx_map_test={i: batch_idxs_test[i] for i in range(n_nets)}

    elif partition == "hetero":#在此处分割数据
        net_dataidx_map_train,net_dataidx_map_train_public=partition_data_dataset(X_train,y_train,n_nets,alpha,partition_rate)
        net_dataidx_map_test=partition_data_dataset(X_test,y_test,n_nets,alpha)[0]
    else:
        raise Exception("partition arg error")
    print("加载成功")
    #未划分的训练集、未划分训练的标签、测试集、测试集的标签、字典存储每个客户端私有训练集的索引、字典存储每个客户端测试集的索引、字典存储每个客户端共有训练集的索引
    return X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test,net_dataidx_map_train_public



#数据预处理方式
def _data_transforms_FashionMNIST():
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.expand((3, 28, 28))),
        transforms.Normalize([0.1307,0.1307,0.1307],[0.3081,0.3081,0.3081]),
    ])

    valid_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.expand((3, 28, 28))),
        transforms.Normalize([0.1307,0.1307,0.1307],[0.3081,0.3081,0.3081]),
    ])

    return train_transform, valid_transform


class FashionMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # print("download = " + str(self.download))
        FashionMNIST_dataobj = torchvision.datasets.FashionMNIST(root=self.root, train=self.train, transform=self.transform, download=self.download)
        data = None
        target = None
        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = FashionMNIST_dataobj.train_data
            data = FashionMNIST_dataobj.data
            target = np.array(FashionMNIST_dataobj.targets)
        else:
            data = FashionMNIST_dataobj.data
            target = np.array(FashionMNIST_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img=img.reshape(1,28,28).type(torch.float32)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.data)


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs_train=None,dataidxs_test=None):
    # 返回值是训练集和测试集的数据加载器
    return get_dataloader_FashionMNIST(datadir, train_bs, test_bs, dataidxs_train,dataidxs_test)


def get_dataloader_FashionMNIST(datadir, train_bs, test_bs, dataidxs_train=None,dataidxs_test=None):
    dl_obj = FashionMNIST_truncated

    transform_train, transform_test = _data_transforms_FashionMNIST()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl

def load_FashionMNIST_data(datadir):
    train_transform, test_transform = _data_transforms_FashionMNIST()
    train_data=FashionMNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    test_data = FashionMNIST_truncated(datadir, train=False, download=True, transform=test_transform)

    
    X_train, y_train = train_data.data, np.array(train_data.target)
    X_test, y_test = test_data.data, np.array(test_data.target)

    return (X_train, y_train, X_test, y_test)

def load_partition_data_FashionMNIST(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size,partition_rate):
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test,net_dataidx_map_train_public = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha,
                                                                                            partition_rate)
    # 训练集和测试集的类别数量，即不同的类别数目。
    #计算训练集和测试集划分后各个客户端的样本数量之和。
    class_num_train = len(np.unique(y_train))
    class_num_test = len(np.unique(y_test))
    train_data_num = sum([len(net_dataidx_map_train[r]) for r in range(client_number)])
    test_data_num = sum([len(net_dataidx_map_test[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    trian_data_public=get_dataloader(dataset,data_dir,batch_size,batch_size,dataidxs_train=net_dataidx_map_train_public)[0]
    # 获取数据加载器的大小
    # print("trian_data_public number = " + str(len(trian_data_public)))
    # print("trian_data_public number = " + str(    [batch_idx for batch_idx, (images, labels) in enumerate(trian_data_public)]))

    # print("train_dl_global number = " + str(len(train_data_global)))
    # print("test_dl_global number = " + str(len(test_data_global)))

    data_local_num_dict_train = dict()
    data_local_num_dict_test = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()


    for client_idx in range(client_number):
        #获取客户端client_idx的私有数据集
        dataidxs_train = net_dataidx_map_train[client_idx]
        dataidxs_test =  net_dataidx_map_test[client_idx]


        # 计算本地私有训练集和测试集的样本数量
        local_data_num_train = len(dataidxs_train)
        local_data_num_test = len(dataidxs_test)


        data_local_num_dict_train[client_idx] = local_data_num_train
        data_local_num_dict_test[client_idx] = local_data_num_test

        # print("client_idx = %d, train_local_sample_number = %d" % (client_idx, local_data_num_train))
        # print("client_idx = %d, test_local_sample_number = %d" % (client_idx, local_data_num_test))

        # 划分的关键之处
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs_train,dataidxs_test)
        # print("客户端{}训练集的{}".format(client_idx, train_data_local))
        # print("客户端{}测试集的{}".format(client_idx, test_data_local))
        # print("客户端{}训练集的shape{}".format(client_idx, np.array(train_data_local).shape))
        # print("客户端{}测试集的shape{}".format(client_idx, np.array(test_data_local).shape))

        # print("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
        # 将生成的本地数据加载器存储在字典中
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    # 这段代码的作用是根据客户端的数据索引，为每个客户端生成本地训练集和测试集的数据加载器，并记录本地数据的相
    # 关信息（样本数量等）。这些信息和加载器将在分布式训练中被使用，每个客户端都有自己的本地数据加载器和相关信息。

    return train_data_num, test_data_num, train_data_global, test_data_global,trian_data_public, \
           data_local_num_dict_train, data_local_num_dict_test,train_data_local_dict, test_data_local_dict, class_num_train,class_num_test
    #训练集数据数量、测试集数据数量、全局训练集提取器、全局测试集提取器、公共数据集合的loader、每个clients本例训练数据数量、每个client本地测试数据数量、每个客户端训练集的dataloader、每个客户端测试机的dataloader、训练集和测试集类的数目


