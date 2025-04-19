import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import svhn_truncated  # 真正运行时采用

# from datasets import CIFAR10_truncated#测试本文件采用

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


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


def _data_transforms_svhn():
    CIFAR_MEAN = [0.4377, 0.4438, 0.4728]
    CIFAR_STD = [0.1201, 0.1231, 0.1052]

    train_transform = transforms.Compose([

        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_svhn_data(datadir):
    train_transform, test_transform = _data_transforms_svhn()

    svhn_train_ds = svhn_truncated(datadir, split="train", transform=train_transform, download=True)
    svhn_test_ds = svhn_truncated(datadir, split="test", transform=test_transform, download=True)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data_dataset(X_train, y_train, n_nets, alpha):
    min_size = 0
    K = 10
    N = y_train.shape[0]
    # 追加一个额外的比例作为公共服务器训练数据
    percent = 0
    n_nets = n_nets + percent
    logging.info("N = " + str(N))
    net_dataidx_map = {}
    public_ditaidx_map = {}
    while min_size < 10:
        # print(min_size)
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset

        # n_nets表示模型的数量
        for k in range(K):  # K个类别
            # print("K",k)
            idx_k = np.where(y_train == k)[0]  # label为k的样本的index
            # print("idx_k",idx_k)
            # print(len(idx_k))
            np.random.seed(k)  # 设置随机种子，希望train和test的划分一样
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))  # 返回一个比例，应该是每个net的数据的比例，例如[0.1,0.1,0.2,0.2,0.4]
            # if k<3:
            #    print(k,proportions) #此处能确保，来任意一个类别，给的proportion都一样 但是为什么叠加起来就不一样呢？

            np.random.shuffle(idx_k)

            # print("proportions1",proportions)
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            # print("proportions2",proportions)
            proportions = proportions / proportions.sum()
            # print("proportions3",proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # print("proportions4",proportions)#此处的proportions应该都不改变
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    for i in range(percent):
        public_ditaidx_map[i] = idx_batch[n_nets - 1 - i]
    return net_dataidx_map, public_ditaidx_map


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    if partition == "homo":
        public_dataidx_map_train = {}
        public_dataidx_map_test = {}
        total_num = n_train
        test_total_num = n_test
        idxs = np.random.permutation(total_num)
        idxs_test = np.random.permutation(test_total_num)

        # 追加一个额外的比例作为公共服务器训练数据
        percent = 1
        n_nets = n_nets + percent
        batch_idxs = np.array_split(idxs, n_nets)
        batch_idxs_test = np.array_split(idxs_test, n_nets)
        net_dataidx_map_train = {i: batch_idxs[i] for i in range(n_nets)}
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_nets)}
        for i in range(percent):
            public_dataidx_map_train[i] = batch_idxs[n_nets - i -1]
            public_dataidx_map_test[i] = batch_idxs_test[n_nets - i -1]

    elif partition == "hetero":  # 在此处分割数据
        net_dataidx_map_train, public_dataidx_map_train = partition_data_dataset(X_train, y_train, n_nets, alpha)
        net_dataidx_map_test, public_dataidx_map_test = partition_data_dataset(X_test, y_test, n_nets, alpha)

        # print(net_dataidx_map_test[0])
        # print(type(net_dataidx_map_test[0]))#表示了第0个客户端的测试数据 但是如何训练和测试的类别分布一样呢？
        # print(np.array(net_dataidx_map_test[0]).shape)

        # train_len,test_len=[],[]
        # for i in range(5):
        #    train_len.append(len(net_dataidx_map_train[i]) if i in net_dataidx_map_train.keys() else 0)
        #    test_len.append(len(net_dataidx_map_test[i]) if i in net_dataidx_map_test.keys() else 0)
        # print(train_len)
        # print(test_len)
        # while True:
        #    pass
    else:
        raise Exception("partition args error")

    return X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, public_dataidx_map_train, public_dataidx_map_test


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    return get_dataloader_svhn(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_svhn(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_svhn(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = svhn_truncated

    transform_train, transform_test = _data_transforms_svhn()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, split="train", transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, split="test", transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_svhn(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = svhn_truncated

    transform_train, transform_test = _data_transforms_svhn()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, split="train", transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, split="test", transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


# def load_partition_data_distributed_cifar10(process_id, dataset, data_dir, partition_method, partition_alpha,
#                                             client_number, batch_size):
#     X_train, y_train, X_test, y_test, net_dataidx_map_train, traindata_cls_counts, public_dataidx_map_train, public_dataidx_map_test  = partition_data(dataset,
#                                                                                                    data_dir,
#                                                                                                    partition_method,
#                                                                                                    client_number,
#                                                                                                    partition_alpha)
#     class_num = len(np.unique(y_train))
#     logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
#     train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
#
#     # get global test data
#     if process_id == 0:
#         train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
#         logging.info("train_dl_global number = " + str(len(train_data_global)))
#         logging.info("test_dl_global number = " + str(len(test_data_global)))
#         train_data_local = None
#         test_data_local = None
#         local_data_num = 0
#     else:
#         # get local dataset
#         dataidxs = net_dataidx_map[process_id - 1]
#         local_data_num = len(dataidxs)
#         logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
#         # training batch size = 64; algorithms batch size = 32
#         train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
#                                                            dataidxs)
#         logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
#             process_id, len(train_data_local), len(test_data_local)))
#         train_data_global = None
#         test_data_global = None
#     return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_svhn(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, public_dataidx_map_train, public_dataidx_map_test = partition_data(
        dataset,
        data_dir,
        partition_method,
        client_number,
        partition_alpha)
    class_num_train = len(np.unique(y_train))
    class_num_test = len(np.unique(y_test))
    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map_train[r]) for r in range(client_number)])
    test_data_num = sum([len(net_dataidx_map_test[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    # test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict_train = dict()
    data_local_num_dict_test = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    # 公共数据的训练和测试字典
    public_train_data_local_dict = dict()
    public_test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs_train = net_dataidx_map_train[client_idx]
        dataidxs_test = net_dataidx_map_test[client_idx]

        local_data_num_train = len(dataidxs_train)
        local_data_num_test = len(dataidxs_test)

        data_local_num_dict_train[client_idx] = local_data_num_train
        data_local_num_dict_test[client_idx] = local_data_num_test

        logging.info("client_idx = %d, train_local_sample_number = %d" % (client_idx, local_data_num_train))
        logging.info("client_idx = %d, test_local_sample_number = %d" % (client_idx, local_data_num_test))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                           dataidxs_train, dataidxs_test)

        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    # 分配公共数据的字典
    public_data_num_train = 0
    public_data_num_test = 0
    for i in range(len(public_dataidx_map_train)):
        public_dataidxs_train = public_dataidx_map_train[i]
        public_dataidxs_test = public_dataidx_map_test[i]

        public_data_num_train = len(public_dataidxs_train) + public_data_num_train
        public_data_num_test = len(public_dataidxs_test) + public_data_num_test

        public_train_data, public_test_data = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                             public_dataidxs_train, public_dataidxs_test)

        public_train_data_local_dict[i] = public_train_data
        public_test_data_local_dict[i] = public_test_data

    logging.info("train_public_sample_number = %d" % public_data_num_train)
    logging.info("test_public_sample_number = %d" % public_data_num_test)

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict_train, data_local_num_dict_test, train_data_local_dict, test_data_local_dict, class_num_train, class_num_test, public_train_data_local_dict, public_test_data_local_dict
