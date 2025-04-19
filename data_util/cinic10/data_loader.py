import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL.Image import Image

from .datasets import ImageFolderTruncated

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


def _data_transforms_cinic10():
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    # Transformer for train set: random crops and horizontal flip
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Lambda(
                                              lambda x: F.pad(x.unsqueeze(0),
                                                              (4, 4, 4, 4),
                                                              mode='reflect').data.squeeze()),
                                          transforms.ToPILImage(),
                                          transforms.RandomCrop(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=cinic_mean,
                                                               std=cinic_std),
                                          ])

    # Transformer for test set
    valid_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Lambda(
                                              lambda x: F.pad(x.unsqueeze(0),
                                                              (4, 4, 4, 4),
                                                              mode='reflect').data.squeeze()),
                                          transforms.ToPILImage(),
                                          transforms.RandomCrop(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=cinic_mean,
                                                               std=cinic_std),
                                          ])
    return train_transform, valid_transform


class CINIC10_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # Load the CINIC-10 dataset using your file organization and naming scheme
        data = []  # Load your dataset images
        target = []  # Load your dataset labels

        if self.dataidxs is not None:
            data = [data[i] for i in self.dataidxs]
            target = [target[i] for i in self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)  # Convert to PIL image
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


def load_cinic10_data(datadir):
    _train_dir = datadir + str('/CINIC-10/train')
    logging.info("_train_dir = " + str(_train_dir))
    _test_dir = datadir + str('/CINIC-10/test')
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    trainset = ImageFolderTruncated(_train_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                                              transforms.Lambda(
                                                                                  lambda x: F.pad(x.unsqueeze(0),
                                                                                                  (4, 4, 4, 4),
                                                                                                  mode='reflect').data.squeeze()),
                                                                              transforms.ToPILImage(),
                                                                              transforms.RandomCrop(32),
                                                                              transforms.RandomHorizontalFlip(),
                                                                              transforms.ToTensor(),
                                                                              transforms.Normalize(mean=cinic_mean,
                                                                                                   std=cinic_std),
                                                                              ]))

    testset = ImageFolderTruncated(_test_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Lambda(
                                                                                lambda x: F.pad(x.unsqueeze(0),
                                                                                                (4, 4, 4, 4),
                                                                                                mode='reflect').data.squeeze()),
                                                                            transforms.ToPILImage(),
                                                                            transforms.RandomCrop(32),
                                                                            transforms.RandomHorizontalFlip(),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize(mean=cinic_mean,
                                                                                                 std=cinic_std),
                                                                            ]))
    X_train, y_train = trainset.imgs, trainset.targets
    X_test, y_test = testset.imgs, testset.targets
    return (X_train, y_train, X_test, y_test)


def partition_data_dataset(X_train, y_train, n_nets, alpha):
    min_size = 0
    K = 10
    N = y_train.shape[0]
    # 追加一个额外的客户作为公共服务器训练数据
    n_nets = n_nets + 1
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
    public_ditaidx_map[0] = idx_batch[n_nets - 1]
    return net_dataidx_map, public_ditaidx_map


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    X_train, y_train, X_test, y_test = load_cinic10_data(datadir)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    n_train = len(X_train)
    n_test = len(X_test)

    if partition == "homo":
        total_num = n_train
        test_total_num = n_test
        idxs = np.random.permutation(total_num)
        idxs_test = np.random.permutation(test_total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        batch_idxs_test = np.array_split(idxs_test, n_nets)
        net_dataidx_map_train = {i: batch_idxs[i] for i in range(n_nets)}
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_nets)}

    elif partition == "hetero":
        net_dataidx_map_train, public_dataidx_map_train = partition_data_dataset(X_train, y_train, n_nets, alpha)
        net_dataidx_map_test, public_dataidx_map_test = partition_data_dataset(X_test, y_test, n_nets, alpha)
    else:
        raise Exception("partition args error")

    return X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, public_dataidx_map_train, public_dataidx_map_test


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    return get_dataloader_cinic10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_cinic10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_cinic10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_cinic10()

    traindir = os.path.join(datadir, 'CINIC-10\\train')
    valdir = os.path.join(datadir, 'CINIC-10\\test')

    train_ds = dl_obj(traindir, dataidxs=dataidxs_train, transform=transform_train)
    test_ds = dl_obj(valdir, dataidxs=dataidxs_test, transform=transform_train)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_cinic10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_cinic10()

    traindir = os.path.join(datadir, 'CINIC-10\\train')
    valdir = os.path.join(datadir, 'CINIC-10\\test')

    train_ds = dl_obj(traindir, dataidxs=dataidxs_train, transform=transform_train)
    test_ds = dl_obj(valdir, dataidxs=dataidxs_test, transform=transform_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def load_partition_data_distributed_cinic10(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts,public_dataidx_map_train, public_dataidx_map_test = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(train_data_global)))
        test_data_num = len(test_data_global)
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                           dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        test_data_num = 0
        train_data_global = None
        test_data_global = None

    return train_data_num, test_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_cinic10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, public_dataidx_map_train, public_dataidx_map_test = partition_data(dataset,
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
        public_train_data, public_test_data = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                             public_dataidx_map_train[0], public_dataidx_map_test[0])

        public_train_data_local_dict[0] = public_train_data
        public_test_data_local_dict[0] = public_test_data
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict_train, data_local_num_dict_test, train_data_local_dict, test_data_local_dict, class_num_train, class_num_test, public_train_data_local_dict, public_test_data_local_dict
