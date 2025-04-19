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


# class Cutout(object):
#     def __init__(self, length):
#         self.length = length

    # def __call__(self, img):
    #     h, w = img.size(1), img.size(2)
    #     mask = np.ones((h, w), np.float32)
    #     y = np.random.randint(h)
    #     x = np.random.randint(w)
    #
    #     y1 = np.clip(y - self.length // 2, 0, h)
    #     y2 = np.clip(y + self.length // 2, 0, h)
    #     x1 = np.clip(x - self.length // 2, 0, w)
    #     x2 = np.clip(x + self.length // 2, 0, w)
    #
    #     mask[y1: y2, x1: x2] = 0.
    #     mask = torch.from_numpy(mask)
    #     mask = mask.expand_as(img)
    #     img *= mask
    #     return img

#def read_data(train_data_dir, test_data_dir):
#    '''parses data in given train and test data directories

#    assumes:
#    - the data in the input directories are .json files with 
#        keys 'users' and 'user_data'
#    - the set of train set users is the same as the set of test set users

#    Return:
#        clients: list of non-unique client ids
#        groups: list of group ids; empty list if none found
#        train_data: dictionary of train data
#        test_data: dictionary of test data
#    '''
#    clients = []
#    groups = []
#    train_data = {}
#    test_data = {}

#    train_files = os.listdir(train_data_dir)
#    train_files = [f for f in train_files if f.endswith('.json')]
#    for f in train_files:
#        file_path = os.path.join(train_data_dir, f)
#        with open(file_path, 'r') as inf:
#            cdata = json.load(inf)
#        clients.extend(cdata['users'])
#        if 'hierarchies' in cdata:
#            groups.extend(cdata['hierarchies'])
#        train_data.update(cdata['user_data'])

#    test_files = os.listdir(test_data_dir)
#    test_files = [f for f in test_files if f.endswith('.json')]
#    for f in test_files:
#        file_path = os.path.join(test_data_dir, f)
#        with open(file_path, 'r') as inf:
#            cdata = json.load(inf)
#        test_data.update(cdata['user_data'])

#    clients = sorted(cdata['users'])

#    return clients, groups, train_data, test_data


#def batch_data(data, batch_size):
#    '''
#    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
#    returns x, y, which are both numpy array of length: batch_size
#    '''
#    data_x = data['x']
#    data_y = data['y']

#    # randomly shuffle data
#    np.random.seed(100)
#    rng_state = np.random.get_state()
#    np.random.shuffle(data_x)
#    np.random.set_state(rng_state)
#    np.random.shuffle(data_y)

#    # loop through mini-batches
#    batch_data = list()
#    for i in range(0, len(data_x), batch_size):
#        batched_x = data_x[i:i + batch_size]
#        batched_y = data_y[i:i + batch_size]
#        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
#        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
#        batch_data.append((batched_x, batched_y))
#    return batch_data


# def load_partition_data_mnist_by_device_id(batch_size,
#                                            device_id,
#                                            train_path="MNIST_mobile",
#                                            test_path="MNIST_mobile"):
#     train_path += '/' + device_id + '/' + 'train'
#     test_path += '/' + device_id + '/' + 'test'
#     return load_partition_data_mnist(batch_size, train_path, test_path)


def partition_data_dataset(X_train,y_train, n_nets, alpha):
    

    min_size = 0
    K = 10
    N = y_train.shape[0]
    # 追加一个额外的比例作为公共服务器训练数据
    percent = 10
    n_nets = n_nets + percent
    logging.info("N = " + str(N))
    net_dataidx_map = {}
    public_ditaidx_map = {}

    while min_size < 10:
        #print(min_size)
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset

        #n_nets表示模型的数量
        for k in range(K):#K个类别
            #print("K",k)
            idx_k = np.where(y_train == k)[0]#label为k的样本的index
            #print("idx_k",idx_k)
            #print(len(idx_k))
            np.random.seed(k)#设置随机种子，希望train和test的划分一样
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets)) #返回一个比例，应该是每个net的数据的比例，例如[0.1,0.1,0.2,0.2,0.4]
            #if k<3:
            #    print(k,proportions) #此处能确保，来任意一个类别，给的proportion都一样 但是为什么叠加起来就不一样呢？

            np.random.shuffle(idx_k)
            
            #print("proportions1",proportions)
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            #print("proportions2",proportions)
            proportions = proportions / proportions.sum()
            #print("proportions3",proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            #print("proportions4",proportions)#此处的proportions应该都不改变
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])


    for j in range(n_nets - percent):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    for i in range(percent):
        public_ditaidx_map[i] = idx_batch[n_nets - 1 - i]
    return net_dataidx_map, public_ditaidx_map


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        test_total_num= n_test
        idxs = np.random.permutation(total_num)
        idxs_test= np.random.permutation(test_total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        batch_idxs_test = np.array_split(idxs_test, n_nets)
        net_dataidx_map_train = {i: batch_idxs[i] for i in range(n_nets)}
        net_dataidx_map_test={i: batch_idxs_test[i] for i in range(n_nets)}

    elif partition == "hetero":#在此处分割数据
        net_dataidx_map_train, public_dataidx_map_train=partition_data_dataset(X_train,y_train,n_nets,alpha)
        net_dataidx_map_test, public_dataidx_map_test =partition_data_dataset(X_test,y_test,n_nets,alpha)


        #print(net_dataidx_map_test[0])
        #print(type(net_dataidx_map_test[0]))#表示了第0个客户端的测试数据 但是如何训练和测试的类别分布一样呢？
        #print(np.array(net_dataidx_map_test[0]).shape)
        
        
        #train_len,test_len=[],[]
        #for i in range(5):
        #    train_len.append(len(net_dataidx_map_train[i]) if i in net_dataidx_map_train.keys() else 0)
        #    test_len.append(len(net_dataidx_map_test[i]) if i in net_dataidx_map_test.keys() else 0)
        #print(train_len)
        #print(test_len)
        #while True:
        #    pass
    else:
        raise Exception("partition args error")

    return X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, public_dataidx_map_train, public_dataidx_map_test



def expand_img(img):
    return img.expand((3, 28, 28))


def _data_transforms_mnist():
    train_transform = transforms.Compose([
        transforms.Lambda(expand_img),
        transforms.Normalize([0.1307,0.1307,0.1307],[0.3081,0.3081,0.3081]),
    ])

    valid_transform = transforms.Compose([
        transforms.Lambda(expand_img),
        transforms.Normalize([0.1307,0.1307,0.1307],[0.3081,0.3081,0.3081]),
    ])

    return train_transform, valid_transform


class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        mnist_dataobj = torchvision.datasets.MNIST(root=self.root, train=self.train, transform=self.transform, download=self.download)
        data = None
        target = None
        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = mnist_dataobj.train_data
            data = mnist_dataobj.data
            target = np.array(mnist_dataobj.targets)
        else:
            data = mnist_dataobj.data
            target = np.array(mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
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
    return get_dataloader_MNIST(datadir, train_bs, test_bs, dataidxs_train,dataidxs_test)


def get_dataloader_MNIST(datadir, train_bs, test_bs, dataidxs_train=None,dataidxs_test=None):
    dl_obj = MNIST_truncated

    transform_train, transform_test = _data_transforms_mnist()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=True,num_workers=4)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True,num_workers=4)

    return train_dl, test_dl

def load_mnist_data(datadir):
    train_transform, test_transform = _data_transforms_mnist()
    train_data=MNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    test_data = MNIST_truncated(datadir, train=False, download=True, transform=test_transform)

    
    X_train, y_train = train_data.data, np.array(train_data.target)
    X_test, y_test = test_data.data, np.array(test_data.target)

    return (X_train, y_train, X_test, y_test)

def load_partition_data_mnist(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, public_dataidx_map_train, public_dataidx_map_test = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)


    class_num_train = len(np.unique(y_train))
    class_num_test = len(np.unique(y_test))
    train_data_num = sum([len(net_dataidx_map_train[r]) for r in range(client_number)])
    test_data_num = sum([len(net_dataidx_map_test[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))

    data_local_num_dict_train = dict()
    data_local_num_dict_test = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    # 公共数据的训练和测试字典
    public_train_data_local_dict = dict()
    public_test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs_train = net_dataidx_map_train[client_idx]
        dataidxs_test =  net_dataidx_map_test[client_idx]


        local_data_num_train = len(dataidxs_train)
        local_data_num_test = len(dataidxs_test)


        data_local_num_dict_train[client_idx] = local_data_num_train
        data_local_num_dict_test[client_idx] = local_data_num_test

        logging.info("client_idx = %d, train_local_sample_number = %d" % (client_idx, local_data_num_train))
        logging.info("client_idx = %d, test_local_sample_number = %d" % (client_idx, local_data_num_test))

        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs_train,dataidxs_test)


        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    # 分配公共数据的字典
    public_data_num_train = 0
    public_data_num_test = 0
    for i in range(len(public_dataidx_map_train)):
        public_dataidxs_train = public_dataidx_map_train[i]
        public_dataidxs_test =  public_dataidx_map_test[i]

        public_data_num_train = len(public_dataidxs_train) + public_data_num_train
        public_data_num_test = len(public_dataidxs_test) + public_data_num_test

        public_train_data, public_test_data = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 public_dataidxs_train,public_dataidxs_test)

        public_train_data_local_dict[i] = public_train_data
        public_test_data_local_dict[i] = public_test_data

    logging.info("train_public_sample_number = %d" % public_data_num_train)
    logging.info("test_public_sample_number = %d" % public_data_num_test)

        
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict_train, data_local_num_dict_test,train_data_local_dict, test_data_local_dict, class_num_train,class_num_test, public_train_data_local_dict, public_test_data_local_dict



