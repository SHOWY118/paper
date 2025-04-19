# from mpi4py import MPI
# from GKTServerTrainer import GKTServerTrainer
# from GKTClientTrainer import GKTClientTrainer
from torch.nn import functional as F
import copy
import torch
import os
import numpy as np
import wandb
import utils
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from heapq import heappop as pop
from heapq import heappush as push
import matplotlib.pyplot as plt

import copy
import queue

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from heapq import heappop as pop
from heapq import heappush as push
import copy
import queue


def tensor_cross_entropy(output: torch.Tensor, target: torch.Tensor):
    return -1.0 * (output.log() * target).mean()


np.random.seed(0)


def knowledge_avg(knowledge, weights):
    result = []
    for k_ in knowledge:
        result.append(knowledge_avg_single(k_, weights))
    return torch.Tensor(np.array(result)).cuda()


def knowledge_avg_single(knowledge, weights):
    result = torch.zeros_like(knowledge[0].knowledge)
    sum = 0
    for _k, _w in zip(knowledge, weights):
        result = result + _k.knowledge * _w
        sum = sum + _w
    result = result / sum
    return np.array(result.detach().cpu())

def predict(model,dataarray):
    model.eval()
    out= []
    bs = 8
    with torch.no_grad():
        for ind in range(0,len(dataarray),bs):
            data = dataarray[ind:(ind+bs)]
            logit = model(data)
            out.append(logit.cpu().numpy())
    out = np.concatenate(out)
    # (num_batches, batch_size, num_classes)
    # print("out_shape:{}".format(out.shape))
    return out

def generate_alignment_data(X, N_alignment=3000):
    index = np.random.choice(range(len(X)), N_alignment,replace=False).astype(int)

    if N_alignment == "all":
        alignment_data = {}
        alignment_data["idx"] = np.arange(len(X))
        alignment_data["X"] = X
        return alignment_data
    else:
        # X_alignment = np.array(X)[index]
        alignment_data = {}
        alignment_data["idx"] = index
        alignment_data["X"] = [X[i] for i in index]
        alignment_data["X"]=torch.stack(alignment_data["X"],dim=0)
        # torch.Size([5000, 3, 28, 28])
        return alignment_data


class DSFL_standalone_API:
    def __init__(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                 train_data_local_dict, test_data_local_dict, args, test_data_global,train_data_public):
        self.client_models = client_models
        self.test_data_global = test_data_global
        self.criterion_KL = utils.KL_Loss(args.temperature)
        self.criterion_CE = F.cross_entropy
        self.train_data_public=train_data_public
        self.global_logits_dict=dict()
        self.public_logits_dict=dict()


    def do_dsfl_stand_alone(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                          train_data_local_dict, test_data_local_dict,train_data_public,args):

        print("*********start initializing with DSFL***************")

        #初始化公共数据集合的logits
        for i in range(len(train_data_public)*args.batch_size):
            self.public_logits_dict[i]=torch.Tensor(np.array([1.0/args.class_num for _ in range(args.class_num)]))

        # 为每个客户端遍历其本地训练数据加载器，逐批次地将图像和标签存储在列表train_data_local_dict_seq[client_index]中。
        train_data_local_dict_seq = {}
        for client_index in range(args.client_number):
            train_data_local_dict_seq[client_index] = []
            for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
                train_data_local_dict_seq[client_index].append((images, labels))

        train_data_public_list_seq=[]
        for batch_idx, (images, labels) in enumerate(train_data_public):
            for image in images:
                train_data_public_list_seq.append(image)

        print("*********start training with DSFL***************")
        for global_epoch in range(args.comm_round):  # 表示进行多少次客户端与服务器之间的交互，设置为无限大则一直不停
            print("开始训练第" + str(global_epoch) + "轮次训练")
            metrics_all = {'test_loss': [], 'test_accTop1': [], 'test_accTop5': [], 'f1': []}

            '''开始步骤一：在local本地进行模型训练'''
            for client_index, client_model in enumerate(self.client_models):
                print("开始训练第" + str(client_index) + "个客户端")
                #若想加入epoch，则效仿fedmd在此处设立一个train_one_model函数
                client_model = self.client_models[client_index]
                # client_model=client_model.cuda()
                client_model.train()
                optim = torch.optim.SGD(client_model.parameters(), lr=args.lr, momentum=0.9,
                                        weight_decay=args.wd)
                for batch_idx, (images, labels) in enumerate(train_data_local_dict_seq[client_index]):
                    labels = torch.tensor(labels, dtype=torch.long)
                    images, labels = images, labels
                    # print(images.shape)
                    # images, labels = images.cuda(), labels.cuda()
                    log_probs = client_model(images)
                    loss = F.cross_entropy(log_probs, labels)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

            '''开始步骤二：用local model预测公共数据集的logits'''
            aligment_data = generate_alignment_data(train_data_public_list_seq, args.N_alignment)

            logits_public=[]
            for client_index, client_model in enumerate(self.client_models):
                logits_public.append(predict(client_model,aligment_data["X"]))


            # batch_num = args.N_alignment / args.batchsize
            # batch_indices = np.random.choice(range(len(train_data_public)), batch_num)
            # #暂存当前回合所有客户端的平均logits    logits对应的batchsize_index由index确定
            # list_log_probs = torch.zeros(args.client_number, args.batch_size, args.class_num)
            # for client_index, client_model in enumerate(self.client_models):
            #     data_iter = iter(train_data_public)
            #     for batch_idx in batch_indices:
            #         (images, labels) = next(data_iter)
            #         log_probs = client_model(images)
            #         list_log_probs[client_index] = torch.add(log_probs, list_log_probs)
            #     list_log_probs[client_index]/=args.N_alignment
            '''开始步骤三、四：所有客户端的本地预测结果上传至中心服务器进行ERA和SA聚合'''
            logits_public = torch.from_numpy(np.array(logits_public))
            #先进行SA。
            list_log_probs_avg = torch.mean(logits_public, axis=0)
            for index_count, data_idx in enumerate(aligment_data["idx"]):
                self.public_logits_dict[data_idx] = list_log_probs_avg[index_count]

            #在SA的基础上进行ERA
            list_log_probs_soft=torch.softmax(list_log_probs_avg/args.public_T, dim=1)
            for index_count, data_idx in enumerate(aligment_data["idx"]):
                self.public_logits_dict[data_idx] = list_log_probs_soft[index_count]

            '''开始步骤五、六:将全局logits分发给本地客户端 每个本地客户端基于全局logits进行本地蒸馏'''
            for client_index, client_model in enumerate(self.client_models):
                client_model.train()
                optim = torch.optim.SGD(client_model.parameters(), lr=args.lr, momentum=0.9,
                                        weight_decay=args.wd)
                log_probs=client_model(aligment_data["X"])
                loss = F.cross_entropy(log_probs,list_log_probs_soft)
                optim.zero_grad()
                loss.backward()
                optim.step()


            print("*********start verifying with FedMD***************")
            if global_epoch % args.interval == 0:
                acc_all = []
                for client_index, client_model in enumerate(self.client_models):
                    if client_index % args.sel != 0:
                        continue
                        # 验证客户端的准确性
                    print("开始验证第" + str(client_index) + "个客户端")
                    client_model.eval()
                    loss_avg = utils.RunningAverage()
                    accTop1_avg = utils.RunningAverage()
                    accTop5_avg = utils.RunningAverage()
                    for batch_idx, (images, labels) in enumerate(test_data_local_dict[client_index]):
                        images, labels = images, labels
                        # images, labels = images.cuda(), labels.cuda()

                        labels = torch.tensor(labels, dtype=torch.long)
                        # log_probs, extracted_features = client_model(images)
                        log_probs = client_model(images)

                        loss = self.criterion_CE(log_probs, labels)
                        # Update average loss and accuracy
                        metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                        # only one element tensors can be converted to Python scalars
                        accTop1_avg.update(metrics[0].item())
                        accTop5_avg.update(metrics[1].item())
                        loss_avg.update(loss.item())
                    # print(loss_avg,type(loss_avg))

                    # compute mean of all metrics in summary
                    test_metrics = {str(client_index) + ' test_loss': loss_avg.value(),
                                    str(client_index) + ' test_accTop1': accTop1_avg.value(),
                                    str(client_index) + ' test_accTop5': accTop5_avg.value(),
                                    }
                    # wandb.log({str(client_index)+" Test/Loss": test_metrics[str(client_index)+' test_loss']})
                    # wandb.log({str(client_index)+" Test/AccTop1": test_metrics[str(client_index)+' test_accTop1']})
                    acc = accTop1_avg.value()
                    print("mean Test/AccTop1 on client", client_index, ":", acc)
                    acc_all.append(acc)
                    # acc_all.append(accTop1_avg.value())
                # wandb.log({"mean Test/AccTop1": float(np.mean(np.array(acc_all)))})
                # metrics=self.eval_on_the_client()
                print("mean Test/AccTop1 on all clients:", float(np.mean(np.array(acc_all))))

