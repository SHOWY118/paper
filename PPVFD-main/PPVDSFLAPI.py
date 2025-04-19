# from mpi4py import MPI
# from GKTServerTrainer import GKTServerTrainer
# from GKTClientTrainer import GKTClientTrainer
import argparse
import pickle
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from typing import List, Any
import pickle
import torchvision
from torch.nn import functional as F
import copy
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR100, EMNIST
from torchvision.transforms import Compose, Normalize
import ATTACKS
import random
import wandb
import utils
import lcc
import torch
from torch import nn, Tensor

from data_util.FMNIST.fashionmnist_data_loader import _data_transforms_FashionMNIST
from data_util.constants import MEAN, STD

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

PROJECT_DIR = Path(__file__).absolute().parent


def get_ppvfedmd_argparser(args) -> ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.set_defaults(**vars(args))
    parser.add_argument("--digest_epoch", type=int, default=1)
    parser.add_argument("--local_epoch", type=int, default=2)
    parser.add_argument("--public_epoch", type=int, default=5)
    parser.add_argument("--communicat_epoch", type=int, default=20)
    return parser


class PPVDSFL_standalone_API:
    def __init__(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                 train_data_local_dict, test_data_local_dict, args, test_data_global,train_data_public):
        self.client_models = client_models
        self.test_data_global = test_data_global
        self.criterion_KL = utils.KL_Loss(args.public_T)
        self.criterion_CE = F.cross_entropy
        self.args = get_ppvfedmd_argparser(args).parse_args()

        self.train_data_public=train_data_public
        self.mse_criterion = torch.nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def custom_cross_entropy(self,raw_output, true_labels):
        loss_per_sample = -torch.sum(true_labels * raw_output, dim=1)
        average_loss = torch.mean(loss_per_sample)
        return average_loss

    def do_ppvdsfl_stand_alone(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, args):
        # wandb.login(key="4651ed05b06c424a6d1a6c4d9b2135a47edfbc39")
        # wandb.init(project='FDMD', config=args)
        wandb.login(key="913a0944b78830edbb2fdd338acc245686e13363")
        wandb.init(project='DSFL', config=args)
        local_knowledge={}
        local_knowledge_hash={}
        global_knowledge={}

        #初始化知识
        for client_index in range(len(self.client_models)):
            local_knowledge_hash[client_index] = {}
            for c in range(args.class_num):
                local_knowledge_hash[client_index][c] = torch.Tensor(np.array([1 / args.class_num for _ in range(args.class_num)]))
            local_knowledge_hash[client_index] = np.stack(
                [tensor.numpy() for tensor in local_knowledge_hash[client_index].values()],axis=0)


        mal_idxs=utils.mal_select_clients(len(self.client_models),args.mal_rate)
        print("恶意客户端ID：{}".format(mal_idxs))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for global_epoch in range(args.comm_round):  # 表示进行多少次客户端与服务器之间的交互，设置为无限大则一直不停
            print("开始训练第" + str(global_epoch) + "轮次训练")
            metrics_all = {'test_loss': [], 'test_accTop1': [], 'test_accTop5': [], 'f1': []}

            if args.attack_method_id == 7:
                # print("执行参数反转攻击")
                w_locals = []  # 存储客户端的模型参数
                for client_index, client_model in enumerate(self.client_models):
                    w_locals.append({name: param.clone().detach() for name, param in client_model.state_dict().items()})
                attacker = ATTACKS.ParameterFlipAttack()
                attacker.attack(w_locals, mal_idxs, device)
                for mal_idx in mal_idxs:
                    self.client_models[mal_idx].load_state_dict(w_locals[mal_idx])
            elif args.attack_method_id == 6:
                # print("执行IPM攻击")
                w_locals = []  # 存储客户端的模型参数
                for client_index, client_model in enumerate(self.client_models):
                    w_locals.append({name: param.clone().detach() for name, param in client_model.state_dict().items()})
                attacker = ATTACKS.IPMAttack(epsilon=0.1)
                attacker.attack(w_locals, mal_idxs, device)
                for mal_idx in mal_idxs:
                    self.client_models[mal_idx].load_state_dict(w_locals[mal_idx])

            # 每个客户端在私有上训练
            for client_index, model in enumerate(self.client_models):
                # print("开始本地训练第" + str(client_index) + "个客户端")
                model.to(self.device)
                model.train()
                optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
                for _ in range(self.args.local_epoch):
                    for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
                        images, labels = images.to(self.device), labels.to(self.device).long()
                        if client_index in mal_idxs:
                            if args.attack_method_id == 8:
                                # print("执行标签翻转攻击")
                                label_flip_attack = ATTACKS.LabelFlipAttack(args.class_num)
                                labels = label_flip_attack.attack(client_index, labels)
                            elif args.attack_method_id == 9:
                                # print("执行女巫攻击")
                                witch_attack = ATTACKS.WitchAttack(args.class_num)
                                labels = witch_attack.attack(client_index, labels,mal_idxs)
                        log_probs = model(images)
                        loss = self.criterion_CE(log_probs, labels)
                        # wandb.log({f"public top1 train Model {client_index} Accuracy": public_train_accTop1_avg.value()})
                        # wandb.log({f"public loss train Model {client_index} Accuracy": public_train_loss_avg.value()})
                        optim.zero_grad()
                        loss.backward()
                        optim.step()



            selected_batches = sorted(random.sample(range(len(self.train_data_public)), args.batch_num))
            print("选择的batch：{}".format(selected_batches))

            for client_index, model in enumerate(self.client_models):
                tmp_logits = {}
                for c in range(args.class_num):
                    tmp_logits[c] = []
                local_knowledge_tem = []
                model.eval()
                with torch.no_grad():
                    for batch_idx, (images, labels) in enumerate(self.train_data_public):
                        if batch_idx not in selected_batches:
                            continue
                        images, labels = images.to(self.device), labels.to(self.device).long()
                        log_probs = model(images)
                        if client_index in mal_idxs:
                            if 1 <= args.attack_method_id <= 5:
                                # print("攻击前知识：{}".format(log_probs))
                                logits_attack = ATTACKS.LogitsProcessor(attack_method_id=args.attack_method_id)
                                log_probs = logits_attack.attack(log_probs)
                                # print("攻击后知识：{}".format(log_probs))
                        local_knowledge_tem.append(log_probs.detach())
                        for logit, label in zip(log_probs, labels):
                            c = int(label)
                            tmp_logits[c].append(logit.cpu().detach().numpy())

                for c in range(args.class_num):
                    if len(tmp_logits[c]) != 0:
                        local_knowledge_hash[client_index][c] = torch.mean(torch.Tensor(np.array(tmp_logits[c])),0)
                    else:
                        local_knowledge_hash[client_index][c] = torch.Tensor(np.array([1 / args.class_num for _ in range(args.class_num)]))
                local_knowledge[client_index]=local_knowledge_tem
                # print("客户端：{}本地知识：{}".format(client_index, local_knowledge_hash[client_index]))

            logical_graph = lcc.LogicalGraphOracle(
                local_knowledges=local_knowledge_hash.copy(),
                mal_idxs=mal_idxs,
                args=args
            )
            client_connection,sim_matrix,attack_success_rate=logical_graph.logit_com_graph()
            mean_rate=0
            for i in range(args.client_number):
                if i not in mal_idxs:
                    mean_rate += attack_success_rate[i]
            print("平均成功率：{}".format(mean_rate / (len(mal_idxs))))
            wandb.log({f"Attack Success Rate": float(mean_rate)},step=global_epoch)

            oracle = lcc.ALCCOracle(
                X=local_knowledge,
                client_connection=client_connection,
                num_workers_origin=args.client_number,
                num_stragglers=0,
                security_guarantee=0,
                privacy_guarantee=10,
                beta=1.15,
                sigma=10 ** 3,
                theta=6,
                sim_matrix=sim_matrix,
                fit_intercept=False,
                args=args
            )
            # 获取每个客户端加密后发送数据
            local_encoded_knowledge = oracle.ALcc_encoded()
            # 获取每个客户端接收到的数据
            received_encoded_knowledge = oracle.encoded_knowledge_distribute()
            # 每个客户端执行本地计算 要上传的logits
            fun_encoded_knowledge = oracle.worker_fun()

            # 中心服务器接收数据进行解码计算、聚合、签名计算等 更新global_knowledge并把相关的信息发送给本地客户端。
            Y_decoded_global_knowledges, self.sum_k, self.sum_T = oracle.ALcc_decoded()
            debug_global_knowledge, debug_global_sum_knowledge = oracle.debug_ALcc()
            # print("解码全局知识1：{}，全局知识2：{}".format(Y_decoded_global_knowledges[0][0],Y_decoded_global_knowledges[1][0]))
            # print("原始全局知识1：{}，全局知识2：{}".format(debug_global_knowledge[0][0],debug_global_knowledge[1][0]))
            summed_loss_1 = lcc.rel_err(Y_decoded_global_knowledges, debug_global_knowledge)
            # summed_loss_2 = lcc.rel_err(self.sum_k, debug_global_sum_knowledge)
            print("summed_loss_1:{}".format(summed_loss_1))
            # print("summed_loss_2:{}".format(summed_loss_2))
            wandb.log({f"mean Loss on all clients": float(np.mean(np.array(summed_loss_1)))},
                       step=global_epoch)
            # 更新全局知识
            for client_index in range(args.client_number):
                global_knowledge[client_index] = torch.Tensor(np.real(Y_decoded_global_knowledges[client_index])/(args.R+1)).cuda()
            # print("全局知识1：{}，全局知识2：{}".format(global_knowledge[0],global_knowledge[1]))

            print("开始下载知识并本地训练")
            for client_index, model in enumerate(self.client_models):
                # 开始digest
                model = model.cuda()
                model.train()
                optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                        weight_decay=args.wd)
                for batch_idx, (images, labels) in enumerate(self.train_data_public):
                    if batch_idx not in selected_batches:
                        continue
                    images, labels = images.to(self.device), labels.to(self.device).long()
                    log_probs = model(images)
                    # loss = self.mse_criterion(log_probs, self.consensus[batch_idx])
                    loss=self.criterion_KL(log_probs,global_knowledge[client_index][selected_batches.index(batch_idx)]/args.public_T)
                    # loss= F.cross_entropy(log_probs, labels)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

            # 验证客户端的准确性
            acc_top1_all = []
            acc_top5_all = []
            honest_top1_all = []
            honest_top5_all = []
            for client_index, client_model in enumerate(self.client_models):
                # 验证客户端的准确性
                # print("开始验证第" + str(client_index) + "个客户端")
                client_model.eval()
                loss_avg = utils.RunningAverage()
                accTop1_avg = utils.RunningAverage()
                accTop5_avg = utils.RunningAverage()
                for batch_idx, (images, labels) in enumerate(test_data_local_dict[client_index]):
                    images, labels = images.to(self.device), labels.to(self.device).long()
                    log_probs = client_model(images)
                    loss = self.criterion_CE(log_probs, labels)
                    # Update average loss and accuracy
                    metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                    # only one element tensors can be converted to Python scalars
                    accTop1_avg.update(metrics[0].item())
                    accTop5_avg.update(metrics[1].item())
                    loss_avg.update(loss.item())
                # compute mean of all metrics in summary
                test_metrics = {str(client_index) + ' test_loss': loss_avg.value(),
                                str(client_index) + ' test_accTop1': accTop1_avg.value(),
                                str(client_index) + ' test_accTop5': accTop5_avg.value(),
                                }
                wandb.log({str(client_index) + " Test/AccTop1": test_metrics[str(client_index) + ' test_accTop1']},
                          step=global_epoch)
                wandb.log({str(client_index) + " Test/AccTop5": test_metrics[str(client_index) + ' test_accTop5']},
                          step=global_epoch)
                acc_top1 = accTop1_avg.value()
                acc_top5 = accTop5_avg.value()
                acc_top1_all.append(acc_top1)
                acc_top5_all.append(acc_top5)

                # 收集所有参与者的数据
                if client_index not in mal_idxs:
                    honest_top1_all.append(acc_top1)
                    honest_top5_all.append(acc_top5)
                # 收集未感染者的数据
            # print("客户端的准确率：{}".format(honest_top1_all))
            print("客户端的平均准确率：{}".format(np.mean(honest_top1_all)))

            wandb.log({f"mean Test/AccTop1 on all honest clients": float(np.mean(np.array(honest_top1_all)))},
                       step=global_epoch)
            wandb.log({f"mean Test/AccTop5 on all honest clients": float(np.mean(np.array(honest_top5_all)))},
                       step=global_epoch)

        wandb.finish()
