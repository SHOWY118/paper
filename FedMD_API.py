
import argparse
import pickle
from argparse import ArgumentParser
from pathlib import Path
import pickle
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
import torch
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


def get_fedmd_argparser(args) -> ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.set_defaults(**vars(args))
    parser.add_argument("--digest_epoch", type=int, default=1)
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--public_epoch", type=int, default=5)
    parser.add_argument("--communicat_epoch", type=int, default=100)
    return parser

class FedMD_standalone_API:
    def __init__(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                 train_data_local_dict, test_data_local_dict, args, test_data_global,train_data_public):
        self.client_models = client_models
        self.test_data_global = test_data_global
        self.criterion_KL = utils.KL_Loss(args.public_T)
        self.criterion_CE = F.cross_entropy
        self.args = get_fedmd_argparser(args).parse_args()

        self.train_data_public=train_data_public
        self.mse_criterion = torch.nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def custom_cross_entropy(self,raw_output, true_labels):
        loss_per_sample = -torch.sum(true_labels * raw_output, dim=1)
        average_loss = torch.mean(loss_per_sample)
        return average_loss

    def do_fedMD_stand_alone(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, args):
        # wandb.login(key="4651ed05b06c424a6d1a6c4d9b2135a47edfbc39")
        # wandb.init(project='FDMD', config=args)

        wandb.login(key="913a0944b78830edbb2fdd338acc245686e13363")
        wandb.init(project='FMD', config=args)

        local_knowledge={}
        global_knowledge={}
        mal_idxs=utils.mal_select_clients(len(self.client_models),args.mal_rate)
        print("恶意客户端ID：{}".format(mal_idxs))

        # train_data_num = {}
        # for client_index, client_model in enumerate(self.client_models):
        #     cur_idx = 0
        #     client_train_data_dist = [0 for _ in range(args.class_num)]
        #     for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
        #         images, labels=images.cuda(), labels.cuda()
        #         # 转换label的tensor类别为list
        #         batch_idx_list_label = list(labels)
        #         for i in batch_idx_list_label:
        #             client_train_data_dist[i] += 1
        #     train_data_num[client_index]=sum(client_train_data_dist)
        #     print("客户端{}的标签分布{}".format(client_index, client_train_data_dist))
        #     print("客户端{}的数量{}".format(client_index, train_data_num[client_index]))
        #     print("客户端{}的数量{}".format(client_index, train_data_local_num_dict[client_index]))
        # print("数量{}".format(sum(list(train_data_num.values()))))

        # # 每个客户端在公共数据集和私有上训练
        # for client_index, model in enumerate(self.client_models):
        #     # compute on public
        #     print("开始公共训练第" + str(client_index) + "个客户端")
        #     model.to(self.device)
        #     model.train()
        #     optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=args.wd)
        #     for _ in range(self.args.public_epoch):
        #         for batch_idx, (images, labels) in enumerate(self.train_data_public):
        #             images, labels = images.to(self.device), torch.tensor(labels, dtype=torch.long).to(self.device)
        #             if client_index in mal_idxs:
        #                 if args.attack_method_id == 8:
        #                     # print("执行标签翻转攻击")
        #                     label_flip_attack = ATTACKS.LabelFlipAttack(args.class_num)
        #                     labels = label_flip_attack.attack(client_index, labels, mal_idxs)
        #                 elif args.attack_method_id == 9:
        #                     # print("执行女巫攻击")
        #                     witch_attack = ATTACKS.WitchAttack(args.class_num)
        #                     labels = witch_attack.attack(client_index, labels, mal_idxs)
        #             log_probs = model(images)
        #             loss = self.criterion_CE(log_probs, labels)
        #             optim.zero_grad()
        #             loss.backward()
        #             optim.step()
        #
        #     for _ in range(self.args.local_epoch):
        #         for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
        #             images, labels = images.to(self.device), labels.to(self.device).long()
        #             if client_index in mal_idxs:
        #                 if args.attack_method_id == 8:
        #                     # print("执行标签翻转攻击")
        #                     label_flip_attack = ATTACKS.LabelFlipAttack(args.class_num)
        #                     labels = label_flip_attack.attack(client_index, labels, mal_idxs)
        #                 elif args.attack_method_id == 9:
        #                     # print("执行女巫攻击")
        #                     witch_attack = ATTACKS.WitchAttack(args.class_num)
        #                     labels = witch_attack.attack(client_index, labels, mal_idxs)
        #             log_probs = model(images)
        #             loss = self.criterion_CE(log_probs, labels)
        #             # wandb.log({f"public top1 train Model {client_index} Accuracy": public_train_accTop1_avg.value()})
        #             # wandb.log({f"public loss train Model {client_index} Accuracy": public_train_loss_avg.value()})
        #             optim.zero_grad()
        #             loss.backward()
        #             optim.step()

        with open('client_models1.pkl', 'rb') as f:
            self.client_models = pickle.load(f)
        # 验证敛散性  在公共测试集上的准确性
        public_acc_all = []
        for client_index, client_model in enumerate(self.client_models):
            # 验证客户端的准确性
            # print("开始验证第" + str(client_index) + "个客户端的敛散性")
            client_model.eval()
            public_test_loss_avg = utils.RunningAverage()
            public_test_accTop1_avg = utils.RunningAverage()

            for batch_idx, (images, labels) in enumerate(test_data_local_dict[client_index]):
                images, labels = images.to(self.device), labels.to(self.device).long()
                log_probs = client_model(images)
                loss = self.criterion_CE(log_probs, labels)
                # Update average loss and accuracy
                pubic_test_metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                # only one element tensors can be converted to Python scalars
                public_test_accTop1_avg.update(pubic_test_metrics[0].item())
                public_test_loss_avg.update(loss.item())
                # wandb.log({f"public top1 test Model {client_index} Accuracy": public_test_accTop1_avg.value()})
                # wandb.log({f"public loss test Model {client_index} Accuracy": public_test_loss_avg.value()})
            # print(loss_avg,type(loss_avg))
            public_acc_all.append(public_test_accTop1_avg.value())
        print("敛散性：{}".format(public_acc_all))
        print("平均敛散性：{}".format(np.mean(public_acc_all)))
        # wandb.log({"public mean Test/AccTop1": float(np.mean(np.array(public_acc_all)))},step=0)

        # with open('client_models1.pkl', 'wb') as f:
        #     pickle.dump(self.client_models, f)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for com_round in range(self.args.communicat_epoch):
            print("开始训练第"+str(com_round)+"轮次")
            if args.attack_method_id == 7 :
                # print("执行参数反转攻击")
                w_locals = []  # 存储客户端的模型参数
                for client_index, client_model in enumerate(self.client_models):
                    w_locals.append({name: param.clone().detach() for name, param in client_model.state_dict().items()})
                attacker = ATTACKS.ParameterFlipAttack()
                attacker.attack(w_locals, mal_idxs, device)
                for mal_idx in mal_idxs:
                    self.client_models[mal_idx].load_state_dict(w_locals[mal_idx])
            elif args.attack_method_id==6:
                # print("执行IPM攻击")
                w_locals = []  # 存储客户端的模型参数
                for client_index, client_model in enumerate(self.client_models):
                    w_locals.append({name: param.clone().detach() for name, param in client_model.state_dict().items()})
                attacker = ATTACKS.IPMAttack(epsilon=0.1)
                attacker.attack(w_locals, mal_idxs, device)
                for mal_idx in mal_idxs:
                    self.client_models[mal_idx].load_state_dict(w_locals[mal_idx])

            selected_batches = sorted(random.sample(range(len(self.train_data_public)), args.batch_num))
            print("选择的batch：{}".format(selected_batches))

            for client_index, model in enumerate(self.client_models):
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
                                # print("攻击前知识：{}".format(log_probs[0]))
                                logits_attack = ATTACKS.LogitsProcessor(attack_method_id=args.attack_method_id)
                                log_probs = logits_attack.attack(log_probs)
                                # print("攻击后知识：{} 另一个：{}".format(log_probs[0],log_probs[1]))

                        local_knowledge_tem.append(log_probs.detach())

                local_knowledge[client_index]=local_knowledge_tem
                # print("本地知识：{}".format(local_knowledge[client_index]))

            #聚合过程
            for id, batch_idx in enumerate(selected_batches):
                tmp = []
                for client_id, knowledge in local_knowledge.items():
                    tmp.append(knowledge[id])
                global_knowledge[batch_idx] = torch.mean(torch.stack(tmp), dim=0)
            # print("全局知识：{}".format(global_knowledge[0]))

            # digest
            print("开始digest")
            for digest_num in range(self.args.digest_epoch):
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
                        loss=self.criterion_KL(log_probs,global_knowledge[batch_idx]/args.public_T)
                        # loss= F.cross_entropy(log_probs, labels)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

            acc_top1_all = []
            honest_top1_all = []
            for client_index, client_model in enumerate(self.client_models):
                # 验证客户端的准确性
                # print("开始验证digest第" + str(client_index) + "个客户端")
                client_model.eval()
                loss_avg = utils.RunningAverage()
                accTop1_avg = utils.RunningAverage()
                for batch_idx, (images, labels) in enumerate(test_data_local_dict[client_index]):
                    images, labels = images.to(self.device), labels.to(self.device).long()
                    log_probs = client_model(images)
                    loss = self.criterion_CE(log_probs, labels)
                    # Update average loss and accuracy
                    metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                    # only one element tensors can be converted to Python scalars
                    accTop1_avg.update(metrics[0].item())
                    loss_avg.update(loss.item())
                acc_top1 = accTop1_avg.value()
                acc_top1_all.append(acc_top1)
                # 收集所有参与者的数据
                if client_index in mal_idxs:
                    honest_top1_all.append(acc_top1)
                # 收集未感染者的数据
            # print("digest后客户端的准确率：{}".format(honest_top1_all))
            print("digest后客户端的平均准确率：{}".format(np.mean(honest_top1_all)))

            # revisit
            for _ in range(self.args.local_epoch):
                print("开始revisit")
                for client_index, model in enumerate(self.client_models):
                    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
                    model.train()
                    for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
                        images, labels = images.to(self.device), labels.to(self.device).long()
                        if client_index in mal_idxs:
                            if args.attack_method_id == 8:
                                # print("执行标签翻转攻击")
                                # print("反赚钱：{}".format(labels))
                                label_flip_attack = ATTACKS.LabelFlipAttack(args.class_num)
                                labels = label_flip_attack.attack(client_index, labels)
                                # print("反赚钱：{}".format(labels))
                            elif args.attack_method_id == 9:
                                # print("执行女巫攻击")
                                witch_attack = ATTACKS.WitchAttack(args.class_num)
                                labels = witch_attack.attack(client_index, labels)
                        log_probs = model(images)
                        loss = self.criterion_CE(log_probs, labels)
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
                          step=com_round)
                wandb.log({str(client_index) + " Test/AccTop5": test_metrics[str(client_index) + ' test_accTop5']},
                          step=com_round)

                acc_top1 = accTop1_avg.value()
                acc_top5 = accTop5_avg.value()
                acc_top1_all.append(acc_top1)
                acc_top5_all.append(acc_top5)

                # 收集所有参与者的数据
                honest_top1_all.append(acc_top1)
                honest_top5_all.append(acc_top5)
                if client_index not in mal_idxs:
                    honest_top1_all.append(acc_top1)
                    honest_top5_all.append(acc_top5)
                # 收集未感染者的数据
            # print("客户端的准确率：{}".format(honest_top1_all))
            print("客户端的平均准确率：{}".format(np.mean(honest_top1_all)))

            wandb.log({f"mean Test/AccTop1 on all honest clients": float(np.mean(np.array(honest_top1_all)))},
                      step=com_round)
            wandb.log({f"mean Test/AccTop5 on all honest clients": float(np.mean(np.array(honest_top5_all)))},
                      step=com_round)

        wandb.finish()
