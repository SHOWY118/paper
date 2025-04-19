import gc
import os
import numpy as np
import utils
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import hnswlib
from scipy import spatial
import sys
import matplotlib.pyplot as plt
import wandb
import ATTACKS


def knowledge_avg(knowledge, weights):
    result = []
    for k_ in knowledge:
        result.append(knowledge_avg_single(k_, weights))
    # return torch.Tensor(np.array(result)).cuda()
    return torch.Tensor(np.array(result))


def knowledge_avg_single(knowledge, weights):
    result = torch.zeros_like(knowledge[0]).cpu()
    sum = 0
    for _k, _w in zip(knowledge, weights):
        result.add_(_k.cpu() * _w)
        sum = sum + _w
    result = result / sum
    return torch.tensor(np.array(result.detach().cpu()))


class KnowledgeCache:
    def __init__(self, n_classes, R):
        self.n_classes = n_classes
        self.cache = {}
        self.idx_to_hash = {}
        self.relation = {}
        for i in range(n_classes):
            self.cache[i] = {}
        self.R = R
        pass

    def add_hash(self, hash, label, idx):
        for k_, l_, i_ in zip(hash, label, idx):
            self.add_hash_single(k_, l_, i_)

    def add_hash_single(self, hash, label, idx):
        # 此处的idx为（client_index,cur_idx）
        self.cache[int(label)][idx] = torch.Tensor(np.array([1.0 / self.n_classes for _ in range(self.n_classes)]))
        # {
        #     2: [torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2])]
        # }
        self.idx_to_hash[idx] = hash
        # {"(client_index,cur_idx)":hash_calue;}

    def build_relation(self):
        hnsw_sim = 0
        for c in range(self.n_classes):
            # 获取c类别样本的所有索引  此处c为类别， 其keys为一个个（client_index,idx）
            idx_vectors = [key for key in self.cache[c].keys()]
            data = list()
            # 将索引样本对应的hash值转化为numpy data数组
            data = np.array([self.idx_to_hash[key].cpu().numpy() for key in idx_vectors])
            # c类别样本的数量 和 每个样本对应哈希值的维度（相同）
            num_elements = data.shape[0]
            dim = data.shape[1]

            data_labels = np.arange(num_elements)

            index = hnswlib.Index(space='cosine', dim=dim)
            index.init_index(max_elements=num_elements, ef_construction=1000, M=64)
            index.add_items(data, data_labels)
            index.set_ef(1000)  # 控制查询时的搜索速度和精度。
            labels, distances = index.knn_query(data, self.R + 1)

            for idx, ele in enumerate(labels):
                self.relation[idx_vectors[int(idx)]] = []
                for x in ele[1:]:
                    self.relation[idx_vectors[int(idx)]].append(idx_vectors[x])

    def set_knowledge(self, knowledge, label, idx):
        for k_, l_, i_ in zip(knowledge, label, idx):
            self.set_knowledge_single(k_, l_, i_)

    def set_knowledge_single(self, knowledge, label, idx):
        self.cache[int(label)][idx] = knowledge

    # 获取与给定样本关联的其他样本的知识信息。 接受多个样本的标签和索引
    def fetch_knowledge(self, label, idx):
        result = []
        for l_, i_ in zip(label, idx):
            result.append(self.fetch_knowledge_single(l_, i_))
        return result

    def fetch_knowledge_single(self, label, idx):
        result = []
        pairs = self.relation[idx]
        for pair in pairs:
            result.append(self.cache[int(label)][pair])
        return result


class FedCache_standalone_API:
    def __init__(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                 train_data_local_dict, test_data_local_dict, args, test_data_global):
        self.client_models = client_models
        self.test_data_global = test_data_global
        self.global_logits_dict = dict()
        self.global_labels_dict = dict()
        self.global_extracted_feature_dict_test = dict()
        self.global_labels_dict_test = dict()
        self.criterion_KL = utils.KL_Loss(args.temperature)
        self.criterion_CE = F.cross_entropy

    def do_fedcache_stand_alone(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                                train_data_local_dict, test_data_local_dict, args):
        # 创建数据预处理流水线，该流水线只包含一个resize（224）的操作
        image_scaler = transforms.Compose([
            transforms.Resize(224),
        ])
        print("*********start training with FedCache***************")
        wandb.login(key="913a0944b78830edbb2fdd338acc245686e13363")
        wandb.init(project='FedCACHE', config=args)

        train_data_local_dict_seq = {}
        # 为每个客户端遍历其本地训练数据加载器，逐批次地将图像和标签存储在列表
        # train_data_local_dict_seq[client_index]中。
        for client_index in range(args.client_number):
            train_data_local_dict_seq[client_index] = []
            for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
                train_data_local_dict_seq[client_index].append((images, labels))
        knowledge_cache = KnowledgeCache(args.class_num, args.R)

        # 创建了一个加载了预训练权重的MobileNetV3Small模型，并将其移动到GPU
        # 上。然后，它通过去掉最后一层来创建一个新的序列模型，用于特征提取。最后，它将模型设置为评估模式，以备用于推理过程。
        # encoder加载的模型输出将是原模型倒数第二层的输出（去掉分类器）
        encoder = mobilenet_v3_small(weights='IMAGENET1K_V1').cuda()
        encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]))
        encoder.eval()

        for client_index, client_model in enumerate(self.client_models):
            cur_idx = 0
            client_train_data_dist = [0 for _ in range(args.class_num)]
            for batch_idx, (images, labels) in enumerate(train_data_local_dict_seq[client_index]):
                images, labels = images.cuda(), labels.cuda()
                # images, labels=images, labels
                batch_idx_list_label = list(labels)
                for i in batch_idx_list_label:
                    client_train_data_dist[i] += 1
                # 这行代码使用之前创建的 encoder 模型处理图像数据，并获取其编码的哈希值。
                hash_code = encoder(image_scaler(images)).detach().cuda()
                hash_code = torch.tensor(hash_code.reshape((hash_code.shape[0], hash_code.shape[1])))
                for img, hash, label in zip(images, hash_code, labels):
                    # 遍历批次中的每张图像、哈希值和标签。
                    knowledge_cache.add_hash_single(hash, label, (client_index, cur_idx))
                    # 代表一张图片
                    cur_idx = cur_idx + 1
            # print("客户端{}的标签分布{}".format(client_index,client_train_data_dist))
        knowledge_cache.build_relation()

        mal_idxs = utils.mal_select_clients(len(self.client_models), args.mal_rate)
        print("恶意客户端：{}".format(mal_idxs))

        print("*********knowledge cache initialized successfully***************")

        # 下面开始训练
        for global_epoch in range(args.comm_round):
            print("*********communication round", global_epoch, "***************")
            metrics_all = {'test_loss': [], 'test_accTop1': [], 'test_accTop5': [], 'f1': []}
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            tem_logit = {}
            for client_index, client_model in enumerate(self.client_models):
                client_model = self.client_models[client_index]
                # print("*********start training on client",client_index,"***************")
                client_model = client_model.cuda()
                client_model.train()
                optim = torch.optim.SGD(client_model.parameters(), lr=args.lr, momentum=0.9,
                                        weight_decay=args.wd)
                cur_idx = 0
                # 初始化存储logits
                for batch_idx, (images, labels) in enumerate(train_data_local_dict_seq[client_index]):
                    images, labels = images.cuda(), labels.cuda().long()
                    if client_index in mal_idxs:
                        if args.attack_method_id == 8:
                            # print("执行标签翻转攻击")
                            label_flip_attack = ATTACKS.LabelFlipAttack(args.class_num)
                            labels_att = label_flip_attack.attack(client_index, labels)
                        elif args.attack_method_id == 9:
                            # print("执行女巫攻击")
                            witch_attack = ATTACKS.WitchAttack(args.class_num)
                            labels_att = witch_attack.attack(client_index, labels, mal_idxs)

                    log_probs = client_model(images)
                    if client_index in mal_idxs:
                        if 1 <= args.attack_method_id <= 5:
                            # print("攻击前知识：{}".format(log_probs[0]))
                            logits_attack = ATTACKS.LogitsProcessor(attack_method_id=args.attack_method_id)
                            log_probs = logits_attack.attack(log_probs)
                            # print("攻击后知识：{} 另一个：{}".format(log_probs[0], log_probs[1]))
                    # 如果是att 8 9 则使用lables_att

                    if client_index in mal_idxs and (args.attack_method_id == 8 or args.attack_method_id == 9):
                        loss_true = F.cross_entropy(log_probs, labels_att)
                    else:
                        loss_true = F.cross_entropy(log_probs, labels)
                    loss = None
                    # 存储老师模型的知识
                    teacher_knowledge = []
                    for img, logit, label in zip(images, log_probs, labels):
                        # 从知识缓存中获取与当前图像关联的知识信息。
                        fetched_knowledge_single = knowledge_cache.fetch_knowledge_single(label,
                                                                                          (client_index, cur_idx))
                        # if global_epoch==1:
                        #     print("全局知识:{}".format(fetched_knowledge_single))
                        # 将当前图像的预测概率（logit）存储为知识信息，方便后续更新知识缓存。
                        tem_logit[(client_index, cur_idx)] = logit
                        # knowledge_cache.set_knowledge_single(logit,label,(client_index,cur_idx))
                        cur_idx = cur_idx + 1
                        # 计算平均知识，用于知识蒸馏
                        avg_knowledge_single = knowledge_avg_single(fetched_knowledge_single,
                                                                    [1 for _ in range(args.R)])

                        # total_sum = torch.zeros_like(fetched_knowledge_single[0])
                        # if global_epoch == 1:
                        #     for i in fetched_knowledge_single:
                        #         total_sum += i
                        #     print("求和：{}".format(total_sum / 30))
                        # if global_epoch==1:
                        #     print("单个全局知识:{}".format(avg_knowledge_single))
                        teacher_knowledge.append(avg_knowledge_single.detach().cpu().numpy())
                    teacher_knowledge = torch.tensor(np.array(teacher_knowledge)).cuda()
                    # 本地模型预测的概率和蒸馏的知识之间的KL
                    # print(teacher_knowledge)
                    # print("客户端:{}的教师的知识:{}".format(client_index,teacher_knowledge[0]))
                    loss_kd = self.criterion_KL(log_probs, teacher_knowledge / args.T)
                    loss = loss_true + args.alpha * loss_kd
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

            for client_index, client_model in enumerate(self.client_models):
                cur_idxs = 0
                for batch_idx, (images, labels) in enumerate(train_data_local_dict_seq[client_index]):
                    for label in labels:
                        logit = tem_logit[client_index, cur_idxs]
                        # 更新全局知识
                        knowledge_cache.set_knowledge_single(logit, label, (client_index, cur_idxs))
                        cur_idxs = cur_idxs + 1

            # 在每个训练周期结束后对客户端模型进行测试并计算精度
            if global_epoch % args.interval == 0:
                # 存储每个客户端的精度
                acc_top1_all = []
                acc_top5_all = []
                honest_top1_all = []
                honest_top5_all = []
                for client_index, client_model in enumerate(self.client_models):
                    # 如果当前客户端索引不是被选中用于测试的索引，继续下一个循环迭代。
                    # sel=1 每个客户端都被选中
                    # print("*********start tesing on client",client_index,"***************")
                    # 将当前客户端模型设置为评估模式，以确保在测试过程中不进行模型更新。
                    client_model.cuda()
                    client_model.eval()
                    loss_avg = utils.RunningAverage()
                    accTop1_avg = utils.RunningAverage()
                    accTop5_avg = utils.RunningAverage()
                    for batch_idx, (images, labels) in enumerate(test_data_local_dict[client_index]):
                        images, labels = images.cuda(), labels.cuda().long()
                        log_probs = client_model(images)
                        loss = self.criterion_CE(log_probs, labels)
                        metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                        accTop1_avg.update(metrics[0].item())
                        accTop5_avg.update(metrics[1].item())
                        loss_avg.update(loss.item())
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
                    # 收集参与者的数据
                    if client_index not in mal_idxs:
                        honest_top1_all.append(acc_top1)
                        honest_top5_all.append(acc_top5)

                print("mean Test/AccTop1 on all clients:", float(np.mean(np.array(honest_top1_all))))
                wandb.log({f"mean Test/AccTop1 on all honest clients": float(np.mean(np.array(honest_top1_all)))},
                          step=global_epoch)
                wandb.log({f"mean Test/AccTop5 on all honest clients": float(np.mean(np.array(honest_top5_all)))},
                          step=global_epoch)
            gc.collect()
        wandb.finish()

