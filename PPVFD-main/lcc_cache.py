import numpy as np
import copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.random_projection import SparseRandomProjection
import matplotlib.pyplot as plt
import torch
import gc
def rel_err(approxs, trues, ord=None):
    rel_err_list=[]
    for client in approxs.keys():
        true=np.real(trues[client])
        approx=np.real(approxs[client])
        eps = np.finfo(true.dtype).eps
        enum = np.linalg.norm(np.array(approx - true), ord=ord)
        denom = np.linalg.norm(true, ord=ord) + eps
        rel_err_list.append(enum / denom)
    # print("rel_err_list:{}".format(rel_err_list))
    return rel_err_list

def rel_err_batch(approxs, trues, batch_num ,ord=None):
    rel_err_list=[]
    for client in range(len(approxs)):
        true=np.real(trues[client][:batch_num])
        approx=np.real(approxs[client][:batch_num])
        eps = np.finfo(true.dtype).eps
        enum = np.linalg.norm(np.array(approx - true), ord=ord)
        denom = np.linalg.norm(true, ord=ord) + eps
        rel_err_list.append(enum / denom)
    # print("rel_err_list:{}".format(rel_err_list))
    return rel_err_list

def modular_inv(a, p):
    x, y, m = 1, 0, p
    while a > 1:
        q = a // m
        t = m

        m = np.mod(a, m)
        a = t
        t = y

        y, x = x - np.int64(q) * np.int64(y), t

        if x < 0:
            x = np.mod(x, p)
    return np.mod(x, p)

def divmodp(_num, _den, _p):
    # compute num / denom modulo prime p
    _num = np.mod(_num, _p)
    _den = np.mod(_den, _p)
    _inv = modular_inv(_den, _p)
    return np.mod(np.int64(_num) * np.int64(_inv), _p)

def PI(vals, p):  # upper-case PI -- product of inputs
    #实数域运算 并保证在伽罗域
    accum = 1
    for v in vals:
        tmp = np.mod(v, p)
        accum = np.mod(accum * tmp, p)
    return accum

def gen_Lagrange_coeffs(alpha_s, beta_s, p=None, is_K1=False):
    #p相知在素数伽罗语模运算
    num_alpha = 1 if is_K1 else len(alpha_s)
    is_analog = np.iscomplexobj(alpha_s) or p is None
    field_type = "complex128" if is_analog else "int64"
    U = np.zeros((len(beta_s), num_alpha), dtype=field_type)   #beta_s行 alpha_s列 存储拉格朗日系数矩阵
    denom = np.zeros((len(beta_s)), dtype=field_type)  # 用于存储分母的连乘积
    for i in range(len(beta_s)):
        cur_beta = beta_s[i]
        denom_betas = [cur_beta - o for o in beta_s if cur_beta != o]
        denom[i] = np.prod(denom_betas) if is_analog else PI(denom_betas, p)

    enum = np.zeros((num_alpha), dtype=field_type)  # prod (alpha_j - beta_l)求得每一行的分子乘积 每一行都相同
    for j in range(num_alpha):
        enum_alphas = [alpha_s[j] - o for o in beta_s]
        enum[j] = np.prod(enum_alphas) if is_analog else PI(enum_alphas, p)

    for i in range(len(beta_s)):
        for j in range(num_alpha):
            if is_analog:
                current_denom = (alpha_s[j] - beta_s[i]) * denom[i]
                U[i][j] = enum[j] / current_denom
            else:
                current_denom = np.mod(np.mod(alpha_s[j] - beta_s[i], p) * denom[i], p)
                U[i][j] = divmodp(enum[j], current_denom, p)  # enum / denom
    return U.astype(field_type)

def quantize(X, q_bit, p):
    X_q= copy.deepcopy(X)
    for key,knowledge_list in X_q.items():
        X_int=torch.round(knowledge_list * (2**q_bit)).to(torch.int64)
        is_negative = (torch.abs(torch.sign(X_int)) - torch.sign(X_int)) / 2
        out = X_int + p * is_negative
        X_q[key] = out.to(torch.int64)
    return X_q

#与量化过程相反
def dequantize(X_q, q_bit, p):
    flag = X_q - (p - 1) // 2
    is_negative = (np.abs(np.sign(flag)) + np.sign(flag)) / 2
    X_q_sign = X_q - p * is_negative
    return X_q_sign.astype("double") / (2**q_bit)

def split(X,num_batches,args,train_data_local_num_dict):
    X_split={}
    for key,value in X.items():
        X_split[key]=np.array(value.clone().detach().cpu())
    for key in X_split.keys():
        cols = X_split[key].shape[0]
        k_parts = np.array([np.zeros(cols,dtype=np.complex128) for _ in range(num_batches)])
        for j in range(cols):
            value = X_split[key][j]
            if value != 0:  # 确保有值进行分割
                splits = np.sort(np.random.rand(num_batches - 1)) * value
                splits = np.concatenate(([0.0], splits, [value]))
                for part_index in range(num_batches):
                    k_parts[part_index][j] = splits[part_index + 1] - splits[part_index]
            else:
                for part_index in range(num_batches):
                    k_parts[part_index][j] = 0
        X_split[key] = k_parts
    return X_split

def concatenate(X_split,privacy_guarantee,theta=6,sigma=None):
    X_to_encode = {}  # Avoid deep copy
    for key in X_split.keys():
        if privacy_guarantee:
            if sigma:
                std = np.sqrt(sigma ** 2 / privacy_guarantee / 2)
                Z = (
                    (std * np.random.randn(privacy_guarantee, *X_split[key].shape[1:], 2))
                        .clip(-theta * std, theta * std)
                        .view(np.complex128)
                        .squeeze(-1)
                )
                X_to_encode[key] = np.concatenate([X_split[key], Z])
    return X_to_encode

def hash_map(logits,n_components=7,random_state=42):
    # 使用LSH进行哈希映射
    lsh = SparseRandomProjection(n_components=n_components,random_state=random_state)
    lsh.fit(logits)
    # 对样本的每个logits向量进行变换
    hashed_logits = lsh.transform(logits)
    return hashed_logits

# 获取每个客户端初始化数据集的标签分布并绘图
def label_dist(client_models,class_num,train_data_local_dict):
    client_label_distributions = {}
    for client_index, client_model in enumerate(client_models):
        client_label_distributions[client_index] = [0 for _ in range(class_num)]
        for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
            # images, labels=images.cuda(), labels.cuda()
            images, labels = images, labels
            # 转换label的tensor类别为list
            for label in labels:
                client_label_distributions[client_index][label] += 1
            # print("客户端{}输出当前批次{}的标签{}类别{}".format(client_index,batch_idx,np.array(labels).shape,labels))
            # print("客户端{}输出当前批次{}的标签分布{}".format(client_index,batch_idx,client_train_data_dist))
        print("客户端{}的数据标签分布{}".format(client_index, client_label_distributions[client_index]))
        # 绘制柱状图
        plt.bar(range(class_num), client_label_distributions[client_index])
        # 添加标签和标题
        plt.xlabel('标签类别')
        plt.ylabel('数量')
        plt.title('客户端{}训练数据的标签分布'.format(client_index))
        plt.show()
    return client_label_distributions

class LogicalGraphOracle():
    def __init__(
        self,
        local_knowledges,
        mal_idxs,
        args,
    ):
        self.local_knowledges=local_knowledges
        self.args=args
        self.mal_idsx=mal_idxs
        self.hashed_logits()
        self.sim_matrix()

    def hashed_logits(self):
        print("*********start distributing hash_global_logits with FD***************")
        # 每个客户端将local_logits通过hash映射
        self.global_hashed_logits = {}
        for client_index in range(self.args.client_number):
            # print("客户端：{}本地知识：{}".format(client_index,self.local_knowledges[client_index]))
            self.global_hashed_logits[client_index] = hash_map(self.local_knowledges[client_index])
    def sim_matrix(self):
        self.similarity_matrix = np.zeros((self.args.client_number, self.args.client_number))
        for i in range(self.args.client_number):
            for j in range(i, self.args.client_number):
                if i != j:
                    # 计算余弦相似度
                    A = self.global_hashed_logits[i].flatten()  # 第 i 行
                    B = self.global_hashed_logits[j].flatten()  # 第 j 行
                    # print(A,B)
                    cosine_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-8)  # 直接使用点积
                    self.similarity_matrix[i, j] = cosine_sim
                    self.similarity_matrix[j, i] = cosine_sim
            self.similarity_matrix[i,i]=-1
        # print("相似度矩阵：{}".format(self.similarity_matrix))

    def logit_com_graph(self):
        # 构建每个客户端的亲密度表（按相似度排序）
        intimacy_table = {i: np.argsort(-self.similarity_matrix[i]) for i in range(self.args.client_number)}
        # 记录每个客户端的邻居
        neighbors_act = {i: intimacy_table[i][:self.args.R] for i in range(self.args.client_number)}
        # print("关系列表：{}".format(neighbors_act))
        appearance_count = [0 for _ in range(self.args.client_number)]
        attack_success_count = np.array([0 for _ in range(self.args.client_number)])
        for key,array in neighbors_act.items():
            if key in self.mal_idsx:
                continue
            for person in array:
                if person in self.mal_idsx:
                    attack_success_count[key]+=1
                # 更新出现次数
                appearance_count[person] += 1
        print("攻击后出现次数:{}".format(appearance_count))

        attack_success_rate=attack_success_count/(self.args.client_number-len(self.mal_idsx))
        return neighbors_act,self.similarity_matrix,attack_success_rate

class ALCCOracle():
    """
    ALCC oracle for PPVFD
    """

    def __init__(
        self,
        X,
        num_workers_origin,
        args,
        train_data_local_num_dict,
        client_connection,
        num_stragglers,
        security_guarantee,
        privacy_guarantee,
        beta,
        sigma,
        theta,
        sim_matrix,
        sample_connect,
        cache,
        fit_intercept=False,
    ):
        self.X = np.hstack((np.ones((len(X), 1)), X)) if fit_intercept else X
        self.client_connection=client_connection
        self.num_workers = num_workers_origin  # N
        self.num_stragglers = num_stragglers  # S
        self.security_guarantee = security_guarantee  # A
        self.privacy_guarantee = privacy_guarantee  # T
        self.beta = beta
        self.sigma = sigma
        self.theta=theta
        self.sim_matrix=sim_matrix
        self.cache=cache
        self.fit_intercept = fit_intercept
        self.workers_id = [i for i in range(num_workers_origin)]
        self.args = args
        self.train_data_local_num_dict=train_data_local_num_dict
        self.sample_connect=sample_connect
        self._init_num_batches()  # 确定K
        self._init_w()
        self.init_client_connections()

        # print("num_batches:{}".format(self.num_batches))
        # 对量化后的结果进行分割成K份
        self.X_split = split(self.X, self.num_batches,self.args,self.train_data_local_num_dict)
        # 生成为每个客户端生成扰动矩阵      size  20*10
        self.X_to_encode = concatenate(self.X_split, self.privacy_guarantee, theta=self.theta,sigma=self.sigma)
        # print("完成预处理")

    def _init_w(self):
        self.w_init={}
        for client_index in self.workers_id:
            self.w_init[client_index]=np.ones(self.args.R)

    def _init_num_batches(self):
     self.num_batches=20

    def init_client_connections(self):
        self.client_connections={node: sorted(list(self.client_connection[node])+[node]) for node in self.workers_id}
        # print("关系列表：{}".format(self.client_connections))

    def ALcc_encoded(self):
        n_betas = self.num_batches + self.privacy_guarantee
        # print("n_betas:{}".format(n_betas))
        # self.alphas_total=np.exp(2 * np.pi * 1j * np.arange(n_alphas) / n_alphas)
        self.betas = self.beta * np.exp(2 * np.pi * 1j * np.arange(n_betas) / n_betas)
        self.alphas={}
        X_encoded={}
        for key_act in self.X_to_encode.keys():
            # print("完成：{}".format(key_act))
            #每个sample的alpha都是一样的。
            self.alphas[key_act] = np.exp(2 * np.pi * 1j * np.arange(len(self.sample_connect[key_act]))/len(self.sample_connect[key_act]))
            #每个客户端的矩阵
            U= gen_Lagrange_coeffs(self.alphas[key_act], self.betas)
            X_encoded[key_act] = np.zeros((len(self.sample_connect[key_act]),U.shape[1],10),dtype=np.complex128) # 根据需要的形状初始化
            for i , idx in enumerate(self.sample_connect[key_act]):
                # client_index 对应的connection（一种矩阵U）下 每个客户端的编码知识    key * 60 * 60 *10
                X_encoded[key_act][i] = np.einsum("kn,kd->nd", U, self.X_to_encode[idx])
        total_memory = sum(value.nbytes for value in X_encoded.values())
        print("Memory size of my_array:", total_memory)
        gc.collect()
        return X_encoded

    def encoded_knowledge_distribute(self,X_encoded):
        for key_act in self.X_to_encode.keys():
            X_encoded[key_act]=np.transpose(X_encoded[key_act], (1, 0, 2))
            # for client_distribute, distribute_data in enumerate(X_encoded[key_act]):
            #     # for client_receive, data in distribute_data.items():
            #     #     if client_receive not in self.X_received[key_act]:
            #     #         self.X_received[key_act][client_receive] = {}
            #     #         # 将数据添加到X_encoded_received中
            #     #     self.X_received[key_act][client_receive][client_distribute]=data
            #     for i, client_receive in enumerate(self.sample_connect[key_act]):
            #         X_received[key_act][i][client_distribute] = distribute_data[i]

            # for receiver in self.X_received[key_act].keys():
            #     # 将接收者按照自定义顺序排序
            #     self.X_received[key_act][receiver]=[self.X_received[key_act][receiver][key] for key in self.sample_connect[key_act]]
        gc.collect()
        return X_encoded

    def worker_fun(self,X_received):
        Y_received={}
        for key_act in self.X_to_encode.keys():
            Y_received[key_act]= np.zeros((len(self.sample_connect[key_act]),10),dtype=np.complex128) # 根据需要的形状初始化
            for id in range(self.args.R):
                #给成聚合函数 加上权重
                # print("权重长度：{}，接受知识的长度：{}".format(len(self.w_init[0]),len(self.X_received[key_act][client_receive])))
                Y_received[key_act][id]=np.tensordot(self.w_init[0],X_received[key_act][id],axes=(0,0))
            # print(local_knowledge_upload[key_act].shape)
        # print("编码输出：{}".format(self.local_knowledge_upload[0][0]))
        # print("原始输出：{}".format(self.test[0][0]))
        del X_received
        gc.collect()
        return Y_received

    def ALcc_decoded(self,local_knowledge_upload):
        #从 有“key”str的词典中找到客户端id作为插值
        self.Y_decoded={}
        self.sum_k={}
        self.sum_T={}
        U_dec = gen_Lagrange_coeffs(self.betas,self.alphas[(0,0)])
        for client_index in self.X_to_encode.keys():
            if self.security_guarantee > 0:
                malicious_workers = np.random.choice(
                    list(set(range(self.num_workers)) - set(succeeded_workers)),
                    self.security_guarantee,
                    replace=False,
                )
                raise NotImplementedError("Adversarial decoding not yet supported!")
            else:
                print("Y_received shape:", local_knowledge_upload[client_index].shape)
                self.Y_decoded[client_index] = np.einsum("nk,nd->kd", U_dec, local_knowledge_upload[client_index])#k+t* ld

            self.sum_k[client_index]=np.sum(self.Y_decoded[client_index][:self.num_batches], axis=0)
        gc.collect()
        return self.Y_decoded,self.sum_k

    def debug_ALcc(self):
        #这里用的x_to_encoded已经是被污染过的。
        k_silence={}
        k_silence_sum={}
        for client in self.X_to_encode.keys():
            if self.args.api=="PPVFDAPI":
                k_silence[client] = np.zeros((self.num_batches+self.privacy_guarantee, self.args.class_num, self.args.class_num), dtype=np.complex128)
            elif self.args.api=="PPVFMDAPI":
                k_silence[client] = np.zeros((self.num_batches+self.privacy_guarantee, self.args.batch_size, self.args.class_num), dtype=np.complex128)
            elif self.args.api=="PPVFEDCACHEAPI":
                k_silence[client] = np.zeros((self.num_batches+self.privacy_guarantee, self.args.class_num), dtype=np.complex128)
            for i in range(self.num_batches+self.privacy_guarantee):
                k_silence[client][i] = np.tensordot(np.array([self.X_to_encode[index][i] for index in self.sample_connect[client]]), self.w_init[0], axes=(0, 0))
                # k_silence[client][i] = sum(self.X_to_encode[index][i] for index in self.client_connections[client])
            k_silence_sum[client]=np.sum(k_silence[client][:self.num_batches], axis=0)
        return k_silence,k_silence_sum