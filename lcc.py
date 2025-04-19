import numpy as np
import copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.random_projection import SparseRandomProjection
import matplotlib.pyplot as plt
import torch

def rel_err(approxs, trues, ord=None):
    rel_err_list=[]
    for client in range(len(approxs)):
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

def split(X,num_batches,args):
    X_split = {key: X[key].copy() for key in X.keys()}  # Avoid deep copy
    if args.api=="PPVFDAPI":
        for key in X_split.keys():
            rows, cols = X_split[key].shape
            k_parts = np.array([np.zeros((rows, cols), dtype=np.complex128) for _ in range(num_batches)])
            for i in range(rows):
                for j in range(cols):
                    value = X_split[key][i][j]
                    if value != 0:  # 确保有值进行分割
                        splits = np.sort(np.random.rand(num_batches - 1)) * value
                        splits = np.concatenate(([0.0], splits, [value]))
                        for part_index in range(num_batches):
                            k_parts[part_index][i][j] = splits[part_index + 1] - splits[part_index]
                    else:
                        for part_index in range(num_batches):
                            k_parts[part_index][i][j] = 0
            X_split[key] = k_parts
    elif args.api=="PPVFMDAPI" or args.api=="PPVDSFLAPI":
        for key in X_split.keys():
            rows, cols = X_split[key][0].shape
            k_parts = np.array([np.zeros((rows, cols), dtype=np.complex128) for _ in range(num_batches)])
            for bat_idx in range(num_batches):
                k_parts[bat_idx] = np.array(X_split[key][bat_idx].cpu().numpy(), dtype=np.complex128)
            X_split[key]=k_parts
    return X_split

def concatenate(X_split,privacy_guarantee,prime_number=2**25-39,theta=6,sigma=None):
    X_to_encode = {key: X_split[key].copy() for key in X_split.keys()}  # Avoid deep copy
    for key in X_to_encode.keys():
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
            else:
                Z = np.random.randint(0, prime_number, (privacy_guarantee, *X_to_encode[key].shape[1:])).astype("int64")
                X_to_encode[key] = np.concatenate([X_to_encode[key], Z])
    return X_to_encode

def hash_map(logits,n_components=7,random_state=42):
    # 使用LSH进行哈希映射
    lsh = SparseRandomProjection(n_components=n_components,random_state=random_state)
    lsh.fit(logits)
    # 对样本的每个logits向量进行变换
    hashed_logits = lsh.transform(logits)
    return hashed_logits


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

class LCCOracle():
    def __init__(
        self,
        X,
        prime_number,
        G,
        num_workers_origin,
        k_regular_graph,
        num_stragglers=5,
        security_guarantee=2,
        privacy_guarantee=2,
        precision=16,
        fit_intercept=False,
    ):
        self.X = np.hstack((np.ones((len(X), 1)), X)) if fit_intercept else X
        self.G=G
        self.workers_id = list(G.nodes())
        self.num_workers = num_workers_origin  # N
        self.num_stragglers = num_stragglers  # S
        self.security_guarantee = security_guarantee  # A
        self.privacy_guarantee = privacy_guarantee  # T
        self.prime_number = prime_number
        self.precision = precision
        self.fit_intercept = fit_intercept
        self.k_regular_graph=k_regular_graph

        self.init_num_batches()  # 确定K
        self.init_client_connections() #确定交互规则
        self.init_weight_sum()    #确定权重
        self.init_client_connection() #确定关系列表

        #数据转换成伽罗域的整数
        self.Xq = quantize(self.X, precision, prime_number)
        #对量化后的结果进行分割成K份
        self.Xq_split=split(self.Xq,self.num_batches)
        #生成为每个客户端生成扰动矩阵
        self.X_to_encode=concatenate(self.Xq_split,self.privacy_guarantee, self.prime_number)

    # 初始化K的值
    def init_num_batches(self):
        self.num_batches=10 # K=10

    # def init_num_batches(self):      #初始化K的值
    #     self.poly_degree = 1  # X.T.dot(X)
    #     self.num_batches = (
    #         (self.num_workers - 1 - 2 * self.security_guarantee - self.num_stragglers)
    #         // self.poly_degree
    #         + 1
    #         - self.privacy_guarantee
    #     )      #计算K值
    #     m, _ = self.X.shape
    #     while m % self.num_batches:
    #         self.num_batches -= 1    #保证满足约束的情况下K能把原始数据整分
    #     self.n_bound = (
    #         self.num_batches + self.privacy_guarantee - 1
    #     ) * self.poly_degree + 1     #N-S-2A
    #     if self.verbose:
    #         print(f"Initialized K={self.num_batches} batches")
    #         print(f"Number of succedeed to restore correctly: N>={self.n_bound}")

    def init_client_connections(self):
        self.client_connections={node: sorted(list(self.G.neighbors(node))+[node]) for node in self.G.nodes()}

    #设定客户端聚合权重
    def init_weight_sum(self):
        self.aggregate_weight=[1 for i in range(len(self.X))]

    def init_client_connection(self):
        # for i, client_index in self.workers_id:
        #     client_connections = self.client_connections[client_index]
        #     client_connections_list = sorted(client_connections.append(client_index))
        #     self.client_connections[client_index]=client_connections_list
        #向关系列表中添加本人并排序
        self.neighbors_to_keep={}
        for i, node in enumerate(self.workers_id):
            # 获取前 k 个邻居的索引，按距离当前节点的顺序排列
            neighbors_to_keep_indices_before = [(i - j) % len(self.workers_id) for j in range(self.k_regular_graph, 0, -1)]
            # 当前节点的索引
            current_node_index = [i]
            # 获取后 k 个邻居的索引，按距离当前节点的顺序排列
            neighbors_to_keep_indices_after = [(i + j) % len(self.workers_id) for j in range(1, self.k_regular_graph + 1)]
            # 合并前当前和后邻居的索引并取值
            neighbors_to_keep_indices = neighbors_to_keep_indices_before + current_node_index + neighbors_to_keep_indices_after
            # self.neighbors_to_keep[node] =self.workers_id[neighbors_to_keep_indices]
            self.neighbors_to_keep[node] =[self.workers_id[u] for u in neighbors_to_keep_indices]
    # 对数据集合进行拉格朗日加密划分 得到编码后的X  #拉格朗日加密 得到N x (m // K) x d
    def Lcc_encoded(self):      #对数据集合进行拉格朗日加密划分
        n_betas = self.num_batches + self.privacy_guarantee
        n_alphas = self.num_workers
        #保证两者不相交
        self.alphas ={}
        self.betas = (np.arange(n_alphas,n_alphas+n_betas)).astype("int64")
        self.U={}
        self.X_encoded={}
        for client_index in self.workers_id:
            self.alphas[client_index]=np.array(self.client_connections[client_index]).astype("int64")
            self.U[client_index] = gen_Lagrange_coeffs(self.alphas[client_index], self.betas, self.prime_number)
            self.X_encoded[client_index] = np.mod(np.einsum("kn,kmd->nmd", self.U[client_index], self.X_to_encode[client_index]), self.prime_number)  # N x (m // K) x d m样本行数 10000-->class_num
            converted_dict = {}
            for i, client_id in enumerate(self.client_connections[client_index]):
                converted_dict[client_id]= self.X_encoded[client_index][i]
            self.X_encoded[client_index]=converted_dict
            #数据是两层词典、最内层为二维list表示 a->b的data
        return self.X_encoded


    def encoded_knowledge_distribute(self):
        self.X_received={}
        for client_distribute, distribute_data in self.X_encoded.items():
            for client_receive, data in distribute_data.items():
                if client_receive not in self.X_received:
                    self.X_received[client_receive] = {}
                    # 将数据添加到X_encoded_received中
                self.X_received[client_receive][client_distribute] = data
        return self.X_received

    def worker_fun(self):
        #客户端本地按照正则树指导进行聚合
        #两层词典 外层是client 内层是 该客户端收到的所有次序的求和
        self.local_knowledge_upload={}
        self.strkey_to_member={}
        for i, node in enumerate(self.workers_id):
            self.local_knowledge_upload[node] = {}
            # 使用滑窗技术处理 neighbors_to_keep_indices
            for j in range(self.k_regular_graph + 1):
                # 获取滑窗内的 k+1 个数
                window = [self.workers_id[u] for u in self.neighbors_to_keep[node][j:j + self.k_regular_graph +1]]
                # 排序并拼接作为键
                key_list = sorted(window)
                str_key = ''.join(map(str, key_list))
                self.strkey_to_member[str_key]=key_list
                # 将键值对添加到字典中，值为空
                self.local_knowledge_upload[node][str_key] =np.mod(sum(self.X_received[node][index] for index in key_list),self.prime_number)
        return self.local_knowledge_upload

    #加密后再解密恢复的数据集合
    def Lcc_decoded(self):
        #从 有“key”str的词典中找到客户端id作为插值
        succeeded_workers = np.random.choice(
            self.num_workers, self.num_workers - self.num_stragglers, replace=False
        )
        Y_received={}        #是对self.local_knowledge_upload格式的转换
        self.Y_decoded={}
        self.U_dec={}

        #转换格式 词典嵌套 str_key client
        for client_index , client_upload_data in self.local_knowledge_upload.items():
            for str_key ,data in client_upload_data.items():
                if str_key not in Y_received:
                    Y_received[str_key]={}
                Y_received[str_key][client_index]=data
        #把Y_received[str_key]转换成list
        for str_key , sub_dict in Y_received.items():
            sub_list=np.stack([np.array(sub_dict[ordered_index]) for ordered_index in self.strkey_to_member[str_key]])
            Y_received[str_key]=sub_list

        for str_key , member in self.strkey_to_member.items():
            if self.security_guarantee > 0:
                malicious_workers = np.random.choice(
                    list(set(range(self.num_workers)) - set(succeeded_workers)),
                    self.security_guarantee,
                    replace=False,
                )
                raise NotImplementedError("Adversarial decoding not yet supported!")
            else:
                self.U_dec[str_key] = gen_Lagrange_coeffs(
                    self.betas,
                    member,
                    self.prime_number,
                )
                self.Y_decoded[str_key] = np.mod(
                    np.einsum("nk,nld->kld", self.U_dec[str_key], Y_received[str_key]), self.prime_number
                )    #k+t* ld

        self.Y_decoded_global=np.mod(sum([u for u in self.Y_decoded.values()])/self.k_regular_graph,self.prime_number)
        self.Y_decoded_global_deq = dequantize(self.Y_decoded_global, self.precision, self.prime_number)
        self.sum_k=np.mod(np.sum(self.Y_decoded_global_deq[:self.num_batches], axis=0),self.prime_number)
        self.sum_T=np.mod((np.sum(self.Y_decoded_global_deq[-self.privacy_guarantee:], axis=0)),self.prime_number)
        return self.Y_decoded_global_deq,self.sum_k,self.sum_T

    #真实X数据集合
    def debug_Lcc(self):
        return sum([u for u in self.X.values()])


class ALCCOracle():
    """
    ALCC oracle for PPVFD
    """

    def __init__(
        self,
        X,
        num_workers_origin,
        args,
        client_connection,
        num_stragglers,
        security_guarantee,
        privacy_guarantee,
        beta,
        sigma,
        theta,
        sim_matrix,
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
        self.fit_intercept = fit_intercept
        self.workers_id = [i for i in range(num_workers_origin)]
        self.args = args

        self._init_num_batches()  # 确定K
        self._init_w()
        self.init_client_connections()

        # print("num_batches:{}".format(self.num_batches))
        # 对量化后的结果进行分割成K份
        self.X_split = split(self.X, self.num_batches,self.args)
        # 生成为每个客户端生成扰动矩阵
        self.X_to_encode = concatenate(self.X_split, self.privacy_guarantee, theta=self.theta,sigma=self.sigma)

    def _init_w(self):
        self.w_init={}
        for client_index in self.workers_id:
            self.w_init[client_index]=np.ones(self.args.R+1)

    # def _init_w(self):
    #     self.w_init = {}
    #     self.tem_weight={}
    #     alpha=0.1
    #     for client_index in self.workers_id:
    #         neighbors = self.client_connection[client_index]
    #         sim_scores = np.array([self.sim_matrix[client_index, j] for j in neighbors])  # 获取相似度分数
    #         # 计算未归一化的权重：w_j = e^(-α(1 - a_ij))
    #         unnormalized_weights = np.exp(-alpha * (1 - sim_scores))
    #
    #         # 归一化邻居的权重（不包括客户端本人）
    #         normalized_weights = unnormalized_weights / np.sum(unnormalized_weights)
    #         # print("邻居节点的权重：{}".format(normalized_weights))
    #
    #         # 生成客户端i本人的随机权重
    #         self.tem_weight[client_index]=np.random.rand()
    #
    #         # 合并客户端i本人及其邻居的编号
    #         all_ids = np.append(neighbors, client_index)
    #         all_weights = np.append(normalized_weights, self.tem_weight[client_index])
    #
    #         # 根据客户端序号升序排列
    #         sorted_indices = np.argsort(all_ids)
    #         sorted_ids = all_ids[sorted_indices]
    #         sorted_weights = all_weights[sorted_indices]
    #
    #         # 保存升序排列的权重
    #         # self.w_init[client_index] = {sorted_ids[j]: sorted_weights[j] for j in range(len(sorted_ids))}
    #         self.w_init[client_index] = np.array([sorted_weights[j] for j in range(len(sorted_ids))])
    #         self.w_init[client_index] = np.ones(self.args.R + 1)
    #     print("权重：{}".format(self.w_init))
    #

    def _init_num_batches(self):
        # self.poly_degree = 2  # X.T.dot(X)
        # self.num_batches = (
        #     (self.num_workers - 1 - 2 * self.security_guarantee - self.num_stragglers)
        #     // self.poly_degree
        #     + 1
        #     - self.privacy_guarantee
        # )
        # m, _ = self.X.shape
        # while m % self.num_batches:
        #     self.num_batches -= 1
        # self.n_bound = (
        #     self.num_batches + self.privacy_guarantee - 1
        # ) * self.poly_degree + 1
        # print(f"Initialized K={self.num_batches} batches")
        # print(f"Numbers of succeded workers to restore correctly: N>={self.n_bound}")
        if self.args.api=="PPVFDAPI":
            self.num_batches=self.args.class_num
        elif self.args.api=="PPVFMDAPI" or self.args.api=="PPVDSFLAPI":
            self.num_batches=self.args.batch_num

    def init_client_connections(self):
        self.client_connections={node: sorted(list(self.client_connection[node])+[node]) for node in self.workers_id}
        # print("关系列表：{}".format(self.client_connections))

    def ALcc_encoded(self):
        n_alphas = self.num_workers
        n_betas = self.num_batches + self.privacy_guarantee
        # print("n_betas:{}".format(n_betas))
        self.alphas_total=np.exp(2 * np.pi * 1j * np.arange(n_alphas) / n_alphas)
        self.betas = self.beta * np.exp(2 * np.pi * 1j * np.arange(n_betas) / n_betas)
        self.alphas={}
        self.U={}
        self.X_encoded={}

        #a>b-[data]
        for client_index in self.workers_id:
            self.alphas[client_index] = self.alphas_total[self.client_connections[client_index]]
            #每个客户端的矩阵
            self.U[client_index] = gen_Lagrange_coeffs(self.alphas[client_index], self.betas)
            self.X_encoded[client_index]={}
            # client_index 对应的connection（一种矩阵U）下 每个客户端的编码知识
            for client_interact_id in self.client_connections[client_index]:
                self.X_encoded[client_index][client_interact_id] = np.einsum("kn,kmd->nmd", self.U[client_index], self.X_to_encode[client_interact_id])
                converted_dict = {}
                for i, client_id in enumerate(self.client_connections[client_index]):
                    converted_dict[client_id] = self.X_encoded[client_index][client_interact_id][i]
                self.X_encoded[client_index][client_interact_id] = converted_dict
        return self.X_encoded

    def encoded_knowledge_distribute(self):
        self.X_received={}
        for client_index in self.workers_id:
            self.X_received[client_index]={}
            for client_distribute, distribute_data in self.X_encoded[client_index].items():
                for client_receive, data in distribute_data.items():
                    if client_receive not in self.X_received[client_index]:
                        self.X_received[client_index][client_receive] = {}
                        # 将数据添加到X_encoded_received中
                    self.X_received[client_index][client_receive][client_distribute]=data

            for receiver in self.X_received[client_index].keys():
                # 将接收者按照自定义顺序排序
                self.X_received[client_index][receiver]=[self.X_received[client_index][receiver][key] for key in self.client_connections[client_index]]

        return self.X_received

    def worker_fun(self):
        self.local_knowledge_upload = {}

        self.k_slice={}
        self.test={}

        for i, client_index in enumerate(self.workers_id):
            self.local_knowledge_upload[client_index] = {}
            for client_receive in self.X_received[client_index].keys():
                #给成聚合函数 加上权重
                self.local_knowledge_upload[client_index][client_receive]=np.tensordot(self.w_init[client_index],self.X_received[client_index][client_receive],axes=(0,0))
            if self.args.api=="PPVFDAPI":
                self.k_slice[client_index] = np.zeros((self.num_batches+self.privacy_guarantee, self.args.class_num,self.args.class_num ), dtype=np.complex128)
            elif self.args.api == "PPVFMDAPI" or self.args.api == "PPVDSFLAPI":
                self.k_slice[client_index] = np.zeros((self.num_batches+self.privacy_guarantee, self.args.batch_size,self.args.class_num ), dtype=np.complex128)
            for i in range(self.num_batches + self.privacy_guarantee):
                self.k_slice[client_index][i] = sum(self.X_to_encode[index][i] for index in self.client_connections[client_index])
            self.test[client_index]=np.einsum("kn,kmd->nmd", self.U[client_index], self.k_slice[client_index])

        # print("编码输出：{}".format(self.local_knowledge_upload[0][0]))
        # print("原始输出：{}".format(self.test[0][0]))
        return self.local_knowledge_upload

    def ALcc_decoded(self):
        #从 有“key”str的词典中找到客户端id作为插值
        Y_received={}        #是对self.local_knowledge_upload格式的转换
        self.Y_decoded={}
        self.Y_decoded_global={}
        self.U_dec={}
        self.sum_k={}
        self.sum_T={}

        for client_index in self.local_knowledge_upload.keys():
            Y_received[client_index]=list(self.local_knowledge_upload[client_index].values())
            if self.security_guarantee > 0:
                malicious_workers = np.random.choice(
                    list(set(range(self.num_workers)) - set(succeeded_workers)),
                    self.security_guarantee,
                    replace=False,
                )
                raise NotImplementedError("Adversarial decoding not yet supported!")
            else:
                self.U_dec[client_index] = gen_Lagrange_coeffs(
                    self.betas,
                    self.alphas_total[self.client_connections[client_index]],
                )
                self.Y_decoded[client_index] = np.einsum("nk,nld->kld", self.U_dec[client_index], Y_received[client_index])#k+t* ld

            self.sum_k[client_index]=np.sum(self.Y_decoded[client_index][:self.num_batches], axis=0)
            self.sum_T[client_index]=np.sum(self.Y_decoded[client_index][-self.privacy_guarantee:], axis=0)
        return self.Y_decoded,self.sum_k,self.sum_T

    def debug_ALcc(self):
        #这里用的x_to_encoded已经是被污染过的。
        k_silence={}
        k_silence_sum={}
        for client in self.workers_id:
            if self.args.api=="PPVFDAPI":
                k_silence[client] = np.zeros((self.num_batches+self.privacy_guarantee, self.args.class_num, self.args.class_num), dtype=np.complex128)
            elif self.args.api == "PPVFMDAPI" or self.args.api == "PPVDSFLAPI":
                k_silence[client] = np.zeros((self.num_batches+self.privacy_guarantee, self.args.batch_size, self.args.class_num), dtype=np.complex128)
            for i in range(self.num_batches+self.privacy_guarantee):
                k_silence[client][i] = np.tensordot(np.array([self.X_to_encode[index][i] for index in self.client_connections[client]]), self.w_init[client], axes=(0, 0))
                # k_silence[client][i] = sum(self.X_to_encode[index][i] for index in self.client_connections[client])
            k_silence_sum[client]=np.sum(k_silence[client][:self.num_batches], axis=0)
        return k_silence,k_silence_sum