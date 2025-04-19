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
        self.num_workers = len(self.workers_id)  # N
        self.num_stragglers = num_stragglers  # S
        self.security_guarantee = security_guarantee  # A
        self.privacy_guarantee = privacy_guarantee  # T
        self.prime_number = prime_number
        self.precision = precision
        self.fit_intercept = fit_intercept
        self.num_workers_origin=num_workers_origin
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
        n_alphas = self.num_workers_origin
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
        self.sum_k=np.mod(np.sum(self.Y_decoded_global_deq[:self.num_batches], axis=0),self.prime_number)/self.num_workers
        self.sum_T=np.mod((np.sum(self.Y_decoded_global_deq[-self.privacy_guarantee:], axis=0)),self.prime_number)
        return self.Y_decoded_global_deq,self.sum_k,self.sum_T

        # {keystr1: {client1: data,
        #            client2: data,
        #            }
        #  keystr2: {client1: data,
        #            client2: data,
        #             }
        # }

    #真实X数据集合
    def debug_Lcc(self):
        return sum([u for u in self.X.values()])