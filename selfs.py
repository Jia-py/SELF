import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import random

class selfs(nn.Module):

    def __init__(self, args, unique_values, features):
        super(selfs, self).__init__()

        self.feature_num = len(unique_values)
        self.device = args.device
        self.args = args
        self.features = np.array(features)

        if args.dataset == 'movielens-1m':
            gpt4o = np.load('selected_features_gpt-4o_movielens.npy').tolist()
            gpt4 = np.load('selected_features_gpt-4-turbo_movielens.npy').tolist()
            gpt35 = np.load('selected_features_gpt-3.5-turbo-instruct_movielens.npy').tolist()
            gemini15pro = np.load('selected_features_gemini-1.5-pro_movielens.npy').tolist()
        elif args.dataset == 'aliccp':
            gpt4o = np.load('selected_features_gpt-4o_aliccp.npy').tolist()
            gpt4 = np.load('selected_features_gpt-4-turbo_aliccp.npy').tolist()
            gpt35 = np.load('selected_features_gpt-3.5-turbo-instruct_aliccp.npy').tolist()
            gemini15pro = np.load('selected_features_gemini-1.5-pro_aliccp.npy').tolist()
        elif args.dataset == 'kuairand-pure':
            gpt4o = np.load('selected_features_gpt-4o_kuairand.npy').tolist()
            gpt4 = np.load('selected_features_gpt-4-turbo_kuairand.npy').tolist()
            gpt35 = np.load('selected_features_gpt-3.5-turbo-instruct_kuairand.npy').tolist()
            gemini15pro = np.load('selected_features_gemini-1.5-pro_kuairand.npy').tolist()
            # remove features that not in self.features
            for lis in [gpt4o, gpt4, gpt35]:
                for ele in lis:
                    if ele not in self.features:
                        lis.remove(ele)
            for lis in [gpt4o, gpt4, gpt35]:
                for ele in lis:
                    if ele not in self.features:
                        lis.remove(ele)
            print(len(gpt4o), len(gpt4), len(gpt35))

        llm_ls = np.array([gpt4o, gpt4, gpt35])
        print(llm_ls.shape)
        feature_weight = np.zeros_like(llm_ls, dtype=float)
        for row_idx in range(llm_ls.shape[0]):
            for col_idx in range(llm_ls.shape[1]):

                index = np.where(llm_ls[row_idx] == self.features[col_idx])[0][0]
                feature_weight[row_idx][col_idx] = 1 - index * (1/self.feature_num)

                # feature_weight[row_idx][col_idx] = 1
        print(feature_weight)
        self.mask_ratio = 0.2

        self.feature_weight = torch.Tensor(feature_weight).to(args.device) # n, feature_num

        self.agent_weight_bb = nn.Parameter(torch.zeros(3,1).to(args.device))
        self.t_bb = torch.tensor([4.0]).to(args.device)

        self.batchnorm_bb = nn.BatchNorm1d(args.embedding_dim)

        self.mode = 'train'
        self.optimizer_method = 'normal'
        self.load_checkpoint = False

    def forward(self, x, current_epoch, current_step, raw_data):
        b,f,e = x.shape
        if self.mode == 'test':
            return x
        x = self.batchnorm_bb(x.transpose(1,2)).transpose(1,2)
        agent_weight = torch.softmax(self.agent_weight_bb * torch.exp(self.t_bb), dim=0).reshape(-1,1) # n,1
        batch_mask = torch.ones_like(self.feature_weight).to(self.device)
        
        for row_idx in range(self.feature_weight.shape[0]):
            random_number = random.randint(1, int(self.feature_num * self.mask_ratio))
            sorted_indices = torch.argsort(self.feature_weight[row_idx], descending=True)
            batch_mask[row_idx, sorted_indices[-random_number:]] = 0
        print(batch_mask)

        gate = torch.sum(agent_weight * batch_mask, dim=0, keepdim=True) # 1, feature_num
        x = x * gate.reshape(1, -1, 1)
        print(agent_weight)
        return x