from contextlib import suppress
import pickle
import pandas as pd
import random
import os
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


import sys

def load_data(dataset):
    print('Dataset: ' + dataset)
    if dataset == 'ciao':
        return CiaoDataLoader("data/ciao/", 'train'), CiaoDataLoader("data/ciao/", 'test')
    if dataset == 'Epinions':
        return EpinionsDataLoader("data/Epinions/", 'train'), EpinionsDataLoader("data/Epinions/", 'test')
    elif dataset == 'Dianping':
        return Dianping("data/Dianping/", 'train'), Dianping("data/Dianping/", 'test')
    elif dataset == 'ml-1m':
        return ML_1m("data/ml-1m/", 'train'), ML_1m("data/ml-1m/", 'test')
    elif dataset == 'ml-10m':
        return ML_1m("data/ml-10m/", 'train'), ML_1m("data/ml-10m/", 'test')
    else:
        raise NotImplementedError


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def n_items(self):
        raise NotImplementedError
    
    @property
    def pad(self):
        raise NotImplementedError
    
    def load_item_freq(self, path):
        item_freq_dict = {}
        with open(path, 'r') as f:
            for line in f:
                item_id, freq = line.strip().split()
                item_freq_dict[int(item_id)] = int(freq)
        item_freq_dict[self.pad] = 0
        
        item_freq_tensor = torch.zeros(self.n_items+2)
        for item_id, freq in item_freq_dict.items():
            item_freq_tensor[item_id] = freq
        return item_freq_tensor
    
    def item_freq_preprocess_fn(self, wf):
        wf = wf + 1
        wf = wf.log()
        wf = wf / wf.max()
        # range: 0 - 1
        return wf
    
    
class CiaoDataLoader(BasicDataset):
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.load_info(path+'/info.pkl')
        self.item_freq_tensor = self.load_item_freq(path+'/item_freq_3.txt')
        self.item_freq_tensor = self.item_freq_preprocess_fn(self.item_freq_tensor)
        
        if mode == 'train':
            print('Loading train data')
            print('Number of users: ', self.num_users, 'Number of items: ', self.num_items)
        train_file = path + '/train_3.txt'
        train_data = []
        trainUser, trainItem = [], []
        self.allPos = {}
        self.candidate_neg_items = {}
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    processed = self.process_item_ids(uid, items, mode)
                    train_data.extend(processed)
                    self.allPos[uid] = items
                    if self.mode == 'train':
                        trainUser.extend([uid] * len(items))
                        trainItem.extend(items)
        
        self.train_data = pd.DataFrame(train_data, columns=['userID', 'ItemIDs'])

        self.train_data['ItemFreq'] = self.train_data['ItemIDs'].apply(
            lambda item_list: self.item_freq_tensor[torch.tensor(item_list)].tolist())
        self.train_data['Mask'] = self.train_data['ItemIDs'].apply(
            lambda items: [0 if item == self.pad else 1 for item in items])
        
        self.trainUser = np.array(trainUser)  # train user -------------------------------
        self.trainItem = np.array(trainItem)  # train item -------------------------------

        if self.mode == 'train':
            self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.num_users, self.num_items))
            self.interactionGraph = None
            self.getInterGraph()
            print('Train data loaded')
        

        if self.mode == 'test':
            print('Load test data')
            test_file = path + '/test_3.txt'
            test_data = []
            with open(test_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        if len(items) > 0:
                            test_data.extend([(uid, items)])
            self.test_data = pd.DataFrame(test_data, columns=['userID', 'ItemIDs'])

            user_item_dict = self.train_data.groupby('userID')['ItemIDs'].sum().to_dict()
            self.test_data['train_data'] = self.test_data['userID'].map(user_item_dict)

            self.test_data['ItemFreq'] = self.test_data['train_data'].apply(
            lambda item_list: self.item_freq_tensor[torch.tensor(item_list)].tolist())

            self.test_data['allPos'] = self.test_data['userID'].map(self.allPos)

            # print(self.test_data.head(10))
            # has_nan = self.test_data.isna().any().any()
            # print(has_nan)
            
            print(f"UI-Graph Sparsity : {(len(self.train_data) + len(self.test_data)) / self.n_users / self.n_items}")
                        
        
    def load_info(self, path):
        with open(path, 'rb') as f:
            info = pickle.load(f)
        # print(info)
        self.num_users = info['num_users']
        self.num_items = info['num_items']
        self.pad_item = info['pad_item']

        self.num_users = self.num_users + 1
        self.num_items = self.num_items + 1
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.train_data.iloc[idx]
        else:
            raise NotImplementedError
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        else:
            raise NotImplementedError
    
    @property
    def length(self):
        return 20
    
    @property
    def n_users(self):
        return self.num_users
    
    @property
    def n_items(self):
        return self.num_items
    
    @property
    def pad(self):
        return self.pad_item
    
    def process_item_ids(self, user_id, item_ids, mode):
        if mode == 'train':
            # all_item_set = set(range(self.n_items))
            # candidate_neg_items_ = list(all_item_set - set(item_ids))
            item_ids_ = np.array(item_ids)
            candidate_neg_items_ = np.setdiff1d(np.arange(self.n_items), item_ids_, assume_unique=True)

            if user_id not in self.candidate_neg_items:
                self.candidate_neg_items[user_id] = candidate_neg_items_
            if len(item_ids) > self.length:
                num_groups = len(item_ids) // self.length
                split_items = [item_ids[i * self.length: (i + 1) * self.length] for i in range(num_groups)]
                return [(user_id, group) for group in split_items]
            elif len(item_ids) < self.length and len(item_ids)>0:
                return [(user_id, item_ids + [self.pad] * (self.length - len(item_ids)))]
            return [(user_id, item_ids)]

        if mode == 'test':
            if len(item_ids) > self.length:
                item_ids = random.sample(item_ids, self.length)
            elif len(item_ids) < self.length and len(item_ids)>0:
                return [(user_id, item_ids + [self.pad] * (self.length - len(item_ids)))]
            return [(user_id, item_ids)]
    
    def getInterGraph(self):
        if self.interactionGraph is None:
            if os.path.exists(self.path + '/adj_mat.npz'):
                norm_adj = sp.load_npz(self.path + '/adj_mat.npz')

            else:
                # build up matrix with n.users+n.items, n.users+n.items
                adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R  
                adj_mat[self.n_users:, :self.n_users] = R.T  

                adj_mat = adj_mat.todok()
                
                # calculate D
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                sp.save_npz(self.path + '/adj_mat.npz', norm_adj)
                
            self.interactionGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.interactionGraph = self.interactionGraph.coalesce()

        return self.interactionGraph
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)

        # return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

class EpinionsDataLoader(CiaoDataLoader):
    def __init__(self, path, mode):
        super().__init__(path, mode)

    @property
    def length(self):
        return 5  #5


class ML_1m(CiaoDataLoader):
    def __init__(self, path, mode):
        super().__init__(path, mode)

    @property
    def length(self):
        return 10 # 10 #  30  # 20

class ML_10m(CiaoDataLoader):
    def __init__(self, path, mode):
        super().__init__(path, mode)

    @property
    def length(self):
        return 20  
    
class Dianping(CiaoDataLoader):
    def __init__(self, path, mode):
        super().__init__(path, mode)

    @property
    def length(self):
        return 20 # 10 #  30  # 20