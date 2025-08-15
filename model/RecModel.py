import torch
from torch import nn
from torch.nn import TransformerEncoder
import torch.nn.functional as F
import math
from functools import partial
import os
import world

import sys

def create_model(world, n_items, n_users, pad, interGraph):
    model = DiscreteDenoiseRecModel(world, n_items, n_users, pad, interGraph)
    target_model = TargetDenoiseRecModel(world, pad)
    return model, target_model

class DiscreteDenoiseRecModel(nn.Module):
    def __init__(self,
                 world,
                 num_items,
                 num_users,
                 padding_idx,
                 interGraph):
        super(DiscreteDenoiseRecModel, self).__init__()
        print("Initialize Rec Model")
        
        self.world = world
        self.num_items = num_items
        self.num_users = num_users
        self.padding_idx = padding_idx
        self.latent_dim = world.args.emb_dim
        
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items+1, embedding_dim=self.latent_dim)
        self.embedding_user = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)

        self.denoiser_layer = torch.nn.TransformerEncoderLayer(
                                d_model = self.latent_dim,
                                nhead = self.world.config_denoiser['nhead'],
                                dim_feedforward= self.world.config_denoiser['dim_feedforward'],
                                batch_first = True,
                                )
        self.denoiser = TransformerEncoder(self.denoiser_layer, num_layers=self.world.config_denoiser['num_layers'])

        self.aggregation_type = self.world.args.aggregation_type
        self.timestep = self.world.args.timestep

        self.f = partial(F.softmax, dim=1)

        self.interGraph = interGraph.to(self.world.device)
        self.inter_n_layers = self.world.args.ui_n_layers
        self.ui_dropout = self.world.args.ui_dropout
        self.keep_prob_ui = self.world.args.keep_prob_ui

        self.__init_weight()

    def __init_weight(self):
        path = 'model/'+self.world.args.dataset+'/'+str(self.world.args.weight_decay)+str(self.world.args.emb_dim)+str(self.world.args.ui_n_layers) + str(self.world.pretrain_epoch) +'embedding_weights.pth'
        print(path)
        
        if os.path.exists(path):
            # self.embedding_item.weight.data = torch.load(path)['embedding_item_weight']
            # self.embedding_user.weight.data = torch.load(path)['embedding_user_weight']
            weights = torch.load(path)
            self.embedding_user.weight.data.copy_(weights['embedding_user_weight'])
            self.embedding_item.weight.data.copy_(weights['embedding_item_weight'])
            world.pretrain_epoch = 0
            print('Use pre-trained embeddings of users and items')
        else:
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            print('Use NORMAL distribution initilizer for embeddings of users')   
        
    def get_padding(self):
        return self.embedding_item.weight[self.padding_idx]
    
    def __dropout(self, keep_prob):
        size = self.interGraph.size()
        index = self.interGraph.indices().t()
        values = self.interGraph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        # g = torch.sparse.FloatTensor(index.t(), values, size)
        g = torch.sparse_coo_tensor(index.t(), values, size)
        return g
    
    def InterEncode(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight[:self.padding_idx]  ########
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.training and self.ui_dropout:
            g_droped = self.__dropout(self.keep_prob_ui)
        else:
            g_droped = self.interGraph  
        
        for layer in range(self.inter_n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        self.users, self.items = torch.split(light_out, [self.num_users, self.num_items])
        return self.users, self.items
    
    def get_bpr_embeddings(self, users, pos_items, neg_items):
        all_users, all_items = self.InterEncode()
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb,\
                users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def getItemEmbedding(self, items):
        all_items = torch.cat([self.items, self.get_padding().unsqueeze(0)], dim=0)
        items_emb = all_items[items.long()]
        return items_emb

    def get_user_embeddings(self, users):
        all_users, _ = self.InterEncode()
        users_emb = all_users[users.long()]
        return users_emb

    def get_embedding(self, x_t, t, x_t2, t2, users_emb):
        mask = (x_t == self.padding_idx).bool()
        mask2 = (x_t2 == self.padding_idx).bool()
        # all_users, all_items = self.InterEncode()
        all_items = torch.cat([self.items, self.get_padding().unsqueeze(0)], dim=0)
        
        # users_emb = all_users[users]
        items_emb_i = all_items[x_t.long()]
        items_emb_i2 = all_items[x_t2.long()]

        if self.timestep == 'none':
            items_emb_i = torch.cat([users_emb.unsqueeze(1), items_emb_i], dim=1)
            prefix = torch.zeros((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)
            mask = torch.cat([prefix, mask], dim=1)

            items_emb_i2 = torch.cat([users_emb.unsqueeze(1), items_emb_i2], dim=1)
            prefix2 = torch.zeros((mask2.shape[0],1), dtype=mask2.dtype, device=mask2.device)
            mask2 = torch.cat([prefix2, mask2], dim=1)

        else:
            t_emb = timestep_embedding(t, self.latent_dim, self.world.device).unsqueeze(1)
            t_emb2 = timestep_embedding(t2, self.latent_dim, self.world.device).unsqueeze(1)

            if self.timestep == 'layerwise':
                users_emb = users_emb.unsqueeze(1) + t_emb
                items_emb_i = torch.cat([users_emb, items_emb_i], dim=1)
                prefix = torch.zeros((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)
                mask = torch.cat([prefix, mask], dim=1)
                
                users_emb2 = users_emb + t_emb2
                items_emb_i2 = torch.cat([users_emb2, items_emb_i2], dim=1)
                prefix2 = torch.zeros((mask2.shape[0],1), dtype=mask2.dtype, device=mask2.device)
                mask2 = torch.cat([prefix2, mask2], dim=1)
            elif self.timestep == 'token':
                items_emb_i = torch.cat([users_emb.unsqueeze(1), t_emb, items_emb_i], dim=1)
                prefix = torch.zeros((mask.shape[0], 2), dtype=mask.dtype, device=mask.device)
                mask = torch.cat([prefix, mask], dim=1)
                
                items_emb_i2 = torch.cat([users_emb2.unsqueeze(1), t_emb2, items_emb_i2], dim=1)
                prefix2 = torch.zeros((mask2.shape[0],2), dtype=mask2.dtype, device=mask2.device)
                mask2 = torch.cat([prefix2, mask2], dim=1)
            else:
                raise NotImplementedError
        return items_emb_i, mask, items_emb_i2, mask2
        
    def forward(self, x_t, mask):

        if mask is not None:
            if mask.shape == x_t.shape:
                mask = mask.any(dim=-1)
            else:
                mask = mask
            model_output = self.denoiser(x_t, src_key_padding_mask=mask)
        else:
            model_output = self.denoiser(x_t)
        return model_output

    def item_decoder(self, item_emb, log_operation=False, softmax=True):
        logits = torch.matmul(item_emb, self.items.T)

        if log_operation:
            return F.log_softmax(logits, dim=2)
        else:
            if softmax:
                return F.softmax(logits, dim=2)
            else:
                return logits

    def get_embeddings_neg(self, items, c):
        # items_emb = torch.cat([self.items, self.get_padding().unsqueeze(0)], dim=0)
        items_emb = self.items
        items_emb_i = items_emb[items.long()]
        items_emb_i = torch.cat([c, items_emb_i], dim=1)
        return items_emb_i

    def get_embeddings_neg2(self, items):
        items_emb = self.items
        items_emb_i = items_emb[items.long()]
        emb = items_emb_i.mean(dim=1, keepdim=True)
        items_emb_i = torch.cat([emb, items_emb_i], dim=1)
        
        return items_emb_i
    
    def test(self, items):
        emb = self.getItemEmbedding(items)
        emb = torch.mean(emb, dim=1, keepdim=True)
        
        return emb



    '''
    def getItemEmbedding(self, items, t, users_emb_i):
        mask = (items != self.padding_idx).long()  #################################################################################
        mask = mask.unsqueeze(-1).expand(-1, -1, self.latent_dim)
        
        items_emb = self.embedding_item.weight
        items_emb_i = items_emb[items.long()]

        # users_emb_i = self.getUserEmbedding(items_emb_i, mask)

        if self.timestep == 'none':
            items_emb_i = torch.cat([users_emb_i, items_emb_i], dim=1)
            prefix = torch.ones((mask.shape[0], 1, mask.shape[-1]), dtype=mask.dtype, device=mask.device)
            mask = torch.cat([prefix, mask], dim=1)
        else:
            t_emb = timestep_embedding(t, self.latent_dim, items.device).unsqueeze(1)
            if self.timestep == 'layerwise':
                users_emb_i = users_emb_i + t_emb
                items_emb_i = torch.cat([users_emb_i, items_emb_i], dim=1)
                prefix = torch.ones((mask.shape[0], 1, mask.shape[-1]), dtype=mask.dtype, device=mask.device)
                mask = torch.cat([prefix, mask], dim=1)

            elif self.timestep == 'token':
                items_emb_i = torch.cat([users_emb_i, t_emb, items_emb_i], dim=1)
                prefix = torch.ones((mask.shape[0], 2, mask.shape[-1]), dtype=mask.dtype, device=mask.device)
                mask = torch.cat([prefix, mask], dim=1)
            else:
                raise NotImplementedError
        return items_emb_i, mask

    def forward(self, x_t, mask):

        if mask is not None:
            if mask.shape == x_t.shape:
                mask = mask.any(dim=-1).bool()
            else:
                mask = mask.bool()
            model_output = self.denoiser(x_t, src_key_padding_mask=mask)
        else:
            model_output = self.denoiser(x_t)
        return model_output
    
    def getUserEmbedding(self, users_emb_i, mask):
        if self.aggregation_type == 'sum':
            emb = (users_emb_i * mask).sum(dim=1, keepdim=True)
            # print(emb.shape)
        elif self.aggregation_type == 'mean':
            print(users_emb_i[9])
            print(mask[9])

            mask_sum = (users_emb_i * mask).sum(dim=1)
            valid_counts = mask.sum(dim=1)
            emb = (mask_sum / valid_counts).unsqueeze(1)

            print("any NaN:", torch.isnan(emb).any())
            print("any Inf:", torch.isinf(emb).any())
            nan_mask = torch.isnan(emb)
            nan_indices = torch.nonzero(nan_mask)
            print(nan_indices)
            sys.exit()
        else:
            raise NotImplementedError
        return emb

    
    def get_embeddings(self, items, mask = None, user_emb_only = False):
        if mask is not None:
            mask = mask.unsqueeze(-1).expand(-1, -1, self.latent_dim)
        
        items_emb = self.embedding_item.weight
        items_emb_i = items_emb[items.long()]

        if self.aggregation_type == 'sum':
            if mask is not None:
                emb = (items_emb_i * mask).sum(dim=1, keepdim=True)
            else:
                emb = items_emb_i.sum(dim=1, keepdim=True)
        elif self.aggregation_type == 'mean':
            if mask is not None:
                mask_sum = (items_emb_i * mask).sum(dim=1)
                valid_counts = mask.sum(dim=1)
                emb = (mask_sum / valid_counts).unsqueeze(1)
            else:
                emb = items_emb_i.mean(dim=1, keepdim=True)
        
        items_emb_i = torch.cat([emb, items_emb_i], dim=1)

        if user_emb_only:
            return emb
        else:
            return items_emb_i
    
    def get_embeddings_neg(self, items, c):
        items_emb = self.embedding_item.weight
        items_emb_i = items_emb[items.long()]
        items_emb_i = torch.cat([c, items_emb_i], dim=1)
        return items_emb_i
    '''

    def emb_to_item(self, emb):
        emb_flat = emb.reshape(-1, self.latent_dim)

        emb_flat_norm = F.normalize(emb_flat, p=2, dim=1)
        item_emb_norm = F.normalize(self.items, p=2, dim=1)

        # similarity = torch.matmul(emb_flat_norm, self.items.T)
        similarity = torch.matmul(emb_flat_norm, item_emb_norm.T)
        pred_ids = torch.argmax(similarity, dim=1)

        # pred_ids = pred_ids.view(1024, 10)
        return pred_ids






class TargetDenoiseRecModel(nn.Module):
    def __init__(self,
                 world,
                 padding_idx):
        super(TargetDenoiseRecModel, self).__init__()
        # print("Initialize Target Model")

        self.world = world
        self.padding_idx = padding_idx
        self.latent_dim = world.args.emb_dim

        self.denoiser_layer = torch.nn.TransformerEncoderLayer(
                                d_model = self.latent_dim,
                                nhead = self.world.config_denoiser['nhead'],
                                dim_feedforward= self.world.config_denoiser['dim_feedforward'],
                                batch_first = True,
                                norm_first = True
                                )
        self.denoiser = TransformerEncoder(self.denoiser_layer, num_layers=self.world.config_denoiser['num_layers'])

    def forward(self, x_t, mask):
        mask = None
        if mask is not None:
            mask = mask.any(dim=-1).bool()
            model_output = self.denoiser(x_t, src_key_padding_mask=mask)
        else:
            model_output = self.denoiser(x_t)
        return model_output

def timestep_embedding(timesteps, dim, device, max_period=10000):
    half = dim // 2
    seq = torch.arange(start=0, end=half, dtype=torch.float32)
    freqs = torch.exp(
        -math.log(max_period) *  seq/ half
    ).to(device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding