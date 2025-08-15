import torch
import numpy as np
import random
from model.diffusion_lib  import *
import world

import sys

def set_seed(random_seed):
    torch.manual_seed(random_seed) # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed) # gpu
        torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed) # numpy
    random.seed(random_seed) # random and transforms
    torch.backends.cudnn.deterministic=True # cudnn deterministic calculation   
    print(f'Random seed: ' + str(random_seed))

def process_fn_in_collate(item_freq):
    return item_freq-item_freq.mean()
    
def collate_fn(batch_input):
    users = torch.stack([torch.tensor(d['userID']) for d in batch_input])
    items = torch.stack([torch.tensor(d['ItemIDs']) for d in batch_input])
    # item_freq = torch.stack([torch.tensor(process_fn_in_collate(torch.tensor(d['ItemFreq']))) for d in batch_input])
    item_freq = torch.stack([ process_fn_in_collate(torch.tensor(d['ItemFreq'])).detach()  for d in batch_input])
    mask = torch.stack([torch.tensor(d['Mask']) for d in batch_input])
    return {
        'users': users,
        'items': items,
        'item_freq': item_freq,
        'mask': mask,
    }
    
def create_diffusion(world, pad_dim):
    sigma_min = world.args.sigma_min
    sigma_max = world.args.sigma_max
    schedule_type = world.args.schedule_type
    noise_schedule = create_noise_schedule(sigma_min, sigma_max, schedule_type)
    
    if world.diffusion_type == "absorb":
        return DiscreteDiffusionAbsorb(world, noise_schedule, pad_dim)

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    
    for targ, src in zip(target_params, source_params):

        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def minibatch(data_frame, batch_size):
    num_rows = len(data_frame)
    for start in range(0, num_rows, batch_size):
        end = start + batch_size
        yield data_frame.iloc[start:end]

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)
    
def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}