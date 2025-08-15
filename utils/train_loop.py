import time
import copy
from tqdm import tqdm
import datetime
from utils.utils import *
from utils.losses import *

from torch import optim

import sys

class TrainLoop:
    def __init__(self, 
                 world, 
                 train_data, 
                 test_data,
                 model,
                 target_model,
                 target_model_init = True):
        self.world = world
        self.epoches = world.args.epoches
        self.batch_size = world.args.batch_size
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True)
        self.test_data = test_data
        self.device = world.device
        
        self.diffusion = create_diffusion(world, train_data.pad)
        self.model = model
        self.lr = self.world.args.lr
        self.opt = optim.Adam(self.model.parameters(), 
                              lr=self.lr)
        self.consistency_loss_fn = get_loss_fn(self.world, self.diffusion)
        self.ss_loss_fn = get_ss_loss_fn(self.world)
        self.dpo_loss_fn = get_dpo_loss_fn(self.world)
        self.bpr_loss_fn = get_bpr_loss_fn(self.world)
        
        self.target_model = target_model
        self.target_model.requires_grad_(False)

        self.ema_rate = world.args.ema_rate
        self.ema_params = copy.deepcopy(list(self.model.denoiser.parameters()))

        # self.neg_sample_n = train_data.length
        # self.neg_sample_n = 1   # 0-219
        self.neg_sample_n = 10  # 2-221
        self.candidate_neg_items = train_data.candidate_neg_items
        self.update_neg_items()
        
        self.pretrain_epoch = world.pretrain_epoch

        if target_model_init:
            self.load_ema_to_target()
            print('Initialize Target Model')
        
        self.best = {}
        self.best['recall'] = [0.0] * len(self.world.topks)
        self.best['precision'] = [0.0] * len(self.world.topks)
        self.best['ndcg'] = [0.0] * len(self.world.topks)

    def update_neg_items(self):
        self.neg_items= {user_id: random.sample(list(items), self.neg_sample_n)
                            for user_id, items in self.candidate_neg_items.items()
                            if len(items) >= self.neg_sample_n}

    def run_loop(self):
        self.start_time = time.time()
        self.loss_save = []
        self.performance_save = []
        total_batch = len(self.train_loader)
        
        for epoch in range (1, self.epoches+1+self.pretrain_epoch):
            self.model.train()
            self.target_model.train()

            aver_diff_loss = 0.
            aver_dpo_loss = 0.
            aver_ssl_loss = 0.
            aver_bpr_loss = 0.

            # self.update_neg = 30
            # self.update_neg = 15 # 0   5
            # self.update_neg = 5 # 2    5
            # self.update_neg = 1   # 7    5
            self.update_neg = 5 # 3    5
            if epoch % self.update_neg == 0:
                self.update_neg_items()
            
            # for i, batch in enumerate(tqdm(self.train_loader, total=total_batch, leave=False)):
            with tqdm(total=total_batch, desc="Training", leave=False) as pbar:
                for batch in self.train_loader:
                    self.opt.zero_grad()

                    users = batch['users'].to(self.device)
                    items = batch['items'].to(self.device)
                    item_freq = batch['item_freq'].to(self.device)
                    mask_original = batch['mask'].to(self.device)
                    
                    pos_items = items[:,0]
                    neg_items = torch.tensor([self.neg_items[int(u)] for u in users])
                    neg_items_ = neg_items[:,0].to(self.device)

                    (users_emb, pos_emb, neg_emb,
                    userEmb0,  posEmb0, negEmb0) = self.model.get_bpr_embeddings(users, pos_items, neg_items_)

                    bpr_loss = self.bpr_loss_fn(users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0)
                    
                    if epoch == self.pretrain_epoch:
                        torch.save({
                            'embedding_user_weight': self.model.embedding_user.weight.data,
                            'embedding_item_weight': self.model.embedding_item.weight.data
                        }, 'model/'+self.world.args.dataset+'/'+str(self.world.args.weight_decay)+str(self.world.args.emb_dim)+str(self.world.args.ui_n_layers) + str(self.world.pretrain_epoch) +'embedding_weights.pth')
                        epoch = epoch + 1 - self.pretrain_epoch

                        self.update_neg = 5

                    if epoch > self.pretrain_epoch:
                        # self.model.embedding_item.weight.requires_grad = False
                        # self.model.embedding_user.weight.requires_grad = False

                        t = self.diffusion.sample_t((items.shape[0],))
                        x_t, x_t2, t2 = self.diffusion.get_qt2_qt_from_x0(items, item_freq, t)
                        
                        items_emb, mask, items_emb2, mask2 = self.model.get_embedding( x_t, t, x_t2, t2, users_emb)

                        items_emb_original = self.model.getItemEmbedding(items)
                        items_emb_original = torch.cat([users_emb.unsqueeze(1), items_emb_original], dim=1)
                        
                        consistency_loss, x_pred = self.consistency_loss_fn(self.model, items_emb, mask, t, self.target_model, items_emb2, mask2, t2, items, mask_original, items_emb_original)
                        
                        x_negs = self.model.get_embeddings_neg2(neg_items)
                        ssl_loss = self.ss_loss_fn(self.model, users, x_pred, items_emb_original, x_negs, users_emb)
                        

                    '''
                    x_pos = self.model.get_embeddings(items, mask_original)
                    
                    neg_items = torch.tensor([self.neg_items[int(u)] for u in users])
                    x_negs = self.model.get_embeddings_neg(neg_items, c= users_emb)
                    x_negs_pred = self.diffusion.predict_x_neg(self.model, x_negs, t)
                    '''
                    # dpo_loss = self.dpo_loss_fn(x_pos,x_pred, x_negs, x_negs_pred)
                    
                    # loss = consistency_loss + ssl_loss + bpr_loss
                    if epoch > self.pretrain_epoch:
                        
                        # loss = bpr_loss + self.world.args.consistency_loss_lambda * consistency_loss + self.world.args.ssl_loss_lambda * ssl_loss
                        loss = bpr_loss + self.world.args.ssl_loss_lambda * ssl_loss +  self.world.args.consistency_loss_lambda *consistency_loss
                    else:
                        loss = bpr_loss
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.opt.step()
                    self._update_ema()
                    self._update_target_ema()
                    # print('------- ------- -------')
                    if epoch > self.pretrain_epoch:
                        aver_diff_loss += consistency_loss.detach().cpu().item() / total_batch
                        aver_ssl_loss += ssl_loss.detach().cpu().item() / total_batch
                    aver_bpr_loss += bpr_loss.detach().cpu().item() / total_batch
                    # aver_dpo_loss += dpo_loss.mean().detach().cpu().item() / total_batch
                    pbar.update(1)
            duration = time.time()-self.start_time
            minutes = int(duration / 60)
            seconds = int(duration % 60)
            # print(f"Epoch: {epoch}, ConsistencyLoss: {aver_diff_loss}, DPOLoss: {aver_dpo_loss}")
            # print(f"Epoch: {epoch}, ConsistencyLoss: {aver_diff_loss}, SSLLoss: {aver_ssl_loss}, BPRLoss: {aver_bpr_loss}")
            if epoch > self.pretrain_epoch:
                print(f"Epoch: {epoch-self.pretrain_epoch}, ConsistencyLoss: {self.world.args.consistency_loss_lambda * aver_diff_loss}, SSLLoss: {self.world.args.ssl_loss_lambda * aver_ssl_loss}")
            else:
                print(f"Epoch: {epoch}, BPRLoss: {aver_bpr_loss}, ConsistencyLoss: {self.world.args.consistency_loss_lambda * aver_diff_loss}, SSLLoss: {self.world.args.ssl_loss_lambda * aver_ssl_loss}")
            print(f"{minutes}:{seconds}")

            if epoch % 3 == 0:
                duration = time.time()-self.start_time
                minutes, seconds = divmod(duration, 60)
                formatted_time = f"{int(minutes):02d}:{seconds:04.1f}"
                print(formatted_time)
                self.test()
                duration = time.time()-self.start_time
                minutes, seconds = divmod(duration, 60)
                formatted_time = f"{int(minutes):02d}:{seconds:04.1f}"
                print(formatted_time)
        self.save()

    def save(self):
        
        combined_dict = {**vars(self.world.args), **self.best}
        
        path = 'log/' + self.world.dataset+ str(datetime.datetime.now().strftime("%d-%H%M")) + '_' +  \
               '_' +str(self.world.args.num_step) + str(self.best['recall'][1])+str(self.best['ndcg'][1]) \
               + '_' + str(self.neg_sample_n) + str(self.world.pretrain_epoch) +'.txt'
        with open(path, 'w') as f:
            for i in combined_dict:
                f.write(i + ":" + str(combined_dict[i]) + '\n')
        print('saved')

    def _update_ema(self):
        # for params in self.ema_params:
        update_ema(self.ema_params, self.model.denoiser.parameters(), rate=self.ema_rate)    
    
    def _update_target_ema(self):
        with th.no_grad():
            update_ema(
                self.target_model.denoiser.parameters(),
                self.model.denoiser.parameters(),
                rate=self.ema_rate,
            )

    def update_best(self, results):
        for key in results:
            for i in range(len(results[key])):
                if results[key][i] > self.best[key][i]:
                    self.best[key][i] = float('%.6f' % results[key][i])

    @torch.no_grad()
    def load_ema_to_target(self, ema_index=0):
        ema_params = self.ema_params
        target_params = list(self.target_model.denoiser.parameters())

        assert len(ema_params) == len(target_params), "EMA's number of parameters and  target_model.denoiser's number of parameters are different"

        for p_tgt, p_ema in zip(target_params, ema_params):
            p_tgt.data.copy_(p_ema.data)

    def test(self):
        print('Test over all items')
        self.model.eval()
        max_K = max(self.world.topks)
        test_batch_size = self.world.args.test_batch_size
        
        

        results = {'precision': np.zeros(len(self.world.topks)),
                    'recall': np.zeros(len(self.world.topks)),
                    'ndcg': np.zeros(len(self.world.topks))}
        
        with torch.no_grad():
            total_batch = self.test_data.test_data.shape[0] // test_batch_size + 1
            groundTrue_list = []
            rating_list = []

            
            for batch in minibatch(self.test_data.test_data, batch_size=test_batch_size):
                groundTrue = batch['ItemIDs'].tolist()
                batch_data = batch['train_data'].tolist()
                batch_data = torch.Tensor(batch_data).long().to(self.device)
                item_freq = torch.Tensor(batch['ItemFreq'].tolist()).to(self.device)
                users = torch.tensor(batch['userID'].tolist()).to(self.device)

                users_emb_i = self.diffusion.sampling(users, batch_data, item_freq, self.model).squeeze(1)
                # users_emb_i = self.model.test(batch_data).squeeze(1)
                
                users_emb_i = F.normalize(users_emb_i, dim=1)

                users_emb_original = self.model.embedding_user.weight[users].squeeze(1)
                # users_emb_original = self.model.get_user_embeddings(users)  ##############################################################
                users_emb_i = torch.stack([users_emb_original, users_emb_i], dim=0).mean(dim=0)
                
                # users_emb_i = self.model.get_user_embeddings(users)
                
                all_items = self.model.items
                rating = self.model.f(torch.matmul(users_emb_i, all_items.t()))
                allPos = batch['allPos'].tolist()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                    
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                del rating
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            X = zip(rating_list, groundTrue_list)    
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x, self.world.topks))
            
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(self.test_data.test_data.shape[0])
            results['precision'] /= float(self.test_data.test_data.shape[0])
            results['ndcg'] /= float(self.test_data.test_data.shape[0])  
            print(f"recall:   \t{results['recall']},\nprecision:\t{results['precision']},\nndcg:     \t{results['ndcg']}") 
            self.performance_save.append(results)

            self.update_best(results)
        
            print(f"Best:\n recall: {self.best['recall']},\n precision: {self.best['precision']},\n ndcg{self.best['ndcg']}")