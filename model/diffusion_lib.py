import torch

from model.RecModel import timestep_embedding

import sys

class DiffusionSchedule:
    def __init__(self, schedule_fn, num_steps, is_constant=True):
        self._schedule_fn = schedule_fn
        self.num_steps = num_steps

    def __call__(self, step):
        return self._schedule_fn(step)

    def __repr__(self):
        return f"DiffusionSchedule(steps: {self.num_steps})"
    
def create_noise_schedule(sigma_min, sigma_max, schedule_type):
    # print(sigma_min, sigma_max, schedule_type)
    if schedule_type == 'linear':
        def step_fn(step):
            return torch.linspace(sigma_min, sigma_max, steps=step)
    else:
        raise NotImplementedError
    return step_fn


class DiscreteDiffusionAbsorb():
    def __init__(self, 
                 world, 
                 noise_schedule, 
                 pad_dim):
        self.world = world
        self.pad_dim = pad_dim
        self.noise_schedule = noise_schedule
        self.num_steps= world.args.num_step
        self.device = world.device
        self.consistency_mode = world.args.consistency_mode
        self.consistency_scale = world.args.consistency_scale
        self.consistency_step = int(self.num_steps/ self.world.args.consistency_scale)
        
        # self.sigma = torch.cat((torch.tensor([0.0]), self.noise_schedule(self.num_steps))).double()
        self.sigma = self.noise_schedule(self.num_steps).double().to(self.device)
        self.item_freq_lambda = world.args.item_freq_lambda
        self.gaussian_sigma = self.num_steps / world.args.gaussian_scale
        print('Gaussian sigma: ', self.gaussian_sigma )
        
        
        # formatted = [f"{x.item():.4f}" for x in self.sigma]
        # print(formatted)
    
    def sample_t(self, size=(1,)):
        """Samples batches of time steps to use."""
        if self.consistency_mode == 'freq_inference':
            return torch.randint(low=1, high=self.num_steps, size=size, device=self.device)
        else:
            return torch.randint(low=self.consistency_step, high=self.num_steps, size=size, device=self.device)
        
    def calculate_absorb_prob(self, item_freq, t):
        t = t - int(self.num_steps/2)
        t = t.unsqueeze(1).expand_as(item_freq)
        prob = t - item_freq
        prob = torch.exp(- (prob ** 2) / (2 * self.gaussian_sigma ** 2))
        prob = prob * item_freq
        prob = prob* self.item_freq_lambda
        return prob
    
    def predict_x_neg(self, model, x_negs, t):
        sigma_t = self.sigma[t]
        absorb_prob = sigma_t.unsqueeze(1).expand(-1, x_negs.shape[1]-1)
        absorb_mask = (torch.rand_like(absorb_prob) < absorb_prob).int()
        zero_col = torch.zeros(absorb_mask.size(0), 1, device=absorb_mask.device)
        absorb_mask = torch.cat([zero_col, absorb_mask], dim=1)
        absorb_mask = absorb_mask.unsqueeze(-1).expand(-1, -1, x_negs.shape[-1])
        pad = model.get_padding()
        pad_expanded = pad.view(1, 1, 128)
        x_negs_t = torch.where(absorb_mask.bool(), pad_expanded, x_negs)
        x_negs_pred = model(x_negs_t, absorb_mask.bool())
        return x_negs_pred
        # has_nan = torch.isnan(x_negs_pred).any()
        # print("Contains NaN:", has_nan.item())
        # sys.exit()

    def noise_fn(self, x_start, t, item_freq):
        # n = 5
        # t[n] = 10
        sigma_t = self.sigma[t]
        absorb_prob = self.calculate_absorb_prob(item_freq, t)
        absorb_prob = sigma_t.unsqueeze(1).expand_as(absorb_prob) - absorb_prob
        absorb_indices = torch.rand(*x_start.shape, device=item_freq.device) < absorb_prob
        x_t = torch.where(absorb_indices, torch.tensor(self.pad_dim, device=x_start.device), x_start)
        # print(x_start[n])
        # print(item_freq[n])
        # print(t[n])
        # print(sigma_t[n])
        # print(absorb_prob[n])
        # print(absorb_indices[n])
        # print(x_t[n])
        return x_t, absorb_prob
        
    def one_step_denoise_fn(self, x_start, prob_at_time_t, t, item_freq, absorb_prob=None):
        if self.consistency_mode == 'freq_inference':
            t2 = t - self.consistency_step
            t2 = torch.clamp(t2, min=0, max=self.num_steps)
            mask = (prob_at_time_t==self.pad_dim).long()
            denoise = item_freq.masked_fill(mask == 0, float('-inf'))
            max_indices = denoise.argmax(dim=1)
            denoise_indices = torch.nn.functional.one_hot(max_indices, num_classes=denoise.size(1))
            prob_at_time_t2 = torch.where(denoise_indices == 1, x_start, prob_at_time_t)
            return prob_at_time_t2, t2
        elif self.consistency_mode == 'euler_solver':
            # print(self.consistency_scale)
            t2 = t - self.consistency_step
            t2 = torch.clamp(t2, min=0, max=self.num_steps)
            denoise_prob = 1 - (absorb_prob - (absorb_prob / self.consistency_scale))
            denoise_indices = torch.rand(*x_start.shape, device=item_freq.device) < denoise_prob
            prob_at_time_t2 = torch.where(denoise_indices, x_start, prob_at_time_t)
            return prob_at_time_t2, t2
        elif self.consistency_mode == 'p_reverse':
            t2 = t - self.consistency_step
            t2 = torch.clamp(t2, min=0, max=self.num_steps)
            prob_at_time_t2,_ = self.noise_fn(x_start, t2, item_freq)
            return prob_at_time_t2, t2
            # m = 799
            # print(absorb_prob[m])
            # print(denoise_prob[m])
            # print(denoise_indices[m])
            # print(t[m])
            # print(t2[m])
            # print(prob_at_time_t[m])
            # print(mask[m])
            # print(denoise[m])
            # print(denoise_indices.shape)
            # print(denoise_indices[m])
            # print(x_start[m])
            # print(prob_at_time_t[m])
            # print(prob_at_time_t2[m])
            # sys.exit()
    
    def sampling(self, users, batch_data, item_freq, model):
        mask = (batch_data != model.padding_idx).long()
        user_emb = model.get_user_embeddings(users).unsqueeze(1)
        
        if self.world.config_sampling['sampling_scale'] == 0:
            t = torch.full((batch_data.shape[0],), self.num_steps).to(self.world.device)
            t_emb = timestep_embedding(t, user_emb.shape[-1], t.device).unsqueeze(1)
            
            user_emb = user_emb + t_emb
            x_t = model.get_padding().view(1, 1, user_emb.shape[-1])
            x_t = x_t.expand(batch_data.shape[0], batch_data.shape[1], user_emb.shape[-1])

            mask = torch.ones(batch_data.shape, dtype=torch.int, device=batch_data.device)
            x_t = torch.cat([user_emb, x_t], dim=1)
            zero_column = torch.zeros(mask.size(0), 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([zero_column, mask], dim=1)
            output = model(x_t, mask.bool())
        else:
            sampling_steps = torch.linspace(0, self.world.config_sampling['sampling_num_step']-1, steps=self.world.config_sampling['sampling_scale']).long()
            x_start = batch_data.to(self.world.device)
            for i in reversed(range(sampling_steps.size(0))):
                t = torch.full((batch_data.shape[0],), sampling_steps[i]).to(self.world.device)
                batch_data, _ = self.noise_fn(x_start, t, item_freq)
                # print(x_start[0])
                mask = (batch_data == model.padding_idx).long()
                t_emb = timestep_embedding(t, user_emb.shape[-1], t.device).unsqueeze(1)
                user_emb_t = user_emb + t_emb

                x_t = model.getItemEmbedding(batch_data)
                # x_t = model.embedding_item.weight[torch.tensor(batch_data).long()]                  
                # mask = mask.unsqueeze(-1).repeat(1, 1, x_t.shape[-1])
                x_t = torch.cat([user_emb_t, x_t], dim=1)
                zero_column = torch.zeros(mask.size(0), 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([zero_column, mask], dim=1)
                output = model(x_t, mask.bool())

                if sampling_steps[i] != 0:
                    item_emb = output[:, 1:, :]
                    x_start = model.emb_to_item(item_emb)
                    x_start = x_start.view(batch_data.shape)


        if self.world.config_sampling['sampling_type'] == 'first':
            user_emb_i = output[:, :1, :]
        elif self.world.config_sampling['sampling_type'] == 'agg':
            items_emb = output[:, 1:, :]
            if self.world.args.aggregation_type == 'mean':
                user_emb_i = items_emb.mean(dim=1, keepdim=True)
            elif self.world.args.aggregation_type == 'sum':
                user_emb_i = items_emb.sum(dim=1, keepdim=True)
        elif self.world.config_sampling['sampling_type'] == 'recover':
            items_emb = output[:, 1:, :]
            x_recover = model.emb_to_item(items_emb)
            x_recover = x_recover.view(batch_data.shape)
            items_emb = model.embedding_item.weight[x_recover.long()]
            if self.world.args.aggregation_type == 'mean':
                user_emb_i = items_emb.mean(dim=1, keepdim=True)
            elif self.world.args.aggregation_type == 'sum':
                user_emb_i = items_emb.sum(dim=1, keepdim=True)
        return user_emb_i

        
    def get_qt2_qt_from_x0(self, x_start, item_freq, t,
                           make_one_hot = False,
                           noise_t_0 = True):
        if make_one_hot:
            x_start = torch.nn.functional.one_hot(x_start, num_classes=self.pad_dim)
        
        prob_at_time_t, absorb_prob = self.noise_fn(x_start, t, item_freq)
        prob_at_time_t = torch.where(t.unsqueeze(1).expand_as(x_start) == 0, x_start, prob_at_time_t)
        prob_at_time_t2, t2 = self.one_step_denoise_fn(x_start, prob_at_time_t, t, item_freq, absorb_prob)
        # prob_at_time_t2 = torch.where(t2.unsqueeze(1).expand_as(x_start) == 0, x_start, prob_at_time_t2)
        prob_at_time_t2 = torch.where(t2.unsqueeze(1).expand_as(x_start) < self.consistency_step, x_start, prob_at_time_t2)
        mask = (prob_at_time_t2 == self.pad_dim).any(dim=1).int() 
        t2[mask == 0] = 0

        del absorb_prob
            
        return prob_at_time_t, prob_at_time_t2, t2
    
    def denoise(self, model, x_t, mask, t):
        model_output = model(x_t, mask)
        return model_output
        











        