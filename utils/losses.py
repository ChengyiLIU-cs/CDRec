import torch
import torch as th
import torch.nn.functional as F
import world

import sys

def get_loss_fn(world, diffusion, reduce_mean=True):
    def get_l2_loss():
        def loss_fn(distiller, distiller_target, mask_original):
            prefix = torch.ones((mask_original.shape[0], 1), dtype=mask_original.dtype, device=mask_original.device)
            mask = torch.cat([prefix, mask_original], dim=1)
            valid_counts = mask.sum(dim=1).clamp(min=1e-8)
            losses = (distiller - distiller_target) ** 2
            mask = mask.unsqueeze(-1).float()
            losses = losses * mask
            losses = losses.mean(dim=2)
            loss = losses.sum(dim=1) / valid_counts
            return loss
        return loss_fn
    def get_kl_loss():
        def loss_fn(distiller, distiller_target, mask, model):
            if world.args.timestep == 'token':
                distiller_item_emb = distiller[:, 2:, :]
                distiller_target_item_emb = distiller_target[:, 2:, :]
            else:
                distiller_item_emb = distiller[:, 1:, :]
                distiller_target_item_emb = distiller_target[:, 1:, :]
            distiller_prob_log = model.item_decoder(distiller_item_emb, log_operation=True)
            distiller_target_prob = model.item_decoder(distiller_target_item_emb, log_operation=False)

            # distiller_target_prob = distiller_target_prob.clamp(min=1e-8)  #########################################################
            
            losses = F.kl_div(distiller_prob_log, distiller_target_prob, reduction='none')

            losses = torch.nn.functional.relu(losses)
            
            losses = losses.sum(dim=-1)
            losses = losses * mask
            
            valid_counts = mask.sum(dim=1).clamp(min=1e-8)
            
            loss = losses.sum(dim=1) / valid_counts
            return loss
        return loss_fn
    def get_cross_entropy_loss():
        def loss_fn(distiller, distiller_target, mask, model):
            if world.args.timestep == 'token':
                distiller_item_emb = distiller[:, 2:, :]
                distiller_target_item_emb = distiller_target[:, 2:, :]
            else:
                distiller_item_emb = distiller[:, 1:, :]
                distiller_target_item_emb = distiller_target[:, 1:, :]
            distiller_logit = model.item_decoder(distiller_item_emb, log_operation=False, softmax=False)

        
            distiller_logit = distiller_logit.view(-1, distiller_logit.shape[-1])
            distiller_target_logit = model.emb_to_item(distiller_target_item_emb)
            
            losses = F.cross_entropy(distiller_logit, distiller_target_logit, reduction='none')
            losses = losses.view(mask.shape[0], -1)
            losses = losses * mask
            valid_counts = mask.sum(dim=1).clamp(min=1e-8)
            
            loss = losses.sum(dim=1) / valid_counts
            return loss
        return loss_fn

    reduce_op = th.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    # if world.args.consistency_loss_type == 'l2':
    #     loss_cal = get_l2_loss()

    def loss_fn(model, x_t, mask, t, target_model, x_t2, mask2, t2, x_start = None, mask_original = None, items_emb_original=None):
        

        def denoise_fn(x_t, mask):
            return model(x_t, mask)

        @th.no_grad()
        def target_denoise_fn(x_t2, mask2):
            return target_model(x_t2, mask2)
        
        distiller = denoise_fn(x_t, mask)
        
        distiller_target = target_denoise_fn(x_t2, mask2)
        distiller_target = th.where(t2.view(x_t2.shape[0],1,1).expand_as(x_t2) <= world.args.num_step/world.args.consistency_scale, x_t2, distiller_target)
        

        if world.args.consistency_loss_type == 'l2':
            loss_cal = get_l2_loss()
            losses_consistency = loss_cal(distiller, distiller_target, mask_original)
        elif world.args.consistency_loss_type == 'kl':
            loss_cal = get_kl_loss()
            losses_consistency = loss_cal(distiller, distiller_target, mask_original, model)
        elif world.args.consistency_loss_type == 'cross_entropy':
            loss_cal = get_cross_entropy_loss()
            losses_consistency = loss_cal(distiller, distiller_target, mask_original, model)
        loss_consistency = reduce_op(losses_consistency)
        
        world.args.consistency_loss_type = 'cross_entropy'
        # world.args.consistency_loss_type = 'kl'
        if world.args.consistency_loss_type == 'l2':
            loss_cal = get_l2_loss()
            losses_original = loss_cal(distiller, items_emb_original, mask_original)
        elif world.args.consistency_loss_type == 'kl':
            loss_cal = get_kl_loss()
            losses_original = loss_cal(distiller, items_emb_original, mask_original, model)
        elif world.args.consistency_loss_type == 'cross_entropy':
            loss_cal = get_cross_entropy_loss()
            losses_original = loss_cal(distiller, items_emb_original, mask_original, model)

        loss_original = reduce_op(losses_original)
        
        beta_consistency = world.args.beta_consistency

        # print(loss_original, loss_consistency)

        loss = beta_consistency * loss_original + (1 - beta_consistency) * loss_consistency
        # loss = 0.1 * loss_original
        return loss, distiller

    return loss_fn

def get_bpr_loss_fn(world, reduce_mean=True):
    reduce_op = th.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    def loss_fn(users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0):
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        bpr_loss = reduce_op(torch.nn.functional.softplus(neg_scores - pos_scores))

        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                                    posEmb0.norm(2).pow(2)  +
                                    negEmb0.norm(2).pow(2)
                                    )/float(users_emb.shape[0])
        # print(bpr_loss)
        # print(world.args.weight_decay)
        # print(reg_loss)
        return bpr_loss + world.args.weight_decay * reg_loss

    return loss_fn
    
def get_ss_loss_fn(world, reduce_mean=True):

    reduce_op = th.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def cal_pos_scores_single(emb1, emb2, normalization = True):
        if normalization:
            emb1 = F.normalize(emb1, p=2, dim=1, eps=1e-12)
            emb2 = F.normalize(emb2, p=2, dim=1, eps=1e-12)
        pos_scores = torch.exp(torch.div(F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8), world.config_ssl['tau']))  
        return pos_scores

    def cal_neg_scores_single(emb1, emb2, normalization = True):
        if normalization:
            emb1 = F.normalize(emb1, p=2, dim=1, eps=1e-12)
            emb2 = F.normalize(emb2, p=2, dim=-1, eps=1e-12)
        emb1 = emb1.unsqueeze(1)
        neg_scores = torch.exp(torch.div(F.cosine_similarity(emb1, emb2, dim=-1), world.config_ssl['tau']) )
        neg_scores = neg_scores.sum(dim=1)
        return neg_scores
    
    def cal_pres_score(emb1, emb2, normalization = True):
        if normalization:
            emb1 = F.normalize(emb1, p=2, dim=1, eps=1e-12)
            emb2 = F.normalize(emb2, p=2, dim=1, eps=1e-12)
        pos_scores = torch.exp(torch.div(F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8), world.config_ssl['tau'])) 
        
        sim_matrix = torch.matmul(emb1, emb2.T)
        sim_matrix = torch.exp(torch.div(sim_matrix, world.config_ssl['tau']))
        mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        neg_scores = sim_matrix[mask].view(sim_matrix.size(0), -1).sum(dim=1)
        return pos_scores, neg_scores

    def loss_fn(model,users, x_pred, pos_item, neg_item, users_emb):
        # print(x_pred.shape)
        # print(pos_item.shape)
        # print(neg_item.shape)
        
        if world.config_sampling['sampling_type'] == 'first':
            user_preference1 = x_pred[:, :1, :]
        # elif world.config_sampling['sampling_type'] == 'agg':
        else:
            x_pred = x_pred[:, 1:, :]
            if world.args.aggregation_type == 'mean':
                user_preference1 = x_pred.mean(dim=1, keepdim=True).squeeze()
            elif world.args.aggregation_type == 'sum':
                user_preference1 = x_pred.sum(dim=1, keepdim=True).squeeze()
        # elif world.config_sampling['sampling_type'] == 'recover':
        #     x_pred = x_pred[:, 1:, :]
        #     x_recover = model.emb_to_item(x_pred)
        #     x_recover = x_recover.view(x_pred.shape[0:2])
        #     items_emb = model.embedding_item.weight[x_recover.long()]
        #     if world.args.aggregation_type == 'mean':
        #         user_preference1 = items_emb.mean(dim=1, keepdim=True)
        #     elif world.args.aggregation_type == 'sum':
        #         user_preference1 = items_emb.sum(dim=1, keepdim=True)
        
        # users_emb_original = model.get_user_embeddings(users)
        users_emb_original =model.embedding_user.weight[users].squeeze(1)
        user_preference1 = torch.stack([users_emb_original, user_preference1], dim=0).mean(dim=0)  # the representation for the user 

        user_preference2 = pos_item[:, :1, :].squeeze()  # the positive item
        
        if user_preference1.shape == user_preference2.shape:
            pos_scores = cal_pos_scores_single(user_preference1, user_preference2)
        
        user_neg_preference = neg_item[:, 1:, :]
        # user_neg_preference = neg_item
        
        neg_scores = cal_neg_scores_single(user_preference1, user_neg_preference)  # the negative items
        
        # losses = -torch.log(pos_scores / neg_scores)    #0.05
        losses = -torch.log(pos_scores / (pos_scores + neg_scores))

        pos_scores2, neg_scores2 = cal_pres_score(user_preference1, users_emb)
        losses2 = -torch.log(pos_scores2 / (pos_scores2 + neg_scores2))

        loss = reduce_op(losses)
        loss2 = reduce_op(losses2)
        
        # weight = 0.7 best
        weight = 0.7
            
        return weight * loss + (1 - weight) * loss2

    return loss_fn









def get_dpo_loss_fn(world, reduce_mean=True):
    reduce_op = th.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    def get_l2_loss():
        def loss_fn(x_start, predicted_x):
            loss = F.mse_loss(x_start, predicted_x, reduction='none').sum(dim=2)  ######################################################
            return loss
        return loss_fn
    if world.args.dpo_loss_type == 'l2':
        loss_cal = get_l2_loss()


    def loss_fn(x_pos,x_pred, x_negs, x_negs_pred):
        loss_pos = loss_cal(x_pos, x_pred)
        # loss_neg = loss_cal(x_negs, x_negs_pred)
        loss_neg = loss_cal(x_negs, x_pred)
        model_diff = loss_pos - loss_neg
        
        
        if world.config_dpo['dpo_mode'] == 'neg':
            losses = -(1 - world.config_dpo['dpo_lamda']) * F.logsigmoid(-world.config_dpo['dpo_gamma'] * model_diff + 1e-8) + loss_pos * world.config_dpo['dpo_lamda']
        elif world.config_dpo['dpo_mode'] == 'neg_sample':
            losses = -F.logsigmoid(-world.config_dpo['dpo_lamda'] * model_diff + 1e-8)
            # print(loss.shape)
            # print("any NaN:", th.isnan(loss).any())
            # print("any Inf:", th.isinf(loss).any())

        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = th.mean(losses)

        return loss

    return loss_fn