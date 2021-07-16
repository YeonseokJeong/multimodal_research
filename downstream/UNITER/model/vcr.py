"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VCR model
"""
from collections import defaultdict
import numpy as np
import os
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

# from .layer import GELU
from .model import (
    UniterPreTrainedModel, UniterModel)


class UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 2)
        )
        self.apply(self.init_weights)
        '''
        ### compute confounder dictionary : prepare initialized confounder dictionary & prior
        self.conf_dict = np.zeros((1601, img_dim))
        self.conf_dict_gt = np.zeros((1601, img_dim))
        self.prior = np.zeros(1601)
        self.prior_gt = np.zeros(1601)
        ###
        '''

    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.uniter.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embeddings.word_embeddings = new_emb

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)
        '''
        ### compute confounder dictionary 2 : extract soft label
        img_soft_label = batch['img_soft_label']
        img_gt_soft_label = batch['img_gt_soft_label']
        img_tot_soft_label = batch['img_tot_soft_label']
        
        for batch_idx in range(len(img_feat)):
            img_set = img_feat[batch_idx]
            if img_set.shape[1] < len(img_gt_soft_label[batch_idx]) + len(img_soft_label[batch_idx]):
                print("error : image < image(gt+nongt)")
            for gt_idx in range(len(img_gt_soft_label[batch_idx])):
                label_gt = img_gt_soft_label[batch_idx][gt_idx].argmax()
                self.prior_gt[label_gt] += 1
                self.conf_dict_gt[label_gt] += img_set[gt_idx].cpu().numpy()
            for nongt_idx in range(len(img_soft_label[batch_idx])):
                label = img_soft_label[batch_idx][nongt_idx].argmax()
                self.prior[label] += 1
                self.conf_dict[label] += img_set[gt_idx+1+nongt_idx].cpu().numpy()
        ###
        '''
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.vcr_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            vcr_loss = F.cross_entropy(
                    rank_scores, targets.squeeze(-1),
                    reduction='mean')
            return vcr_loss
        else:
            rank_scores = rank_scores[:, 1:]
            return rank_scores
    '''
    def save_conf_prior(self, opts):
        #self.conf_dict = self.conf_dict / prior[: np.newaxis]
        #self.conf_dict_gt = self.conf_dict_gt / prior_gt[: np.newaxis]
        # os.makedirs('./conf_and_prior', exist_ok=True)
        np.save(f'./conf_and_prior/{opts.split}_dic_vcr_nongt_uniter.npy', self.conf_dict)
        np.save(f'./conf_and_prior/{opts.split}_dic_vcr_gt_uniter.npy', self.conf_dict_gt)
        #self.prior = self.prior / np.sum(self.prior)
        #self.prior_gt = self.prior_gt / np.sum(self.prior_gt)
        np.save(f'./conf_and_prior/{opts.split}_stat_prob_vcr_nongt_uniter.npy', self.prior)
        np.save(f'./conf_and_prior/{opts.split}_stat_prob_vcr_gt_uniter.npy', self.prior_gt)
    '''