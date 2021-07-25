from .pretrain import UniterForPretraining
from torch import nn
from .layer import BertOnlyMLMHead
from collections import defaultdict
from torch.nn import functional as F
import torch
import numpy as np
from .do_calculus import Matcher, BoxCoder, BalancedPositiveNegativeSampler, FastRCNNLossComputation

class UniterForPretrainingForVCR(UniterForPretraining):
    """ 2nd Stage Pretrain UNITER for VCR
    """
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
        self.cls = BertOnlyMLMHead(
            self.uniter.config, self.uniter.embeddings.word_embeddings.weight)

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        txt_lens = batch['txt_lens']
        num_bbs = batch['num_bbs']
        img_soft_labels = batch['img_soft_labels']
        if task == 'mlm':
            txt_labels = batch['txt_labels']
            return self.forward_mlm(input_ids, position_ids,
                                    txt_type_ids, img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    txt_labels, txt_lens, num_bbs, img_soft_labels, compute_loss)
        elif task == 'mrfr':
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrfr_feat_target = batch['feat_targets']
            ### pretrain by vc_feat
            vc_feat = batch['vc_feat']
            mrfr_vc_feat_target = batch['vc_feat_targets']

            return self.forward_mrfr(input_ids, position_ids,
                                     txt_type_ids, img_feat, img_pos_feat,
                                     attention_mask, gather_index,
                                     img_masks, img_mask_tgt,
                                     mrfr_feat_target, vc_feat, mrfr_vc_feat_target, txt_lens, num_bbs, img_soft_labels, compute_loss)
        elif task.startswith('mrc'):
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.forward_mrc(input_ids, position_ids,
                                    txt_type_ids, img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    img_masks, img_mask_tgt,
                                    mrc_label_target, txt_lens, num_bbs, img_soft_labels, task, compute_loss)
        elif task.startswith('dc'):
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            '''
            return self.forward_dc_1(input_ids, position_ids,
                                    txt_type_ids, img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    img_masks, img_mask_tgt,
                                    mrc_label_target, txt_lens, num_bbs, img_soft_labels, task, compute_loss)

            return self.forward_dc_2(input_ids, position_ids,
                                    txt_type_ids, img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    img_masks, img_mask_tgt,
                                    mrc_label_target, txt_lens, num_bbs, img_soft_labels, task, compute_loss)
            '''
            return self.forward_dc_3(input_ids, position_ids,
                                    txt_type_ids, img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    img_masks, img_mask_tgt,
                                    mrc_label_target, txt_lens, num_bbs, img_soft_labels, task, compute_loss)

        else:
            raise ValueError('invalid task')
    
    ### use 'do-calculus' in UNITER pretrain : make method
    def do_calculus_1(self, sequence_output, img_feats, proposals, txt_lens, num_bbs):
        """
        Arguments:
        - sequence_output : "img + txt" output

        Returns:

        """
        image_uniter_outputs = []
        img_feat_list = []
        i = 0
        for (output, txt_len) in zip(sequence_output, txt_lens):
            image_uniter_output = output[txt_len:txt_len+num_bbs[i]]
            image_uniter_outputs.append(image_uniter_output)
            i+=1
        for (img_feat, num_bb) in zip(img_feats, num_bbs):
            real_img_feat = img_feat[:num_bb]
            img_feat_list.append(real_img_feat)
        
        assert len(image_uniter_outputs) == len(num_bbs)
        assert len(image_uniter_outputs) == len(img_feat_list)
        # class_logits_list = [self.predictor(self_feature) for self_feature in image_uniter_outputs]
        zs = [self.causal_predictor(img_feat_list[i], [num_bbs[i]]) for i in range(len(image_uniter_outputs))]

        return image_uniter_outputs, zs

    def do_calculus_loss_1(self, class_logits_causal_list, proposals, img_soft_labels, compute_loss):

        matcher = Matcher(
            0.7, # cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.7
            0.3, # cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.3
            allow_low_quality_matches=False,
        )

        bbox_reg_weights = (10., 10., 5., 5.) # cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
        box_coder = BoxCoder(weights=bbox_reg_weights)

        fg_bg_sampler = BalancedPositiveNegativeSampler(
            512, 0.25
        )
        # cfg = MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        # cfg = MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

        cls_agnostic_bbox_reg = False # cfg.MODEL.CLS_AGNOSTIC_BBOX_REG = False

        loss_evaluator = FastRCNNLossComputation(
            matcher,
            fg_bg_sampler,
            box_coder,
            cls_agnostic_bbox_reg
        )
        loss_causal, prediction_list = loss_evaluator(
            class_logits_causal_list, proposals, img_soft_labels, compute_loss
        )
        return loss_causal, prediction_list

    ###

    ### use 'do-calculus' in UNITER pretrain version 2 : make method
    def do_calculus_2(self, sequence_output, img_feats, proposals, txt_lens, num_bbs):

        image_uniter_outputs = []

        i = 0
        for (output, txt_len) in zip(sequence_output, txt_lens):
            image_uniter_output = output[txt_len:txt_len+num_bbs[i]]
            image_uniter_outputs.append(image_uniter_output)
            i+=1

        
        assert len(image_uniter_outputs) == len(num_bbs)

        # class_logits_list = [self.predictor(self_feature) for self_feature in image_uniter_outputs]
        zs = [self.causal_predictor_2(image_uniter_outputs[i], [num_bbs[i]]) for i in range(len(image_uniter_outputs))]

        return zs

    # MLM
    def forward_mlm(self, input_ids, position_ids, txt_type_ids, img_feat,
                    img_pos_feat, attention_mask, gather_index,
                    txt_labels, txt_lens, num_bbs, img_soft_labels, compute_loss=True):
        sequence_output, _ = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)
        '''
        ### use 'do-calculus' in UNITER pretrain : compute loss
        class_logits, class_logits_causal = self.do_calculus(sequence_output, img_pos_feat, txt_lens, num_bbs)
        loss_classifier, loss_causal = self.do_calculus_loss(class_logits, class_logits_causal, img_pos_feat, img_soft_labels)
        ###
        '''
    
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self.cls(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss #, loss_classifier, loss_causal
        else:
            return prediction_scores

    # MRFR
    def forward_mrfr(self, input_ids, position_ids, txt_type_ids,
                     img_feat, img_pos_feat,
                     attention_mask, gather_index, img_masks, img_mask_tgt,
                     feat_targets, vc_feat, mrfr_vc_feat_target, txt_lens, num_bbs, img_soft_labels, compute_loss=True):
        sequence_output, _ = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks,
                                      txt_type_ids=txt_type_ids)
        '''
        ### use 'do-calculus' in UNITER pretrain : compute loss
        class_logits, class_logits_causal = self.do_calculus(sequence_output, img_pos_feat, txt_lens, num_bbs)
        loss_classifier, loss_causal = self.do_calculus_loss(class_logits, class_logits_causal, img_pos_feat, img_soft_labels)
        ###
        '''

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output, img_mask_tgt)

        if vc_feat.shape[-1]==1024:
            prediction_feat = self.feat_regress_vc(masked_output)
            feat_targets = mrfr_vc_feat_target
        else:
            prediction_feat = self.feat_regress(masked_output)

        if compute_loss:
            mrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                                   reduction='none')
            return mrfr_loss#, loss_classifier, loss_causal
        else:
            return prediction_feat

    # MRC
    def forward_mrc(self, input_ids, position_ids, txt_type_ids,
                    img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks, img_mask_tgt,
                    label_targets, txt_lens, num_bbs, img_soft_labels, task, compute_loss=True):
        sequence_output, _ = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks,
                                      txt_type_ids=txt_type_ids)
        '''
        ### use 'do-calculus' in UNITER pretrain : compute loss
        class_logits, class_logits_causal = self.do_calculus(sequence_output, img_pos_feat, txt_lens, num_bbs)
        loss_classifier, loss_causal = self.do_calculus_loss(class_logits, class_logits_causal, img_pos_feat, img_soft_labels)
        ###
        '''

        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_soft_label = self.region_classifier(masked_output)

        if compute_loss:
            if "kl" in task:
                prediction_soft_label = F.log_softmax(
                    prediction_soft_label, dim=-1)
                mrc_loss = F.kl_div(
                    prediction_soft_label, label_targets, reduction='none')
            else:
                # background class should not be the target
                label_targets = torch.max(label_targets[:, 1:], dim=-1)[1] + 1
                mrc_loss = F.cross_entropy(
                    prediction_soft_label, label_targets,
                    ignore_index=0, reduction='none')
            return mrc_loss #, loss_classifier, loss_causal
        else:
            return prediction_soft_label

    # DC 1 (Do-Calculus 1)
    def forward_dc_1(self, input_ids, position_ids, txt_type_ids,
                    img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks, img_mask_tgt,
                    label_targets, txt_lens, num_bbs, img_soft_labels, task, compute_loss=True):
        sequence_output, _ = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)

        ### use 'do-calculus' in UNITER pretrain : compute loss
        device = img_pos_feat.device
        image_uniter_outputs, zs = self.do_calculus(sequence_output, img_feat, img_pos_feat, txt_lens, num_bbs)
        
        batch_zs = pad_tensors(zs, num_bbs).to(device)
        attention_mask_list, gather_index_list = [], []
        for i in range(input_ids.size(0)):
            attention_mask_list.append(attention_mask[i][txt_lens[i]:])
            #gather_index_list.append(gather_index[i][txt_lens[i]:])
        #attention_mask = attention_mask[:, input_ids.size(1):]
        #gather_index = gather_index[:, input_ids.size(1):]
     
        attention_mask = pad_tensors(attention_mask_list, num_bbs).to(device)
        # gather_index = pad_tensors(gather_index_list, num_bbs).to(device)
        gather_index = gather_index[:, input_ids.size(1)]
        zs_output, _ = self.uniter(None, position_ids,
                                      batch_zs, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)
        causal_logits_list = []
        yzs = []
        i = 0
        # 밑에서 일렬로 쫙 펴버리자, 그러면서 label도 펴버리자, 그리고 한 번에 loss를 계산하자
        for (uniter_output, z_output) in zip(image_uniter_outputs, zs_output):
            z = z_output[:num_bbs[i]]
            i += 1
            assert len(uniter_output) == len(z)
            length = len(uniter_output)
            yz = torch.cat((self.Wx(uniter_output).unsqueeze(1).repeat(1, length, 1), z.unsqueeze(0).repeat(length, 1, 1)), 2).view(-1, 2*self.Wx(uniter_output).size(1))
            yzs.append(yz)
            causal_logits_list.append(self.causal_score(yz))
        
            

        if compute_loss:
            loss_causal = self.do_calculus_loss(causal_logits_list, img_pos_feat, img_soft_labels, compute_loss)
            return loss_causal
        else:
            prediction_soft_label = self.do_calculus_loss(causal_logits_list, img_pos_feat, img_soft_labels, compute_loss)
            return prediction_soft_label
        ###

    # DC 2 (Do-Calculus 2)
    def forward_dc_2(self, input_ids, position_ids, txt_type_ids,
                    img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks, img_mask_tgt,
                    label_targets, txt_lens, num_bbs, img_soft_labels, task, compute_loss=True):
        sequence_output, _ = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)

        # 
        # sequence_img_output = sequence_output[:, input_ids.size(1):, :]; import ipdb;ipdb.set_trace(context=10)
        sequence_img_output = []
        for i, sequence in enumerate(sequence_output):
            sequence_img_output.append(sequence[txt_lens[i]:, :])
        class_logits_causal_list = self.causal_predictor_2(sequence_img_output, num_bbs)

        ### use 'do-calculus' in UNITER pretrain : compute loss
        device = img_pos_feat.device
        

        # batch_zs = pad_tensors(zs, num_bbs).to(device)
        '''
        attention_mask_list, gather_index_list = [], []
        for i in range(input_ids.size(0)):
            attention_mask_list.append(attention_mask[i][txt_lens[i]:])
            #gather_index_list.append(gather_index[i][txt_lens[i]:])
        #attention_mask = attention_mask[:, input_ids.size(1):]
        #gather_index = gather_index[:, input_ids.size(1):]
     
        attention_mask = pad_tensors(attention_mask_list, num_bbs).to(device)
        # gather_index = pad_tensors(gather_index_list, num_bbs).to(device)
        gather_index = gather_index[:, input_ids.size(1)]
        zs_output = self.uniter(None, position_ids,
                                      batch_zs, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)
        causal_logits_list = []
        yzs = []
        i = 0
        # 밑에서 일렬로 쫙 펴버리자, 그러면서 label도 펴버리자, 그리고 한 번에 loss를 계산하자
        for (uniter_output, z_output) in zip(image_uniter_outputs, zs_output):
            z = z_output[:num_bbs[i]]
            i += 1
            assert len(uniter_output) == len(z)
            length = len(uniter_output)
            yz = torch.cat((self.Wx_1(uniter_output).unsqueeze(1).repeat(1, length, 1), z.unsqueeze(0).repeat(length, 1, 1)), 2).view(-1, 2*self.Wx(uniter_output).size(1))
            yzs.append(yz)
            causal_logits_list.append(self.causal_score(yz))
        
            

        if compute_loss:
            loss_causal = self.do_calculus_loss_1(class_logits_causal_list, img_pos_feat, img_soft_labels, compute_loss)
            return loss_causal
        else:
            prediction_soft_label = self.do_calculus_loss_1(class_logits_causal_list, img_pos_feat, img_soft_labels, compute_loss)
            return prediction_soft_label
        '''
        loss_causal = self.do_calculus_loss_1(class_logits_causal_list, img_pos_feat, img_soft_labels, True)
        prediction_soft_label = self.do_calculus_loss_1(class_logits_causal_list, img_pos_feat, img_soft_labels, False)
        return loss_causal, prediction_soft_label
        ###
    # DC 3 (Do-Calculus 3)
    def forward_dc_3(self, input_ids, position_ids, txt_type_ids,
                    img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks, img_mask_tgt,
                    label_targets, txt_lens, num_bbs, img_soft_labels, task, compute_loss=True):
        
        _, embedding_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)

        # 
        # sequence_img_output = sequence_output[:, input_ids.size(1):, :]; import ipdb;ipdb.set_trace(context=10)
        img_emb_list = []
        for i, sequence in enumerate(embedding_output):
            img_emb_list.append(sequence[txt_lens[i]:, :])
        class_logits_causal_list, label_list = self.causal_predictor_2(img_emb_list, num_bbs, img_soft_labels)

        ### use 'do-calculus' in UNITER pretrain : compute loss
        device = img_pos_feat.device
        

        # batch_zs = pad_tensors(zs, num_bbs).to(device)
        '''
        if compute_loss:
            loss_causal = self.do_calculus_loss_1(class_logits_causal_list, img_pos_feat, img_soft_labels, compute_loss)
            return loss_causal
        else:
            prediction_soft_label = self.do_calculus_loss_1(class_logits_causal_list, img_pos_feat, img_soft_labels, compute_loss)
            return prediction_soft_label'''
        ###
        loss_causal, prediction_soft_label = self.do_calculus_loss_1(class_logits_causal_list, img_pos_feat, img_soft_labels, True)

        return loss_causal, prediction_soft_label, label_list

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        pass
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, hid, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    if len(tensors[0].shape) > 1:
        for i, (t, l) in enumerate(zip(tensors, lens)):
            output.data[i, :l, ...] = t.data
    else:
        lens = [tensors[i].size(0) for i in range(len(tensors))]
        # max_len = max(lens)
        output = torch.zeros(bs, max_len, dtype=dtype)
        for i, (t, l) in enumerate(zip(tensors, lens)):
            if l > max_len:
                l = max_len
            output.data[i, :l] = t.data[:l]
    return output
