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
        else:
            raise ValueError('invalid task')
    
    ### use 'do-calculus' in UNITER pretrain : make method
    def do_calculus(self, sequence_output, proposals, txt_lens, num_bbs):
        """
        Arguments:
        - sequence_output : "img + txt" output

        Returns:

        """
        features = []
        i = 0
        for (output, txt_len) in zip(sequence_output, txt_lens):
            output_img = output[txt_len:txt_len+num_bbs[i]]
            features.append(output_img)
            i+=1
        
        class_logits_list = [self.predictor(feature) for feature in features]
        # class_logits = np.array(class_logits_list).sum() # (주의 : 아직 평균을 안 낸 상태이다.)
        class_logits_causal_list = [self.causal_predictor(feature, [num_bb]) for (feature, num_bb) in zip(features, num_bbs)]
        # class_logits_causal = np.array((class_logits_causal_list)).sum() # (주의 : 아직 평균을 안 낸 상태이다.)

        return class_logits_list, class_logits_causal_list

    def do_calculus_loss(self, class_logits_list, class_logits_causal_list, proposals, img_soft_labels):

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
        loss_classifier, loss_causal = loss_evaluator(
            class_logits_list, class_logits_causal_list, proposals, img_soft_labels
        )

        return loss_classifier, loss_causal
    ###

    # MLM
    def forward_mlm(self, input_ids, position_ids, txt_type_ids, img_feat,
                    img_pos_feat, attention_mask, gather_index,
                    txt_labels, txt_lens, num_bbs, img_soft_labels, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)
        
        ### use 'do-calculus' in UNITER pretrain : compute loss
        class_logits, class_logits_causal = self.do_calculus(sequence_output, img_pos_feat, txt_lens, num_bbs)
        loss_classifier, loss_causal = self.do_calculus_loss(class_logits, class_logits_causal, img_pos_feat, img_soft_labels)
        ###
        
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
            return masked_lm_loss, loss_classifier, loss_causal
        else:
            return prediction_scores

    # MRFR
    def forward_mrfr(self, input_ids, position_ids, txt_type_ids,
                     img_feat, img_pos_feat,
                     attention_mask, gather_index, img_masks, img_mask_tgt,
                     feat_targets, vc_feat, mrfr_vc_feat_target, txt_lens, num_bbs, img_soft_labels, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks,
                                      txt_type_ids=txt_type_ids)

        ### use 'do-calculus' in UNITER pretrain : compute loss
        class_logits, class_logits_causal = self.do_calculus(sequence_output, img_pos_feat, txt_lens, num_bbs)
        loss_classifier, loss_causal = self.do_calculus_loss(class_logits, class_logits_causal, img_pos_feat, img_soft_labels)
        ###

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
            return mrfr_loss, loss_classifier, loss_causal
        else:
            return prediction_feat

    # MRC
    def forward_mrc(self, input_ids, position_ids, txt_type_ids,
                    img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks, img_mask_tgt,
                    label_targets, txt_lens, num_bbs, img_soft_labels, task, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks,
                                      txt_type_ids=txt_type_ids)

        ### use 'do-calculus' in UNITER pretrain : compute loss
        class_logits, class_logits_causal = self.do_calculus(sequence_output, img_pos_feat, txt_lens, num_bbs)
        loss_classifier, loss_causal = self.do_calculus_loss(class_logits, class_logits_causal, img_pos_feat, img_soft_labels)
        ###

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
            return mrc_loss, loss_classifier, loss_causal
        else:
            return prediction_soft_label
