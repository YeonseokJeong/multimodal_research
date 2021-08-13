"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VCR dataset
"""
import copy
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
from .data import (DetectFeatTxtTokDataset, TxtTokLmdb, DetectFeatLmdb,
                   TxtLmdb, get_ids_and_lens, pad_tensors,
                   get_gather_index)
import numpy as np
from transformers import AutoTokenizer
from nltk import pos_tag

class VcrTxtTokLmdb(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len=120, task="qa,qar"):
        assert task == "qa" or task == "qar" or task == "qa,qar",\
            "VCR only support the following tasks: 'qa', 'qar' or 'qa,qar'"
        self.task = task
        if task == "qa,qar":
            id2len_task = "qar"
        else:
            id2len_task = task
        if max_txt_len == -1:
            self.id2len = json.load(
                open(f'{db_dir}/id2len_{id2len_task}.json'))
        else:
            self.id2len = {
                id_: len_
                for id_, len_ in json.load(
                    open(f'{db_dir}/id2len_{id2len_task}.json')
                    ).items()
                if len_ <= max_txt_len
            }

        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']


class VcrDetectFeatTxtTokDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db_gt=None, img_db=None):
        assert not (img_db_gt is None and img_db is None),\
            "img_db_gt and img_db cannot all be None"
        assert isinstance(txt_db, VcrTxtTokLmdb)
        assert img_db_gt is None or isinstance(img_db_gt, DetectFeatLmdb)
        assert img_db is None or isinstance(img_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.img_db_gt = img_db_gt
        self.task = self.txt_db.task
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        txt_lens, self.ids = get_ids_and_lens(txt_db)

        txt2img = txt_db.txt2img

        if self.img_db and self.img_db_gt:
            self.lens = [tl+self.img_db_gt.name2nbb[txt2img[id_][0]] +
                         self.img_db.name2nbb[txt2img[id_][1]]
                         for tl, id_ in zip(txt_lens, self.ids)]
        elif self.img_db:
            self.lens = [tl+self.img_db.name2nbb[txt2img[id_][1]]
                         for tl, id_ in zip(txt_lens, self.ids)]
        else:
            self.lens = [tl+self.img_db_gt.name2nbb[txt2img[id_][0]]
                         for tl, id_ in zip(txt_lens, self.ids)]

    def _get_img_feat(self, fname_gt, fname):
        
        if self.img_db and self.img_db_gt:
            img_feat_gt, bb_gt = self.img_db_gt[fname_gt]
            img_bb_gt = torch.cat([bb_gt, bb_gt[:, 4:5]*bb_gt[:, 5:]], dim=-1)

            img_feat, bb = self.img_db[fname]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
            ### extract uniter bbox
            #real_name = '_'.join(fname.split('_')[2:])
            #np.save("./bbox_val/"+real_name+".npy", bb.numpy())
            #np.save("./bbox_gt_val/"+real_name+".npy", bb_gt.numpy())
            ###
            img_feat = torch.cat([img_feat_gt, img_feat], dim=0)
            img_bb = torch.cat([img_bb_gt, img_bb], dim=0)
            num_bb = img_feat.size(0)
        elif self.img_db:
            img_feat, bb = self.img_db[fname]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
            num_bb = img_feat.size(0)
        elif self.img_db_gt:
            img_feat, bb = self.img_db_gt[fname_gt]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
            num_bb = img_feat.size(0)
        return img_feat, img_bb, num_bb


class VcrDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.task != "qa,qar",\
            "loading training dataset with each task separately"

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        if self.task == "qar":
            input_ids_rs = txt_dump['input_ids_rs']
            answer_label = txt_dump['qa_target']
            assert answer_label >= 0, "answer_label < 0"
            input_ids_gt_a = [self.txt_db.sep] + copy.deepcopy(
                input_ids_as[answer_label])
            type_ids_gt_a = [2] * len(input_ids_gt_a)
            type_ids_q += type_ids_gt_a
            input_ids_q += input_ids_gt_a
            input_ids_for_choices = input_ids_rs
        else:
            input_ids_for_choices = input_ids_as
        return input_ids_q, input_ids_for_choices, type_ids_q

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        input_ids_q, input_ids_for_choices, type_ids_q = self._get_input_ids(
            example)
        label = example['%s_target' % (self.task)]

        outs = []
        for index, input_ids_a in enumerate(input_ids_for_choices):
            if index == label:
                target = torch.tensor([1]).long()
            else:
                target = torch.tensor([0]).long()
            input_ids = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            # type_id
            # 0 -- question
            # 1 -- region
            # 2 -- answer
            # 3 -- rationale
            type_id_for_choice = 3 if type_ids_q[-1] == 2 else 2
            txt_type_ids = [0] + type_ids_q + [type_id_for_choice]*(
                len(input_ids_a)+2)
            attn_masks = torch.ones(
                len(input_ids) + num_bb, dtype=torch.long)
            input_ids = torch.tensor(input_ids)
            txt_type_ids = torch.tensor(txt_type_ids)

            outs.append(
                (input_ids, txt_type_ids,
                 img_feat, img_pos_feat,
                 attn_masks, target))

        return tuple(outs)


def vcr_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, targets) = map(list, unzip(concat(inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_type_ids = pad_sequence(
        txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'txt_type_ids': txt_type_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch


class VcrEvalDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        assert self.task == "qa,qar",\
            "loading evaluation dataset with two tasks together"

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_for_choices = []
        type_ids_for_choices = []
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        input_ids_rs = txt_dump['input_ids_rs']
        for index, input_ids_a in enumerate(input_ids_as):
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q + [2]*(
                len(input_ids_a)+2)
            input_ids_for_choices.append(curr_input_ids_qa)
            type_ids_for_choices.append(curr_type_ids_qa)
        for index, input_ids_a in enumerate(input_ids_as):
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q + [2]*(
                len(input_ids_a)+1)
            if (self.split == "val" and index == txt_dump["qa_target"]) or\
                    self.split == "test":
                for input_ids_r in input_ids_rs:
                    curr_input_ids_qar = copy.deepcopy(curr_input_ids_qa) +\
                        input_ids_r + [self.txt_db.sep]
                    curr_type_ids_qar = copy.deepcopy(curr_type_ids_qa) +\
                        [3]*(len(input_ids_r)+2)
                    input_ids_for_choices.append(curr_input_ids_qar)
                    type_ids_for_choices.append(curr_type_ids_qar)
        return input_ids_for_choices, type_ids_for_choices
    
    ### calculate confounder dictionary & prior 2 : prepare method to extract label
    def _get_img_feat_for_db(self, img_db, fname):
        img_dump = img_db.get_dump(fname)
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        img_soft_label = torch.tensor(img_dump['soft_labels'])
        return img_feat, img_bb, img_soft_label

    def _get_img_feat(self, fname_gt, fname):
        if self.img_db and self.img_db_gt:
            (img_feat_gt, img_bb_gt,
             img_soft_label_gt) = self._get_img_feat_for_db(
                 self.img_db_gt, fname_gt)

            (img_feat, img_bb,
             img_soft_label) = self._get_img_feat_for_db(
                 self.img_db, fname)

            img_feat = torch.cat([img_feat_gt, img_feat], dim=0)
            img_bb = torch.cat([img_bb_gt, img_bb], dim=0)
            img_soft_label = torch.cat(
                [img_soft_label_gt, img_soft_label], dim=0)
        elif self.img_db:
            (img_feat, img_bb,
             img_soft_label) = self._get_img_feat_for_db(
                 self.img_db, fname)
        else:
            (img_feat, img_bb,
             img_soft_label) = self._get_img_feat_for_db(
                 self.img_db_gt, fname_gt)
        num_bb = img_feat.size(0)
        return img_feat, img_bb, img_soft_label, num_bb
    ###
    
    def __getitem__(self, i):
        qid = self.ids[i]
        example = super().__getitem__(i)
        img_feat, img_pos_feat, img_tot_soft_label, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        ### compute confounder dictionary : extract soft label
        img_soft_label = self.img_db.get_dump(example['img_fname'][1])['soft_labels']
        img_gt_soft_label = self.img_db_gt.get_dump(example['img_fname'][0])['soft_labels']
        ###
        input_ids_for_choices, type_ids_for_choices = self._get_input_ids(
            example)
        qa_target = torch.tensor([int(example["qa_target"])])
        qar_target = torch.tensor([int(example["qar_target"])])

        outs = []
        for index, input_ids in enumerate(input_ids_for_choices):

            attn_masks = torch.ones(
                len(input_ids) + num_bb, dtype=torch.long)

            input_ids = torch.tensor(input_ids)
            txt_type_ids = torch.tensor(
                type_ids_for_choices[index])

            outs.append(
                (input_ids, txt_type_ids,
                 img_feat, img_pos_feat,
                 attn_masks, img_soft_label, img_gt_soft_label, img_tot_soft_label)) ### compute confounder dictionary : extract soft label

        return tuple(outs), qid, qa_target, qar_target


def vcr_eval_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, img_soft_label, img_gt_soft_label, img_tot_soft_label) = map(
         list, unzip(concat(outs for outs, _, _, _ in inputs)))
    # import ipdb;ipdb.set_trace(context=10)

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_type_ids = pad_sequence(
        txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    
    qa_targets = torch.stack(
        [t for _, _, t, _ in inputs], dim=0)
    qar_targets = torch.stack(
        [t for _, _, _, t in inputs], dim=0)
    qids = [id_ for _, id_, _, _ in inputs]
    
    ### compute confounder dictionary : extract soft label
    #img_tot_soft_label = [torch.Tensor(np.concatenate((img_soft_label[i], img_gt_soft_label[i]), axis=0)) for i in range(len(img_soft_label))]
    #img_tot_soft_label = pad_tensors(img_tot_soft_label, num_bbs)
    #img_soft_label = torch.Tensor(img_soft_label)
    #img_gt_soft_label = torch.Tensor(img_gt_soft_label)
    ### 

    return {'qids': qids,
            'input_ids': input_ids,
            'txt_type_ids': txt_type_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            'qa_targets': qa_targets,
            'qar_targets': qar_targets,
            'img_soft_label': img_soft_label,
            'img_gt_soft_label': img_gt_soft_label,
            'img_tot_soft_label' : img_tot_soft_label,
            'txt_lens': txt_lens} ### compute confounder dictionary : extract soft label

class VcrDcDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.task != "qa,qar",\
            "loading training dataset with each task separately"

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_q = txt_dump['input_ids']#;import ipdb;ipdb.set_trace(context=10)
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        if self.task == "qar":
            input_ids_rs = txt_dump['input_ids_rs']
            answer_label = txt_dump['qa_target']
            assert answer_label >= 0, "answer_label < 0"
            input_ids_gt_a = [self.txt_db.sep] + copy.deepcopy(input_ids_as[answer_label])
            type_ids_gt_a = [2] * len(input_ids_gt_a)
            type_ids_q += type_ids_gt_a
            input_ids_q += input_ids_gt_a
            input_ids_for_choices = input_ids_rs
            input_ids_for_choices_dc = input_ids_rs
        else:
            # base #
            input_ids_for_choices = input_ids_as
            # do-calculus # 
            num_choices = len(input_ids_as)
            # 1> extract object
            object_list = txt_dump['objects']
            # 2> extract noun of object

            noun_object_list = []
            for element in pos_tag(object_list):
                if element[1]=='NN' or element[1]=='NNS' or element[1]=='NNP':
                    idx = self.tokenizer.encode(element[0])[1]
                    noun_object_list.append(idx)
            # noun_object_list = [element[0] for element in pos_tag(object_list) if element[1]=='NN' or element[1]=='NNS' or element[1]=='NNP']
            input_ids_for_choices_dc = num_choices*[noun_object_list]
        return input_ids_q, input_ids_for_choices, input_ids_for_choices_dc, type_ids_q

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        input_ids_q, input_ids_for_choices, input_ids_for_choices_dc, type_ids_q = self._get_input_ids(
            example)
        label = example['%s_target' % (self.task)]

        outs = []
        for index, input_ids_a in enumerate(input_ids_for_choices):
            if index == label:
                target = torch.tensor([1]).long()
            else:
                target = torch.tensor([0]).long()
            # base #
            input_ids = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            # type_id
            # 0 -- question
            # 1 -- region
            # 2 -- answer
            # 3 -- rationale
            type_id_for_choice = 3 if type_ids_q[-1] == 2 else 2
            # base #
            txt_type_ids = [0] + type_ids_q + [type_id_for_choice]*(
                len(input_ids_a)+2)
            attn_masks = torch.ones(
                len(input_ids) + num_bb, dtype=torch.long)
            input_ids = torch.tensor(input_ids)
            txt_type_ids = torch.tensor(txt_type_ids)
            # do-calculus # 

            input_ids_dc = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_for_choices_dc[index] + [self.txt_db.sep]
            txt_type_ids_dc = [0] + type_ids_q + [type_id_for_choice]*(
                len(input_ids_for_choices_dc[index])+2)
            attn_masks_dc = torch.ones(
                len(input_ids_dc) + num_bb, dtype=torch.long)

            input_ids_dc = torch.tensor(input_ids_dc)
            txt_type_ids_dc = torch.tensor(txt_type_ids_dc)


            outs.append(
                (input_ids, txt_type_ids,
                 img_feat, img_pos_feat,
                 attn_masks, target, input_ids_dc, txt_type_ids_dc, attn_masks_dc))

        return tuple(outs)


def vcr_dc_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, targets, input_ids_dc, txt_type_ids_dc, attn_masks_dc) = map(list, unzip(concat(inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_type_ids = pad_sequence(
        txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    # if input_ids_dc[0] is not None:
        # do-calculus #
    txt_lens_dc = [i.size(0) for i in input_ids_dc]
    input_ids_dc = pad_sequence(input_ids_dc, batch_first=True, padding_value=0)
    txt_type_ids_dc = pad_sequence(txt_type_ids_dc, batch_first=True, padding_value=0)
    position_ids_dc = torch.arange(0, input_ids_dc.size(1), dtype=torch.long).unsqueeze(0)
    attn_masks_dc = pad_sequence(attn_masks_dc, batch_first=True, padding_value=0)
    bs_dc, max_tl_dc = input_ids_dc.size()
    out_size_dc = attn_masks_dc.size(1)
    gather_index_dc = get_gather_index(txt_lens_dc, num_bbs, bs_dc, max_tl_dc, out_size_dc)
    '''else:
        txt_lens_dc = None
        input_ids_dc = None
        txt_type_ids_dc = None
        position_ids_dc = None
        attn_masks_dc = None
        gather_index_dc = None'''


    batch = {'input_ids': input_ids,
             'txt_type_ids': txt_type_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets,
             'input_ids_dc': input_ids_dc,
             'txt_type_ids_dc': txt_type_ids_dc,
             'position_ids_dc': position_ids_dc,
             'attn_masks_dc': attn_masks_dc,
             'gather_index_dc': gather_index_dc,
             }
    return batch

class VcrDcEvalDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        assert self.task == "qa,qar",\
            "loading evaluation dataset with two tasks together"

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_for_choices = []
        type_ids_for_choices = []
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        input_ids_rs = txt_dump['input_ids_rs']
        # do-calculus # 
        input_ids_for_choices_dc = []
        type_ids_for_choices_dc = []        
        num_choices = len(input_ids_as)
        # 1> extract object
        object_list = txt_dump['objects']
        # 2> extract noun of object
        noun_object_list = []
        for element in pos_tag(object_list):
            if element[1]=='NN' or element[1]=='NNS' or element[1]=='NNP':
                idx = self.tokenizer.encode(element[0])[1]
                noun_object_list.append(idx)
        input_ids_as_dc = num_choices*[noun_object_list]



        for index, input_ids_a in enumerate(input_ids_as):
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q + [2]*(
                len(input_ids_a)+2)
            input_ids_for_choices.append(curr_input_ids_qa)
            type_ids_for_choices.append(curr_type_ids_qa)

            curr_input_ids_qa_dc = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_as_dc[index] + [self.txt_db.sep]
            curr_type_ids_qa_dc = [0] + type_ids_q + [2]*(len(input_ids_as_dc[index])+2)
            input_ids_for_choices_dc.append(curr_input_ids_qa_dc)
            type_ids_for_choices_dc.append(curr_type_ids_qa_dc)

        for index, input_ids_a in enumerate(input_ids_as):
            '''
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q + [2]*(
                len(input_ids_a)+1)
            '''
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) # + [self.txt_db.sep] # + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q # + [2]*(len(input_ids_a)+1)
            
            curr_input_ids_qa_dc = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) # + [self.txt_db.sep] # + input_ids_as_dc[index] + [self.txt_db.sep]
            curr_type_ids_qa_dc = [0] + type_ids_q # + [2]*(len(input_ids_as_dc[index])+2)

            if (self.split == "val" and index == txt_dump["qa_target"]) or\
                    self.split == "test":
                for input_ids_r in input_ids_rs:
                    curr_input_ids_qar = copy.deepcopy(curr_input_ids_qa) + [self.txt_db.sep] + input_ids_a + [self.txt_db.sep] + input_ids_r + [self.txt_db.sep]
                    curr_type_ids_qar = copy.deepcopy(curr_type_ids_qa) + [2]*(len(input_ids_a)+1) + [3]*(len(input_ids_r)+2)
                    input_ids_for_choices.append(curr_input_ids_qar)
                    type_ids_for_choices.append(curr_type_ids_qar)
                    # import ipdb;ipdb.set_trace(context=10)
                    curr_input_ids_qar_dc = copy.deepcopy(curr_input_ids_qa_dc) + [self.txt_db.sep] + input_ids_as_dc[index] + [self.txt_db.sep] + input_ids_r + [self.txt_db.sep]
                    curr_type_ids_qar_dc = copy.deepcopy(curr_type_ids_qa_dc) + [2]*(len(input_ids_as_dc[index])+1) + [3]*(len(input_ids_r)+2)
                    input_ids_for_choices_dc.append(curr_input_ids_qar_dc)
                    type_ids_for_choices_dc.append(curr_type_ids_qar_dc)
            else:
                curr_input_ids_qa = copy.deepcopy(curr_input_ids_qa) + [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
                curr_type_ids_qa = curr_type_ids_qa + [2]*(len(input_ids_a)+2)
                
                curr_input_ids_qa_dc = copy.deepcopy(curr_input_ids_qa_dc) + [self.txt_db.sep] + input_ids_as_dc[index] + [self.txt_db.sep]
                curr_type_ids_qa_dc = curr_type_ids_qa_dc + [2]*(len(input_ids_as_dc[index])+2)


                    
        return input_ids_for_choices, type_ids_for_choices, input_ids_for_choices_dc, type_ids_for_choices_dc
    
    ### calculate confounder dictionary & prior 2 : prepare method to extract label
    def _get_img_feat_for_db(self, img_db, fname):
        img_dump = img_db.get_dump(fname)
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        img_soft_label = torch.tensor(img_dump['soft_labels'])
        return img_feat, img_bb, img_soft_label

    def _get_img_feat(self, fname_gt, fname):
        if self.img_db and self.img_db_gt:
            (img_feat_gt, img_bb_gt,
             img_soft_label_gt) = self._get_img_feat_for_db(
                 self.img_db_gt, fname_gt)

            (img_feat, img_bb,
             img_soft_label) = self._get_img_feat_for_db(
                 self.img_db, fname)

            img_feat = torch.cat([img_feat_gt, img_feat], dim=0)
            img_bb = torch.cat([img_bb_gt, img_bb], dim=0)
            img_soft_label = torch.cat(
                [img_soft_label_gt, img_soft_label], dim=0)
        elif self.img_db:
            (img_feat, img_bb,
             img_soft_label) = self._get_img_feat_for_db(
                 self.img_db, fname)
        else:
            (img_feat, img_bb,
             img_soft_label) = self._get_img_feat_for_db(
                 self.img_db_gt, fname_gt)
        num_bb = img_feat.size(0)
        return img_feat, img_bb, img_soft_label, num_bb
    ###
    
    def __getitem__(self, i):
        qid = self.ids[i]
        example = super().__getitem__(i)
        img_feat, img_pos_feat, img_tot_soft_label, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        ### compute confounder dictionary : extract soft label
        img_soft_label = self.img_db.get_dump(example['img_fname'][1])['soft_labels']
        img_gt_soft_label = self.img_db_gt.get_dump(example['img_fname'][0])['soft_labels']
        ###
        input_ids_for_choices, type_ids_for_choices, input_ids_for_choices_dc, type_ids_for_choices_dc = self._get_input_ids(
            example)
        qa_target = torch.tensor([int(example["qa_target"])])
        qar_target = torch.tensor([int(example["qar_target"])])

        outs = []
        for index, input_ids in enumerate(input_ids_for_choices):

            attn_masks = torch.ones(
                len(input_ids) + num_bb, dtype=torch.long)

            input_ids = torch.tensor(input_ids)
            txt_type_ids = torch.tensor(
                type_ids_for_choices[index])
            #if input_ids_for_choices_dc is not None:
                # do-calculus #
            attn_masks_dc = torch.ones(len(input_ids_for_choices_dc[index]) + num_bb, dtype=torch.long)
            input_ids_dc = torch.tensor(input_ids_for_choices_dc[index])
            txt_type_ids_dc = torch.tensor(type_ids_for_choices_dc[index])
            #else: 
            #    attn_masks_dc, input_ids_dc, txt_type_ids_dc = None, None, None
            

            outs.append(
                (input_ids, txt_type_ids,
                 img_feat, img_pos_feat,
                 attn_masks, img_soft_label, img_gt_soft_label, img_tot_soft_label,
                 attn_masks_dc, input_ids_dc, txt_type_ids_dc)) ### compute confounder dictionary : extract soft label

        return tuple(outs), qid, qa_target, qar_target


def vcr_dc_eval_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, img_soft_label, img_gt_soft_label, img_tot_soft_label,
     attn_masks_dc, input_ids_dc, txt_type_ids_dc) = map(
         list, unzip(concat(outs for outs, _, _, _ in inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_type_ids = pad_sequence(
        txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    # import ipdb;ipdb.set_trace(context=10)
    if attn_masks_dc is not None:
        # do-calculus #
        txt_lens_dc = [i.size(0) for i in input_ids_dc]
        input_ids_dc = pad_sequence(input_ids_dc, batch_first=True, padding_value=0)
        txt_type_ids_dc = pad_sequence(txt_type_ids_dc, batch_first=True, padding_value=0)
        position_ids_dc = torch.arange(0, input_ids_dc.size(1), dtype=torch.long).unsqueeze(0)
        attn_masks_dc = pad_sequence(attn_masks_dc, batch_first=True, padding_value=0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    if attn_masks_dc is not None:
        # do-calculus #
        bs_dc, max_tl_dc = input_ids_dc.size()
        out_size_dc = attn_masks_dc.size(1)
        gather_index_dc = get_gather_index(txt_lens_dc, num_bbs, bs_dc, max_tl_dc, out_size_dc)
    else:
        input_ids_dc, txt_type_ids_dc, txt_type_ids_dc, position_ids_dc, attn_masks_dc = None, None, None, None, None
    
    qa_targets = torch.stack(
        [t for _, _, t, _ in inputs], dim=0)
    qar_targets = torch.stack(
        [t for _, _, _, t in inputs], dim=0)
    qids = [id_ for _, id_, _, _ in inputs]

    return {'qids': qids,
        'input_ids': input_ids,
        'txt_type_ids': txt_type_ids,
        'position_ids': position_ids,
        'img_feat': img_feat,
        'img_pos_feat': img_pos_feat,
        'attn_masks': attn_masks,
        'gather_index': gather_index,
        'qa_targets': qa_targets,
        'qar_targets': qar_targets,
        'img_soft_label': img_soft_label,
        'img_gt_soft_label': img_gt_soft_label,
        'img_tot_soft_label' : img_tot_soft_label,
        'txt_lens_dc': txt_lens_dc,
        'input_ids_dc': input_ids_dc,
        'txt_type_ids_dc': txt_type_ids_dc,
        'position_ids_dc': position_ids_dc,
        'attn_masks_dc': attn_masks_dc,
        'gather_index_dc': gather_index_dc} ### compute confounder dictionary : extract soft label

class VcrConfPriorDataset(DetectFeatTxtTokDataset):
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        self.img_db_gt = None
        #assert self.task == "qa,qar",\
        #    "loading evaluation dataset with two tasks together"

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_for_choices = []
        type_ids_for_choices = []
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        input_ids_rs = txt_dump['input_ids_rs']
        for index, input_ids_a in enumerate(input_ids_as):
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q + [2]*(
                len(input_ids_a)+2)
            input_ids_for_choices.append(curr_input_ids_qa)
            type_ids_for_choices.append(curr_type_ids_qa)
        for index, input_ids_a in enumerate(input_ids_as):
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q + [2]*(
                len(input_ids_a)+1)
            if (self.split == "val" and index == txt_dump["qa_target"]) or\
                    self.split == "test":
                for input_ids_r in input_ids_rs:
                    curr_input_ids_qar = copy.deepcopy(curr_input_ids_qa) +\
                        input_ids_r + [self.txt_db.sep]
                    curr_type_ids_qar = copy.deepcopy(curr_type_ids_qa) +\
                        [3]*(len(input_ids_r)+2)
                    input_ids_for_choices.append(curr_input_ids_qar)
                    type_ids_for_choices.append(curr_type_ids_qar)
        return input_ids_for_choices, type_ids_for_choices
    
    ### calculate confounder dictionary & prior 2 : prepare method to extract label
    def _get_img_feat_for_db(self, img_db, fname):
        img_dump = img_db.get_dump(fname)
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        img_soft_label = torch.tensor(img_dump['soft_labels'])
        return img_feat, img_bb, img_soft_label

    def _get_img_feat(self, fname_gt, fname):
        if self.img_db and self.img_db_gt:
            (img_feat_gt, img_bb_gt,
             img_soft_label_gt) = self._get_img_feat_for_db(
                 self.img_db_gt, fname_gt)

            (img_feat, img_bb,
             img_soft_label) = self._get_img_feat_for_db(
                 self.img_db, fname)

            img_feat = torch.cat([img_feat_gt, img_feat], dim=0)
            img_bb = torch.cat([img_bb_gt, img_bb], dim=0)
            img_soft_label = torch.cat(
                [img_soft_label_gt, img_soft_label], dim=0)
        elif self.img_db:
            (img_feat, img_bb,
             img_soft_label) = self._get_img_feat_for_db(
                 self.img_db, fname)
        else:
            (img_feat, img_bb,
             img_soft_label) = self._get_img_feat_for_db(
                 self.img_db_gt, fname_gt)
        num_bb = img_feat.size(0)
        return img_feat, img_bb, img_soft_label, num_bb
    ###
    
    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        input_ids = torch.tensor([self.txt_db.cls_]+ example['input_ids']+[self.txt_db.sep])

        # img input
        img_feat, img_pos_feat, img_soft_label, num_bb = self._get_img_feat(
            None, example['img_fname'])

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, img_soft_label


def vcr_conf_prior_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, img_soft_labels
     ) = map(list, unzip(inputs))
    # import ipdb;ipdb.set_trace(context=10)

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    #img_soft_label = pad_tensors(img_soft_labels, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    
    '''
    qa_targets = torch.stack(
        [t for _, _, t, _ in inputs], dim=0)
    qar_targets = torch.stack(
        [t for _, _, _, t in inputs], dim=0)
    qids = [id_ for _, id_, _, _ in inputs]
    '''
    ### compute confounder dictionary : extract soft label
    #img_tot_soft_label = [torch.Tensor(np.concatenate((img_soft_label[i], img_gt_soft_label[i]), axis=0)) for i in range(len(img_soft_label))]
    #img_tot_soft_label = pad_tensors(img_tot_soft_label, num_bbs)
    #img_soft_label = torch.Tensor(img_soft_label)
    #img_gt_soft_label = torch.Tensor(img_gt_soft_label)
    ### 

    return {'input_ids': input_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            'img_soft_label': img_soft_labels,
            'txt_lens': txt_lens} ### compute confounder dictionary : extract soft label



'''
class VcrConfPriorDataset(DetectFeatTxtTokDataset):
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        self.img_db_gt = None
        #assert self.task == "qa,qar",\
        #    "loading evaluation dataset with two tasks together"

    def _get_input_ids(self, txt_dump):
        # text input
        input_ids_for_choices = []
        type_ids_for_choices = []
        input_ids_q = txt_dump['input_ids']
        type_ids_q = [0]*len(input_ids_q)
        input_ids_as = txt_dump['input_ids_as']
        input_ids_rs = txt_dump['input_ids_rs']
        for index, input_ids_a in enumerate(input_ids_as):
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q + [2]*(
                len(input_ids_a)+2)
            input_ids_for_choices.append(curr_input_ids_qa)
            type_ids_for_choices.append(curr_type_ids_qa)
        for index, input_ids_a in enumerate(input_ids_as):
            curr_input_ids_qa = [self.txt_db.cls_] + copy.deepcopy(input_ids_q) +\
                [self.txt_db.sep] + input_ids_a + [self.txt_db.sep]
            curr_type_ids_qa = [0] + type_ids_q + [2]*(
                len(input_ids_a)+1)
            if (self.split == "val" and index == txt_dump["qa_target"]) or\
                    self.split == "test":
                for input_ids_r in input_ids_rs:
                    curr_input_ids_qar = copy.deepcopy(curr_input_ids_qa) +\
                        input_ids_r + [self.txt_db.sep]
                    curr_type_ids_qar = copy.deepcopy(curr_type_ids_qa) +\
                        [3]*(len(input_ids_r)+2)
                    input_ids_for_choices.append(curr_input_ids_qar)
                    type_ids_for_choices.append(curr_type_ids_qar)
        return input_ids_for_choices, type_ids_for_choices
    
    ### calculate confounder dictionary & prior 2 : prepare method to extract label
    def _get_img_feat_for_db(self, img_db, fname):
        img_dump = img_db.get_dump(fname)
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        img_soft_label = torch.tensor(img_dump['soft_labels'])
        return img_feat, img_bb, img_soft_label

    def _get_img_feat(self, fname_gt, fname):
        if self.img_db and self.img_db_gt:
            (img_feat_gt, img_bb_gt,
             img_soft_label_gt) = self._get_img_feat_for_db(
                 self.img_db_gt, fname_gt)

            (img_feat, img_bb,
             img_soft_label) = self._get_img_feat_for_db(
                 self.img_db, fname)

            img_feat = torch.cat([img_feat_gt, img_feat], dim=0)
            img_bb = torch.cat([img_bb_gt, img_bb], dim=0)
            img_soft_label = torch.cat(
                [img_soft_label_gt, img_soft_label], dim=0)
        elif self.img_db:
            (img_feat, img_bb,
             img_soft_label) = self._get_img_feat_for_db(
                 self.img_db, fname)
        else:
            (img_feat, img_bb,
             img_soft_label) = self._get_img_feat_for_db(
                 self.img_db_gt, fname_gt)
        num_bb = img_feat.size(0)
        return img_feat, img_bb, img_soft_label, num_bb
    ###
    
    def __getitem__(self, i):
        qid = self.ids[i]
        example = super().__getitem__(i)
        img_feat, img_pos_feat, img_tot_soft_label, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'])
        ### compute confounder dictionary : extract soft label
        img_soft_label = self.img_db.get_dump(example['img_fname'])['soft_labels']
        # img_gt_soft_label = self.img_db_gt.get_dump(example['img_fname'][0])['soft_labels']
        ###
        input_ids_for_choices, type_ids_for_choices = self._get_input_ids(
            example)
        qa_target = torch.tensor([int(example["qa_target"])])
        qar_target = torch.tensor([int(example["qar_target"])])

        outs = []
        for index, input_ids in enumerate(input_ids_for_choices):

            attn_masks = torch.ones(
                len(input_ids) + num_bb, dtype=torch.long)

            input_ids = torch.tensor(input_ids)
            txt_type_ids = torch.tensor(
                type_ids_for_choices[index])

            outs.append(
                (input_ids, txt_type_ids,
                 img_feat, img_pos_feat,
                 attn_masks, img_soft_label, img_tot_soft_label)) ### compute confounder dictionary : extract soft label

        return tuple(outs), qid, qa_target, qar_target


def vcr_conf_prior_collate(inputs):
    (input_ids, txt_type_ids, img_feats,
     img_pos_feats, attn_masks, img_soft_label, img_tot_soft_label) = map(
         list, unzip(concat(outs for outs, _, _, _ in inputs)))
    # import ipdb;ipdb.set_trace(context=10)

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_type_ids = pad_sequence(
        txt_type_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    
    qa_targets = torch.stack(
        [t for _, _, t, _ in inputs], dim=0)
    qar_targets = torch.stack(
        [t for _, _, _, t in inputs], dim=0)
    qids = [id_ for _, id_, _, _ in inputs]
    
    ### compute confounder dictionary : extract soft label
    #img_tot_soft_label = [torch.Tensor(np.concatenate((img_soft_label[i], img_gt_soft_label[i]), axis=0)) for i in range(len(img_soft_label))]
    #img_tot_soft_label = pad_tensors(img_tot_soft_label, num_bbs)
    #img_soft_label = torch.Tensor(img_soft_label)
    #img_gt_soft_label = torch.Tensor(img_gt_soft_label)
    ### 

    return {'qids': qids,
            'input_ids': input_ids,
            'txt_type_ids': txt_type_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            'qa_targets': qa_targets,
            'qar_targets': qar_targets,
            'img_soft_label': img_soft_label,
            'img_tot_soft_label' : img_tot_soft_label,
            'txt_lens': txt_lens} ### compute confounder dictionary : extract soft label
'''