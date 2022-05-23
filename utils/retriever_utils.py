"""MathQA utils.
"""
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import random
import enum
import six
import copy
from six.moves import map
from six.moves import range
from six.moves import zip
from tqdm import tqdm
from utils.utils import *

_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)

class MathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "filename_id question paragraphs table_descriptions \
            pos_sent_ids pos_table_ids"
        )):
    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 filename_id,
                 retrieve_ind,
                 tokens,
                 input_ids,
                 segment_ids,
                 input_mask,
                 label):

        self.filename_id = filename_id
        self.retrieve_ind = retrieve_ind
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.

    Args:
      tokenizer: a tokenizer from bert.tokenization.FullTokenizer
      text: text to tokenize
      apply_basic_tokenization: If True, apply the basic tokenization. If False,
        apply the full tokenization (basic + wordpiece).

    Returns:
      tokenized text.

    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """

    _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize

    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.get_vocab():
                tokens.append(token)
            else:
                tokens.append(tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))

    return tokens

def remove_space(text_in):
    res = []

    for tmp in text_in.split(" "):
        if tmp != "":
            res.append(tmp)

    return " ".join(res)


def wrap_single_pair(tokenizer, question, context, label, max_seq_length,
                     cls_token, sep_token):
    '''
    single pair of question, context, label feature
    '''

    question_tokens = tokenize(tokenizer, question)
    this_gold_tokens = tokenize(tokenizer, context)

    tokens = [cls_token] + question_tokens + [sep_token]
    segment_ids = [0] * len(tokens)

    tokens += this_gold_tokens
    segment_ids.extend([0] * len(this_gold_tokens))

    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length-1]
        tokens += [sep_token]
        segment_ids = segment_ids[:max_seq_length]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    this_input_feature = {
        "context": context,
        "tokens": tokens,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "label": label
    }

    return this_input_feature


def convert_single_mathqa_example(example, option, is_training, tokenizer, max_seq_length,
                                  cls_token, sep_token):
    """Converts a single MathQAExample into Multiple Retriever Features."""
    """ option: tf idf or all"""
    """train: 1:3 pos neg. Test: all"""

    pos_features, neg_sent_features, irrelevant_neg_table_features, relevant_neg_table_features = [], [], [], []

    question = example.question

    # positive examples
    # tables = example.tables
    paragraphs = example.paragraphs
    pos_text_ids = example.pos_sent_ids
    pos_table_ids = example.pos_table_ids
    table_descriptions = example.table_descriptions
    
    relevant_table_ids = set([i.split("-")[0] for i in pos_table_ids])

    for sent_idx, sent in enumerate(paragraphs):
        if sent_idx in pos_text_ids:
            this_input_feature = wrap_single_pair(
                tokenizer, example.question, sent, 1, max_seq_length,
                cls_token, sep_token)
        else:
            this_input_feature = wrap_single_pair(
                tokenizer, example.question, sent, 0, max_seq_length,
                cls_token, sep_token)
        this_input_feature["ind"] = sent_idx
        this_input_feature["filename_id"] = example.filename_id
        
        if sent_idx in pos_text_ids:
            pos_features.append(this_input_feature)
        else:
            neg_sent_features.append(this_input_feature)
        
    for cell_idx in table_descriptions:
        this_gold_sent = table_descriptions[cell_idx]
        if cell_idx in pos_table_ids:
            this_input_feature = wrap_single_pair(
                tokenizer, question, this_gold_sent, 1, max_seq_length,
                cls_token, sep_token)
            this_input_feature["ind"] = cell_idx
            this_input_feature["filename_id"] = example.filename_id
            pos_features.append(this_input_feature)
        else:
            ti = cell_idx.split("-")[0]
            this_input_feature = wrap_single_pair(
                tokenizer, question, this_gold_sent, 0, max_seq_length,
                cls_token, sep_token)
            this_input_feature["ind"] = cell_idx
            this_input_feature["filename_id"] = example.filename_id
            if ti in relevant_table_ids:
                relevant_neg_table_features.append(this_input_feature)
            else:
                irrelevant_neg_table_features.append(this_input_feature)
                
    return pos_features, neg_sent_features, irrelevant_neg_table_features, relevant_neg_table_features

def read_examples(input_path, tokenizer, op_list, const_list, log_file):
    """Read a json file into a list of examples."""

    write_log(log_file, "Reading " + input_path)
    with open(input_path) as input_file:
        input_data = json.load(input_file)

    examples = []
    for entry in input_data:
        examples.append(read_mathqa_entry(entry, tokenizer))

    return input_data, examples, op_list, const_list

def read_mathqa_entry(entry, tokenizer):

    question = entry["qa"]["question"]
    
    paragraphs = entry["paragraphs"]
    # tables = entry["tables"]
    
    if 'text_evidence' in entry["qa"]:
        pos_sent_ids = entry["qa"]['text_evidence']
        pos_table_ids = entry["qa"]['table_evidence']
    else: # test set
        pos_sent_ids = []
        pos_table_ids = []

    
    table_descriptions = entry["table_description"]
    filename_id = entry["uid"]

    return MathQAExample(
        filename_id=filename_id,
        question=question,
        paragraphs=paragraphs,
        # tables=tables,
        table_descriptions=table_descriptions,
        pos_sent_ids=pos_sent_ids,
        pos_table_ids=pos_table_ids,
    )


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 option,
                                 is_training,
                                 ):
    """Converts a list of DropExamples into InputFeatures."""
    res, res_neg_sent, res_irrelevant_neg_table, res_relevant_neg_table = [], [], [], []
    for (example_index, example) in tqdm(enumerate(examples)):
        pos_features, neg_sent_features, irrelevant_neg_table_features, relevant_neg_table_features = example.convert_single_example(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            option=option,
            is_training=is_training,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token)

        res.extend(pos_features)
        res_neg_sent.extend(neg_sent_features)
        res_irrelevant_neg_table.extend(irrelevant_neg_table_features)
        res_relevant_neg_table.extend(relevant_neg_table_features)
        
    return res, res_neg_sent, res_irrelevant_neg_table, res_relevant_neg_table



def retrieve_evaluate(all_logits, all_filename_ids, all_inds, ori_file, topn):
    '''
    save results to file. calculate recall
    '''
    
    res_filename = {}
    res_filename_inds = {}
    
    print(len(all_logits))
    for this_logit, this_filename_id, this_ind in zip(all_logits, all_filename_ids, all_inds):
        
        if this_filename_id not in res_filename:
            res_filename[this_filename_id] = []
            res_filename_inds[this_filename_id] = []
            
        if this_ind not in res_filename_inds[this_filename_id]:
            res_filename[this_filename_id].append({
                "score": this_logit[1].item(),
                "ind": this_ind
            })
            res_filename_inds[this_filename_id].append(this_ind)
            
        
        
    with open(ori_file) as f:
        data_all = json.load(f)
        
    # take top ten
    all_recall = 0.0
    # all_recall_3 = 0.0
    
    count_data = 0
    for data in data_all:
        this_filename_id = data["uid"]
        
        if this_filename_id not in res_filename:
            continue
        count_data += 1
        this_res = res_filename[this_filename_id]
        
        sorted_dict = sorted(this_res, key=lambda kv: kv["score"], reverse=True)
        
        # sorted_dict = sorted_dict[:topn]
        
        gold_sent_inds = data["qa"]["text_evidence"]
        gold_table_inds = data["qa"]["table_evidence"]
        
        # table rows
        table_retrieved = []
        text_retrieved = []

        # all retrieved
        table_re_all = []
        text_re_all = []
        
        correct = 0
        # correct_3 = 0
        
        for tmp in sorted_dict[:topn]:
            if type(tmp["ind"]) == str:
                table_retrieved.append(tmp)
                if tmp["ind"] in gold_table_inds:
                    correct += 1
            else:
                text_retrieved.append(tmp)
                if tmp["ind"] in gold_sent_inds:
                    correct += 1
                
        #     if tmp["ind"] in gold_inds:
        #         correct += 1
        # # print(sorted_dict)
        for tmp in sorted_dict:
            if type(tmp["ind"]) == str:
                table_re_all.append(tmp)
            else:
                text_re_all.append(tmp)
                
        # for tmp in sorted_dict[:3]:
        #     if tmp["ind"] in gold_inds:
        #         correct_3 += 1
                
        all_recall += (float(correct) / (len(gold_table_inds) + len(gold_sent_inds)))
        # all_recall_3 += (float(correct_3) / len(gold_inds)) 

        data["table_retrieved_all"] = table_re_all
        data["text_retrieved_all"] = text_re_all
        
    # res_3 = all_recall_3 / len(data_all)
    res = all_recall / len(data_all)
    
    # res_message = "Top 3: " + str(res_3) + "\n" + "Top 5: " + str(res) + "\n"
    res_message = f"Top {topn}: {res}\n"
    
    return res, res_message


def retrieve_inference(all_logits, all_filename_ids, all_inds, output_prediction_file, ori_file):
    '''
    save results to file. calculate recall
    '''
    
    res_filename = {}
    res_filename_inds = {}

    for this_logit, this_filename_id, this_ind in zip(all_logits, all_filename_ids, all_inds):
        if this_filename_id not in res_filename:
            res_filename[this_filename_id] = []
            res_filename_inds[this_filename_id] = []
            
        if this_ind not in res_filename_inds[this_filename_id]:
            res_filename[this_filename_id].append({
                "score": this_logit[1].item(),
                "ind": this_ind
            })
            res_filename_inds[this_filename_id].append(this_ind)
            
        
        
    with open(ori_file) as f:
        data_all = json.load(f)
    

    output_data = []
    for data in data_all:
        table_re_all = []
        text_re_all = []
        this_filename_id = data["uid"]
        
        if this_filename_id not in res_filename:
            continue

        this_res = res_filename[this_filename_id]
        sorted_dict = sorted(this_res, key=lambda kv: kv["score"], reverse=True)
        
        for tmp in sorted_dict:
            if type(tmp["ind"]) == str:
                table_re_all.append(tmp)
            else:
                text_re_all.append(tmp)
                
        data["table_retrieved_all"] = table_re_all
        data["text_retrieved_all"] = text_re_all
        output_data.append(data)
        
    with open(output_prediction_file, "w") as f:
        json.dump(output_data, f, indent=4)
    
    return None