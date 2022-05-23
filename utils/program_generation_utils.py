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
import math
import tqdm
from sympy import simplify
from utils.utils import *

all_ops = ["add", "subtract", "multiply", "divide", "exp"]

sys.path.insert(0, '../utils/')
max_seq_length = 512
max_program_length = 30

class MathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "id original_question question_tokens options answer \
            numbers number_indices original_program program"
        )):

    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)


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


def program_tokenization(original_program):
    original_program = original_program.split(',')
    program = []
    for tok in original_program:
        tok = tok.strip()
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program


def convert_single_mathqa_example(example, is_training, tokenizer, max_seq_length,
                                  max_program_length, op_list, op_list_size,
                                  const_list, const_list_size,
                                  cls_token, sep_token):
    """Converts a single MathQAExample into an InputFeature."""
    features = []
    question_tokens = example.question_tokens
    if len(question_tokens) >  max_seq_length - 2:
        print("too long")
        question_tokens = question_tokens[:max_seq_length - 2]
    tokens = [cls_token] + question_tokens + [sep_token]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)


    input_mask = [1] * len(input_ids)
    for ind, offset in enumerate(example.number_indices):
        if offset < len(input_mask):
            input_mask[offset] = 2
        else:
            if is_training == True:
                return features

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    number_mask = [tmp - 1 for tmp in input_mask]
    for ind in range(len(number_mask)):
        if number_mask[ind] < 0:
            number_mask[ind] = 0
    option_mask = [1, 0, 0, 1] + [1] * (len(op_list) + len(const_list) - 4)
    option_mask = option_mask + number_mask
    option_mask = [float(tmp) for tmp in option_mask]

    for ind in range(len(input_mask)):
        if input_mask[ind] > 1:
            input_mask[ind] = 1

    numbers = example.numbers
    number_indices = example.number_indices
    program = example.program
    if program is not None and is_training:
        program_ids = prog_token_to_indices(program, numbers, number_indices,
                                            max_seq_length, op_list, op_list_size,
                                            const_list, const_list_size)
        if not program_ids:
            return None
        
        program_mask = [1] * len(program_ids)
        program_ids = program_ids[:max_program_length]
        program_mask = program_mask[:max_program_length]
        if len(program_ids) < max_program_length:
            padding = [0] * (max_program_length - len(program_ids))
            program_ids.extend(padding)
            program_mask.extend(padding)
    else:
        program = ""
        program_ids = [0] * max_program_length
        program_mask = [0] * max_program_length
    assert len(program_ids) == max_program_length
    assert len(program_mask) == max_program_length
    
    this_input_features = {
        "id": example.id,
        "unique_id": -1,
        "example_index": -1,
        "tokens": tokens,
        "question": example.original_question,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "option_mask": option_mask,
        "segment_ids": segment_ids,
        "options": example.options,
        "answer": example.answer,
        "program": program,
        "program_ids": program_ids,
        "program_weight": 1.0,
        "program_mask": program_mask
    }

    features.append(this_input_features)
    return features


def read_mathqa_entry(entry, tokenizer, entity_name):
    if entry["qa"][entity_name] != "arithmetic":
        return None
    
    
    context = ""
    for idx in entry["model_input"]:
        if type(idx) == int:
            context += entry["paragraphs"][idx][:-1]
            context += " "

        else:
            context += entry["table_description"][idx][:-1]
            context += " "

    question = entry["qa"]["question"]
    this_id = entry["uid"]

    original_question = question + " " + tokenizer.sep_token + " " + context.strip()

    options = entry["qa"]["answer"] if "answer" in entry["qa"] else None
    answer = entry["qa"]["answer"] if "answer" in entry["qa"] else None

    original_question_tokens = original_question.split(' ')
    numbers = []
    number_indices = []
    question_tokens = []

    # TODO
    for i, tok in enumerate(original_question_tokens):
        num = str_to_num(tok)
        if num is not None:
            if num != "n/a":
                numbers.append(str(num))
            else:
                numbers.append(tok)
            number_indices.append(len(question_tokens))
            if tok and tok[0] == '.':
                numbers.append(str(str_to_num(tok[1:])))
                number_indices.append(len(question_tokens) + 1)
        tok_proc = tokenize(tokenizer, tok)
        question_tokens.extend(tok_proc)



    original_program = entry["qa"]['program'] if "program" in entry["qa"] else None
    if original_program:
        program = program_tokenization(original_program)
    else:
        program = None


    return MathQAExample(
        id=this_id,
        original_question=original_question,
        question_tokens=question_tokens,
        options=options,
        answer=answer,
        numbers=numbers,
        number_indices=number_indices,
        original_program=original_program,
        program=program)


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if scores == None:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def read_examples(input_path, tokenizer, op_list, const_list, log_file):
    """Read a json file into a list of examples."""
    with open(input_path) as input_file:
        input_data = json.load(input_file)

    examples = []
    for entry in input_data:
        examples.append(read_mathqa_entry(entry, tokenizer))
        program = examples[-1].program
    return input_data, examples, op_list, const_list

def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 max_program_length,
                                 is_training,
                                 op_list,
                                 op_list_size,
                                 const_list,
                                 const_list_size,
                                 verbose=True):
    """Converts a list of DropExamples into InputFeatures."""
    unique_id = 1000000000
    res = []
    for (example_index, example) in enumerate(examples):
        features = example.convert_single_example(
            is_training=is_training,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_program_length=max_program_length,
            op_list=op_list,
            op_list_size=op_list_size,
            const_list=const_list,
            const_list_size=const_list_size,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token)

        if features:
            for feature in features:
                feature["unique_id"] = unique_id
                feature["example_index"] = example_index
                res.append(feature)
                unique_id += 1

    return res


RawResult = collections.namedtuple(
    "RawResult",
    "unique_id logits loss")


def compute_prog_from_logits(logits, max_program_length, example,
                             template=None):
    pred_prog_ids = []
    op_stack = []
    loss = 0
    for cur_step in range(max_program_length):
        cur_logits = logits[cur_step]
        cur_pred_softmax = _compute_softmax(cur_logits)
        cur_pred_token = np.argmax(cur_logits.cpu())
        loss -= np.log(cur_pred_softmax[cur_pred_token])
        pred_prog_ids.append(cur_pred_token)
        if cur_pred_token == 0:
            break
    return pred_prog_ids, loss


def compute_predictions(all_examples, all_features, all_results, n_best_size,
                        max_program_length, tokenizer, op_list, op_list_size,
                        const_list, const_list_size):
    """Computes final predictions based on logits."""
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature["example_index"]].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index", "logits"
        ])

    all_predictions = collections.OrderedDict()
    all_predictions["pred_programs"] = collections.OrderedDict()
    all_predictions["ref_programs"] = collections.OrderedDict()
    all_nbest = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        if example_index not in example_index_to_features:
            continue
        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            if feature["unique_id"] not in unique_id_to_result:
                continue
            result = unique_id_to_result[feature["unique_id"]]
            logits = result.logits
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    logits=logits))

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", "options answer program_ids program")

        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            program = example.program
            pred_prog_ids, loss = compute_prog_from_logits(pred.logits,
                                                           max_program_length,
                                                           example)
            pred_prog = indices_to_prog(pred_prog_ids,
                                        example.numbers,
                                        example.number_indices,
                                        max_seq_length,
                                        op_list, op_list_size,
                                        const_list, const_list_size
                                        )
            nbest.append(
                _NbestPrediction(
                    options=example.options,
                    answer=example.answer,
                    program_ids=pred_prog_ids,
                    program=pred_prog))

        # assert len(nbest) >= 1
        if len(nbest) == 0:
            continue
        
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["id"] = example.id
            output["options"] = entry.options
            output["ref_answer"] = entry.answer
            output["pred_prog"] = [str(prog) for prog in entry.program]
            output["ref_prog"] = example.program
            output["question_tokens"] = example.question_tokens
            output["numbers"] = example.numbers
            output["number_indices"] = example.number_indices
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions["pred_programs"][example_index] = nbest_json[0]["pred_prog"]
        all_predictions["ref_programs"][example_index] = nbest_json[0]["ref_prog"]
        all_nbest[example_index] = nbest_json

    return all_predictions, all_nbest


def write_predictions(all_predictions, output_prediction_file):
    """Writes final predictions in json format."""

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")


def process_row(row_in):

    row_out = []
    invalid_flag = 0

    for num in row_in:
        num = num.replace("$", "").strip()
        num = num.split("(")[0].strip()

        num = str_to_num(num)

        if num == "n/a":
            invalid_flag = 1
            break

        row_out.append(num)

    if invalid_flag:
        return "n/a"

    return row_out


def reprog_to_seq(prog_in, is_gold):
    '''
    predicted recursive program to list program
    ["divide(", "72", "multiply(", "6", "210", ")", ")"]
    ["multiply(", "6", "210", ")", "divide(", "72", "#0", ")"]
    '''

    st = []
    res = []

    try:
        num = 0
        for tok in prog_in:
            if tok != ")":
                st.append(tok)
            else:
                this_step_vec = [")"]
                for _ in range(3):
                    this_step_vec.append(st[-1])
                    st = st[:-1]
                res.extend(this_step_vec[::-1])
                st.append("#" + str(num))
                num += 1
    except:
        if is_gold:
            raise ValueError

    return res


def eval_program(program):
    '''
    calculate the numerical results of the program
    '''

    invalid_flag = 0
    this_res = "n/a"

    try:
        program = program[:-1]  # remove EOF
        # check structure
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"

        program = "|".join(program)
        steps = program.split(")")[:-1]

        res_dict = {}

        for ind, step in enumerate(steps):
            step = step.strip()

            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()

            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            if "#" in arg1:
                arg1 = res_dict[int(arg1.replace("#", ""))]
            else:
                arg1 = str_to_num(arg1)
                if arg1 == "n/a":
                    invalid_flag = 1
                    break

            if "#" in arg2:
                arg2 = res_dict[int(arg2.replace("#", ""))]
            else:
                arg2 = str_to_num(arg2)
                if arg2 == "n/a":
                    invalid_flag = 1
                    break

            if op == "add":
                this_res = arg1 + arg2
            elif op == "subtract":
                this_res = arg1 - arg2
            elif op == "multiply":
                this_res = arg1 * arg2
            elif op == "divide":
                this_res = arg1 / arg2
            elif op == "exp":
                this_res = arg1 ** arg2

            res_dict[ind] = this_res

        if this_res != "n/a":
            this_res = round(this_res, 5)

    except:
        invalid_flag = 1

    return invalid_flag, this_res


def evaluate_result(all_nbest, json_ori, program_mode):
    '''
    execution acc
    program acc
    '''

    data = all_nbest

    with open(json_ori) as f_in:
        data_ori = json.load(f_in)

    data_dict = {}
    for each_data in data_ori:
        assert each_data["uid"] not in data_dict
        data_dict[each_data["uid"]] = each_data

    exe_correct = 0

    res_list = []
    all_res_list = []

    for tmp in data:
        each_data = data[tmp][0]
        each_id = each_data["id"]

        each_ori_data = data_dict[each_id]
        gold_res = each_ori_data["qa"]["answer"]

        pred = each_data["pred_prog"]
        gold = each_data["ref_prog"]

        if program_mode == "nest":
            if pred[-1] == "EOF":
                pred = pred[:-1]
            pred = reprog_to_seq(pred, is_gold=False)
            pred += ["EOF"]
            gold = gold[:-1]
            gold = reprog_to_seq(gold, is_gold=True)
            gold += ["EOF"]

        invalid_flag, exe_res = eval_program(pred)

        if invalid_flag == 0:
            if exe_res == gold_res:
                exe_correct += 1

        each_ori_data["qa"]["predicted"] = pred

        if exe_res != gold_res:
            res_list.append(each_ori_data)
        all_res_list.append(each_ori_data)

    exe_acc = float(exe_correct) / len(data)

    print("All: ", len(data))
    print("Exe acc: ", exe_acc)

    return exe_acc
