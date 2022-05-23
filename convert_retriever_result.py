import argparse
import collections
import json
import os
import sys
import random


### for single sent retrieve
def convert_train(json_in, json_out, topn, max_len = 256):
    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        try:
            gold_inds = []
            cur_len = 0
            table_retrieved = each_data["table_retrieved_all"]
            text_retrieved = each_data["text_retrieved_all"]
            all_retrieved = table_retrieved + text_retrieved

            gold_table_inds = each_data["qa"]["table_evidence"]
            gold_text_inds = each_data["qa"]["text_evidence"]
            for ind in gold_table_inds:
                gold_inds.append(ind)
                cur_len += len(each_data["table_description"][ind].split())

            for ind in gold_text_inds:
                gold_inds.append(ind)
                try:
                    cur_len += len(each_data["paragraphs"][ind].split())
                except:
                    continue

            false_retrieved = []
            for tmp in all_retrieved:
                if tmp["ind"] not in gold_inds:
                    false_retrieved.append(tmp)

            sorted_dict = sorted(false_retrieved, key=lambda kv: kv["score"], reverse=True)
            res_n = topn - len(gold_inds)
            
            other_cands = []
            while res_n > 0 and cur_len < max_len:
                next_false_retrieved = sorted_dict.pop(0)
                if next_false_retrieved["score"] < 0:
                    break

                if type(next_false_retrieved["ind"]) == int:
                    cur_len += len(each_data["paragraphs"][next_false_retrieved["ind"]].split())
                    other_cands.append(next_false_retrieved["ind"])
                    res_n -= 1
                else:
                    cur_len += len(each_data["table_description"][next_false_retrieved["ind"]].split())
                    other_cands.append(next_false_retrieved["ind"])
                    res_n -= 1
            
            # recover the original order in the document
            input_inds = gold_inds + other_cands
            context = get_context(each_data, input_inds)
            each_data["model_input"] = context
            del each_data["table_retrieved_all"]
            del each_data["text_retrieved_all"]
        except:
            print(each_data["uid"])

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

def convert_test(retriever_json_in, question_classification_json_in, json_out, topn, max_len = 256):
    with open(retriever_json_in) as f_in:
        data = json.load(f_in)
    
    with open(question_classification_json_in) as f_in:
        qc_data = json.load(f_in)

    qc_map = {}
    for example in qc_data:
        qc_map[example["uid"]] = example["pred"]

    for each_data in data:
        cur_len = 0
        table_retrieved = each_data["table_retrieved_all"]
        text_retrieved = each_data["text_retrieved_all"]
        all_retrieved = table_retrieved + text_retrieved

        cands_retrieved = []
        for tmp in all_retrieved:
            cands_retrieved.append(tmp)

        sorted_dict = sorted(cands_retrieved, key=lambda kv: kv["score"], reverse=True)
        res_n = topn
        
        other_cands = []

        while res_n > 0 and cur_len < max_len:
            next_false_retrieved = sorted_dict.pop(0)
            if next_false_retrieved["score"] < 0:
                break

            if type(next_false_retrieved["ind"]) == int:
                cur_len += len(each_data["paragraphs"][next_false_retrieved["ind"]].split())
                other_cands.append(next_false_retrieved["ind"])
                res_n -= 1
            else:
                cur_len += len(each_data["table_description"][next_false_retrieved["ind"]].split())
                other_cands.append(next_false_retrieved["ind"])
                res_n -= 1
        
        # recover the original order in the document
        input_inds = other_cands
        context = get_context(each_data, input_inds)
        each_data["model_input"] = context

        each_data["qa"]["predicted_question_type"] = qc_map[each_data["uid"]]
        del each_data["table_retrieved_all"]
        del each_data["text_retrieved_all"]


    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

def get_context(each_data, input_inds):
    context = []
    table_sent_map = get_table_sent_map(each_data["paragraphs"])
    inds_map = {}
    for ind in input_inds:
        if type(ind) == str:
            table_ind = int(ind.split("-")[0])
            sent_ind = table_sent_map[table_ind]
            if sent_ind not in inds_map:
                inds_map[sent_ind] = [ind]
            else:
                if type(inds_map[sent_ind]) == int:
                    inds_map[sent_ind] = [ind]
                else:
                    inds_map[sent_ind].append(ind)
        else:
            if ind not in inds_map:
                inds_map[ind] = ind
    
    for sent_ind in sorted(inds_map.keys()):
        if type(inds_map[sent_ind]) != list:
            context.append(sent_ind)
        else:
            for table_ind in sorted(inds_map[sent_ind]):
                context.append(table_ind)
    
    return context

def get_table_sent_map(paragraphs):
    table_index = 0
    table_sent_map = {}
    for i, sent in enumerate(paragraphs):
        if sent.startswith("## Table "):
            table_sent_map[table_index] = i
            table_index += 1
    return table_sent_map



if __name__ == '__main__':
    
    json_dir_in = "output/retriever_output"
    question_classification_json_dir_in = "output/question_classification_output"
    json_dir_out = "dataset/reasoning_module_input"
    os.makedirs(json_dir_out, exist_ok = True)
    
    topn, max_len = 10, 256
    
    mode_names = ["train", "test", "dev"]
    for mode in mode_names:
        json_in = os.path.join(json_dir_in, f"{mode}.json")
        question_classification_json_in = os.path.join(question_classification_json_dir_in, f"{mode}.json")
        json_out_train = os.path.join(json_dir_out, mode + "_training.json")
        json_out_inference = os.path.join(json_dir_out, mode + "_inference.json")

        if mode == "train":
            convert_train(json_in, json_out_train, topn, max_len)
        if mode == "dev":
            convert_train(json_in, json_out_train, topn, max_len)
            convert_test(json_in, question_classification_json_in, json_out_inference, topn, max_len)
        elif mode == "test":
            convert_test(json_in, question_classification_json_in, json_out_inference, topn, max_len)
        
        print(f"Convert {mode} set done")
        
