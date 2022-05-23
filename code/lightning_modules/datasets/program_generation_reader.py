import json
import logging
import sys
import os
import torch

from typing import Dict, Iterable, List, Any, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from lightning_modules.models.seq2seq_model_util import get_model, left_pad_sequences

from transformers import BertTokenizer, RobertaTokenizer
import program_generation_utils
from program_generation_utils import *
from utils import *
from torch.utils.data import DataLoader

os.environ['TOKENIZERS_PARALLELISM']='0'
op_list_file = "/home/lily/yz979/code/LogicNLG_2/retriever/operation_list.txt"
log_file = "log.txt"
const_list_file = "/home/lily/yz979/code/LogicNLG_2/retriever/constant_list.txt"
op_list = read_txt(op_list_file, log_file)
op_list = [op + '(' for op in op_list]
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
const_list = read_txt(const_list_file, log_file)
const_list = [const.lower().replace('.', '_') for const in const_list]
reserved_token_size = len(op_list) + len(const_list)
valid_file = "dataset/data/val.json"

class FinQADataset(Dataset):
    def __init__(
        self, 
        model_name: str,
        file_path: str,
        max_seq_length: int,
        max_program_length: int,
        max_instances: int,
        mode: str = "train", 
        **kwargs):
        super().__init__(**kwargs)

        assert mode in ["train", "test", "valid"]

        self.max_seq_length = max_seq_length
        self.max_program_length = max_program_length
        
        if model_name.startswith("bert"):
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        elif model_name.startswith("roberta"):
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            
        self.max_instances = max_instances
        self.mode = mode
        self.instances = self.read(file_path, self.tokenizer)


    def read(self, input_path: str, tokenizer) -> Iterable[Dict[str, Any]]:
        with open(input_path) as input_file:
            input_data = json.load(input_file)[:self.max_instances]

        examples = []
        for entry in input_data:
            example = program_generation_utils.read_mathqa_entry(entry, tokenizer)
            if example:
                examples.append(example)


        kwargs = {
                "examples": examples,
                "tokenizer": tokenizer,
                "max_seq_length": self.max_seq_length,
                "max_program_length": self.max_program_length,
                "is_training": True,
                "op_list": op_list,
                "op_list_size": len(op_list),
                "const_list": const_list,
                "const_list_size": len(const_list),
                "verbose": True
            }
        
        if self.mode != "train":
            kwargs["is_training"] = False
            
        data = convert_examples_to_features(**kwargs)
        return data

    def __getitem__(self, idx: int):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def truncate(self, max_instances):
        truncated_instances = self.instances[max_instances:]
        self.instances = self.instances[:max_instances]
        return truncated_instances

    def extend(self, instances):
        self.instances.extend(instances)
        
def customized_collate_fn(examples: List) -> Dict[str, Any]:
    result_dict = {}
    for k in examples[0].keys():
        try:
            result_dict[k] = left_pad_sequences([torch.tensor(ex[k]) for ex in examples], 
                                    batch_first=True, padding_value=0)
        except:
            result_dict[k] = [ex[k] for ex in examples]
    return result_dict

class FinQADataModule(LightningDataModule):
    def __init__(self, 
                model_name: str,
                max_seq_length: int,
                max_program_length: int,
                batch_size: int = 1, 
                val_batch_size: int = 1,
                train_file_path: str = None,
                val_file_path: str = None,
                test_file_path: str = None,
                train_max_instances: int = sys.maxsize,
                val_max_instances: int = sys.maxsize):
        super().__init__()
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.max_program_length = max_program_length
        
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.test_file_path = test_file_path

        self.train_max_instances = train_max_instances
        self.val_max_instances = val_max_instances

        self.train_data = None
        self.val_data = None

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        assert stage in ["fit", "validate", "test"]

        train_data = FinQADataset(model_name=self.model_name,
                                file_path=self.train_file_path, 
                                  max_seq_length = self.max_seq_length, 
                                  max_program_length = self.max_program_length, 
                                  max_instances = self.train_max_instances, 
                                  mode = "train")

        self.train_data = train_data

        val_data = FinQADataset(model_name=self.model_name,
                                file_path=self.val_file_path,
                                max_seq_length = self.max_seq_length, 
                                max_program_length = self.max_program_length, 
                                max_instances=self.val_max_instances, 
                                mode="valid")
        self.val_data = val_data 
        
        test_data = FinQADataset(model_name=self.model_name,
                                file_path=self.test_file_path,
                                max_seq_length = self.max_seq_length, 
                                max_program_length = self.max_program_length, 
                                max_instances=self.val_max_instances, 
                                mode="test")
        self.test_data = test_data

    def train_dataloader(self):
        if self.train_data is None:
            self.setup(stage="fit")

        dtloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=customized_collate_fn)
        return dtloader

    def val_dataloader(self):
        if self.val_data is None:
            self.setup(stage="validate")

        dtloader = DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=False, drop_last=False, collate_fn=customized_collate_fn)
        return dtloader

    def test_dataloader(self):
        if self.test_data is None:
            self.setup(stage="test")
            
        dtloader = DataLoader(self.test_data, batch_size=self.val_batch_size, shuffle=False, drop_last=False, collate_fn=customized_collate_fn)
        return dtloader