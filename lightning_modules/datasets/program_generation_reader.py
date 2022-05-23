import json
import logging
import sys
import os
import torch

from typing import Dict, Iterable, List, Any, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from utils.program_generation_utils import *
from utils.utils import *
from utils.datasets_util import right_pad_sequences
from torch.utils.data import DataLoader

os.environ['TOKENIZERS_PARALLELISM']='0'
op_list, const_list = get_op_const_list()
reserved_token_size = len(op_list) + len(const_list)

class ProgramGenerationDataset(Dataset):
    def __init__(
        self, 
        model_name: str,
        file_path: str,
        max_seq_length: int,
        max_program_length: int,
        max_instances: int,
        mode: str = "train", 
        entity_name: str = "question_type",
        **kwargs):
        super().__init__(**kwargs)

        assert mode in ["train", "test", "valid"]

        self.max_seq_length = max_seq_length
        self.max_program_length = max_program_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        self.max_instances = max_instances
        self.mode = mode
        self.entity_name = entity_name
        self.instances = self.read(file_path, self.tokenizer, self.entity_name)

        print(f"read {len(self.instances)} {self.mode} examples") 

    def read(self, input_path: str, tokenizer, entity_name: str) -> Iterable[Dict[str, Any]]:
        with open(input_path) as input_file:
            if self.max_instances > 0:
                input_data = json.load(input_file)[:self.max_instances]
            else:
                input_data = json.load(input_file)

        examples = []
        for entry in input_data:
            example = read_mathqa_entry(entry, tokenizer, entity_name)
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
            result_dict[k] = right_pad_sequences([torch.tensor(ex[k]) for ex in examples], 
                                    batch_first=True, padding_value=0)
        except:
            result_dict[k] = [ex[k] for ex in examples]
    return result_dict

class ProgramGenerationDataModule(LightningDataModule):
    def __init__(self, 
                model_name: str,
                max_seq_length: int,
                max_program_length: int,
                batch_size: int = 1, 
                val_batch_size: int = 1,
                train_file_path: str = None,
                val_file_path: str = None,
                train_max_instances: int = sys.maxsize,
                val_max_instances: int = sys.maxsize,
                entity_name: str = "question_type"):
        super().__init__()
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.max_program_length = max_program_length
        
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.train_file_path = train_file_path
        self.val_file_path = val_file_path

        self.train_max_instances = train_max_instances
        self.val_max_instances = val_max_instances

        self.entity_name = entity_name

        self.train_data = None
        self.val_data = None

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        assert stage in ["fit", "validate", "test"]

        train_data = ProgramGenerationDataset(model_name=self.model_name,
                                file_path=self.train_file_path, 
                                  max_seq_length = self.max_seq_length, 
                                  max_program_length = self.max_program_length, 
                                  max_instances = self.train_max_instances, 
                                  mode = "train", 
                                  entity_name = self.entity_name)

        self.train_data = train_data

        val_data = ProgramGenerationDataset(model_name=self.model_name,
                                file_path=self.val_file_path,
                                max_seq_length = self.max_seq_length, 
                                max_program_length = self.max_program_length, 
                                max_instances=self.val_max_instances, 
                                mode="valid", 
                                entity_name = self.entity_name)
        self.val_data = val_data 

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


class ProgramGenerationPredictionDataModule(LightningDataModule):
    def __init__(self, 
                model_name: str,
                max_seq_length: int,
                max_program_length: int,
                batch_size: int = 1, 
                test_file_path: str = None,
                test_max_instances: int = sys.maxsize,
                entity_name: str = "question_type"):
        super().__init__()
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.max_program_length = max_program_length
        
        self.batch_size = batch_size
        
        self.test_file_path = test_file_path
        self.test_max_instances = test_max_instances

        self.test_data = None

        self.entity_name = entity_name

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        assert stage in ["test", "predict"]
        
        test_data = ProgramGenerationDataset(model_name=self.model_name,
                                file_path=self.test_file_path,
                                max_seq_length = self.max_seq_length, 
                                max_program_length = self.max_program_length, 
                                max_instances=self.test_max_instances, 
                                mode="test",
                                entity_name = self.entity_name)
        self.test_data = test_data

    def test_dataloader(self):
        if self.test_data is None:
            self.setup(stage="test")
            
        dtloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=customized_collate_fn)
        return dtloader
    
    def predict_dataloader(self):
        if self.test_data is None:
            self.setup(stage="predict")
            
        dtloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=customized_collate_fn)
        return dtloader