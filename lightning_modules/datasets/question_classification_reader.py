import json
import logging
import sys
import os
import torch

from typing import Dict, Iterable, List, Any, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from utils.datasets_util import right_pad_sequences

from transformers import AutoTokenizer
from utils.utils import *
from torch.utils.data import DataLoader

os.environ['TOKENIZERS_PARALLELISM']='0'

class QuestionClassificationDataset(Dataset):
    def __init__(
        self, 
        model_name: str,
        file_path: str,
        max_instances: int,
        mode: str = "train", 
        **kwargs):
        super().__init__(**kwargs)

        assert mode in ["train", "test", "valid", "predict"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.max_instances = max_instances
        self.mode = mode
        self.instances = self.read(file_path, self.tokenizer, mode)

        print(f"read {len(self.instances)} {self.mode} examples")

    def read(self, input_path: str, tokenizer, mode) -> Iterable[Dict[str, Any]]:
        with open(input_path) as input_file:
            if self.max_instances > 0:
                input_data = json.load(input_file)[:self.max_instances]
            else:
                input_data = json.load(input_file)

        data = []
        for entry in input_data:
            feature = {}
            input_text_encoded = tokenizer.encode_plus(entry["qa"]["question"],
                                    max_length=128,
                                    pad_to_max_length=True)
            input_ids = input_text_encoded["input_ids"]
            input_mask = input_text_encoded["attention_mask"]

            feature = {
                "uid": entry["uid"],
                "question": entry["qa"]["question"],
                "input_ids": input_ids,
                "input_mask": input_mask,
            }

            if mode != "predict":
                feature["labels"] = 1 if entry["qa"]["question_type"] == "arithmetic" else 0
            
            data.append(feature)

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
        
def customized_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    result_dict = {}
    for k in examples[0].keys():
        try:
            if k == "labels":
                result_dict[k] = torch.tensor([example[k] for example in examples])
            else:   
                result_dict[k] = right_pad_sequences([torch.tensor(ex[k]) for ex in examples], 
                                        batch_first=True, padding_value=0)
        except:
            result_dict[k] = [ex[k] for ex in examples]
    return result_dict

class QuestionClassificationDataModule(LightningDataModule):
    def __init__(self, 
                model_name: str,
                batch_size: int = 1, 
                val_batch_size: int = 1,
                train_file_path: str = None,
                val_file_path: str = None,
                num_workers: int = 8,
                train_max_instances: int = sys.maxsize,
                val_max_instances: int = sys.maxsize):
        super().__init__()
        self.transformer_model_name = model_name

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path

        self.train_max_instances = train_max_instances
        self.val_max_instances = val_max_instances
        
        self.train_data = None
        self.val_data = None

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        assert stage in ["fit", "validate", "test"]

        train_data = QuestionClassificationDataset(model_name = self.transformer_model_name,
                                file_path=self.train_file_path,
                                max_instances=self.train_max_instances, 
                                mode="train")
        self.train_data = train_data

        val_data = QuestionClassificationDataset(model_name = self.transformer_model_name,
                                file_path=self.val_file_path,
                                max_instances=self.val_max_instances, 
                                mode="valid")
        self.val_data = val_data 

    def train_dataloader(self):
        if self.train_data is None:
            self.setup(stage="fit")

        dtloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=customized_collate_fn, num_workers = self.num_workers)
        return dtloader

    def val_dataloader(self):
        if self.val_data is None:
            self.setup(stage="validate")

        dtloader = DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=True, drop_last=False, collate_fn=customized_collate_fn, num_workers = self.num_workers)
        return dtloader
    

class QuestionClassificationPredictionDataModule(LightningDataModule):
    def __init__(self, 
                model_name: str,
                batch_size: int = 1, 
                num_workers: int = 8,
                test_file_path: str = None,
                test_max_instances: int = sys.maxsize):
        super().__init__()
        self.transformer_model_name = model_name

        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.test_file_path = test_file_path

        self.test_max_instances = test_max_instances
        
        self.test_data = None

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        assert stage in ["test", "predict"]
        
        test_data = QuestionClassificationDataset(model_name = self.transformer_model_name,
                                file_path=self.test_file_path,
                                max_instances=self.test_max_instances, 
                                mode=stage)
        self.test_data = test_data

    def test_dataloader(self):
        if self.test_data is None:
            self.setup(stage="test")
            
        dtloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=customized_collate_fn, num_workers = self.num_workers)
        return dtloader

    def predict_dataloader(self):
        if self.test_data is None:
            self.setup(stage="predict")
            
        dtloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=customized_collate_fn, num_workers = self.num_workers)
        return dtloader