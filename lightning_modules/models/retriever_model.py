import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from utils.retriever_utils import *
from utils.utils import *
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup


class RetrieverModel(LightningModule):
    
    def __init__(self, 
                 transformer_model_name: str,
                 topn: int, 
                 dropout_rate: float, 
                 warmup_steps: int = 0,
                 optimizer: Dict[str, Any] = None,
                 lr_scheduler: Dict[str, Any] = None,
                 ) -> None:

        super().__init__()
        
        self.topn = topn
        self.dropout_rate = dropout_rate
        self.transformer_model_name = transformer_model_name

        self.model = AutoModel.from_pretrained(self.transformer_model_name)
        self.warmup_steps = warmup_steps
        self.model_config = AutoConfig.from_pretrained(self.transformer_model_name)
        
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        self.predictions: List[Dict[str, Any]] = []

        self.opt_params = optimizer["init_args"]
        self.lrs_params = lr_scheduler

        hidden_size = self.model_config.hidden_size
        self.cls_prj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.cls_dropout = nn.Dropout(self.dropout_rate)

        self.cls_final = nn.Linear(hidden_size, 2, bias=True)
        
        self.predictions = []

    def forward(self, input_ids, attention_mask, segment_ids, metadata) -> List[Dict[str, Any]]:

        input_ids = torch.tensor(input_ids).to("cuda")
        attention_mask = torch.tensor(attention_mask).to("cuda")
        segment_ids = torch.tensor(segment_ids).to("cuda")
        
        bert_outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        bert_sequence_output = bert_outputs.last_hidden_state

        bert_pooled_output = bert_sequence_output[:, 0, :]

        pooled_output = self.cls_prj(bert_pooled_output)
        pooled_output = self.cls_dropout(pooled_output)

        logits = self.cls_final(pooled_output)
        output_dicts = []
        for i in range(len(metadata)):
            output_dicts.append({"logits": logits[i], "filename_id": metadata[i]["filename_id"], "ind": metadata[i]["ind"]})
        return output_dicts


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]
        labels = batch["label"]
        labels = torch.tensor(labels).to("cuda")
        
        metadata = [{"filename_id": filename_id, "ind": ind} for filename_id, ind in zip(batch["filename_id"], batch["ind"])]
        
        output_dicts = self(input_ids, attention_mask, segment_ids, metadata)
        
        logits = []
        for output_dict in output_dicts:
            logits.append(output_dict["logits"])
        logits = torch.stack(logits)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        self.log("loss", loss.sum(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss.sum()}

    def on_fit_start(self) -> None:
        # save the code using wandb
        if self.logger: 
            # if logger is initialized, save the code
            self.logger[0].log_code()
        else:
            print("logger is not initialized, code will not be saved")  

        return super().on_fit_start()

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        input_ids = batch["input_ids"]
        attention_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]
        
        labels = batch["label"]
        labels = torch.tensor(labels).to("cuda")
        
        metadata = [{"filename_id": filename_id, "ind": ind} for filename_id, ind in zip(batch["filename_id"], batch["ind"])]
        
        output_dicts = self(input_ids, attention_mask, segment_ids, metadata)
        
        logits = []
        for output_dict in output_dicts:
            logits.append(output_dict["logits"])
        logits = torch.stack(logits)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        self.log("val_loss", loss)
        return output_dicts

    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        input_ids = batch["input_ids"]
        attention_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]
        
        metadata = [{"filename_id": filename_id, "ind": ind} for filename_id, ind in zip(batch["filename_id"], batch["ind"])]
        
        output_dicts = self(input_ids, attention_mask, segment_ids, metadata)
        return output_dicts
    

    def predict_step_end(self, outputs: List[Dict[str, Any]]) -> None:
        self.predictions.extend(outputs)

        
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.opt_params)
        if self.lrs_params["name"] == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, **self.lrs_params["init_args"])
        elif self.lrs_params["name"] == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, **self.lrs_params["init_args"])
        elif self.lrs_params["name"] == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer, **self.lrs_params["init_args"])
        else:
            raise ValueError(f"lr_scheduler {self.lrs_params} is not supported")

        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step"
                    }
                }