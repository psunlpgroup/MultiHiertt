import torch
import os, json
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
import datasets


class QuestionClassificationModel(LightningModule):
    
    def __init__(self, 
                 model_name: str,
                 warmup_steps: int = 0,
                 optimizer: Dict[str, Any] = None,
                 lr_scheduler: Dict[str, Any] = None,
                 test_set: str = "dev",
                 ) -> None:

        super().__init__()
        self.transformer_model_name = model_name

        self.model_config = AutoConfig.from_pretrained(self.transformer_model_name, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.transformer_model_name, config=self.model_config)
        self.metric = datasets.load_metric('precision')

        self.test_set = test_set
        
        self.warmup_steps = warmup_steps
        self.opt_params = optimizer["init_args"]
        self.lrs_params = lr_scheduler

    def forward(self, **inputs) -> List[Dict[str, Any]]:
        return self.model(**inputs)


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
        labels = torch.tensor(batch["labels"]).to("cuda")

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("loss", loss, on_step=True, on_epoch=True)
        
        return loss

    def on_fit_start(self) -> None:
# save the code using wandb
        if self.logger: 
            # if logger is initialized, save the code
            self.logger[0].log_code()
        else:
            print("logger is not initialized, code will not be saved")  

        return super().on_fit_start()

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")
        labels = torch.tensor(batch["labels"]).to("cuda")

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        self.log("val_loss", loss)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        labels = batch["labels"]
        uids = batch["uid"]
        
        return {"preds": preds, "labels": labels, "uids": uids}
    
    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()

        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        print("precision:", self.metric.compute(predictions=preds, references=labels))

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        self.validation_epoch_end(outputs)
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["input_mask"]).to("cuda")

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=None)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        uids = batch["uid"]
        
        return {"preds": preds, "uids": uids}
        
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