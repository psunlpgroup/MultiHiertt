import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from utils.span_selection_utils import *
from utils.utils import *
from transformers import T5ForConditionalGeneration, AutoTokenizer
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup

class SpanSelectionModel(LightningModule):
    
    def __init__(self, 
                 model_name: str,
                 optimizer: Dict[str, Any] = None,
                 lr_scheduler: Dict[str, Any] = None,
                 load_ckpt_file: str = None,
                 test_set: str = "dev_training.json",
                 input_dir: str = "dataset/reasoning_module_input",
                 ) -> None:

        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        self.lrs_params = lr_scheduler
        self.opt_params = optimizer["init_args"]
        
        self.test_set = test_set
        self.input_dir = input_dir

        self.predictions = []

    def forward(self, input_ids, attention_mask, label_ids) -> List[Dict[str, Any]]:
        input_ids = torch.tensor(input_ids).to("cuda")
        attention_mask = torch.tensor(attention_mask).to("cuda")
        label_ids = torch.tensor(label_ids).to("cuda")
        
        loss = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels = label_ids).get("loss")

        return {"loss": loss}


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["input_mask"]
        label_ids = batch["label_ids"]
        label_ids = torch.tensor(label_ids).to("cuda")
        
        
        output_dict = self(input_ids, attention_mask, label_ids)
        
        loss = output_dict["loss"]
    
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

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
        input_ids = torch.tensor(input_ids).to("cuda")
        attention_mask = torch.tensor(attention_mask).to("cuda")
        
        labels = batch["label"]
        label_ids = batch["label_ids"]
        label_ids = torch.tensor(label_ids).to("cuda")

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        
        
        output_dict = self(input_ids, attention_mask, label_ids = label_ids)
        
        unique_ids = batch["uid"]
        output_dict["preds"] = {}
        for i, unique_id in enumerate(unique_ids):
            output_dict["preds"][unique_id] = (preds[i], labels[i])
            
        loss = output_dict["loss"]

        self.log("val_loss", loss)
        return output_dict

    def validation_step_end(self, outputs: List[Dict[str, Any]]) -> None:
        self.predictions.append(outputs)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        all_filename_id = []
        all_preds = []
        all_labels = []
        for output_dict in self.predictions:
            preds = output_dict["preds"]
            for unique_id, pred in preds.items():
                all_filename_id.append(unique_id)
                all_preds.append(pred[0])
                all_labels.append(pred[1])
                
        
        test_file = os.path.join(self.input_dir, self.test_set)
        res = 0
        res = span_selection_evaluate(all_preds, all_filename_id, test_file)
        
        self.log("exact_match", res[0])
        self.log("f1", res[1])
        print(f"exact_match: {res[0]}, f1: {res[1]}")
        # reset the predictions
        self.predictions = []
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        input_ids = batch["input_ids"]
        attention_mask = batch["input_mask"]
        input_ids = torch.tensor(input_ids).to("cuda")
        attention_mask = torch.tensor(attention_mask).to("cuda")

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        
        unique_ids = batch["uid"]
        output_dict = []
        for i, unique_id in enumerate(unique_ids):
            output_dict.append({"uid": unique_id, "preds": preds[i]})
        return output_dict

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