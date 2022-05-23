import os
import json
import torch
import pytorch_lightning as pl

from typing import Any, Dict, Optional, List
from pytorch_lightning.callbacks import Callback
from pathlib import Path
from utils.retriever_utils import *

class SavePredictionCallback(Callback):
    def __init__(self, test_set: str, input_dir: str, output_dir: str):
        self.test_set = test_set
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.predictions = []

    def on_predict_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", 
                                outputs: List[Dict[str, Any]], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.predictions.extend(outputs)

    def on_predict_epoch_end(self, trainer, pl_module, outputs) -> None:
        # save the predictions
        all_logits = []
        all_filename_id = []
        all_ind = []
        for output_dict in self.predictions:
            all_logits.append(output_dict["logits"])
            all_filename_id.append(output_dict["filename_id"])
            all_ind.append(output_dict["ind"])
            
        test_file = os.path.join(self.input_dir, f"{self.test_set}.json")

        os.makedirs(self.output_dir, exist_ok=True)
        output_prediction_file = os.path.join(self.output_dir, f"{self.test_set}.json")

        retrieve_inference(all_logits, all_filename_id, all_ind, output_prediction_file, test_file)
        print(f"generate {self.test_set} inference file in {output_prediction_file}")

        self.predictions = []
