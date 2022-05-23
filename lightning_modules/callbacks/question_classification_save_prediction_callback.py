import os
import json
import torch
import pytorch_lightning as pl

from typing import Any, Dict, Optional, List
from pytorch_lightning.callbacks import Callback
from pathlib import Path


class SavePredictionCallback(Callback):
    def __init__(self, test_set: str, input_dir: str, output_dir: str):
        self.test_set = test_set
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.predictions = []

    def on_predict_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", 
                                outputs: List[Dict[str, Any]], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        preds = outputs["preds"].detach().cpu().numpy()

        for i, uid in enumerate(outputs["uids"]):
            result = {
                "uid": uid,
                "pred": "arithmetic" if int(preds[i]) == 1 else "span_selection",
            }
            self.predictions.append(result)

    def on_predict_epoch_end(self, trainer, pl_module, outputs) -> None:
        # save the predictions
        os.makedirs(self.output_dir, exist_ok=True)
        output_prediction_file = os.path.join(self.output_dir, f"{self.test_set}.json")

        json.dump(self.predictions, open(output_prediction_file, "w"), indent = 4)
        print(f"generate {self.test_set} inference file in {output_prediction_file}")

        self.predictions = []
