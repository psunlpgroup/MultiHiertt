import os
import json
import torch
import pytorch_lightning as pl

from typing import Any, Dict, Optional, List
from pytorch_lightning.callbacks import Callback
from pathlib import Path

class SavePredictionCallback(Callback):
    def __init__(self):
        self.predictions = list()
        self.prediction_save_dir = None
        self.metrics_save_dir = None

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str]) -> None:
        if pl_module.global_rank == 0:
            # make dirs for predictions and metrics saving paths
            pred_save_dir = os.path.join(trainer.log_dir, 'predictions')
            metrics_save_dir = os.path.join(trainer.log_dir, 'metrics')

            Path(pred_save_dir).mkdir(parents=True, exist_ok=True)
            Path(metrics_save_dir).mkdir(parents=True, exist_ok=True)

            self.prediction_save_dir = pred_save_dir
            self.metrics_save_dir = metrics_save_dir

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", 
                                outputs: List[Dict[str, Any]], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.predictions.extend(outputs)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # save the predictions
        save_pred_file_path = os.path.join(self.prediction_save_dir, 
                                f'predictions_step_{trainer.global_step}_rank_{trainer.global_rank}.jsonl')
        with open(save_pred_file_path, 'w+') as f:
            for prediction in self.predictions:
                f.write(json.dumps(prediction)+'\n')
        print(f"{len(self.predictions)} predictions saved to {save_pred_file_path}")
        self.predictions = []


        if pl_module.global_rank == 0:
            self.save_metrics(trainer, pl_module)

        pl_module._rouge_metric.reset()
        pl_module._bleu_metric.reset()
        pl_module._em_metric.reset()
        pl_module._stmt_length.reset()
        pl_module._cell_stmt_num.reset()
        pl_module._edit_distance.reset()

    def save_metrics(self, trainer, pl_module) -> None:
        metrics = {}

        rouge_dict = dict([(k, float(v)) for k, v in pl_module._rouge_metric.compute().items() if k.endswith('fmeasure')])
        metrics.update(rouge_dict)
        metrics["bleu"] = float(pl_module._bleu_metric.compute())
        metrics["cell_exact_match"] = float(pl_module._em_metric.compute())
        metrics["output_stmt_len"] = float(pl_module._stmt_length.compute())
        metrics["output_stmt_num"] = float(pl_module._cell_stmt_num.compute())
        metrics["cell_edit_dist"] = float(pl_module._edit_distance.compute())

        # save the evaluation metrics
        save_metrics_file_path = os.path.join(self.metrics_save_dir, f'metrics_step_{trainer.global_step}.json')
        with open(save_metrics_file_path, 'w+') as f:
            f.write(json.dumps(metrics, indent=4))

        print(f"Eval metrics saved to {save_metrics_file_path}")