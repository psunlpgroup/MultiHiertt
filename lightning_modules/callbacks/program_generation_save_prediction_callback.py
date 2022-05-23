import os
import json
import torch
import pytorch_lightning as pl

from typing import Any, Dict, Optional, List
from pytorch_lightning.callbacks import Callback
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModel
from utils.program_generation_utils import *
from utils.utils import * 

op_list, const_list = get_op_const_list()

class SavePredictionCallback(Callback):
    def __init__(self, test_set: str, input_dir: str, output_dir: str, model_name: str, program_length: int, input_length: int, entity_name: str):
        self.test_set = test_set
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.predictions = []

        self.model_name = model_name
        self.program_length = program_length
        self.input_length = input_length
        self.entity_name = entity_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def on_predict_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", 
                                outputs: List[Dict[str, Any]], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.predictions.extend(outputs)

    def on_predict_epoch_end(self, trainer, pl_module, outputs) -> None:
        test_file = os.path.join(self.input_dir, f"{self.test_set}_inference.json")
        
        with open(test_file) as input_file:
            input_data = json.load(input_file)

        data_ori = []
        for entry in input_data:
            example = read_mathqa_entry(entry, self.tokenizer, self.entity_name)
            if example:
                data_ori.append(example)
            
        kwargs = {
                "examples": data_ori,
                "tokenizer": self.tokenizer,
                "max_seq_length": self.input_length,
                "max_program_length": self.program_length,
                "is_training": False,
                "op_list": op_list,
                "op_list_size": len(op_list),
                "const_list": const_list,
                "const_list_size": len(const_list),
                "verbose": True
            }
            
        data = convert_examples_to_features(**kwargs)
        
        all_results = []

        
        for output_dict in self.predictions:
            all_results.append(
                RawResult(
                    unique_id=output_dict["unique_id"],
                    logits=output_dict["logits"],
                    loss=None
                ))
            
        all_predictions, all_nbest = compute_predictions(
            data_ori,
            data,
            all_results,
            n_best_size=1,
            max_program_length=self.program_length,
            tokenizer=self.tokenizer,
            op_list=op_list,
            op_list_size=len(op_list),
            const_list=const_list,
            const_list_size=len(const_list))
        
        output_data = []
        for i in all_nbest:
            pred = all_nbest[i][0]
            uid = pred["id"]
            pred_prog = pred["pred_prog"]
            invalid_flag, pred_ans = eval_program(pred_prog)
            if invalid_flag == 1:
                pred_ans = -float("inf")
            output_data.append({"uid": uid, "predicted_ans": pred_ans, "predicted_program": pred_prog})
        
        os.makedirs(self.output_dir, exist_ok=True)

        output_file = os.path.join(self.output_dir, f"{self.test_set}_predictions.json")
        json.dump(output_data, open(output_file, "w"), indent = 4)
        print(f"Predictions saved to {output_file}")
        # reset the predictions
        self.predictions = []
