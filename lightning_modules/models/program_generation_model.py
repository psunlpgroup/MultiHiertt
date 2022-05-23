import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from utils.program_generation_utils import *
from utils.utils import *
from transformers import AutoConfig, AutoTokenizer, AutoModel
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup

op_list, const_list = get_op_const_list()
reserved_token_size = len(op_list) + len(const_list)

class ProgramGenerationModel(LightningModule):
    
    def __init__(self, 
                 model_name: str,
                 program_length: int,
                 input_length: int,
                 max_step_ind: int,
                 dropout_rate: float, 
                 num_decoder_layers: int, 
                 n_best_size: int,
                 sep_attention: bool,
                 layer_norm: bool,
                 warmup_steps: int = 0,
                 optimizer: Dict[str, Any] = None,
                 lr_scheduler: Dict[str, Any] = None,
                 test_set: str = "dev_training.json",
                 entity_name: str = "question_type",
                 input_dir: str = "dataset/program_generator_input",
                 load_ckpt_file: str = None,
                 ) -> None:

        super().__init__()
        
        self.model_name = model_name
        self.const_list = const_list
        self.op_list = op_list
        self.op_list_size = len(op_list)
        self.const_list_size = len(const_list)
        self.reserved_token_size = self.op_list_size + self.const_list_size
        self.max_step_ind = max_step_ind
        
        self.program_length = program_length
        self.input_length = input_length
        self.num_decoder_layers = num_decoder_layers
        self.n_best_size = n_best_size
        
        self.test_set = test_set
        self.entity_name = entity_name
        self.input_dir = input_dir

        self.sep_attention = sep_attention
        self.layer_norm = layer_norm
        
        self.reserved_ind = nn.Parameter(torch.arange(
            0, self.reserved_token_size), requires_grad=False)
        self.reserved_go = nn.Parameter(torch.arange(op_list.index(
            'GO'), op_list.index('GO') + 1), requires_grad=False)

        self.reserved_para = nn.Parameter(torch.arange(op_list.index(
            ')'), op_list.index(')') + 1), requires_grad=False)

        # masking for decoidng for test time
        op_ones = nn.Parameter(torch.ones(
            self.op_list_size), requires_grad=False)
        op_zeros = nn.Parameter(torch.zeros(
            self.op_list_size), requires_grad=False)
        other_ones = nn.Parameter(torch.ones(
            input_length + self.const_list_size), requires_grad=False)
        other_zeros = nn.Parameter(torch.zeros(
            input_length + self.const_list_size), requires_grad=False)
        self.op_only_mask = nn.Parameter(
            torch.cat((op_ones, other_zeros), 0), requires_grad=False)
        self.seq_only_mask = nn.Parameter(
            torch.cat((op_zeros, other_ones), 0), requires_grad=False)
        
        # for ")"
        para_before_ones = nn.Parameter(torch.ones(
            op_list.index(')')), requires_grad=False)
        para_after_ones = nn.Parameter(torch.ones(
            input_length + self.reserved_token_size - op_list.index(')') - 1), requires_grad=False)
        para_zero = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.para_mask = nn.Parameter(torch.cat(
            (para_before_ones, para_zero, para_after_ones), 0), requires_grad=False)
        
        # for step embedding
        # self.step_masks = []
        all_tmp_list = self.op_list + self.const_list
        self.step_masks = nn.Parameter(torch.zeros(
            self.max_step_ind, input_length + self.reserved_token_size), requires_grad=False)
        for i in range(self.max_step_ind):
            this_step_mask_ind = all_tmp_list.index("#" + str(i))
            self.step_masks[i, this_step_mask_ind] = 1.0
        

        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_config = AutoConfig.from_pretrained(self.model_name)
            
        self.hidden_size = self.model_config.hidden_size
        
        self.cls_prj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.cls_dropout = nn.Dropout(dropout_rate)

        self.seq_prj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.seq_dropout = nn.Dropout(dropout_rate)
        
        self.reserved_token_embedding = nn.Embedding(
            self.reserved_token_size, self.hidden_size)

        # attentions
        self.decoder_history_attn_prj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.decoder_history_attn_dropout = nn.Dropout(dropout_rate)

        self.question_attn_prj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.question_attn_dropout = nn.Dropout(dropout_rate)

        self.question_summary_attn_prj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.question_summary_attn_dropout = nn.Dropout(dropout_rate)

        if self.sep_attention:
            self.input_embeddings_prj = nn.Linear(
                self.hidden_size*3, self.hidden_size, bias=True)
        else:
            self.input_embeddings_prj = nn.Linear(
                self.hidden_size*2, self.hidden_size, bias=True)
        self.input_embeddings_layernorm = nn.LayerNorm([1, self.hidden_size])

        self.option_embeddings_prj = nn.Linear(
            self.hidden_size*2, self.hidden_size, bias=True)

        # decoder lstm
        self.rnn = torch.nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                 num_layers=self.num_decoder_layers, batch_first=True)

        # step vector
        self.decoder_step_proj = nn.Linear(
            3*self.hidden_size, self.hidden_size, bias=True)
        self.decoder_step_proj_dropout = nn.Dropout(dropout_rate)

        self.step_mix_proj = nn.Linear(
            self.hidden_size*2, self.hidden_size, bias=True)
        
        
        self.warmup_steps = warmup_steps
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        
        self.predictions: List[Dict[str, Any]] = []

        self.opt_params = optimizer["init_args"]
        self.lrs_params = lr_scheduler


    def forward(self, is_training, input_ids, input_mask, segment_ids, option_mask, program_ids, program_mask, metadata) -> List[Dict[str, Any]]:
        
        input_ids = torch.tensor(input_ids).to("cuda")
        input_mask = torch.tensor(input_mask).to("cuda")
        segment_ids = torch.tensor(segment_ids).to("cuda")
        option_mask = torch.tensor(option_mask).to("cuda")
        program_ids = torch.tensor(program_ids).to("cuda")
        program_mask = torch.tensor(program_mask).to("cuda")
        
        bert_outputs = self.model(
            input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        bert_sequence_output = bert_outputs.last_hidden_state
        bert_pooled_output = bert_sequence_output[:, 0, :]
        batch_size, seq_length, bert_dim = list(bert_sequence_output.size())
        split_program_ids = torch.split(program_ids, 1, dim=1)
        
        pooled_output = self.cls_prj(bert_pooled_output)
        pooled_output = self.cls_dropout(pooled_output)

        sequence_output = self.seq_prj(bert_sequence_output)
        sequence_output = self.seq_dropout(sequence_output)
        
        op_embeddings = self.reserved_token_embedding(self.reserved_ind)
        op_embeddings = op_embeddings.repeat(batch_size, 1, 1)
        
        init_decoder_output = self.reserved_token_embedding(self.reserved_go)
        decoder_output = init_decoder_output.repeat(batch_size, 1, 1)
        
        logits = []
        
        # [batch, op + seq len, hidden]
        initial_option_embeddings = torch.cat(
            [op_embeddings, sequence_output], dim=1)

        if self.sep_attention:
            decoder_history = decoder_output
        else:
            decoder_history = torch.unsqueeze(pooled_output, dim=-1)

        decoder_state_h = torch.zeros(1, batch_size, self.hidden_size, device = "cuda")
        decoder_state_c = torch.zeros(1, batch_size, self.hidden_size, device = "cuda")

        float_input_mask = input_mask.float()
        float_input_mask = torch.unsqueeze(float_input_mask, dim=-1)

        this_step_new_op_emb = initial_option_embeddings

        for cur_step in range(self.program_length):

            # decoder history att
            decoder_history_attn_vec = self.decoder_history_attn_prj(
                decoder_output)
            decoder_history_attn_vec = self.decoder_history_attn_dropout(
                decoder_history_attn_vec)

            decoder_history_attn_w = torch.matmul(
                decoder_history, torch.transpose(decoder_history_attn_vec, 1, 2))
            decoder_history_attn_w = F.softmax(decoder_history_attn_w, dim=1)

            decoder_history_ctx_embeddings = torch.matmul(
                torch.transpose(decoder_history_attn_w, 1, 2), decoder_history)

            if self.sep_attention:
                # input seq att
                question_attn_vec = self.question_attn_prj(decoder_output)
                question_attn_vec = self.question_attn_dropout(
                    question_attn_vec)

                question_attn_w = torch.matmul(
                    sequence_output, torch.transpose(question_attn_vec, 1, 2))
                question_attn_w -= 1e6 * (1 - float_input_mask)
                question_attn_w = F.softmax(question_attn_w, dim=1)

                question_ctx_embeddings = torch.matmul(
                    torch.transpose(question_attn_w, 1, 2), sequence_output)

            # another input seq att
            question_summary_vec = self.question_summary_attn_prj(
                decoder_output)
            question_summary_vec = self.question_summary_attn_dropout(
                question_summary_vec)

            question_summary_w = torch.matmul(
                sequence_output, torch.transpose(question_summary_vec, 1, 2))
            question_summary_w -= 1e6 * (1 - float_input_mask)
            question_summary_w = F.softmax(question_summary_w, dim=1)

            question_summary_embeddings = torch.matmul(
                torch.transpose(question_summary_w, 1, 2), sequence_output)

            if self.sep_attention:
                concat_input_embeddings = torch.cat([decoder_history_ctx_embeddings,
                                                     question_ctx_embeddings,
                                                     decoder_output], dim=-1)
            else:
                concat_input_embeddings = torch.cat([decoder_history_ctx_embeddings,
                                                     decoder_output], dim=-1)

            input_embeddings = self.input_embeddings_prj(
                concat_input_embeddings)

            if self.layer_norm:
                input_embeddings = self.input_embeddings_layernorm(
                    input_embeddings)

            question_option_vec = this_step_new_op_emb * question_summary_embeddings
            option_embeddings = torch.cat(
                [this_step_new_op_emb, question_option_vec], dim=-1)

            option_embeddings = self.option_embeddings_prj(option_embeddings)
            option_logits = torch.matmul(
                option_embeddings, torch.transpose(input_embeddings, 1, 2))
            option_logits = torch.squeeze(
                option_logits, dim=2)  # [batch, op + seq_len]
            option_logits -= 1e6 * (1 - option_mask)
            logits.append(option_logits)

            if is_training:
                program_index = torch.unsqueeze(
                    split_program_ids[cur_step], dim=1)
            else:
                # constrain decoding
                if cur_step % 4 == 0 or (cur_step + 1) % 4 == 0:
                    # op round
                    option_logits -= 1e6 * self.seq_only_mask
                else:
                    # number round
                    option_logits -= 1e6 * self.op_only_mask

                if (cur_step + 1) % 4 == 0:
                    # ")" round
                    option_logits -= 1e6 * self.para_mask
                    # print(program_index)

                program_index = torch.argmax(
                    option_logits, axis=-1, keepdim=True)

                program_index = torch.unsqueeze(
                    program_index, dim=1
                )

            if (cur_step + 1) % 4 == 0:
                # update op embeddings
                this_step_index = cur_step // 4
                this_step_list_index = (
                    self.op_list + self.const_list).index("#" + str(this_step_index))
                this_step_mask = self.step_masks[this_step_index, :]

                decoder_step_vec = self.decoder_step_proj(
                    concat_input_embeddings)
                decoder_step_vec = self.decoder_step_proj_dropout(
                    decoder_step_vec)
                decoder_step_vec = torch.squeeze(decoder_step_vec)

                this_step_new_emb = decoder_step_vec  # [batch, hidden]

                this_step_new_emb = torch.unsqueeze(this_step_new_emb, 1)
                this_step_new_emb = this_step_new_emb.repeat(
                    1, self.reserved_token_size+self.input_length, 1)  # [batch, op seq, hidden]

                this_step_mask = torch.unsqueeze(
                    this_step_mask, 0)  # [1, op seq]
                # print(this_step_mask)

                this_step_mask = torch.unsqueeze(
                    this_step_mask, 2)  # [1, op seq, 1]
                this_step_mask = this_step_mask.repeat(
                    batch_size, 1, self.hidden_size)  # [batch, op seq, hidden]

                this_step_new_op_emb = torch.where(
                    this_step_mask > 0, this_step_new_emb, initial_option_embeddings)

            # print(program_index.size())
            program_index = torch.repeat_interleave(
                program_index, self.hidden_size, dim=2)  # [batch, 1, hidden]

            input_program_embeddings = torch.gather(
                option_embeddings, dim=1, index=program_index)

            decoder_output, (decoder_state_h, decoder_state_c) = self.rnn(
                input_program_embeddings, (decoder_state_h, decoder_state_c))
            decoder_history = torch.cat(
                [decoder_history, input_program_embeddings], dim=1)

        
        logits = torch.stack(logits, dim=1)

        output_dicts = []
        for i in range(len(metadata)):
            output_dicts.append({"logits": logits[i], "unique_id": metadata[i]["unique_id"]})
        return output_dicts


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]
        program_ids = batch["program_ids"]
        program_mask = batch["program_mask"]
        option_mask = batch["option_mask"]
        is_training = True
        
        program_ids = torch.tensor(program_ids).to("cuda")
        program_mask = torch.tensor(program_mask).to("cuda")
        
        metadata = [{"unique_id": filename_id} for filename_id in batch["unique_id"]]
        
        output_dicts = self(is_training, input_ids, input_mask, segment_ids, option_mask, program_ids, program_mask, metadata)
        
        logits = []
        for output_dict in output_dicts:
            logits.append(output_dict["logits"])
        logits = torch.stack(logits)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), program_ids.view(-1))
        loss = loss * program_mask.view(-1)
        
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
        input_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]
        option_mask = batch["option_mask"]
        program_ids = batch["program_ids"]
        program_mask = batch["program_mask"]
        is_training = False
        
        program_ids = torch.tensor(program_ids).to("cuda")
        
        metadata = [{"unique_id": filename_id} for filename_id in batch["unique_id"]]
        
        output_dicts = self(is_training, input_ids, input_mask, segment_ids, option_mask, program_ids, program_mask, metadata)
        
        logits = []
        for output_dict in output_dicts:
            logits.append(output_dict["logits"])
        logits = torch.stack(logits)
        
        loss = self.criterion(logits.view(-1, logits.shape[-1]), program_ids.view(-1))
        self.log("val_loss", loss)
        
        return output_dicts
    
    def validation_step_end(self, outputs: List[Dict[str, Any]]) -> None:
        self.predictions.extend(outputs)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        test_file = os.path.join(self.input_dir, self.test_set)
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
            n_best_size=self.n_best_size,
            max_program_length=self.program_length,
            tokenizer=self.tokenizer,
            op_list=op_list,
            op_list_size=len(op_list),
            const_list=const_list,
            const_list_size=len(const_list))

        exe_acc = evaluate_result(all_nbest, test_file, program_mode="seq")
        
        self.log("exe_acc", exe_acc)
        
        # reset the predictions
        self.predictions = []

    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        segment_ids = batch["segment_ids"]
        option_mask = batch["option_mask"]
        program_ids = batch["program_ids"]
        program_mask = batch["program_mask"]
        is_training = False
        
        program_ids = torch.tensor(program_ids).to("cuda")
        
        metadata = [{"unique_id": filename_id} for filename_id in batch["unique_id"]]
        
        output_dicts = self(is_training, input_ids, input_mask, segment_ids, option_mask, program_ids, program_mask, metadata)
        
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