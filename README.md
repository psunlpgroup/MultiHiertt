# MultiHiertt
Data and code for ACL 2022 paper "MultiHiertt: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data"
<https://aclanthology.org/2022.acl-long.454/>

## Requirements
- python 3.9.7
- pytorch 1.10.2, 
- pytorch-lightning 1.5.10
- huggingface transformers 4.18.0
- run `pip install -r requirements.txt` to install rest of the dependencies 

## Leaderboard
- The leaderboard for the private test data is held on [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/6738)

## Main Files Structures
```shell
dataset/
training_configs/ & inference_configs/ # Configuration files for training and inference
lightning_modules/: 
    models/ # Implementation for each module
    datasets/ # Dataloaders
    callbacks/ # Callbacks for saving predictions
utils/ # Utilities for modules
txt_files/ # Txt files such as constant_list.txt, etc
output/ # Predictions and intermediate results
checkpoint/

convert_retriever_result.py # convert inference of Fact Retrieving & Question Type Classification Module into model input of Reasoning Modules.
trainer.py
evaluate.py
```

## Dataset
The dataset is stored as json files [Download Link](https://drive.google.com/drive/folders/1ituEWZ5F7G9T9AZ0kzZZLrHNhRigHCZJ?usp=sharing), each entry has the following format:

```
"uid": unique example id;
"paragraphs": the list of sentences in the document;
"tables": the list of tables in HTML format in the document;
"table_description": the list of table descriptions for each data cell in tables. Generated by the pre-processing script;
"qa": {
  "question": the question;
  "answer": the answer;
  "program": the reasoning program;
  "text_evidence": the list of indices of gold supporting text facts;
  "table_evidence": the list of indices of gold supporting table facts;
}
```

## MT2Net
We provide the model checkpoints in [Hugging Face](https://huggingface.co/datasets/yilunzhao/MultiHiertt/tree/main). Download them (`*.ckpt`) into the directory `checkpoints`.
### 1. Fact Retrieving & Question Type Classification Module
#### 1.1 Training Stage
- Edit `training_configs/retriever_finetuning.yaml` & `training_configs/question_classification_finetuning.yaml` to set your own project and data path. 
- Run the following commands to train the model.
```
export PYTHONPATH=`pwd`; python trainer.py {fit, validate} --config training_configs/*_finetuning.yaml
```
#### 1.2 Inference Stage
- Edit `inference_configs/retriever_inference.yaml` & `inference_configs/retriever_inference.yaml` to set your own project and data path. 
- Run the following commands to get the intermediate results for {Train, Dev, Test} set, respectively.
```
export PYTHONPATH=`pwd`; python trainer.py predict --ckpt_path checkpoints/*_model.ckpt --config inference_configs/*_inference.yaml
```
where `checkpoints/*_model.ckpt` can be replaced by the checkpoint path from training stage. And the inference set or files should be specified in *_inference.yaml. 

### 2. Reasoning Module Input Generation
- Prepare `output/retriever_output/{train, dev, test}.json` & `output/question_classification_output/{train, dev, test}.json` from Step 1. 

- Run the following commands to convert predictions of Fact Retrieving & Question Type Classification Module for {Train, Dev, Test} into model input of Reasoning Module, respectively.
```
python convert_retriever_result.py
```
The output files are stored in `dataset/reasoning_module_input`, where `*_training.json` is used for the training stage and `*_inference.json` is used for the inference stage.

### 3. Reasoning Module
#### 3.1 Training Stage
- Edit `training_configs/program_generation_finetuning.yaml` & `training_configs/span_selection_finetuning.yaml` to set your own project and data path.
- Run the following commands to train the model and generate the prediction files.
```
export PYTHONPATH=`pwd`; python trainer.py fit --config training_configs/*_finetuning.yaml
```
#### 3.2 Inference Stage
- Edit `inference_configs/program_generation_inference.yaml` & `inference_configs/span_selection_inference.yaml` to set your own project and data path. 
- Run the following commands to get the prediction file for {Dev, Test} set
```
export PYTHONPATH=`pwd`; python trainer.py predict --ckpt_path checkpoints/*_model.ckpt --config inference_configs/*_inference.yaml
```
where `checkpoints/*_model.ckpt` can be replaced by the checkpoint path from training stage. And the inference set or files should be specified in *_inference.yaml. 


## Evaluation
Run the following commands to get the prediction file for {Dev, Test} set (and the performance on the Dev set), respectively.
```
python evaluate.py dataset/{test, dev}.json
```
The prediction file with the following format will be generated in the directory `output/final_predictions`:
```
[
    {
        "uid": "bd2ce4dbf70d43e094d93d314b30bd39",
        "predicted_ans": "106.0",
        "predicted_program": []
    },
    ...
]
```
For test set, Please zip the generated test prediction file `test_predictions.json` into `test_predictions.zip`; and submit `test_predictions.zip` to [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/6738) to get the final score. Please exactly match the filename.

## Any Questions?
For any issues or questions, kindly email us at: Yilun Zhao yilun.zhao@yale.edu.

## Citation
```
@inproceedings{zhao-etal-2022-multihiertt,
    title = "{M}ulti{H}iertt: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data",
    author = "Zhao, Yilun  and
      Li, Yunxiang  and
      Li, Chenying  and
      Zhang, Rui",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.454",
    pages = "6588--6600",
}
```
