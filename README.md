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
- The leaderboard for the test data will be hold on [CodaLab](https://codalab.lisn.upsaclay.fr) (We will create it before the end of June)


## Dataset
The dataset is stored as json files [Download Link](https://drive.google.com/drive/folders/1ituEWZ5F7G9T9AZ0kzZZLrHNhRigHCZJ?usp=sharing), each entry has the following format:

```
"uid": unique example id;
"paragraphs": the list of sentences in the document;
"tables": the list of tables in HTML format in the document;
"qa": {
  "question": the question;
  "answer": the answer;
  "program": the reasoning program;
  "text_evidence": the list of indices of gold supporting text facts;
  "table_evidence": the list of indices of gold supporting table facts;
}
```


## Fact Retrieving Module
- Edit `training_configs/retriever_finetuning.yaml` to set your own project and data path. 
- Run the following commands to train the model and generate the prediction files.
```
export PYTHONPATH=`pwd`; python trainer.py fit --config training_configs/retriever_finetuning.yaml
```

## Reasoning Module
- Edit `training_configs/program_generation_finetuning.yaml` & `training_configs/span_selection_finetuning.yaml` to set your own project and data path.
- Run the following commands to train the model and generate the prediction files.
```
export PYTHONPATH=`pwd`; python trainer.py fit --config training_configs/program_generation_finetuning.yaml
```
```
export PYTHONPATH=`pwd`; python trainer.py fit --config training_configs/span_selection_finetuning.yaml
```


## Evaluation
Prepare your prediction file into the following format:
```
[
    {
        "uid": "bd2ce4dbf70d43e094d93d314b30bd39",
        "answer": "106.0"
    },
    ...
]
```
Run the following commands to evaluate
```
python evaluate.py your_prediction_file test.json
```

## Any Questions?
For any issues, kindly email us at: Yilun Zhao yilun.zhao@yale.edu.

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