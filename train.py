#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

from intent_model.preprocessing import NLTKTokenizer
from intent_model.dataset import Dataset
from intent_model.multiclass import KerasMulticlassModel

config_file = sys.argv[1]
data_file = sys.argv[2]

# Reading full data
comment_name = "request"
train_data = pd.read_csv(Path(data_file), sep=',')
print(train_data.head())

values = {"istask": 0, "request": "пропущено"}
train_data.fillna(values, inplace=True)

# Tokenization that splits words and punctuation by space
preprocessor = NLTKTokenizer()
for k in range(3839):
    inds = np.arange(k * 10000, min((k + 1) * 10000, train_data.shape[0]))
    train_data.loc[inds, comment_name] = preprocessor.infer(train_data.loc[inds, comment_name].values)

# Reading parameters of intent_model from json
with open(config_file, "r") as f:
    opt = json.load(f)

# Initializing classes from dataset
columns = list(train_data.columns)
columns.remove(comment_name)
classes = np.array(columns)
opt["classes"] = " ".join(list(classes))
print(classes)

# Constructing data
data_dict = dict()
train_pairs = []
for i in range(train_data.shape[0]):
    train_pairs.append((train_data.loc[i, comment_name], classes[np.where(train_data.loc[i, classes].values == 1)[0]]))
data_dict["train"] = train_pairs

# Building dataset splitting full dataset on train and valid in proportion 9:1
dataset = Dataset(data=data_dict, seed=42, classes=classes,
                  field_to_split="train", splitted_fields="train valid", splitting_proportions="0.9 0.1")

# Initilizing intent_model with given parameters
print("Initializing intent_model")
model = KerasMulticlassModel(opt)

# Training intent_model on the given dataset
print("Training intent_model")
model.train(dataset=dataset)
