# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
from pathlib import Path
import json

from preprocessing import NLTKTokenizer
from dataset import Dataset
from multiclass import KerasMulticlassModel



# Reading full data
comment_name = "request"
path_to_data = "./"
train_data = pd.read_csv(Path(path_to_data).joinpath("intent_full_data.csv"), sep=',')
print(train_data.head())

# Tokenization that splits words and punctuation by space
preprocessor = NLTKTokenizer()
train_data.loc[:, comment_name] = preprocessor.infer(train_data.loc[:, comment_name].values)

# Reading parameters of model from json
with open("./config.json", "r") as f:
    opt = json.load(f)

# Initializing classes from dataset
classes = np.array(['AddToPlaylist', 'BookRestaurant', 'GetWeather',
                    'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent'])
opt["classes"] = classes

# Constructing data
data_dict = dict()
train_pairs = []
for i in range(train_data.shape[0]):
    train_pairs.append((train_data.loc[i, comment_name], classes[np.where(train_data.loc[i, classes].values == 1)[0]]))
data_dict["train"] = train_pairs

# Building dataset splitting full dataset on train and valid in proportion 9:1
dataset = Dataset(data=data_dict, seed=42, classes=classes,
                  field_to_split="train", splitted_fields="train valid", splitting_proportions="0.9 0.1")

# Initilizing model with given parameters
print("Initializing model")
model = KerasMulticlassModel(opt)

# Training model on the given dataset
print("Training model")
model.train(dataset=dataset)
