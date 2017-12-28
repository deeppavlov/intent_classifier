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


# Initializing Tokenizer class that can split words and punctuation by space
preprocessor = NLTKTokenizer()

comment_name = "request"
path_to_data = "/home/dilyara/data/data_files/snips"
train_data = pd.read_csv(Path(path_to_data).joinpath("intent_full_data.csv"), sep=',')
print(train_data.head())

with open("./config.json", "r") as f:
    opt = json.load(f)

classes = np.array(['AddToPlaylist', 'BookRestaurant', 'GetWeather',
                    'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent'])
opt["classes"] = classes
print(opt)

data_dict = dict()

train_pairs = []
# for i in range(100):
for i in range(train_data.shape[0]):
    train_pairs.append((train_data.loc[i, comment_name], classes[np.where(train_data.loc[i, classes].values == 1)[0]]))

test_pairs = []
# for i in range(test_data.shape[0]):
# for i in range(100):
#     test_pairs.append((test_data.loc[i, comment_name], ))

data_dict["train"] = train_pairs
# data_dict["test"] = test_pairs


dataset = Dataset(data=data_dict, seed=42, classes=classes,
                  field_to_split="train", splitted_fields="train valid", splitting_proportions="0.9 0.1")
print("Initializing model")
model = KerasMulticlassModel(opt)
print("Training model")
model.train(dataset=dataset)
