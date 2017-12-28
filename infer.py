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

import numpy as np
import json

from preprocessing import NLTKTokenizer
from multiclass import KerasMulticlassModel


preprocessor = NLTKTokenizer()

with open("./config.json", "r") as f:
    opt = json.load(f)

classes = np.array(['AddToPlaylist', 'BookRestaurant', 'GetWeather',
                    'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent'])
opt["classes"] = classes
print(opt)

print("Initializing model")
model = KerasMulticlassModel(opt)

phrase = "I want you to add 'I love you, baby' to my playlist"
predictions = model.infer(preprocessor.infer([phrase])[0])
print(np.max(predictions), classes[np.argmax(predictions)])
