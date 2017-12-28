#!/usr/bin/env python3

import sys
import numpy as np
import json

from preprocessing import NLTKTokenizer
from multiclass import KerasMulticlassModel


def infer(phrase):
    global preprocessor, classes, model
    predictions = model.infer(preprocessor.infer([phrase])[0])
    return np.max(predictions), classes[np.argmax(predictions)]


preprocessor = NLTKTokenizer()

with open("./config.json", "r") as f:
    opt = json.load(f)

classes = np.array(['AddToPlaylist', 'BookRestaurant', 'GetWeather',
                    'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent'])
opt["classes"] = classes

model = KerasMulticlassModel(opt)

print("\nPlease, enter sentence")
for query in sys.stdin:
    print(infer(query))



