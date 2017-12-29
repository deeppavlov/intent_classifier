#!/usr/bin/env python3

import sys
import numpy as np
import json

from preprocessing import NLTKTokenizer
from multiclass import KerasMulticlassModel


def infer(phrase):
    global preprocessor, classes, model
    try:
        predictions = model.infer(preprocessor.infer(phrase))
    except Exception:
        print('olololo', file=sys.stderr)
        return 0, 'error'
    return np.max(predictions), classes[np.argmax(predictions)]


preprocessor = NLTKTokenizer()

with open("./config.json", "r") as f:
    opt = json.load(f)

classes = np.array(['AddToPlaylist', 'BookRestaurant', 'GetWeather',
                    'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent'])
opt["classes"] = classes
opt["model_from_saved"] = True

model = KerasMulticlassModel(opt)

for query in sys.stdin:
    print(infer(query))



