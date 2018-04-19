#!/usr/bin/env python3

import sys
import numpy as np
import json

from intent_model.preprocessing import NLTKTokenizer
from intent_model.multiclass import KerasMulticlassModel


config_file = sys.argv[1]

def infer(phrase):
    global preprocessor, classes, model
    try:
        predictions = model.infer(preprocessor.infer(phrase))
    except Exception:
        print('Error', file=sys.stderr)
        return 0, 'error'
    return np.max(predictions), classes[np.argmax(predictions)]


# Initializing preprocessor
preprocessor = NLTKTokenizer()

# Reading parameters
with open(config_file, "r") as f:
    opt = json.load(f)


# Infering is possible only for saved intent_model
opt['model_from_saved'] = True

# Initializing intent_model
model = KerasMulticlassModel(opt)

# Initializing classes
classes = model.classes

print("Model is ready! You now can enter requests.")
for query in sys.stdin:
    print(infer(query))



