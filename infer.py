#!/usr/bin/env python3

import numpy as np
import json
import sys

from intent_model.preprocessing import NLTKTokenizer
from intent_model.multiclass import KerasMulticlassModel


config_file = sys.argv[1]

# Reading parameters
with open(config_file, "r") as f:
    opt = json.load(f)

# Infering is possible only for saved intent_model
opt['model_from_saved'] = True

# Initializing intent_model
print("Initializing intent_model")
model = KerasMulticlassModel(opt)

# Initializing classes
classes = model.classes

# Initializing preprocessor
preprocessor = NLTKTokenizer()

phrase = "I want you to add 'I love you, baby' to my playlist"

# Predicting
predictions = model.infer(preprocessor.infer(phrase))

# Result
print(np.max(predictions), classes[np.argmax(predictions)])
