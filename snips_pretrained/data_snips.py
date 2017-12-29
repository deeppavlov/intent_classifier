import json
import pandas as pd
import numpy as np

all_intents_data = pd.DataFrame()
all_intents_full_data = pd.DataFrame()

for intent in ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork',
               'SearchScreeningEvent']:
    with open("./nlu-benchmark/2017-06-custom-intent-engines/" + intent + "/train_" + intent + ".json",
              encoding='cp1251') as data_file:
        data = json.load(data_file)

    print("length:", len(data[intent]))

    texts = []
    for i in range(len(data[intent])):
        text = ''
        for j in range(len(data[intent][i]['data'])):
            text += data[intent][i]['data'][j]['text']
        texts.append(text)

    dftrain = pd.DataFrame(data=texts, columns=['request'])
    dftrain[intent] = np.ones(dftrain.shape[0], dtype='int')
    print(dftrain.head())
    dftrain.to_csv("./intent_data/" + intent + ".csv", index=False)

    all_intents_data = all_intents_data.append(dftrain, ignore_index=True)

    with open("./nlu-benchmark/2017-06-custom-intent-engines/" + intent + "/train_" + intent + "_full.json",
              encoding='cp1251') as data_file:
        full_data = json.load(data_file)

    print('full length:', len(full_data[intent]))
    texts = []
    for i in range(len(full_data[intent])):
        text = ''
        for j in range(len(full_data[intent][i]['data'])):
            text += full_data[intent][i]['data'][j]['text']
        texts.append(text)

    dftrain = pd.DataFrame(data=texts, columns=['request'])
    dftrain[intent] = np.ones(dftrain.shape[0], dtype='int')
    print(dftrain.head())
    dftrain.to_csv("./intent_data/" + intent + "_full.csv", index=False)

    all_intents_full_data = all_intents_full_data.append(dftrain, ignore_index=True)

all_intents_data = all_intents_data.fillna(value=0)
all_intents_full_data = all_intents_full_data.fillna(value=0)
all_intents_data.to_csv("./intent_data/intent_data.csv", index=False)
all_intents_full_data.to_csv("./intent_data/intent_full_data.csv", index=False)

all_intents_full_data = all_intents_full_data.append(all_intents_data, ignore_index=True)
all_intents_full_data.to_csv("./intent_data/all_intent_data.csv", index=False)

