# Neural Networks for Intent Classifier

In this repo one can find code for training and infering intent classification
that is presented as _shallow-and-wide Convolutional Neural Network_ https://arxiv.org/abs/1408.5882.
Also this repo contains pre-trained model for intent classification on SNIPS dataset
https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines

SNIPS dataset considers the following intents:

-AddToPlaylist

-BookRestaurant

-GetWeather

-PlayMusic

-RateBook

-SearchCreativeWork

-SearchScreeningEvent


### How to use

First of all, one have to download this repo:

```
git clone https://github.com/deepmipt/intent_classifier.git

cd intent_classifier
```

The next step is to install requirements:

```
pip install -r requirements.txt
```

Now one is able to infer pre-trained model:

```
./intent_classifier.py
```

The script loads pre-trained model, if necessary downloads pre-trained fastText embedding model [1],
and then it is ready to predict class and probability of given phrase to belong with this class.

Example:
```
./intent_classifier.py
>I want you to add 'I love you, baby' to my playlist
>(0.99991322, 'AddToPlaylist')
```

### References

[1] P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information