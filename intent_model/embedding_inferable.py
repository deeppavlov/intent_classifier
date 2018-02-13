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
from pathlib import Path

from intent_model.utils import download

class EmbeddingInferableModel(object):

    def __init__(self, embedding_dim, embedding_fname=None, embedding_url=None, module="fastText", *args, **kwargs):
        """
        Method initialize the class according to given parameters.
        Args:
            embedding_fname: name of file with embeddings
            embedding_dim: dimension of embeddings
            embedding_url: url link to embedding to try to download if file does not exist
            *args:
            **kwargs:
        """
        self.tok2emb = {}
        self.embedding_dim = embedding_dim
        self.model = None
        self.module = module
        self.load(embedding_fname, embedding_url)

    def add_items(self, sentence_li):
        """
        Method adds new items to tok2emb dictionary from given text
        Args:
            sentence_li: list of sentences

        Returns: None

        """
        for sen in sentence_li:
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            for tok in tokens:
                if self.tok2emb.get(tok) is None:
                    try:
                        if self.module == "fasttext":
                            self.tok2emb[tok] = self.fasttext_model[tok]
                        else:
                            self.tok2emb[tok] = self.fasttext_model.get_word_vector(tok)
                    except KeyError:
                        self.tok2emb[tok] = np.zeros(self.embedding_dim)
        return

    def emb2str(self, vec):
        """
        Method returns string corresponding to the given embedding vectors
        Args:
            vec: vector of embeddings

        Returns:
            string corresponding to the given embeddings
        """
        string = ' '.join([str(el) for el in vec])
        return string

    def load(self, embedding_fname, embedding_url=None, *args, **kwargs):
        """
        Method initializes dict of embeddings from file
        Args:
            fname: file name

        Returns:
            Nothing
        """

        if not embedding_fname:
            raise RuntimeError('Please, provide path to model')
        fasttext_model_file = embedding_fname

        if not Path(fasttext_model_file).is_file():
            emb_path = embedding_url
            if not emb_path:
                raise RuntimeError('Fasttext model file does not exist locally. URL does not contain  fasttext model file')
            embedding_fname = Path(fasttext_model_file).name
            try:
                download(dest_file_path=fasttext_model_file, source_url=embedding_url)
            except Exception as e:
                raise RuntimeError('Looks like the `EMBEDDINGS_URL` variable is set incorrectly', e)

        if self.module == "fastText":
            import fastText
            self.fasttext_model = fastText.load_model(fasttext_model_file)
        if self.module == "fasttext":
            import fasttext
            self.fasttext_model = fasttext.load_model(fasttext_model_file)
        return

    def infer(self, instance, *args, **kwargs):
        """
        Method returns embedded data
        Args:
            instance: sentence or list of sentences

        Returns:
            Embedded sentence or list of embedded sentences
        """
        if type(instance) is str:
            tokens = instance.split(" ")
            self.add_items(tokens)
            embedded_tokens = []
            for tok in tokens:
                embedded_tokens.append(self.tok2emb.get(tok))
            if len(tokens) == 1:
                embedded_tokens = embedded_tokens[0]
            return embedded_tokens
        else:
            embedded_instance = []
            for sample in instance:
                tokens = sample.split(" ")
                self.add_items(tokens)
                embedded_tokens = []
                for tok in tokens:
                    embedded_tokens.append(self.tok2emb.get(tok))
                embedded_instance.append(embedded_tokens)
            return embedded_instance

