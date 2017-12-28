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


import json
import copy
from pathlib import Path
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input
import keras.metrics
import keras.optimizers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Dense, Input, concatenate, Activation, Embedding
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from embedding_inferable import EmbeddingInferableModel
import metrics as metrics_file
from utils import labels2onehot, log_metrics

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))


class KerasMulticlassModel(object):
    """
    Class builds keras model
    """

    def __init__(self, opt, *args, **kwargs):
        """
        Method initializes model using parameters from opt
        Args:
            opt: dictionary of parameters
            *args:
            **kwargs:
        """
        self.opt = copy.deepcopy(opt)

        self.model_path_ = Path(self.opt["model_path"])

        self.classes = np.array(self.opt['classes'])
        self.opt.pop('classes', None)

        self.n_classes = self.classes.shape[0]
        self.confident_threshold = self.opt['confident_threshold']
        if 'add_metrics' in self.opt.keys():
            self.add_metrics = self.opt['add_metrics'].split(' ')
            self.add_metrics_values = len(self.add_metrics) * [0.]
        else:
            self.add_metrics = None

        if self.opt['fasttext_model'] is not None:
            if Path(self.opt['fasttext_model']).is_file():
                self.fasttext_model = EmbeddingInferableModel(embedding_fname=self.opt['fasttext_model'],
                                                              embedding_dim=self.opt['embedding_size'])
            else:
                self.fasttext_model = EmbeddingInferableModel(embedding_dim=self.opt['embedding_size'],
                                                              embedding_url='http://lnsigo.mipt.ru/export/intent/reddit_fasttext_model.tar.gz')
        else:
            raise IOError("Error: FastText model file path is not given")

        if self.opt['model_from_saved']:
            self.model = self.load(model_name=self.opt['model_name'],
                                   fname=self.model_path_,
                                   optimizer_name=self.opt['optimizer'],
                                   lr=self.opt['lear_rate'],
                                   decay=self.opt['lear_rate_decay'],
                                   loss_name=self.opt['loss'],
                                   metrics_names=self.opt['lear_metrics'],
                                   add_metrics_file=metrics_file)
        else:
            self.model = self.init_model_from_scratch(model_name=self.opt['model_name'],
                                                      optimizer_name=self.opt['optimizer'],
                                                      lr=self.opt['lear_rate'],
                                                      decay=self.opt['lear_rate_decay'],
                                                      loss_name=self.opt['loss'],
                                                      metrics_names=self.opt['lear_metrics'],
                                                      add_metrics_file=metrics_file)

        self.metrics_names = self.model.metrics_names
        self.metrics_values = len(self.metrics_names) * [0.]

    def texts2vec(self, sentences):
        embeddings_batch = []
        for sen in sentences:
            embeddings = []
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            if len(tokens) > self.opt['text_size']:
                tokens = tokens[:self.opt['text_size']]
            for tok in tokens:
                embeddings.append(self.fasttext_model.infer(tok))
            if len(tokens) < self.opt['text_size']:
                pads = [np.zeros(self.opt['embedding_size'])
                        for _ in range(self.opt['text_size'] - len(tokens))]
                embeddings = pads + embeddings

            embeddings = np.asarray(embeddings)
            embeddings_batch.append(embeddings)

        embeddings_batch = np.asarray(embeddings_batch)
        return embeddings_batch

    def train_on_batch(self, batch):
        """
        Method trains the model on the given batch
        Args:
            batch - list of tuples (preprocessed text, labels)

        Returns:
            loss and metrics values on the given batch
        """
        texts = list(batch[0])
        labels = list(batch[1])
        features = self.texts2vec(texts)
        onehot_labels = labels2onehot(labels, classes=self.classes)
        metrics_values = self.model.train_on_batch(features, onehot_labels)
        return metrics_values

    def train(self, dataset, *args, **kwargs):
        """
        Method trains the model using batches and validation
        Args:
            dataset: instance of class Dataset

        Returns: None

        """
        updates = 0
        val_loss = 1e100
        val_increase = 0
        epochs_done = 0

        n_train_samples = len(dataset.data['train'])

        valid_iter_all = dataset.iter_all(data_type='valid')
        valid_x = []
        valid_y = []
        for valid_i, valid_sample in enumerate(valid_iter_all):
            valid_x.append(valid_sample[0])
            valid_y.append(valid_sample[1])

        valid_x = self.texts2vec(valid_x)
        valid_y = labels2onehot(valid_y, classes=self.classes)

        # print('\n____Training over {} samples____\n\n'.format(n_train_samples))

        while epochs_done < self.opt['epochs']:
            batch_gen = dataset.batch_generator(batch_size=self.opt['batch_size'],
                                                data_type='train')
            for step, batch in enumerate(batch_gen):
                metrics_values = self.train_on_batch(batch)
                updates += 1

                if self.opt['verbose'] and step % 50 == 0:
                    log_metrics(names=self.metrics_names,
                                values=metrics_values,
                                updates=updates,
                                mode='train')

            epochs_done += 1
            if epochs_done % self.opt['val_every_n_epochs'] == 0:
                if 'valid' in dataset.data.keys():
                    valid_metrics_values = self.model.test_on_batch(x=valid_x, y=valid_y)

                    log_metrics(names=self.metrics_names,
                                values=valid_metrics_values,
                                mode='valid')
                    if valid_metrics_values[0] > val_loss:
                        val_increase += 1
                        # print("__Validation impatience {} out of {}".format(
                        #     val_increase, self.opt['val_patience']))
                        if val_increase == self.opt['val_patience']:
                            # print("___Stop training: validation is out of patience___")
                            break
                    val_loss = valid_metrics_values[0]
            # print('epochs_done: {}'.format(epochs_done))

        self.save()

    def infer(self, data, *args):
        """
        Method returns predictions on the given data
        Args:
            data: sentence or list of sentences
            *args:

        Returns:
            Predictions for the given data
        """
        if type(data) is str:
            features = self.texts2vec([data])
            preds = self.model.predict_on_batch(features)[0]
        else:
            features = self.texts2vec(data)
            preds = self.model.predict_on_batch(features)
        return preds

    def cnn_model(self, params):
        """
        Method builds uncompiled model of shallow-and-wide CNN
        Args:
            params: disctionary of parameters for NN

        Returns:
            Uncompiled model
        """
        if type(self.opt['kernel_sizes_cnn']) is str:
            self.opt['kernel_sizes_cnn'] = [int(x) for x in
                                            self.opt['kernel_sizes_cnn'].split(' ')]

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        outputs = []
        for i in range(len(params['kernel_sizes_cnn'])):
            output_i = Conv1D(params['filters_cnn'], kernel_size=params['kernel_sizes_cnn'][i],
                              activation=None,
                              kernel_regularizer=l2(params['coef_reg_cnn']),
                              padding='same')(inp)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = GlobalMaxPooling1D()(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)

        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def dcnn_model(self, params):
        """
        Method builds uncompiled model of deep CNN
        Args:
            params: disctionary of parameters for NN

        Returns:
            Uncompiled model
        """
        if type(self.opt['kernel_sizes_cnn']) is str:
            self.opt['kernel_sizes_cnn'] = [int(x) for x in
                                            self.opt['kernel_sizes_cnn'].split(' ')]

        if type(self.opt['filters_cnn']) is str:
            self.opt['filters_cnn'] = [int(x) for x in
                                       self.opt['filters_cnn'].split(' ')]

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        output = inp

        for i in range(len(params['kernel_sizes_cnn'])):
            output = Conv1D(params['filters_cnn'][i], kernel_size=params['kernel_sizes_cnn'][i],
                            activation=None,
                            kernel_regularizer=l2(params['coef_reg_cnn']),
                            padding='same')(output)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = MaxPooling1D()(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def init_model_from_scratch(self, model_name, optimizer_name,
                                lr, decay, loss_name, metrics_names=None, add_metrics_file=None,
                                loss_weights=None,
                                sample_weight_mode=None,
                                weighted_metrics=None,
                                target_tensors=None):
        """
        Method initializes model from scratch with given params
        Args:
            model_name: name of model function described as a method of this class
            optimizer_name: name of optimizer from keras.optimizers
            lr: learning rate
            decay: learning rate decay
            loss_name: loss function name (from keras.losses)
            metrics_names: names of metrics (from keras.metrics) as one string
            add_metrics_file: file with additional metrics functions
            loss_weights: optional parameter as in keras.model.compile
            sample_weight_mode: optional parameter as in keras.model.compile
            weighted_metrics: optional parameter as in keras.model.compile
            target_tensors: optional parameter as in keras.model.compile

        Returns:
            compiled model with given network and learning parameters
        """
        # print('[ Initializing model from scratch ]')

        model_func = getattr(self, model_name, None)
        if callable(model_func):
            model = model_func(params=self.opt)
        else:
            raise AttributeError("Model {} is not defined".format(model_name))

        optimizer_func = getattr(keras.optimizers, optimizer_name, None)
        if callable(optimizer_func):
            optimizer_ = optimizer_func(lr=lr, decay=decay)
        else:
            raise AttributeError("Optimizer {} is not callable".format(optimizer_name))

        loss_func = getattr(keras.losses, loss_name, None)
        if callable(loss_func):
            loss = loss_func
        else:
            raise AttributeError("Loss {} is not defined".format(loss_name))

        metrics_names = metrics_names.split(' ')
        metrics_funcs = []
        for i in range(len(metrics_names)):
            metrics_func = getattr(keras.metrics, metrics_names[i], None)
            if callable(metrics_func):
                metrics_funcs.append(metrics_func)
            else:
                metrics_func = getattr(add_metrics_file, metrics_names[i], None)
                if callable(metrics_func):
                    metrics_funcs.append(metrics_func)
                else:
                    raise AttributeError("Metric {} is not defined".format(metrics_names[i]))

        model.compile(optimizer=optimizer_,
                      loss=loss,
                      metrics=metrics_funcs,
                      loss_weights=loss_weights,
                      sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
        return model

    def load(self, model_name, fname, optimizer_name,
             lr, decay, loss_name, metrics_names=None, add_metrics_file=None, loss_weights=None,
             sample_weight_mode=None, weighted_metrics=None, target_tensors=None):
        """
        Method initiliazes model from saved params and weights
        Args:
            model_name: name of model function described as a method of this class
            fname: path and first part of name of model
            optimizer_name: name of optimizer from keras.optimizers
            lr: learning rate
            decay: learning rate decay
            loss_name: loss function name (from keras.losses)
            metrics_names: names of metrics (from keras.metrics) as one string
            add_metrics_file: file with additional metrics functions
            loss_weights: optional parameter as in keras.model.compile
            sample_weight_mode: optional parameter as in keras.model.compile
            weighted_metrics: optional parameter as in keras.model.compile
            target_tensors: optional parameter as in keras.model.compile

        Returns:
            model with loaded weights and network parameters from files
            but compiled with given learning parameters
        """
        # print('___Initializing model from saved___'
        #       '\nModel weights file is %s.h5'
        #       '\nNetwork parameters are from %s_opt.json' % (fname, fname))

        fname = self.model_path_.name
        opt_fname = str(fname) + '_opt.json'
        weights_fname = str(fname) + '.h5'

        opt_path = Path.joinpath(self.model_path_, opt_fname)
        weights_path = Path.joinpath(self.model_path_, weights_fname)

        if Path(opt_path).is_file():
            with open(opt_path, 'r') as opt_file:
                self.opt = json.load(opt_file)
        else:
            raise IOError("Error: config file %s_opt.json of saved model does not exist" % fname)

        model_func = getattr(self, model_name, None)
        if callable(model_func):
            model = model_func(params=self.opt)
        else:
            raise AttributeError("Model {} is not defined".format(model_name))

        # print("Loading wights from `{}`".format(fname + '.h5'))
        model.load_weights(weights_path)

        optimizer_func = getattr(keras.optimizers, optimizer_name, None)
        if callable(optimizer_func):
            optimizer_ = optimizer_func(lr=lr, decay=decay)
        else:
            raise AttributeError("Optimizer {} is not callable".format(optimizer_name))

        loss_func = getattr(keras.losses, loss_name, None)
        if callable(loss_func):
            loss = loss_func
        else:
            raise AttributeError("Loss {} is not defined".format(loss_name))

        metrics_names = metrics_names.split(' ')
        metrics_funcs = []
        for i in range(len(metrics_names)):
            metrics_func = getattr(keras.metrics, metrics_names[i], None)
            if callable(metrics_func):
                metrics_funcs.append(metrics_func)
            else:
                metrics_func = getattr(add_metrics_file, metrics_names[i], None)
                if callable(metrics_func):
                    metrics_funcs.append(metrics_func)
                else:
                    raise AttributeError("Metric {} is not defined".format(metrics_names[i]))

        model.compile(optimizer=optimizer_,
                      loss=loss,
                      metrics=metrics_funcs,
                      loss_weights=loss_weights,
                      sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                     )
        return model

    def save(self, fname=None):
        """
        Method saves the model parameters into <<fname>>_opt.json (or <<model_file>>_opt.json)
        and model weights into <<fname>>.h5 (or <<model_file>>.h5)
        Args:
            fname: file_path to save model. If not explicitly given seld.opt["model_file"] will be used

        Returns:
            nothing
        """
        fname = self.model_path_.name if fname is None else fname
        opt_fname = str(fname) + '_opt.json'
        weights_fname = str(fname) + '.h5'

        opt_path = Path.joinpath(self.model_path_, opt_fname)
        weights_path = Path.joinpath(self.model_path_, weights_fname)
        # print("[ saving model: {} ]".format(str(opt_path)))
        self.model.save_weights(weights_path)

        with open(opt_path, 'w') as outfile:

            json.dump(self.opt, outfile)

        return True

    def reset(self):
        pass
