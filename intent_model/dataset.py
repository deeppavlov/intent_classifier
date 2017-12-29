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


import random
from typing import Generator
from sklearn.model_selection import train_test_split


class Dataset(object):
    def __init__(self, data, seed=None, classes=None,
                 fields_to_merge=None, merged_field=None,
                 field_to_split=None, splitted_fields=None, splitting_proportions=None,
                 *args, **kwargs):

        rs = random.getstate()
        random.seed(seed)
        self.random_state = random.getstate()
        random.setstate(rs)

        self.train = data.get('train', [])
        self.test = data.get('test', [])
        self.data = {
            'train': self.train,
            'test': self.test,
            'all': self.train + self.test
        }

        self.classes = classes
        if fields_to_merge is not None:
            if merged_field is not None:
                # print("Merging fields <<{}>> to new field <<{}>>".format(fields_to_merge, merged_field))
                self._merge_data(fields_to_merge=fields_to_merge.split(' '), merged_field=merged_field)
            else:
                raise IOError("Given fields to merge BUT not given name of merged field")

        if field_to_split is not None:
            if splitted_fields is not None:
                # print("Splitting field <<{}>> to new fields <<{}>>".format(field_to_split, splitted_fields))
                self._split_data(field_to_split=field_to_split,
                                 splitted_fields=splitted_fields.split(" "),
                                 splitting_proportions=[float(s) for s in splitting_proportions.split(" ")])
            else:
                raise IOError("Given field to split BUT not given names of splitted fields")

    def batch_generator(self, batch_size: int, data_type: str = 'train') -> Generator:
        r"""This function returns a generator, which serves for generation of raw (no preprocessing such as tokenization)
         batches
        Args:
            batch_size (int): number of samples in batch
            data_type (str): can be either 'train', 'test', or 'valid'
        Returns:
            batch_gen (Generator): a generator, that iterates through the part (defined by data_type) of the dataset
        """
        data = self.data[data_type]
        data_len = len(data)
        order = list(range(data_len))

        rs = random.getstate()
        random.setstate(self.random_state)
        random.shuffle(order)
        self.random_state = random.getstate()
        random.setstate(rs)

        for i in range((data_len - 1) // batch_size + 1):
            yield list(zip(*[data[o] for o in order[i * batch_size:(i + 1) * batch_size]]))

    def iter_all(self, data_type: str = 'train') -> Generator:
        r"""Iterate through all data. It can be used for building dictionary or
        Args:
            data_type (str): can be either 'train', 'test', or 'valid'
        Returns:
            samples_gen: a generator, that iterates through the all samples in the selected data type of the dataset
        """
        data = self.data[data_type]
        for x, y in data:
            yield (x, y)

    def _split_data(self, field_to_split, splitted_fields, splitting_proportions):
        data_to_div = self.data[field_to_split].copy()
        data_size = len(self.data[field_to_split])
        for i in range(len(splitted_fields) - 1):
            self.data[splitted_fields[i]], data_to_div = train_test_split(data_to_div,
                                                                          test_size=
                                                                          len(data_to_div) -
                                                                          int(data_size * splitting_proportions[i]))
        self.data[splitted_fields[-1]] = data_to_div
        return True

    def _merge_data(self, fields_to_merge, merged_field):
        data = self.data.copy()
        data[merged_field] = []
        for name in fields_to_merge:
            data[merged_field] += self.data[name]
        self.data = data
        return True
