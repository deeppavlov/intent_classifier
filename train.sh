#!/bin/bash

KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=8 python train.py config.json "data13022018/train.csv"
