import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd

from train.train import run as run_train
from preprocessing.preprocessing import utils
from keras.models import load_model

import os
import json
import time

from predict.predict import run
from preprocessing.preprocessing import utils


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class Test_TextPredictionModel(unittest.TestCase):
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_predict(self):
        params = {
            'batch_size': 2,
            'epochs': 1,
            'dense_dim': 2,
            'min_samples_per_label': 1,
            'verbose': 0
        }

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, _ = run_train.train('Fake_Path', params, model_dir, False)

            # load model
            model = load_model(os.path.join(model_dir, 'model.h5'))

            # load params
            with open(os.path.join(model_dir, 'params.json'), 'r') as f:
                params = json.load(f)

            # load labels_to_index
            with open(os.path.join(model_dir, 'labels_index.json'), 'r') as f:
                labels_to_index = json.load(f)

        # create TextPredictionModel
        model = run.TextPredictionModel(model, params, labels_to_index)

        # predict, with title 1
        predictions = model.predict(['Is it possible to execute the procedure of a function in the scope of the caller?'],1)

        print(predictions)
        print(model.labels_index_inv)
        print(model.labels_to_index)
        # -> [array([1, 0])]
        # index 0 is 'ruby-on-rails', index 1 is 'php', use predictions to assert
        index_return = model.labels_index_inv[predictions[0][0]]
        print(index_return)
        # assert with index 1, the label is 'ruby-on-rails' or 'php'
        self.assertEqual(index_return, 'ruby-on-rails' or 'php')











