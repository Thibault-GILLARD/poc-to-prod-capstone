import unittest
import pandas as pd
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils

class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(base._get_num_train_batches(), 4)  

    def test__get_num_test_batches(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(base._get_num_test_batches(), 1)
         

    def test_get_index_to_label_map(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['a', 'b', 'c'])
        self.assertEqual(base.get_index_to_label_map(), {0: 'a', 1: 'b', 2: 'c'})

    def test_index_to_label_and_label_to_index_are_identity(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['a', 'b', 'c'])

        index_to_label = base.get_index_to_label_map()
        label_to_index = base.get_label_to_index_map()

        reversed_label_to_index = {value: key for key, value in label_to_index.items()}

        self.assertEqual(index_to_label, reversed_label_to_index)

    def test_to_indexes(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['a', 'b', 'c'])
        self.assertEqual(base.to_indexes(['a', 'b', 'c']), [0, 1, 2])



class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_a', 'tag_b', 'tag_c'],
            'tag_id': [1, 2, 3, 1, 2, 3],
            'tag_position': [0, 1, 2, 0, 1, 2],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        l = utils.LocalTextCategorizationDataset("fake_path", 1, min_samples_per_label=2)

        self.assertEqual(l._get_num_samples(), 2)



    def test_get_train_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_a', 'tag_b', 'tag_c'],
            'tag_id': [1, 2, 3, 1, 2, 3],
            'tag_position': [0, 1, 2, 0, 1, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        l = utils.LocalTextCategorizationDataset("fake_path", 1, min_samples_per_label=2)
        l._get_num_samples = MagicMock(return_value=2)
        l._get_num_train_batches = MagicMock(return_value=1)
        l._get_num_test_batches = MagicMock(return_value=1)
        l._get_label_list = MagicMock(return_value=['a', 'b', 'c'])
        l.to_indexes = MagicMock(return_value=[0, 1, 2])
        l.x_train = ['title_1', 'title_2']
        l.y_train = [0, 1]
        l.train_batch_index = 0
        l.test_batch_index = 0

        x, y = l.get_train_batch()
        len_x, len_y = len(x), len(y)
        shape = (len_x, len_y)
        self.assertEqual(shape, (1, 1))


    def test_get_test_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_a', 'tag_b', 'tag_c'],
            'tag_id': [1, 2, 3, 1, 2, 3],
            'tag_position': [0, 1, 2, 0, 1, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        l = utils.LocalTextCategorizationDataset("fake_path", 1, min_samples_per_label=2)
        l._get_num_samples = MagicMock(return_value=2)
        l._get_num_train_batches = MagicMock(return_value=1)
        l._get_num_test_batches = MagicMock(return_value=1)
        l._get_label_list = MagicMock(return_value=['a', 'b', 'c'])
        l.to_indexes = MagicMock(return_value=[0, 1, 2])
        l.x_test = ['title_1', 'title_2']
        l.y_test = [0, 1]
        l.train_batch_index = 0
        l.test_batch_index = 0

        x, y = l.get_test_batch()
        len_x, len_y = len(x), len(y)
        shape = (len_x, len_y)
        self.assertEqual(shape, (1, 1))

    def test_get_train_batch_raises_assertion_error(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4'],
            'tag_name': ['tag_a', 'tag_a', 'tag_b', 'tag_b'],
            'tag_id': [1, 2, 1, 1],
            'tag_position': [0, 0, 1, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4']
        }))
        with self.assertRaises(AssertionError):
            utils.LocalTextCategorizationDataset(filename="fake", batch_size=2, train_ratio=0.5,
                                                 min_samples_per_label=1)
             
