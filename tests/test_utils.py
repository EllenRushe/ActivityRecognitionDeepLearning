import unittest
from src.utils import utils
import numpy as np
import pandas as pd
from collections import Counter

class test_utils(unittest.TestCase):

    def setUp(self):
        # Set up mock dataframe for testing.
        test_df_x = np.arange(1, 21)
        test_df_y = np.arange(1, 21)
        test_df_z = np.arange(1, 21)
        test_df_user = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4])
        classes = ['class1', 'class2', 'class3']
        labels = np.random.choice(classes, 20)
        zipped = list(zip(test_df_x, test_df_y, test_df_z, test_df_user, labels))
        self.df = pd.DataFrame(zipped, columns=['x', 'y', 'z', 'user', 'label'])


    def test_split_users_train_val_test(self):
        train_users, test_users, val_users = utils.split_users_train_val_test(self.df['user'])
        # Test that there is no cross-over between users in each set.
        self.assertTrue (np.intersect1d(train_users, val_users).size == 0)
        self.assertTrue (np.intersect1d(train_users, test_users).size == 0)
        self.assertTrue (np.intersect1d(val_users, test_users).size == 0)
        # Check that no users have been omitted.
        self.assertTrue (train_users.size + val_users.size + test_users.size == 4)


    def test_split_data_on_users(self):
        train_users, test_users, val_users = utils.split_users_train_val_test(self.df['user'])
        train, test, val = utils.split_data_on_users(self.df, train_users, test_users, val_users)

        train_df_users = np.unique(train.user)
        test_df_users = np.unique(test.user)
        val_df_users = np.unique(val.user)
        # Check that no users other than those that were allocated to that set are present.
        self.assertTrue (np.setdiff1d(train_df_users, train_users).size == 0)
        self.assertTrue (np.setdiff1d(test_df_users, test_users).size == 0)
        self.assertTrue (np.setdiff1d(val_df_users, val_users).size == 0)


    def test_encode_labels(self):
        new_df = utils.encode_labels(self.df, 'label')
        unique_labels = np.unique(self.df['label'])
        # Check that the one hot encoding is as expected (i.e. columns are aligned properly)
        for label_name in unique_labels:
            test_label = new_df.loc[new_df['label'] == label_name]
            assert ((test_label[label_name].all() == 1))


    def test_one_hot_cols_as_arrays(self):
        new_df = utils.encode_labels(self.df, 'label')
        train_users, test_users, val_users = utils.split_users_train_val_test(new_df['user'])
        train, test, val = utils.split_data_on_users(new_df, train_users, test_users, val_users)
        unique_labels = np.unique(self.df['label'])
        target_train, target_test, target_val = utils.one_hot_cols_as_arrays(unique_labels, train, test, val)

        one_hot_dim = unique_labels.size
        # Check that the one-hot vector has the expected dimensions
        # (test assumes that the there is an array of one-hot vectors, i.e. 2D array)
        self.assertTrue(target_train.shape[1] == one_hot_dim)
        self.assertTrue(target_val.shape[1] == one_hot_dim)
        self.assertTrue(target_test.shape[1] == one_hot_dim)

        # Get number of instances of each class in order for each data set.
        num_per_class_train = [len(train.loc[train['label'] == label]) for label in unique_labels]
        num_per_class_val = [len(val.loc[val['label'] == label]) for label in unique_labels]
        num_per_class_test = [len(test.loc[test['label'] == label]) for label in unique_labels]

        # Get number of instances of each one-hot encoded label in each data set.
        one_hot_train = Counter(map(tuple, target_train))
        one_hot_val = Counter(map(tuple, target_val))
        one_hot_test = Counter(map(tuple, target_test))

        for label_index in range(unique_labels.size):
            encoded_label = np.zeros(unique_labels.size)
            encoded_label[label_index] = 1
            encoded_label = tuple(encoded_label)
            self.assertTrue(num_per_class_train[label_index] == one_hot_train[encoded_label])
            self.assertTrue(num_per_class_val[label_index] == one_hot_val[encoded_label])
            self.assertTrue(num_per_class_test[label_index] == one_hot_test[encoded_label])

if __name__ == '__main__':
    unittest.main()