import numpy as np
import pandas as pd
from skimage.util.shape import view_as_windows
from numpy.random import seed
seed(1)

class utils:


    def split_users_train_val_test(user_col, split_dim = (.7, .9)):
        '''
        :param user_col: Pandas dataframe column containing users.
        :param split_dim: tuple - % of data in [train, test], remainder will be validation set (split in order).
        :return: 3 arrays - Training users, test users, validation users.
        '''
        # Separate users so models can be made personal/impersonal etc.
        unique_users = np.unique(user_col)
        train_users, test_users, val_users, = np.split(unique_users, [int(split_dim[0] * len(unique_users)), int(split_dim[1] * len(unique_users))])
        return train_users, test_users, val_users

    def split_data_on_users(df, train_users, test_users, val_users=None):
        '''
        :param df: Dataframe containing data to be split.
        :param train_users: Users in training set.
        :param test_users: Users in test set.
        :param val_users: (Optional) Users in validation set (default=None)
        :return: training, test and validation dataframe
        '''
        # split based on the users in the train/test/val split.
        train = df.loc[df.user.isin(train_users)]
        test = df.loc[df.user.isin(test_users)]
        if val_users is not None:
            val = df.loc[df.user.isin(val_users)]
            return train, test, val
        return train, test


    def encode_labels( df, label_col):
        '''
        Adds one-hot-encoded columns to dataframe based on a categorical column.
        :param df: Dataframe to add one hot encoding to.
        :param label_col: Categorical column to encode.
        :return: New dataframe with one-hot-encoded columns.
        '''
        one_hot = pd.get_dummies(df[label_col])
        one_hot_df = pd.concat([df, one_hot], axis=1)
        return one_hot_df

    def one_hot_cols_as_arrays( col_subset, train_set, test_set, val_set=None):
        '''
        Changes a subset of pandas dataframe columns into arrays for each dataset passed.
        Used for obtaining one-hot-encoded target vector.
        :param col_subset: subset of columns which are one-hot encoded.
        :param train_set: Dataframe for training set.
        :param test_set: Dataframe for test set.
        :param val_set: (Optional) Dataframe for validation set.
        :return:
        '''
        target_train = np.array(train_set[col_subset])
        target_test = np.array(test_set[col_subset])
        if val_set is not None:
            target_val = np.array(val_set[col_subset])
            return target_train, target_test, target_val
        return target_train, target_test



    def sliding_window(data, window_size, stride):
        '''
        :param data: data to be windowed.
        :param window_size: Window size in each dimension as tuple.
        :param stride: stride for sliding window.
        :return: Multi-dimensional windowed channels
        '''
        # Ensure data is stored contiguously
        data = np.ascontiguousarray(data)
        windowed = view_as_windows(data, window_size, stride)
        return windowed.reshape((windowed.shape[0], windowed.shape[2], windowed.shape[3]))



    def separate_windowed_channels(windowed, num_channels):
        '''
        :param windowed: Data that has been windowed.
        :param num_channels: Number of channels
        :return: Array with (num_channels, number of samples, window_size)
        '''
        return np.array([windowed[...,i] for i in range(num_channels)])



    def extract_labels(windowed_labels, label_index = -1):
        '''
        :param windowed_labels: labelled data windowed in the same way as the data.
        :param label_index: (Optional) Index of label, last one by default.
        :return: array containing labels of data.
        '''
        return np.array([window[-1] for window in windowed_labels])


# Quick test on sliding window and shuffle.
# from sklearn.utils import shuffle
# window_size = 5
# stride = 2
#
# y = np.array([[0,0,1], [0,0,0], [0,1,1], [1,0,0],[1,0,1],[0,0,1], [0,0,0], [0,1,1], [1,0,0],[1,0,1]])
# windowed_labels = utils.sliding_window(y, (window_size, y.shape[1]), stride)
# formatted_labels = utils.extract_labels(windowed_labels)
#
# data = np.array([['x1', 'y1', 'z1', 'd', 'j', 'si', 'st', 'u', 'w'],
#                  ['x2', 'y2', 'z2', 'd', 'j', 'si', 'st', 'u', 'w'],
#                  ['x3', 'y3', 'z3', 'd', 'j', 'si', 'st', 'u', 'w'],
#                  ['x4', 'y4', 'z4', 'd', 'j', 'si', 'st', 'u', 'w'],
#                  ['x5', 'y5', 'z5', 'd', 'j', 'si', 'st', 'u', 'w'],
#                  ['x6', 'y6', 'z6', 'd', 'j', 'si', 'st', 'u', 'w'],
#                  ['x7', 'y7', 'z7', 'd', 'j', 'si', 'st', 'u', 'w'],
#                  ['x8', 'y8', 'z8', 'd', 'j', 'si', 'st', 'u', 'w'],
#                  ['x9', 'y9', 'z9', 'd', 'j', 'si', 'st', 'u', 'w'],
#                  ['x10','y10','z10', 'd', 'j', 'si', 'st', 'u', 'w']])
# windowed_data = utils.sliding_window(data, (window_size, data.shape[1]), stride)
#
# print('Windowed data\n', windowed_data)
# print('Windowed labels\n', windowed_labels)
#
# shuffled_data, shuffled_y = shuffle(windowed_data,formatted_labels)
#
# print('Shuffled data\n', shuffled_data)
# print('Shuffled targets\n', shuffled_y)
#
# channels_extracted = utils.separate_windowed_channels(shuffled_data, 3)
# channels_extracted = np.reshape(channels_extracted, channels_extracted.shape+(1,))
# print(channels_extracted)
#
# x, y, z = channels_extracted
# print('x channel', x)
# print('y, channel', y)
# print('z channel', z)