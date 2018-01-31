# References
# Number of layers, hidden nodes and feature maps were built to resemble that of:
# Ord́oñez, F.J., Roggen, D.: Deep convolutional and lstm recurrent neural networks for multimodal wearable activity recognition. Sensors 16(1), 115 (2016)

# Kernel size, dropout probability and degree of weight decay chosen with guidence from:
# Ronao, C.A., Cho, S.B.: Human activity recognition with smartphone sensors using deep learning neural networks. Expert Systems with Applications 59, 235–244 (2016)

from numpy.random import seed
seed(1)
import numpy as np
import time
import logging
import pandas as pd
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Conv1D,  Dense, Concatenate, Dropout, Flatten
from keras.regularizers import  l2
from keras.callbacks import CSVLogger
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from src.utils import utils
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Do not unnecessarily allocate GPUs not being usedl. 
import os

date_time = time.strftime('%d_%b_%Y_%H:%M:%S')

# For general output log and results.
logger = logging.getLogger('general_log')
handler = logging.FileHandler('./logs/5_fold_cross_validations_logs/multi_cnn3_all_conv_hybrid_model_{}.log'.format(date_time))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Log of performance at each epoch
csv_logger = CSVLogger('./logs/5_fold_cross_validations_logs/per_epoch_multi_cnn3_all_conv_hybrid_model{}'.format(date_time), append=True, separator=';')

###################################################### LOAD DATA #######################################################
logger.info('Loading data...')

# Read in cleaned data, removing erroneous column.
df = pd.read_csv('data/cleaned_WISDM_v1')

# Transform classes into numeric values and add one-hot-encoding and add concatenate to dataframe.
labels = np.unique(df['activity'])
df = utils.encode_labels(df, 'activity')
df = df.drop(['user', 'activity', 'timeastamp'], axis=1)
labels = np.array(df[labels])


#################################################### Hyperparameters #######################################################

window = 128
stride = 64

adam = optimizers.Adam(lr=0.0005)

lambda_reg = 0.0001
num_filters_1 = 64
kernal_size_1 = 20
num_filters_2 = 64
kernal_size_2 = 20
num_filters_3 = 64
kernal_size_3 = 20

pool_size = 2

num_epochs = 500

windowed_df = utils.sliding_window(df[['x','y','z']], (window, df[['x','y','z']].shape[1]), stride)
windowed_labels = utils.sliding_window(labels, (window, labels.shape[1]), stride)
extracted_labels = utils.extract_labels(windowed_labels)
assert (windowed_df.shape[0] == extracted_labels.shape[0])
shuffled_data, shuffled_target = shuffle(windowed_df,extracted_labels, random_state=1)

print('Data shape: {}'.format(shuffled_data.shape))
print('Data shape: {}'.format(shuffled_target.shape))


#################################################### Cross-validation #####################################################

def format_channels(windowed_data, num_channels):
    data_separated = utils.separate_windowed_channels(windowed_data, num_channels)
    data_reshaped = np.reshape(data_separated, data_separated.shape+(1,))
    return data_reshaped

ksplit = KFold(n_splits=5, random_state=1)
fold = 0
for train_split, test_split in ksplit.split(shuffled_data):
    
    train, target_train = shuffled_data[train_split], shuffled_target[train_split]
    test, target_test = shuffled_data[test_split], shuffled_target[test_split]

    print('train shape: {} {}'.format(train.shape, target_train.shape))
    print('test shape: {} {}'.format(test.shape, target_test.shape))

    train_x, train_y, train_z = format_channels(train, 3)
    test_x, test_y, test_z = format_channels(test, 3)

    ######################################################## MODEL #########################################################


    # Input layers
    inp_x = Input(shape=(train_x.shape[1], train_x.shape[2]))
    inp_y = Input(shape=(train_y.shape[1], train_y.shape[2]))
    inp_z = Input(shape=(train_z.shape[1], train_z.shape[2]))
    print(train_x.shape)
    # CNN x channel
    conv_1_x = Conv1D(num_filters_1, kernal_size_1, activation='relu', strides=1, padding='valid',
                      kernel_regularizer=l2(lambda_reg))(inp_x)
    conv_2_x = Conv1D(num_filters_2, kernal_size_2, activation='relu', strides=2, padding='valid',
                      kernel_regularizer=l2(lambda_reg))(conv_1_x)
    conv_3_x = Conv1D(num_filters_3, kernal_size_3, activation='relu', strides=2, padding='valid',
                      kernel_regularizer=l2(lambda_reg))(conv_2_x)
    drop_1_x = Dropout(.5)(conv_3_x)
    flatten_x = Flatten()(drop_1_x)

    # CNN y channel
    conv_1_y = Conv1D(num_filters_1, kernal_size_1, activation='relu', strides=1, padding='valid',
                      kernel_regularizer=l2(lambda_reg))(inp_y)
    conv_2_y = Conv1D(num_filters_2, kernal_size_2, activation='relu', strides=2, padding='valid',
                      kernel_regularizer=l2(lambda_reg))(conv_1_y)
    conv_3_y = Conv1D(num_filters_3, kernal_size_3, activation='relu', strides=2, padding='valid',
                      kernel_regularizer=l2(lambda_reg))(conv_2_y)
    drop_1_y = Dropout(.5)(conv_3_y)
    flatten_y = Flatten()(drop_1_y)

    # CNN z channel
    conv_1_z = Conv1D(num_filters_1, kernal_size_1, activation='relu', strides=1, padding='valid',
                      kernel_regularizer=l2(lambda_reg))(inp_z)
    conv_2_z = Conv1D(num_filters_2, kernal_size_2, activation='relu', strides=2, padding='valid',
                      kernel_regularizer=l2(lambda_reg))(conv_1_z)
    conv_3_z = Conv1D(num_filters_3, kernal_size_3, activation='relu', strides=2, padding='valid',
                      kernel_regularizer=l2(lambda_reg))(conv_2_z)
    drop_1_z = Dropout(.5)(conv_3_z)
    flatten_z = Flatten()(drop_1_z)

    channel_merge = Concatenate()([flatten_x, flatten_y, flatten_z])

    dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(lambda_reg))(channel_merge)
    dense_2 = Dense(128, activation='relu', kernel_regularizer=l2(lambda_reg))(dense_1)
    output = Dense(6, activation='softmax', kernel_regularizer=l2(lambda_reg))(dense_2)

    model = Model(inputs=[inp_x, inp_y, inp_z], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
    model.summary()

    model.fit([train_x, train_y, train_z], target_train,
              callbacks=[csv_logger], epochs=num_epochs)

    logger.info('Evaluating model on test set...\n')

    # Predict based on test set.
    probs = model.predict([test_x, test_y, test_z])

    ############################################################ Evaluation #########################################################################
    # Class is the index with the largest probability
    y_pred= probs.argmax(-1)
    # target_test is the index with 1 (largest value)
    y_true = target_test.argmax(-1)

    # Reporting
    report = classification_report(y_true,y_pred)
    logger.info('F-measure score on test set on fold {}:\n {}\n'.format(fold, report))

    # Confusion matrix
    conf_mat = confusion_matrix(y_true,y_pred)
    logger.info('Confusion matrix on fold {}: \n {} \n '.format(fold, conf_mat))

    metric_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', beta=1)
    # Index 2 contains macro f scores.
    logger.info('Macro f-score  on fold {} (sci-kit learn implementation){}:'.format(fold, metric_macro[2]))

    np.set_printoptions(precision=2)

    # Save model
    model.save('./models/5_fold_cross_validations_models/multi_cnn3_all_conv_hybrid_model_{}_fold_{}'.format(date_time, fold))
    logger.info('precision_recall_fscore_support {} (sci-kit learn implementation){}:'.format(fold, metric_macro))

    fold+=1
