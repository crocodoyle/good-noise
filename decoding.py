import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from mne import read_epochs, concatenate_epochs

from keras.models import Sequential, Model
from keras.layers import CuDNNLSTM, LSTM, Flatten, BatchNormalization, Dense, Dropout, TimeDistributed, Reshape, ConvLSTM2D, Conv3D, AveragePooling1D, Average, Input
from keras.layers import Conv1D

from keras.constraints import max_norm
from keras import regularizers
from keras.optimizers import Adam
from keras.utils import Sequence

import keras.backend as K

import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold

from sklearn.metrics import accuracy_score, confusion_matrix


import itertools, h5py, argparse


participants = ['ft10_p1', 'ft10_p2', 'ft10_p3', 'ft10_p4']
sessions = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
channel_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

preproc_types = ['filtered', 'amplified']

channels_file = 'Glasgow_BioSemi_132.ced'

data_dir = '/home/users/adoyle/data/'

n_channels = 128
n_timepoints = 667


class EEGEpochSequence(Sequence):

    def __init__(self, f, indices, batch_size):
        self.eeg = f['eeg']
        self.labels = f['label']
        self.batch_size = batch_size

        self.indices = indices

    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        return_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size].tolist()
        return self.eeg[return_indices, ...], self.labels[return_indices]

def merge_all_epochs(preproc_type):
    faces, noise, participant_num_face, participant_num_noise = [], [], [], []

    n_face_epochs = 0
    n_noise_epochs = 0

    for participant_idx, participant in enumerate(participants):
        for session_idx, session in enumerate(sessions):
            face_epochs = read_epochs(data_dir + '/epochs/' + preproc_type + '/faces_' + participant + '_' + session + '-epo.fif', proj=False, preload=False, verbose=False)
            noise_epochs = read_epochs(data_dir + '/epochs/' + preproc_type + '/noise_' + participant + '_' + session + '-epo.fif', proj=False, preload=False, verbose=False)

            faces.append(face_epochs)
            noise.append(noise_epochs)

            n_face_epochs += len(face_epochs)
            n_noise_epochs += len(noise_epochs)

    f = h5py.File(data_dir + '/epochs/' + preproc_type + '.hdf5', 'w')
    f.create_dataset('eeg', (n_face_epochs + n_noise_epochs, n_channels, n_timepoints), dtype='float32')
    f.create_dataset('label', (n_face_epochs + n_noise_epochs,), dtype='uint8')
    f.create_dataset('participant', (n_face_epochs + n_noise_epochs,), dtype='uint8')


    idx = 0
    for participant_idx, participant in enumerate(participants):
        for session_idx, session in enumerate(sessions):
            face_epochs = read_epochs(data_dir + '/epochs/' + preproc_type + '/faces_' + participant + '_' + session + '-epo.fif', proj=False, preload=True, verbose=False)
            noise_epochs = read_epochs(data_dir + '/epochs/' + preproc_type + '/noise_' + participant + '_' + session + '-epo.fif', proj=False, preload=True, verbose=False)

            faces = face_epochs.get_data()
            print(faces.shape)

            min, max = np.min(faces), np.max(faces)

            faces = (faces - min) / (max - min) # rescale between 0-1

            f['eeg'][idx:idx + faces.shape[0], 0:n_channels, :] = faces[:, 0:n_channels, :]
            f['participant'][idx:idx + faces.shape[0]] = participant_idx
            f['label'][idx:idx + faces.shape[0]] = 1

            idx += faces.shape[0]

            noises = noise_epochs.get_data()

            min, max = np.min(noises), np.max(noises)
            noises = (noises - min) / (max - min)  # rescale between 0-1

            f['eeg'][idx:idx + noises.shape[0], 0:n_channels, :] = noises[:, 0:n_channels, :]
            f['participant'][idx:idx + noises.shape[0]] = participant_idx
            f['label'][idx:idx + noises.shape[0]] = 0

            idx += noises.shape[0]

    f.close()

    return f

def load_labels(preproc_type):
    labels = np.load(data_dir + '/epochs/' + preproc_type + '_labels.npy')
    participant_nums = np.load(data_dir + '/epochs/' + preproc_type + '_participant_number.npy')

    indices = np.asarray(list(range(len(labels))))

    return labels, participant_nums, indices


def densenet_model(n_timepoints, n_channels):
    from densenet.classifiers.one_d import DenseNet121
    model = DenseNet121(input_shape=(n_timepoints, n_channels))

    model.layers.pop()

    input = model.input
    new_layer = Dense(1, activation='sigmoid')
    output = new_layer(model.layers[-1].output)

    new_model = Model(inputs=[input], output=output)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)

    new_model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy', 'mse'])
    return new_model

def conv_model(n_timepoints, n_channels):
    model = Sequential()

    model.add(Conv1D(32, 5, strides=1, padding='valid', activation='relu', data_format='channels_last', dilation_rate=1, input_shape=(n_timepoints, n_channels)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    model.add(Conv1D(32, 5, strides=1, padding='valid', activation='relu', data_format='channels_last', dilation_rate=1))
    model.add(Conv1D(32, 5, strides=1, padding='valid', activation='relu', data_format='channels_last', dilation_rate=1))
    model.add(Conv1D(32, 5, strides=1, padding='valid', activation='relu', data_format='channels_last', dilation_rate=1))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    model.add(Conv1D(64, 3, strides=1, padding='valid', activation='relu', data_format='channels_last', dilation_rate=1))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    model.add(Conv1D(128, 3, strides=1, padding='valid', activation='relu', data_format='channels_last', dilation_rate=1))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', 'mse'])
    return model

def averaged_LSTM(n_timepoints):
    model = Sequential()

    model.add(BatchNormalization(input_shape=(n_timepoints,)))
    model.add(AveragePooling1D(pool_size=2, strides=None, padding='valid'))
    model.add(CuDNNLSTM(128, return_sequences=True))
    #     model.add(LSTM(128, dropout=0.5, recurrent_constraint=max_norm(2.), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def lstm_conv_model(n_channels, n_timepoints):
    model = Sequential()
    model.add(Reshape((n_channels // 8, n_channels // 16), input_shape=(n_channels, n_timepoints)))
    model.add(ConvLSTM2D(32, (3, 3), return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(32, (3, 3), return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(32, (3, 3), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def lstm_model(n_channels, n_timepoints):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(n_channels, n_timepoints)))
    model.add(AveragePooling1D(pool_size=2, strides=None, padding='valid'))
    model.add(CuDNNLSTM(512, data_format='channels_first', return_sequences=True))
    #     model.add(LSTM(128, dropout=0.5, recurrent_constraint=max_norm(2.), return_sequences=True))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def batch(f, indices, labels, batch_size, average_channels=False):
    eeg = f['eeg']

    while True:
        np.random.shuffle(indices)

        for index in range(0, len(indices), batch_size):
            if index + batch_size > len(indices):
                break

            if average_channels:
                yield np.mean(eeg[index:index + batch_size, ...], axis=1), labels[index:index + batch_size]
            else:
                yield eeg[index:index + batch_size, ...], labels[index:index + batch_size]


def train(remake_hdf5):
    n_epochs = 5
    n_folds = 4

    batch_size = 1024

    test_accuracies = np.zeros(n_folds, dtype='float32')

    for preproc_type in preproc_types:

        if remake_hdf5:
            merge_all_epochs(preproc_type)
        with h5py.File(data_dir + '/epochs/' + preproc_type + '.hdf5', 'r') as f:

            labels = f['label']
            participant_nums = f['participant']
            all_indices = np.asarray(range(len(labels)))

            gkf = GroupKFold(n_splits=n_folds)
            for fold_idx, (train_indices, test_indices) in enumerate(gkf.split(all_indices, labels, participant_nums)):

                train_labels = labels[train_indices.tolist()]
                test_labels = labels[test_indices.tolist()]

                # model = sequential_model(n_channels, n_timepoints)
                model = densenet_model(n_channels, n_timepoints)
                # model.summary(line_length=150)

                # print('Training, testing:', len(train_labels), len(test_indices))
                # print('Training faces fraction:', np.sum(train_labels) / len(train_labels))
                # print('Testing faces fraction:', np.sum(test_labels) / len(test_labels))

                eeg_seq_train = EEGEpochSequence(f, train_indices, batch_size)
                eeg_seq_test = EEGEpochSequence(f, test_indices, batch_size)

                # history = model.fit_generator(batch(f, train_indices, train_labels, batch_size, average_channels=False), steps_per_epoch=len(train_indices) // batch_size - 1, epochs=n_epochs, shuffle=False, use_multiprocessing=True, workers=4)
                history = model.fit_generator(eeg_seq_train, epochs=n_epochs, shuffle=True, use_multiprocessing=True, workers=4)

                predictions = model.predict_generator(eeg_seq_test)
                test_accuracies[fold_idx] = accuracy_score(test_labels, np.argmax(predictions, axis=-1))

            print('Results for preprocessing type:', preproc_type)
            print('Fold testing accuracies:', test_accuracies)

        K.clear_session()

if __name__ == '__main__':
    print('Decoding faces vs. noise')

    parser = argparse.ArgumentParser(description='RNN for face detection')
    parser.add_argument('--remake-hdf5', action='store_true', default=False,
                        help='remake the hdf5 file if preprocessing has changed')

    args = parser.parse_args()

    train(args.remake_hdf5)




