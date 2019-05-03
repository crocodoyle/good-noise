import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from mne import read_epochs, concatenate_epochs

from keras.models import Sequential, Model
from keras.layers import CuDNNLSTM, LSTM, Flatten, BatchNormalization, Dense, Dropout, TimeDistributed, Reshape, ConvLSTM2D, Conv3D, AveragePooling1D, Average, Input
from keras.layers import Conv1D, Permute, multiply, RepeatVector, Lambda

from keras.constraints import max_norm, UnitNorm
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras.utils import Sequence, to_categorical

import keras.backend as K

from sklearn.model_selection import GroupKFold

from sklearn.metrics import accuracy_score

import itertools, h5py, argparse


participants = ['ft10_p1', 'ft10_p2', 'ft10_p3', 'ft10_p4']
sessions = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
channel_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

preproc_types = ['NPA', 'Bandpass', 'Highpass', 'Raw']

channels_file = 'Glasgow_BioSemi_132.ced'

data_dir = '/data1/users/adoyle/eeg_test_retest/'

n_channels = 128
n_timepoints = 667

plot_colours = ['blue', 'red', 'green', 'darkorange']

seeds = [1337, 42, 24, 7331]


class EEGEpochSequence(Sequence):

    def __init__(self, f, indices, batch_size):
        self.eeg = f['eeg']
        self.labels = to_categorical(f['label'])
        self.batch_size = batch_size

        self.indices = indices

    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        return_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size].tolist()
        return np.swapaxes(self.eeg[return_indices, ...], 1, 2), self.labels[return_indices]


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

            # min, max = np.min(faces), np.max(faces)
            #
            # faces = (faces - min) / (max - min) # rescale between 0-1

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


def attention_3d_block(inputs, single_attention_vec):
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = int(inputs.shape[1])
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vec:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def get_activations(model, inputs, layer_name='attention_vec'):
    activations = []
    inp = model.input

    outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs

    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)

    return activations



def lstm_attention_model(n_channels, n_timepoints):
    inputs = Input(shape=(n_timepoints, n_channels))
    attention_mul = attention_3d_block(inputs, True)
    attention_mul = CuDNNLSTM(16, recurrent_constraint=max_norm(2.), return_sequences=False)(attention_mul)

    output = Dense(2, activation='softmax')(attention_mul)

    model = Model(input=[inputs], output=output)

    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # sgd = SGD(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def lstm_model(n_channels, n_timepoints):
    inputs = Input(shape=(n_timepoints, n_channels))
    attention_mul = CuDNNLSTM(16, recurrent_constraint=max_norm(2.), return_sequences=True)(inputs)

    flat = Flatten()(attention_mul)
    output = Dense(2, activation='softmax')(flat)

    model = Model(input=[inputs], output=output)

    # optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    optimizer = SGD(lr=0.0002)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train(attention):
    n_epochs = 25
    n_folds = len(participants)
    n_preproc_types = len(preproc_types)

    batch_size = 1024

    test_accuracies = np.zeros((n_preproc_types, n_folds), dtype='float32')
    train_accuracies = np.zeros((n_preproc_types, n_folds, n_epochs), dtype='float32')

    losses = np.zeros((n_preproc_types, n_folds, n_epochs), dtype='float32')

    if attention:
        model = lstm_attention_model(n_channels, n_timepoints)
    else:
        model = lstm_model(n_channels, n_timepoints)
    model.summary()

    attention_fig, attention_ax = plt.subplots(1, 1)
    results_fig, results_ax = plt.subplots(1, n_folds, figsize=(24, 4))
    loss_fig, loss_ax = plt.subplots(1, n_folds, figsize=(24, 4))

    for preproc_idx, preproc_type in enumerate(preproc_types):
        print('Beginning analysis for', preproc_type, 'pre-processing')

        with h5py.File(data_dir + '/epochs/' + preproc_type + '.hdf5', 'r') as f:

            labels = f['label']
            participant_nums = f['participant']
            all_indices = np.asarray(range(len(labels)))

            gkf = GroupKFold(n_splits=n_folds)
            for fold_idx, (train_indices, test_indices) in enumerate(gkf.split(all_indices, labels, participant_nums)):
                print('Training', preproc_type, 'fold', str(fold_idx+1), '/', str(n_folds))
                # train_labels = labels[train_indices.tolist()]
                # test_labels = labels[test_indices.tolist()]

                np.random.seed(seeds[fold_idx])
                # model = sequential_model(n_channels, n_timepoints)
                if attention:
                    model = lstm_attention_model(n_channels, n_timepoints)
                else:
                    model = lstm_model(n_channels, n_timepoints)

                # print('Training, testing:', len(train_labels), len(test_indices))
                # print('Training faces fraction:', np.sum(train_labels) / len(train_labels))
                # print('Testing faces fraction:', np.sum(test_labels) / len(test_labels))

                eeg_seq_train = EEGEpochSequence(f, train_indices, batch_size)
                eeg_seq_test = EEGEpochSequence(f, test_indices, batch_size)

                history = model.fit_generator(eeg_seq_train, epochs=n_epochs, steps_per_epoch=3, shuffle=True, use_multiprocessing=False)

                metrics = model.evaluate_generator(eeg_seq_test)
                print(model.metrics_names, metrics)

                # for (eeg_input, label) in eeg_seq_test:
                #     test_probs = model.predict_on_batch(eeg_input)
                #
                #     test_predictions = np.argmax(test_probs, axis=-1)
                #     batch_accuracy = accuracy_score(np.argmax(label, axis=-1), test_predictions)
                #     print(label[0], label[-1])
                #     print(eeg_input[0], eeg_input[-1])
                #     print('batch accuracy:', batch_accuracy)

                if attention:
                    attention_vector = np.asarray(get_activations(model, eeg_seq_test[0][0])).squeeze()
                    print('attention:', attention_vector.shape)

                    average_attention = np.mean(np.mean(attention_vector, axis=2), axis=0)

                    time_steps = (np.asarray(range(0, n_timepoints), dtype='float32') / 512) - 0.3
                    attention_ax.plot(time_steps, average_attention, color=plot_colours[preproc_idx], label=preproc_type)

                test_accuracies[preproc_idx, fold_idx] = metrics[1]


                train_accuracies[preproc_idx, fold_idx, :] = np.copy(history.history['acc'])
                losses[preproc_idx, fold_idx, :] = np.copy(history.history['loss'])

            print('Results for preprocessing type:', preproc_type)
            print('Fold testing accuracies:', test_accuracies)

        K.clear_session()

    if attention:
        attention_ax.set_xlabel('Time (s)', fontsize=16)
        attention_ax.set_ylabel('Attention', fontsize=16)
        attention_ax.legend()

        attention_fig.savefig(data_dir + '/results/attention.png', bbox_inches='tight')

    print('Results:')
    for preproc_idx, preproc_type in enumerate(preproc_types):
        print('Pre-processing type:', preproc_type, 'train accuracy:', np.mean(train_accuracies[preproc_idx, :, -1]), 'test accuracy:', np.mean(test_accuracies[preproc_idx, :]))

        for fold_idx in range(n_folds):
            # print(train_accuracies[preproc_idx, fold_idx, :])

            # if fold_idx == 0:
                results_ax[fold_idx].plot(train_accuracies[preproc_idx, fold_idx, :], color=plot_colours[preproc_idx], label=preproc_type)
                loss_ax[fold_idx].plot(losses[preproc_idx, fold_idx, :], color=plot_colours[preproc_idx], label=preproc_type)
                # results_ax[0].plot(losses[preproc_idx, fold_idx, :], color=plot_colours[preproc_idx], linestyle='--', label=preproc_type)
            # else:
            #     results_ax.plot(train_accuracies[preproc_idx, fold_idx, :], color=plot_colours[preproc_idx])
            #     loss_ax.plot(losses[preproc_idx, fold_idx, :], color=plot_colours[preproc_idx])
            #     # results_ax[0].plot(losses[preproc_idx, fold_idx, :], color=plot_colours[preproc_idx], linestyle='--')

                results_ax[fold_idx].legend(loc='center right', shadow=True, fancybox=True)

                results_ax[fold_idx].set_xlabel('Epoch', fontsize=16)
                results_ax[fold_idx].set_ylabel('Train Accuracy', fontsize=16)
                results_ax[fold_idx].set_ylim([0.45, 1.05])

                loss_ax[fold_idx].legend(shadow=True, fancybox=True)

                loss_ax[fold_idx].set_xlabel('Epoch', fontsize=16)
                loss_ax[fold_idx].set_ylabel('Loss', fontsize=16)

    results_fig.savefig(data_dir + '/results/decoding_results.png', dpi=500, bbox_inches='tight')
    loss_fig.savefig(data_dir + '/results/loss.png', dpi=500, bbox_inches='tight')


    test_results_fig, test_results_ax = plt.subplots(1, 1, figsize=(4, 3))

    boxplots = test_results_ax.boxplot(test_accuracies.T, labels=preproc_types, patch_artist=True)
    for patch, colour in zip(boxplots['boxes'], plot_colours):
        patch.set_facecolor(colour)

    test_results_ax.set_xlabel('Pre-Processing Method', fontsize=16)
    test_results_ax.set_ylabel('Test Accuracy', fontsize=16)
    test_results_ax.grid(b=True, which='both')

    test_results_fig.savefig(data_dir + '/results/test_scores.png', dpi=500, bbox_inches='tight')

if __name__ == '__main__':
    print('Decoding faces vs. noise')

    parser = argparse.ArgumentParser(description='RNN for face detection')
    parser.add_argument('--make-hdf5', action='store_true', default=False,
                        help='remake the hdf5 file if preprocessing has changed')
    parser.add_argument('--attention', action='store_true', default=False,
                        help='use the attention model')

    args = parser.parse_args()
    print(args)

    if args.make_hdf5:
        for preproc_type in preproc_types:
            merge_all_epochs(preproc_type)

    train(args.attention)




