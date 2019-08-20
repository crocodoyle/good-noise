import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import mne, npa
from npa import NPA

from mne.io import read_raw_edf
from mne import Epochs, read_epochs, concatenate_epochs

from mne.time_frequency import psd_welch, psd_multitaper

from mne.datasets.sleep_physionet.age import fetch_data

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
from fooof import FOOOF, FOOOFGroup


import argparse, time, os


colours = ['goldenrod', 'forestgreen', 'navy', 'rebeccapurple']

subjects = list(range(20))

mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

event_ids = {'Sleep stage W': 1,
             'Sleep stage 1': 2,
             'Sleep stage 2': 3,
             'Sleep stage 3': 4,
             'Sleep stage 4': 5,
             'Sleep stage R': 6}

stages = sorted(event_ids.keys())

bg_slope = dict()
for stage in stages:
    bg_slope[stage] = []

preproc_types = ['NPA', 'Bandpass', 'Highpass', 'Raw']
data_dir = 'c:/Users/doyle/mne_data/physionet-sleep-data/'


def save_epochs(epochs, subject_id, session_id, preproc):
    for stage in stages:
        try:
            epochs[stage].save(data_dir + '/epochs/' + preproc + '_' + stage[-1] + '_' + str(subject_id) + '_' + str(session_id) + '-epo.fif', overwrite=True)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic Sleep Scoring with NPA.')
    args = parser.parse_args()

    os.makedirs(data_dir + '/epochs/', exist_ok=True)
    os.makedirs(data_dir + '/FOOOFs/', exist_ok=True)

    print('Arguments for this experiment:', args)

    start_all = time.time()
    all_epochs = []

    datafiles = fetch_data(subjects=list(range(20)))

    # fg = FOOOFGroup(peak_width_limits=[1, 12.0])

    for eeg_filename, label_filename in datafiles:
        eeg = mne.io.read_raw_edf(eeg_filename, preload=True, verbose=0)
        labels = mne.read_annotations(label_filename)

        subject_id = int(eeg_filename.split('\\')[-1][3:5])
        session_id = int(eeg_filename.split('\\')[-1][5])

        eeg.set_annotations(labels, emit_warning=False)
        eeg.set_channel_types(mapping)

        chunk_length = 2.
        tmax = chunk_length - 1. / eeg.info['sfreq']

        events, _ = mne.events_from_annotations(eeg, event_id=event_ids, chunk_duration=chunk_length, verbose=0)

        eeg_info = eeg.info.copy()

        try:
            epochs = mne.Epochs(raw=eeg, events=events, event_id=event_ids, tmin=0., tmax=tmax, baseline=None, picks='eeg', verbose=0)
            save_epochs(epochs, subject_id, session_id, 'raw')
        except Exception as e:
            print('RAW:', e)

        freq_range = [1, 45]

        psd, freq = psd_welch(eeg, fmin=freq_range[0], fmax=freq_range[1], picks='eeg', n_jobs=2, verbose=0)
        print(psd.shape, freq.shape)

        ff = FOOOF(peak_threshold=2.0, aperiodic_mode='knee', peak_width_limits=(1.0, 10.0), verbose=False)
        ff.fit(freq, psd[0,:], freq_range=freq_range)
        ff.plot(save_fig=True, file_name=str(subject_id) + '_' + str(session_id), file_path=data_dir+'/FOOOFs/')

        amp = npa.NPA(ff, eeg.info['sfreq'])
        amp.fit_filters(peak_mode='normal')

        time_series = eeg.get_data()
        print('EEG shape:', time_series.shape)

        time_series[0:2, :] = amp.amplify(time_series[0:2, :])

        eeg = mne.io.RawArray(np.float64(time_series), eeg_info, verbose=0)

        try:
            epochs = mne.Epochs(raw=eeg, events=events, event_id=event_ids, tmin=0., tmax=tmax, baseline=None, picks='eeg')
            save_epochs(epochs, subject_id, session_id, 'npa')
        except Exception as e:
            print('NPA:', e)

    #     for stage in stages:
    #         psd, freq = psd_welch(epochs[stage], n_jobs=-1)
    #
    #         try:
    #             fg.fit(freqs=freq, power_spectra=psd[:, 0], freq_range=[2, 45], n_jobs=-1)
    #             exps = fg.get_params('aperiodic_params', 'exponent')
    #
    #             bg_slope[stage].extend(exps)
    #         except Exception as e:
    #             print(e)
    #
    # param_fig, param_axes = plt.subplots(1, 1, sharey=True, sharex=True)
    #
    # for stage_idx, stage in enumerate(stages):
    #     param_axes.hist(bg_slope[stage], density=True, label=stage[-1])
    #
    # param_axes.legend(shadow=True, fancybox=True)
    # param_fig.savefig(data_dir + 'results.png')

    elapsed = time.time() - start_all
    print('Took', elapsed / 60, 'minutes')

