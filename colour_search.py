import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import mne
from mne.io import read_raw_edf, set_bipolar_reference, read_raw_eeglab
from mne import Epochs, pick_types, find_events
from mne.viz.topomap import plot_psds_topomap, plot_topomap
from mne.viz import plot_ica_components
from mne.time_frequency import psd_multitaper, psd_welch
from scipy.signal import welch, lfilter, filtfilt

from mne.parallel import parallel_func
from functools import partial

from mne.preprocessing import compute_proj_eog, ICA, ica_find_ecg_events, ica_find_eog_events
from mne.preprocessing import create_ecg_epochs, create_eog_epochs

from scipy.io import loadmat
from scipy.signal import periodogram, deconvolve, cheby2
from scipy.misc import pade

from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import csv, io, pickle

import numpy as np
from fooof import FOOOF, FOOOFGroup
from fooof.utils import combine_fooofs, get_settings

import argparse, time, os

import sys
from PIL import Image


participants = ['ft10_p1', 'ft10_p2', 'ft10_p3', 'ft10_p4']
sessions = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
channel_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

eog_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4']

# preproc_types = ['raw', 'filtered', 'ica', 'bads2good', 'amplified']

preproc_types = ['filtered', 'amplified']

channels_file = 'Glasgow_BioSemi_132.ced'

data_dir = 'D:/eeg_test_retest/'

n_channels = 132
channel_names = channel_names[0:n_channels]
n_freqs = 256
n_events = 1100
n_bands = 5
n_participants = len(participants)
n_sessions = len(sessions)

# n_freqs_fooofed = 47027

fooof_group = FOOOFGroup()
freq = None


# load channel positions
def load_montage():
    location_reader = csv.reader(open(data_dir + 'Glasgow_BioSemi_132.ced', 'r'), delimiter='\t')

    lines = list(location_reader)[1:]

    points = np.zeros((n_channels, 3), dtype='float32')
    for i, line in enumerate(lines):
        points[i, :] = [line[5], line[4], line[6]]

    montage = mne.channels.Montage(points, channel_names, 'Glasgow_BioSemi_132', list(range(132)))

    return montage


def load_results(participant_idx, session_idx):
    behaviour = loadmat(data_dir + 'ft10_behaviour/ft10_behaviour/' + participants[participant_idx] + sessions[session_idx] + '.mat')
    # print(behaviour['results'])

    #     print(behaviour.keys())
    #     print(behaviour)

    #     for result_array in behaviour['results'][0][0]:
    #         print(result_array[0])

    stim_labels = behaviour['results'][0][0][1] - 1
    correct = behaviour['results'][0][0][3]

    return stim_labels[0], correct[0]


def ssp_preprocessing(eeg, participant, session_name, reject):
    try:
        projs, eog_events = compute_proj_eog(eeg, n_grad=0, n_mag=0, n_eeg=2, reject=reject, average=True)

        eog_projs = projs[-2:]

        f_eog = mne.viz.plot_projs_topomap(eog_projs, info=eeg.info, show=False)
        f_eog.savefig(data_dir + 'results/EOG_projections_' + participant + '_' + session_name + '.png', dpi=500)
        plt.close(f_eog)

        eeg = eeg.add_proj(eog_projs[-1:])
        eeg = eeg.apply_proj()

    except Exception as e:
        print('Problem with EOG plot:', e)

    return eeg

def ica_preprocessing(eeg, filter_eeg, participant, session, eog_channel, reject, f_low, f_high):
    try:
        ica = ICA(n_components=0.98, method='extended-infomax')

        # eeg_copy = eeg.copy()
        # eeg_copy.filter(2, 40, picks=list(range(132)), n_jobs=7, verbose=0)

        reject_only_eeg = dict(eeg=reject['eeg'])

        ica.fit(filter_eeg, picks=list(range(128)), start=10000, stop=30000, decim=4, reject=reject, verbose=0)

        eog_channel_number = eeg.ch_names.index(eog_channel)

        eog_average = create_eog_epochs(filter_eeg, ch_name=eog_channel, tmin=-.5, tmax=.5, l_freq=f_low, reject=reject_only_eeg, picks=list(range(128)), verbose=0).average()
        eog_epochs = create_eog_epochs(filter_eeg, ch_name=eog_channel, tmin=-.5, tmax=.5, l_freq=f_low, reject=reject, picks=list(range(128)) + [eog_channel_number], verbose=0)

        # plot_ica_components(ica, picks=None, res=64, cmap='interactive', sensors=True, colorbar=False, title=None, show=True, outlines='head', contours=6, inst=eeg_copy)

        try:
            eog_inds = []

            threshold = 3
            while len(eog_inds) < 1:
                threshold -= 0.05
                eog_inds, scores = ica.find_bads_eog(eog_epochs, l_freq=f_low, threshold=threshold, verbose=0)
            eog_inds = [eog_inds[0]]
            print('ICA threshhold:', threshold)

            ica_eog_scores_fig = ica.plot_scores(scores, exclude=eog_inds, show=False)
            ica_eog_scores_fig.savefig(data_dir + '/results/ica_eog_scores_' + participant + '_' + session + eog_channel + '.png')
            plt.close(ica_eog_scores_fig)

            sources_fig = ica.plot_sources(eog_average, exclude=eog_inds, show=False)  # look at source time course
            sources_fig.savefig(data_dir + '/results/ica_eog_sources_' + participant + '_' + session + eog_channel + '.png')
            plt.close(sources_fig)

            ica_properties_fig = ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 35., 'n_jobs': 7}, image_args={'sigma': 1.}, show=False)
            for n_fig, fig in enumerate(ica_properties_fig):
                fig.savefig(data_dir + '/results/ica_properties_' + participant + '_' + session + '_' + str(n_fig) + eog_channel + '.png')
                plt.close(fig)

            ica_excluded_fig = ica.plot_overlay(eog_average, exclude=eog_inds, show=False)
            ica_excluded_fig.savefig(data_dir + '/results/ica_excluded_' + participant + '_' + session + eog_channel + '.png')
            plt.close(ica_excluded_fig)

            ica.exclude.extend(eog_inds)
            ica.apply(eeg)

        except ValueError as e:
            print(e)
            print('Problem with ICA')


    except ZeroDivisionError as e:
        print('Zero Division Error in ICA for subj', participant, session, e)

    return eeg


def combine_vertical_eog(eeg):
    EOG_only = eeg.copy().pick_channels(['EXG1', 'EXG2'])
    # print(EOG_only[0][0][0])
    EOG_only[0][0][0] = (EOG_only[0][0][0] + EOG_only[0][1][0]) / 2
    EOG_only.drop_channels(['EXG2'])
    EOG_only.rename_channels({'EXG1': 'EOG'})
    EOG_only.set_channel_types({'EOG': 'eog'})

    eeg.add_channels([EOG_only], force_update_info=True)

    return eeg


def fooof_channel_rejection(eeg, psds, freqs, f_low, f_high, participant, session_name):
    from scipy.stats import bayes_mvs

    n_bads = 0

    fooof_group = FOOOFGroup(max_n_peaks=6, min_peak_amplitude=0.1, peak_width_limits=[1, 12], background_mode='knee')
    fooof_group.fit(freqs, psds, freq_range=[f_low, f_high/2], n_jobs=-1)
    fooof_group_fig = fooof_group.plot(save_fig=True, file_name='FOOOF_group_' + participant + '_' + session_name,
                                       file_path=data_dir + '/results/')

    bg_slope = fooof_group.get_all_data('background_params', col='slope')

    mean_cntr, var_cntr, std_cntr = bayes_mvs(bg_slope, alpha=0.9)

    lower_slope = mean_cntr[1][0] - std_cntr[1][1]
    upper_slope = mean_cntr[1][1] + std_cntr[1][1]

    print('upper and lower slope range (mean, std)', lower_slope, upper_slope, np.mean(bg_slope), np.std(bg_slope))

    for channel_idx, slope in enumerate(bg_slope):
        if slope < lower_slope or slope > upper_slope:
            eeg.info['bads'].append(eeg.ch_names[channel_idx])
            n_bads += 1

    eeg.interpolate_bads(reset_bads=True)

    return eeg, n_bads

def convert_epochs_float32(epochs):
    epoch_data = epochs.get_data()

    min, max = np.min(epoch_data), np.max(epoch_data)
    epoch_data = (epoch_data - min) / (max - min)

    epoch_data_float32 = np.float32(np.copy(epoch_data))

    new_epochs = mne.EpochsArray(epoch_data_float32, epochs.info, verbose=0)

    return new_epochs


def plot_connectivity(epochs, participant, session_name, type, condition):

    fs = epochs.info['sfreq']

    con, freqs, times, n_epochs, n_tapers = spectral_connectivity(epochs, method=['wpli2_debiased'], mode='multitaper', sfreq=fs, fmin=4, fmax=128, faverage=True, mt_adaptive=False, n_jobs=7, verbose=0)
    con = con[0:128, 0:128, 0]

    from mayavi import mlab

    mfig = mlab.figure(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))
    mfig.scene.disable_render = True

    # Plot the sensor locations
    sens_loc = epochs.info['chs'][0:128]
    sens_loc = [[x['loc'][0], x['loc'][1], x['loc'][2]] for x in sens_loc]
    sens_loc = np.array(sens_loc)

    pts = mlab.points3d(sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2], color=(1, 1, 1), opacity=1, scale_factor=0.025)

    # Get the strongest connections
    n_con = 30  # show up to 20 connections
    min_dist = 0.05  # exclude sensors that are less than 5cm apart
    threshold = np.sort(con, axis=None)[-n_con]
    ii, jj = np.where(con >= threshold)

    # Remove close connections
    from scipy import linalg
    con_nodes = list()
    con_val = list()
    for i, j in zip(ii, jj):
        if linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
            con_nodes.append((i, j))
            con_val.append(con[i, j])

    con_val = np.array(con_val)

    # Show the connections as tubes between sensors
    vmax = 0.6
    vmin = 0.2
    for val, nodes in zip(con_val, con_nodes):
        x1, y1, z1 = sens_loc[nodes[0]]
        x2, y2, z2 = sens_loc[nodes[1]]
        points = mlab.plot3d([x1, x2], [y1, y2], [z1, z2], [val, val], vmin=vmin, vmax=vmax, tube_radius=0.01, colormap='Spectral')
        points.module_manager.scalar_lut_manager.reverse_lut = True

    mlab.scalarbar(points, title=None, nb_labels=4, orientation='vertical')

    # Add the sensor names for the connections shown
    nodes_shown = list(set([n[0] for n in con_nodes] + [n[1] for n in con_nodes]))

    for node in nodes_shown:
        point = sens_loc[node]
        mlab.text3d(point[0], point[1], point[2], channel_names[node], scale=0.05, color=(0, 0, 0))

    mfig.scene.disable_render = False
    view = (-88, 40.8)
    mlab.view(*view)
    mlab.savefig(data_dir + '/results/' + participant + session_name + '_' + type + '_' + condition + '_connectivity.png')
    mlab.close()


def save_epochs_as(eeg, preproc_type, events, reject, participant, session_name):
    os.makedirs(data_dir + '/results/' + preproc_type)

    face_epochs = Epochs(eeg, events, [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111], tmin=-0.3, tmax=1,
                             picks=list(range(132)), proj=True, reject=reject, detrend=1, preload=False, verbose=0).drop_bad()

    # face_csd = csd_multitaper(face_epochs, fmin=0, fmax=128, n_jobs=1)
    # face_csd_fig = face_csd.plot(mode='coh', show=False)
    # face_csd_fig.savefig(data_dir + '/results/' + participant + session_name + '_faces_coherence.png')
    # plt.close(face_csd_fig)

    face_epochs = convert_epochs_float32(face_epochs)
    face_epochs.save(data_dir + '/epochs/' + preproc_type + '/faces_' + participant + '_' + session_name + '-epo.fif',
                     verbose=0)

    noise_epochs = Epochs(eeg, events, [12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112], tmin=-0.3, tmax=1,
                              picks=list(range(132)), proj=True, reject=reject, detrend=1, preload=False, verbose=0).drop_bad()
    # noise_csd = csd_multitaper(noise_epochs, fmin=0, fmax=128, n_jobs=1)
    # noise_csd_fig = noise_csd.plot(mode='coh', show=False)
    # noise_csd_fig.savefig(data_dir + '/results/' + participant + session_name + '_noise_coherence.png')
    # plt.close(noise_csd_fig)

    noise_epochs = convert_epochs_float32(noise_epochs)
    noise_epochs.save(data_dir + '/epochs/' + preproc_type + '/noise_' + participant + '_' + session_name + '-epo.fif', verbose=0)

    return face_epochs, noise_epochs


def plot_evoked(face_epochs, noise_epochs, evoked_ax, session_idx):
    face_evoked = face_epochs[0:128].average()
    noise_evoked = noise_epochs[0:128].average()

    face_evoked.plot(spatial_colors=True, time_unit='s', gfp=True, axes=evoked_ax[session_idx][0], window_title=None, selectable=False, show=False)
    noise_evoked.plot(spatial_colors=True, time_unit='s', gfp=True, axes=evoked_ax[session_idx][1], window_title=None, selectable=False, show=False)

    evoked_difference = face_evoked.data - noise_evoked.data
    evoked_diff = face_evoked.copy()
    evoked_diff.data = evoked_difference

    evoked_diff.plot(spatial_colors=True, time_unit='s', gfp=True, axes=evoked_ax[session_idx][2], window_title=None, selectable=False, show=False)


def plot_amplifier(filter_coeffs, amplitudes, sfreq, ideal_gains, fooof):
    from scipy.signal import freqz
    import matplotlib.pyplot as plt

    sfreq = float(sfreq)
    n_points = len(ideal_gains[0])

    # f = np.linspace(0, sfreq/2, n_points)
    # logfreqs = np.logspace(-2, 0, n_points, endpoint=True, base=10)

    title = 'Neural Power Amplifier'

    fig, (ax_filter) = plt.subplots(1, 3, figsize=(24, 6))
    total_mag = np.zeros(n_points)

    for idx, (coeffs, amplitude) in enumerate(zip(filter_coeffs, amplitudes)):
        w, h = freqz(coeffs[0], coeffs[1], worN=n_points)

        freqs = w * sfreq / (2*np.pi)

        mag = np.maximum(np.abs(h), 1e-20)

        if idx > 0:
            npa = 11
        else:
            npa = 1

        total_mag += (mag * amplitude * npa)

        ax_filter[1].plot(freqs, mag*amplitude, 'b', linewidth=2, zorder=4)

    total_mag = 10 * np.log10(total_mag) # to dB

    ideal_mag = np.zeros(n_points)

    for idx, (gain, amplitude) in enumerate(zip(ideal_gains, amplitudes)):
        if idx > 0:
            npa = 11
        else:
            npa = 1

        ideal_mag += (np.abs(gain) * amplitude * npa)
        ax_filter[1].plot(freqs, np.maximum(gain*amplitude, 1e-20), color='r', linestyle='dashed', linewidth=2, zorder=3)

    ideal_mag = 10*np.log10(ideal_mag)

    ax_filter[2].plot(freqs, total_mag, color='b', linewidth=2, zorder=4, label='Actual Filter Response')
    ax_filter[2].plot(freqs, ideal_mag, color='r', linestyle='dashed', linewidth=2, zorder=4, label='Ideal Filter Response')

    ax_filter[2].legend(loc="lower right", fontsize=16, shadow=True, fancybox=True)

    ax_filter[1].set(xlim=[0, 40], ylabel='Gain (V/V)', xlabel='Frequency (Hz)', title='Individual Filters')
    ax_filter[2].set(xlim=[0, 40], ylim=[-20, 10], ylabel='Total Magnitude (dB)', xlabel='Frequency (Hz)', title=title)

    fooof.plot(plt_log=False, save_fig=False, ax=ax_filter[0])

    ax_filter[0].set_xlabel('Frequency (Hz)', fontsize=16)
    ax_filter[1].set_xlabel('Frequency (Hz)', fontsize=16)
    ax_filter[2].set_xlabel('Frequency (Hz)', fontsize=16)

    ax_filter[0].set_title('FOOOF Spectrum', fontsize=20)
    ax_filter[1].set_title('Individual Filter Response', fontsize=20)
    ax_filter[2].set_title('Neural Power Amplifier Response', fontsize=20)

    # plt.legend(shadow=True, fancybox=True)
    plt.tight_layout()

    return fig


def preprocess(args):
    f_high = 128
    f_low = 1

    save_epochs = args.save_epochs

    # number_bad_channels = []

    montage = load_montage()
    montage_fig = montage.plot(scale_factor=20, show_names=True, kind='topomap', show=False)
    montage_fig.savefig(data_dir + '/results/montage_2D.png', dpi=500)
    plt.close(montage_fig)
    montage_fig2 = montage.plot(scale_factor=20, show_names=True, kind='3d', show=False)
    montage_fig2.savefig(data_dir + '/results/montagefig_3D.png', dpi=500)
    plt.close(montage_fig2)

    fooof_fig, fooof_axes = plt.subplots(nrows=10, ncols=4, sharex=True, sharey=False, squeeze=False, figsize=(24, 40))
    # power_fig, power_axes = plt.subplots(nrows=10, ncols=4, sharex=True, sharey=False, squeeze=False, figsize=(24, 40))
    # topo_fig, topo_axes = plt.subplots(nrows=10, ncols=4, sharex=True, sharey=True, squeeze=False, figsize=(24, 40))

    r2s = []

    for participant_idx, participant in enumerate(participants):
        plt.close()

        evoked_fig_filtered, evoked_ax_filtered = plt.subplots(nrows=10, ncols=3, sharex=True, sharey=False, squeeze=False, figsize=(10, 40))
        evoked_fig_amplified, evoked_ax_amplified = plt.subplots(nrows=10, ncols=3, sharex=True, sharey=False, squeeze=False, figsize=(10, 40))

        for session_idx, session_name in enumerate(sessions):
            print('Participant', participant_idx + 1, '/', len(participants), 'session', session_idx + 1, '/', len(sessions))

            filename = data_dir + participant + '/' + participant + session_name + '.bdf'

            eeg = read_raw_edf(filename, montage=montage, eog=eog_channels, preload=True, verbose=0)
            # mne.set_eeg_reference(eeg, ref_channels='average', copy=True, projection=False)

            events = mne.find_events(eeg, stim_channel='STI 014', verbose=0)
            reject = dict(eeg=80e-5, eog=60e-4)   # manually tuned argh

            eeg.pick_channels(channel_names)

            # combine EOG
            # eeg = combine_vertical_eog(eeg)
            # print('Notch Filter')
            # eeg_notch = eeg.copy()
            # eeg_notch.notch_filter(np.arange(50, 128, 50), notch_widths=256/400, n_jobs=7)

            print('Bandpass Filter')
            filter_eeg = eeg.copy()
            filter_eeg.filter(1, 40, picks=list(range(132)), n_jobs=7, verbose=0)          # band-pass

            if save_epochs:
                print('Saving Epochs')
                face_epochs, noise_epochs = save_epochs_as(filter_eeg, 'filtered', events, reject, participant, session_name)
                plot_evoked(face_epochs, noise_epochs, evoked_ax_filtered, session_idx)

                if args.connectivity:
                    plot_connectivity(face_epochs, participant, session_name, 'face', 'filtered')
                    plot_connectivity(noise_epochs, participant, session_name, 'noise', 'filtered')

                del(face_epochs)
                del(noise_epochs)

            print('ICA')
            eeg = ica_preprocessing(eeg, filter_eeg, participant, session_name, eog_channels[0], reject, f_low, f_high)
            # eeg = ica_preprocessing(eeg, participant, session_name, 'EOG', reject=reject)
            # eeg = ica_preprocessing(eeg, participant, session_name, 'EXG2', reject=reject)
            # eeg = ica_preprocessing(eeg, participant, session_name, 'EXG3')

            # eeg = ssp_preprocessing(eeg, participant, session_name, reject)
            del(filter_eeg)

            eeg.set_eeg_reference('average', projection=True, verbose=0)
            eeg.apply_proj()

            # face_epochs, noise_epochs = save_epochs_as(eeg, 'ica', events, reject, participant, session_name)
            # plot_evoked(face_epochs, noise_epochs, evoked_axes['ica'], session_idx)

            print('Plot Power')
            # eeg.plot_psd(tmin=100, fmin=0.1, fmax=256, picks=list(range(128)), ax=power_axes[session_idx][participant_idx], area_mode='std', area_alpha=0.33, dB=True, estimate='auto', average=False, show=False, n_jobs=7, spatial_colors=True, verbose=0)

            # eeg.pick_channels(channel_names[0:128])
            # eeg.plot_psd_topo(tmin=100, dB=True, show=False, block=False, n_jobs=1, axes=topo_axes[session_idx][participant_idx], verbose=0)
            eeg.pick_channels(channel_names[0:132])

            print('Computing power spectrum for entire session...')
            start_time = time.time()
            psds, freqs = psd_multitaper(eeg, f_low, f_high, n_jobs=7, verbose=0)
            # print('Power Spectrum')
            #psds, freqs = psd_welch(eeg, fmin=f_low, fmax=128, tmin=500, tmax=2000, n_fft=2048, n_overlap=512, n_jobs=7)
            # psds, freqs = psd_multitaper(eeg, fmin=f_low, fmax=f_high, tmin=500, tmax=2000, n_jobs=1, verbose=1)

            print('Took', (time.time() - start_time) // 60, 'mins')

            print('Frequencies shape:', freqs.shape, 'Power spectrum distribution shape:', psds.shape)
            #
            # channel_rejected_eeg = eeg.copy()
            # channel_rejected_eeg, n_bads = fooof_channel_rejection(channel_rejected_eeg, psds, freqs, f_low, f_high, participant, session_name)
            # print('Bad channels:', n_bads)
            #
            # number_bad_channels.append(n_bads)
            #
            # face_epochs, noise_epochs = save_epochs_as(channel_rejected_eeg, 'bads2good', events, reject, participant, session_name)
            # plot_evoked(face_epochs, noise_epochs, evoked_axes['bads2good'], session_idx)
            #
            # psds, freqs = psd_welch(channel_rejected_eeg, fmin=0, fmax=n_freqs, tmin=500, tmax=2000, n_fft=2048, n_overlap=512, n_jobs=7)

            print('Fitting FOOOF...')
            start_time = time.time()
            fooof = FOOOF(min_peak_amplitude=0.05, peak_width_limits=[3, 15], background_mode='knee')
            fooof.fit(freqs, np.mean(psds, axis=0), freq_range=[f_low / 2, 45])
            print('FOOOF fit in', (time.time() - start_time) // 60, 'mins')
            fooof.plot(plt_log=False, save_fig=True, file_name='FOOOF_' + participant + '_' + session_name, file_path=data_dir + '/results/')
            fooof.plot(plt_log=False, ax=fooof_axes[session_idx][participant_idx])

            r2s.append(fooof.r_squared_)
            # amplify time series
            print('Amplifying Time')

            eeg_time_series = eeg.get_data(picks=list(range(132)))

            amplified_time_series = np.copy(eeg_time_series)
            print('Learning Filters')
            filter_coeffs, amplitudes, amp_figs, ideal_gains = fooof.learn_filters(512)

            for plot_idx, amp_fig in enumerate(amp_figs):
                amp_fig.savefig(data_dir + '/results/' + participant + session_name + '_amplifier_' + str(plot_idx) + '.png')
                plt.close(amp_fig)

            fig = plot_amplifier(filter_coeffs, amplitudes, 512, ideal_gains, fooof)
            fig.savefig(data_dir + '/results/' + participant + session_name + '_amplifier.png', dpi=500)
            plt.close(fig)

            print('Applying Filters')
            for coeffs, amplitude in zip(filter_coeffs, amplitudes):
                if isinstance(coeffs[0], np.float64):
                    fun = partial(lfilter, b=coeffs, a=[1.0], axis=-1)
                else:
                    fun = partial(filtfilt, b=coeffs[0], a=coeffs[1], axis=-1)

                parallel, p_fun, _ = parallel_func(fun, 7)
                filtered_eeg = parallel(p_fun(x=eeg_time_series[p]) for p in range(132))

                # filtered_eeg = lfilter(coeffs, [1.0], eeg_time_series, axis=-1)*amplitude*10
                for p in range(132):
                    amplified_time_series[p] += (filtered_eeg[p] * amplitude * 10)

            epoch_info = mne.pick_info(eeg.info, sel=(list(range(132))), copy=True)

            eeg = mne.io.RawArray(np.float64(amplified_time_series), epoch_info, verbose=0)

            if save_epochs:
                face_epochs, noise_epochs = save_epochs_as(eeg, 'amplified', events, None, participant, session_name)
                plot_evoked(face_epochs, noise_epochs, evoked_ax_amplified, session_idx)

                if args.connectivity:
                    plot_connectivity(face_epochs, participant, session_name, 'face', 'amplified')
                    plot_connectivity(noise_epochs, participant, session_name, 'noise', 'amplified')


                del(face_epochs)
                del(noise_epochs)

        evoked_fig_filtered.savefig(data_dir + '/results/evoked_filtered' + participant + '.png', dpi=500, bbox_inches='tight')
        evoked_fig_amplified.savefig(data_dir + '/results/evoked_amplified' + participant + '.png', dpi=500, bbox_inches='tight')

        # for preproc_type in preproc_types:
        #     evoked_fig = evoked_fig[preproc_type]
        #     evoked_fig.savefig(data_dir + '/results/evoked_' + preproc_type + participant + '.png', dpi=500, bbox_inches='tight')

    fooof_fig.savefig(data_dir + '/results/fooofs.png', dpi=500, bbox_inches='tight')
    # power_fig.savefig(data_dir + '/results/power.png', dpi=500, bbox_inches='tight')

    print('FOOOF r squared', r2s)
    print(np.mean(r2s))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing to test the Neural Power Amplifier.')

    parser.add_argument('--save-epochs', action='store_true', default=False, help='save epochs')
    parser.add_argument('--connectivity', action='store_true', default=False, help='compute connectivity using phase lag index')

    args = parser.parse_args()

    print('Arguments for this experiment:')
    print(args)

    start_all = time.time()
    preprocess(args)
    elapsed = time.time() - start_all
    print('Took', elapsed / 60, 'minutes')