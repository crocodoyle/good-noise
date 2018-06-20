import colorednoise as cn
import numpy as np

from sklearn import linear_model

from scipy.signal import periodogram

# optionally plot the Power Spectral Density with Matplotlib
from matplotlib import mlab
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

save_dir = 'E:/brains/temp/'


def generate_colours():
    n_samples = 2 ** 17  # number of samples to generate
    colours = ['brown', 'pink', 'white', 'blue', 'violet']
    mpl_colours = ['saddlebrown', 'magenta', 'white', 'blue', 'purple']
    betas = [2, 1, 0, -1, -2]

    fig1, (freq_ax) = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 4))
    fig2, (time_series_ax) = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(6, 3))
    fig3, (pink_ax) = plt.subplots(1, 1, figsize=(12, 2))
    fig4, (blue_ax) = plt.subplots(1, 1, figsize=(12, 2))

    fs = 10000

    pink_time = cn.powerlaw_psd_gaussian(1, n_samples)
    freq_pink, pxx_pink = periodogram(pink_time, fs=fs, detrend='constant', nfft=n_samples / 4)

    pink_ax.loglog(freq_pink, pxx_pink, color='magenta')
    pink_ax.xaxis.set_ticks([])
    pink_ax.yaxis.set_ticks([])

    pink_ax.set_xlim([1, 5000])
    pink_ax.set_ylim([10e-11, 10e8])

    fig3.savefig(save_dir + 'pink.png', dpi=500)

    blue_time = cn.powerlaw_psd_gaussian(-1, n_samples)
    freq_blue, pxx_blue = periodogram(blue_time, fs=fs, detrend='constant', nfft=n_samples / 4)

    blue_ax.loglog(freq_blue, pxx_blue, color='blue')
    blue_ax.xaxis.set_ticks([])
    blue_ax.yaxis.set_ticks([])

    blue_ax.set_xlim([1, 5000])
    blue_ax.set_ylim([10e-11, 10e8])

    fig4.savefig(save_dir + 'blue.png', dpi=500)



    for i, (colour, beta) in enumerate(zip(mpl_colours, betas)):
        y = cn.powerlaw_psd_gaussian(beta, n_samples)

        freq, pxx = periodogram(y, fs=fs, detrend='constant', nfft=n_samples/4)
        theoretical = 1 / (freq ** beta)

        # slope = np.log(pxx[-100] / pxx[100]) / np.log(freq[-100] / freq[100])
        # print(slope)

        log_diff =  np.log(theoretical[fs:]) - np.log(pxx[fs:])
        offset = np.mean(log_diff)

        pxx = (np.exp(offset))*pxx

        time_series_ax[i].plot(y[:800], color=colour)
        time_series_ax[i].xaxis.set_ticks([])
        time_series_ax[i].yaxis.set_ticks([])
        time_series_ax[i].set_ylabel(colours[i])

        freq_ax.loglog(freq, pxx, zorder=1, color=colour)
        freq_ax.loglog(freq, theoretical, lw=0.5, zorder=2, color='k')

    freq_ax.xaxis.set_ticks([])
    freq_ax.yaxis.set_ticks([])

    freq_ax.set_xlim([1, 5000])
    freq_ax.set_ylim([10e-11, 10e8])

    time_series_ax[4].set_xlabel('time', fontsize=20)
    freq_ax.set_xlabel('frequency', fontsize=20)
    freq_ax.set_ylabel('noise power', fontsize=20)


    # ax1.set_xticklabels([])
    # ax1.set_yticklabels([])
    #
    # ax1.set_ylim([0.00000000000000000001, 0.01])
    # ax1.set_xlim([100, 10e7])
    #
    # ax1.set_xlabel('frequency', fontsize=20)
    # ax1.set_ylabel('power spectral density', fontsize=20)

    # extraticks = [8, 13, 14, 30, 60, 0.5, 3, 4, 7]
    #
    #
    # ticks = ax2.axes.get_xaxis().get_ticks()
    # ax2.axes.get_xaxis().set_ticks(ticks + extraticks)

    fig1.tight_layout()
    fig1.savefig(save_dir + 'psd.png', dpi=500)

    fig2.tight_layout()
    fig2.subplots_adjust(wspace=0.01, hspace=0.01)
    fig2.savefig(save_dir + 'time_series.png', dpi=500)

    plt.close()


def signal_and_noise():
    Fs = 8000
    f = 5
    sample = 8000
    x = np.arange(sample)
    y_signal = np.sin(2 * np.pi * f * x / Fs)

    fig1, (time_ax) = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 4))

    time_ax.plot(x, y_signal, color='r')
    time_ax.xaxis.set_ticklabels([])
    time_ax.yaxis.set_ticklabels([])

    time_ax.set_ylim([-1.05, 1.05])

    time_ax.set_xlabel('time', fontsize=20)
    time_ax.set_ylabel('amplitude', fontsize=20)
    time_ax.set_title('$y(t) = x(t)$', fontsize=20)

    plt.tight_layout()
    plt.savefig(save_dir + 'only_signal.png', dpi=500)

    freq, pxx = periodogram(y_signal, fs=Fs, nfft=sample)

    fig1, (freq_ax) = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 4))

    freq_ax.loglog(freq, pxx, color='r')
    freq_ax.set_xlabel('frequency', fontsize=20)
    freq_ax.set_ylabel('power', fontsize=20)
    time_ax.set_title('$x(t)$', fontsize=20)

    plt.tight_layout()
    plt.savefig(save_dir + 'signal_psd.png', dpi=500)

    fig1, (time_ax) = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 4))

    Fs = 8000
    f = 5
    sample = 8000
    x = np.arange(0, sample)

    y = np.sin(2 * np.pi * f * x / Fs)
    y_discretized = np.copy(y)

    decimation = 100

    steps = sample // decimation
    for i in range(steps):
        y_discretized[i*decimation:(i+1)*decimation] = y[i*decimation]

    time_ax.plot(x, y, color='r', alpha=1)
    time_ax.plot(x, y_discretized, color='k')

    time_ax.xaxis.set_ticklabels([])
    time_ax.yaxis.set_ticklabels([])

    time_ax.set_ylim([-1.05, 1.05])

    time_ax.set_xlabel('time', fontsize=20)
    time_ax.set_ylabel('amplitude', fontsize=20)
    time_ax.set_title('$y(t) = x(t) + \epsilon$ (discretization)', fontsize=20)

    plt.tight_layout()
    plt.savefig(save_dir + 'discretized.png', dpi=500)



    fig1, (time_ax) = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 4))

    Fs = 8000
    f = 5
    sample = 8000
    x_orig = np.arange(0, sample)
    x = np.arange(0, sample, 500)

    y_orig = np.sin(2 * np.pi * f * x / Fs)
    y = np.sin(2 * np.pi * f * x / Fs)

    time_ax.plot(x, y, color='k', alpha=1)
    # time_ax.plot(x_orig, y_orig, color='r')

    time_ax.xaxis.set_ticklabels([])
    time_ax.yaxis.set_ticklabels([])

    time_ax.set_ylim([-1.05, 1.05])

    time_ax.set_xlabel('time', fontsize=20)
    time_ax.set_ylabel('amplitude', fontsize=20)
    time_ax.set_title('$y(t) = x(t) + \epsilon$ (subsampled)', fontsize=20)

    plt.tight_layout()
    plt.savefig(save_dir + 'time_discretized.png', dpi=500)



    fig1, (time_ax) = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 4))

    Fs = 8000
    f = 5
    sample = 8000
    x = np.arange(sample)
    y = 0.9*np.sin(2 * np.pi * f * x / Fs)

    f_noise = 60
    noise = 1/10*np.sin(2 * np.pi * f_noise * x / Fs)

    time_ax.plot(x, y+noise, color='k')
    time_ax.plot(x, y, color='r', alpha=1)
    time_ax.xaxis.set_ticklabels([])
    time_ax.yaxis.set_ticklabels([])

    time_ax.set_ylim([-1.05, 1.05])

    time_ax.set_xlabel('time', fontsize=20)
    time_ax.set_ylabel('amplitude', fontsize=20)
    time_ax.set_title('$y(t) = x(t) + \epsilon$ (single freq.)', fontsize=20)

    plt.tight_layout()
    plt.savefig(save_dir + 'noisy.png', dpi=500)


    fig1, (time_ax) = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 4))

    Fs = 8000
    f = 5
    sample = 8000
    x = np.arange(sample)
    y = 0.9*np.sin(2 * np.pi * f * x / Fs)

    noise = (1/10)*cn.powerlaw_psd_gaussian(0, sample)

    time_ax.plot(x, y+noise, color='white')
    time_ax.plot(x, y, color='r', alpha=0.5)
    time_ax.xaxis.set_ticklabels([])
    time_ax.yaxis.set_ticklabels([])

    time_ax.set_ylim([-1.05, 1.05])

    time_ax.set_xlabel('time', fontsize=20)
    time_ax.set_ylabel('amplitude', fontsize=20)
    time_ax.set_title('$y(t) = x(t) + \epsilon$ (white)', fontsize=20)


    plt.tight_layout()
    plt.savefig(save_dir + 'white_noisy.png', dpi=500)


    fig1, (time_ax) = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 4))

    Fs = 8000
    f = 5
    sample = 8000
    x = np.arange(sample)
    y = 0.9*np.sin(2 * np.pi * f * x / Fs)

    noise = (1/10)*cn.powerlaw_psd_gaussian(1, sample)

    time_ax.plot(x, y+noise, color='magenta')
    time_ax.plot(x, y, color='r', alpha=0.5)
    time_ax.xaxis.set_ticklabels([])
    time_ax.yaxis.set_ticklabels([])

    time_ax.set_ylim([-1.05, 1.05])

    time_ax.set_xlabel('time', fontsize=20)
    time_ax.set_ylabel('amplitude', fontsize=20)
    time_ax.set_title('$y(t) = x(t) + \epsilon$ (pink)', fontsize=20)


    plt.tight_layout()
    plt.savefig(save_dir + 'pink_noisy.png', dpi=500)

    fig1, (time_ax) = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 4))

    Fs = 8000
    f = 5
    sample = 8000
    x = np.arange(sample)
    y = 0.9*np.sin(2 * np.pi * f * x / Fs)

    noise = (1/10)*cn.powerlaw_psd_gaussian(-1, sample)

    time_ax.plot(x, y+noise, color='b')
    time_ax.plot(x, y, color='r', alpha=0.5)
    time_ax.xaxis.set_ticklabels([])
    time_ax.yaxis.set_ticklabels([])

    time_ax.set_ylim([-1.05, 1.05])

    time_ax.set_xlabel('time', fontsize=20)
    time_ax.set_ylabel('amplitude', fontsize=20)
    time_ax.set_title('$y(t) = x(t) + \epsilon$ (blue)', fontsize=20)


    plt.tight_layout()
    plt.savefig(save_dir + 'blue_noisy.png', dpi=500)


    fig1, (time_ax) = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 4))

    Fs = 8000
    f = 5
    sample = 8000
    x = np.arange(0, sample)

    y = np.sin(2 * np.pi * f * x / Fs)
    y_discretized = np.copy(y)

    steps = sample // decimation
    for i in range(steps):
        y_discretized[i*decimation:(i+1)*decimation] = y[i*decimation]

    noise = (1 / 10) * cn.powerlaw_psd_gaussian(-1, sample)

    time_ax.plot(x, y_discretized + noise, color='b')
    time_ax.plot(x, y, color='r', alpha=0.75)
    time_ax.plot(x, y_discretized, color='k', alpha=0.75)


    time_ax.xaxis.set_ticklabels([])
    time_ax.yaxis.set_ticklabels([])

    time_ax.set_ylim([-1.05, 1.05])

    time_ax.set_xlabel('time', fontsize=20)
    time_ax.set_ylabel('amplitude', fontsize=20)
    time_ax.set_title('$y(t) = x(t) + \epsilon$ (discretization + blue)', fontsize=20)


    plt.tight_layout()
    plt.savefig(save_dir + 'reconstructed.png', dpi=500)


    plt.close()


def dither_image(input_img, output_filename):
    from dither import Dither
    from PIL import Image

    img = Image.open(input_img)

    quantized = img.quantize(colors=16)
    quantized.save(save_dir + 'img_quantized.bmp')

    img = Dither(input_img).floyd_steinberg_dither(input_img)
    img.save(save_dir + output_filename)


if __name__ == '__main__':
    # dither_image(save_dir + 'conference_bnr.jpg', 'dithered.bmp')
    signal_and_noise()
    # generate_colours()