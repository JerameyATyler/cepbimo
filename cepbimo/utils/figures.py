""" Module for producing consistent figures """

# Use seaborn to manage theme
import seaborn as sns

color_palette = 'viridis'

sns.set_theme(palette=color_palette)

figure_dimensions = dict(square=(6, 6), wrect=(8, 4), hrect=(4, 8), refs=(6, 8))

label_font = dict(size=14, name='Courier', family='serif', style='italic', weight='bold')
suptitle_font = dict(size=20, name='Courier', family='serif', style='normal', weight='bold')
title_font = dict(size=16, name='Courier', family='sans', style='italic', weight='normal')


def basic_figure(shape, **kwargs):
    """Generates a basic figure with the specified shape."""
    import matplotlib.pyplot as plt

    fig = plt.figure(layout="constrained")
    fig.set_size_inches(figure_dimensions[shape])

    ax = fig.add_subplot()

    if "suptitle" in kwargs.keys():
        plt.suptitle(kwargs["suptitle"], **suptitle_font)
    if "title" in kwargs.keys():
        ax.set_title(kwargs["title"], **title_font)

    return plt, fig, ax


def polar_figure(shape, **kwargs):
    """Generates a polar figure with the specified shape."""
    import matplotlib.pyplot as plt

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions[shape])

    ax = fig.add_subplot(projection='polar')

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)

    # Set theta direction to match HRTF direction
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(90)

    ax.tick_params(labelsize=20)

    # Arrow
    ax.annotate('',
                xy=(0.5, 1),
                xytext=(0.5, 0.5),
                xycoords="axes fraction",
                arrowprops=dict(facecolor='red',
                                width=6,
                                headwidth=12,
                                alpha=0.75))
    # Arrow text
    ax.annotate('Face Forward',
                xy=(0.5, 0.75),
                xytext=(0.5, 0.75),
                xycoords="axes fraction",
                ha='center',
                va='center',
                fontsize=20,
                rotation=90)
    return plt, fig, ax


def set_parameters(ax, **kwargs):
    if 'xlim' in kwargs.keys():
        ax.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs.keys():
        ax.set_ylim(kwargs['ylim'])
    if 'zlim' in kwargs.keys():
        ax.set_zlim(kwargs['zlim'])
    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['xlabel'], **label_font)
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'], **label_font)
    if 'zlabel' in kwargs.keys():
        ax.set_zlabel(kwargs['zlabel'], **label_font)


def plot_wave(x, filepath=None, **kwargs):
    """Plot waveform(s)"""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import os

    num_frames, num_channels = x.size, x.ndim

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions['wrect'])

    ax = fig.subplots(num_channels, 1)

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)

    if num_channels == 1:
        ax = [ax]

    for c in range(num_channels):
        ax[c].grid(True)

        if 't' in kwargs.keys():
            t = kwargs['t'][0]
        elif 'fs' in kwargs.keys():
            fs = kwargs['fs'][0]
            t = np.arange(0, num_frames) / fs
        else:
            t = np.linspace(0, 1, num_frames)
        if 'xlim' in kwargs.keys():
            ax[c].set_xlim(kwargs['xlim'][c])
        if 'ylim' in kwargs.keys():
            ax[c].set_ylim(kwargs['ylim'][c])
        if 'xlabel' in kwargs.keys():
            ax[c].set_xlabel(kwargs['xlabel'][c], **label_font)
        if 'ylabel' in kwargs.keys():
            ax[c].set_ylabel(kwargs['ylabel'][c], **label_font)
        if 'title' in kwargs.keys():
            ax[c].set_title(kwargs['title'][c], **title_font)
        ax[c].plot(t, x[:, c])

    if filepath is not None:
        path = Path(filepath)
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])
        plt.savefig(filepath)
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

        return filepath
    return plt


def demo_plot_wave():
    """Demonstrate plot_wave usage"""
    import numpy as np

    t = np.linspace(0, 2 * np.pi, 500)
    x1 = np.sin(t ** 2)
    x2 = np.cos(t ** 2)
    x = np.array([x1, x2]).transpose()

    args = dict(
        t=[t, t],
        suptitle='plot_wave demo',
        title=['Raiders of the lost wave plot', 'The wave plot of doom'],
        xlim=[(0, 2 * np.pi), (0, 2 * np.pi)],
        ylim=[(-1, 1), (-1, 1)],
        xlabel=[r'$2*\pi$', r'$2*\pi$'],
        ylabel=[r'$\sin(x^2)$', r'$\cos(x^2)$']
    )

    plt = plot_wave(x, **args)
    plt.show()
    plt.close()


def plot_waves(x, filepath=None, **kwargs):
    """Plot waveform(s)"""
    import numpy as np
    from pathlib import Path
    import os

    num_frames, num_channels = x.size, x.ndim

    plt, fig, ax = basic_figure('wrect', **kwargs)

    if 't' in kwargs.keys():
        t = kwargs['t']
    elif 'fs' in kwargs.keys():
        fs = kwargs['fs'][0]
        t = np.arange(0, num_frames) / fs
    else:
        t = np.linspace(0, 1, num_frames)

    set_parameters(ax, **kwargs)

    handles = []
    for i in range(num_channels):
        if 'legend_labels' in kwargs.keys() and len(kwargs['legend_labels']) <= num_channels:
            label = kwargs['legend_labels'][i]
        else:
            label = ''
        handle, = ax.plot(t, x[:, i], label=label)
        handles.append(handle)

    if 'legend' in kwargs.keys() \
            and kwargs['legend']:
        ax.legend(handles=handles)

    if filepath is not None:
        path = Path(filepath)
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])
        plt.savefig(filepath)
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

        return filepath
    return plt


def demo_plot_waves():
    """Demonstrate plot_waves usage"""
    import numpy as np

    t = np.linspace(0, 2 * np.pi, 500)
    x1 = np.sin(t ** 2)
    x2 = np.cos(t ** 2)
    x = np.array([x1, x2]).transpose()

    args = dict(
        t=t,
        suptitle='plot_wave demo',
        title='',
        xlim=(0, 2 * np.pi),
        ylim=(-1, 1),
        xlabel=r'$2*\pi$',
        ylabel=r'$\sin(x^2)$ and $\cos(x^2)$',
        legend=True,
        legend_labels=[r"$\sin(x^2)$", r"$\cos(x^2)$"]
    )

    plt = plot_waves(x, **args)
    plt.show()
    plt.close()


def plot_spectrogram(x, filepath=None, **kwargs):
    """Plot spectrogram"""
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions['square'])

    ax = fig.add_subplot()

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)

    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)

    if 'fs' in kwargs.keys():
        ax.specgram(x, cmap=color_palette, Fs=kwargs['fs'])
    else:
        ax.specgram(x, cmap=color_palette)

    if filepath is not None:
        path = Path(filepath)
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])
        plt.savefig(filepath)
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

        return filepath

    return plt


def demo_plot_spectrogram():
    """Demonstrate plot_spectrogram usage"""
    from scipy.signal import chirp
    import numpy as np

    fs = 48000
    T = 4
    t = np.arange(0, int(T * fs)) / fs
    f0 = 1
    f1 = 20000
    w = chirp(t, f0=f0, f1=f1, t1=T, method='logarithmic')

    args = dict(fs=fs, title='The last wave plot', suptitle='plot_spectrogram demo')

    plt = plot_spectrogram(w, **args)

    plt.show()
    plt.close()


def plot_spectrum(x, fs, spectrum='amplitude', filepath=None, **kwargs):
    """Plot the specified spectrum"""
    import numpy as np
    from numpy.fft import fftfreq
    from transforms.spectrum import amplitude_spectrum, power_spectrum, phase_spectrum, log_spectrum
    from pathlib import Path
    import os

    if spectrum == 'amplitude':
        s = amplitude_spectrum(x)
    elif spectrum == 'power':
        s = power_spectrum(x)
    elif spectrum == 'phase':
        s = phase_spectrum(x)
    elif spectrum == 'log':
        s = log_spectrum(x)
    else:
        s = x

    step = 1 / fs
    freqs = fftfreq(x.size, step)
    idx = np.argsort(freqs)

    n = int(np.ceil(x.size / 2))

    s = s[idx][n:]
    freqs = freqs[idx][n:]

    plt, fig, ax = basic_figure('wrect', **kwargs)

    ax.plot(freqs, s)

    if filepath is not None:
        path = Path(filepath)
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])
        plt.savefig(filepath)
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

        return filepath

    return plt


def demo_plot_spectrum():
    """Demonstrate plot_spectrum usage."""
    import numpy as np

    x = np.random.rand(301) - 0.5
    fs = 30

    plt = plot_spectrum(x, fs, **dict(suptitle='Amplitude spectrum', title='The spectral plot'))
    plt.show()
    plt.close()

    plt = plot_spectrum(x, fs, spectrum='power', **dict(suptitle='Power spectrum', title='The spectral plot reloaded'))
    plt.show()
    plt.close()

    plt = plot_spectrum(x, fs, spectrum='phase',
                        **dict(suptitle='Phase spectrum', title='The spectral plot revolutions'))
    plt.show()
    plt.close()

    plt = plot_spectrum(x, fs, spectrum='log',
                        **dict(suptitle='Logarithm of spectrum', title='The spectral plot resurrections'))
    plt.show()
    plt.close()


def plot_cepstrum(x, fs, offset, window_length, filepath=None, **kwargs):
    """Plot the cepstrum."""
    from transforms.spectrum import cepstrum
    from pathlib import Path
    import os

    C, q = cepstrum(x, fs, offset, window_length)

    plt, fig, ax = basic_figure('wrect', **kwargs)

    set_parameters(ax, **kwargs)

    ax.plot(q * 1000, C)

    if filepath is not None:
        path = Path(filepath)
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])
        plt.savefig(filepath)
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

        return filepath

    return plt


def demo_plot_cepstrum():
    """Demonstrate plot_cepstrum usage."""
    import numpy as np

    x = np.random.rand(301) - 0.5
    fs = 30

    offset = 1
    window_length = fs * 10

    args = dict(
        suptitle='Amplitude cepstrum',
        title='The cepstrum Lebowski',
        xlabel='Quefrency, ms',
        ylabel='Amplitude'
    )
    plt = plot_cepstrum(x, fs, offset, window_length, **args)
    plt.show()
    plt.close()


def plot_hrtfs(zeniths, azimuths, filepath=None, **kwargs):
    """Plot head-related transfer functions (HRTF)"""
    import numpy as np
    from pathlib import Path
    import os

    plt, fig, ax = polar_figure('square', **kwargs)

    ax.set_rmax(90)
    ax.set_rticks(np.linspace(-30, 75, 8))
    ax.set_rlim(bottom=90, top=-40)

    ax.scatter(np.radians(azimuths), zeniths)

    if filepath is not None:
        path = Path(filepath)
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])
        plt.savefig(filepath)
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

        return filepath

    return plt


def demo_plot_hrtfs():
    """Demonstrate plot_hrtfs usage."""
    import numpy as np
    from data_loader import list_hrtf_data

    hrtfs = list_hrtf_data()

    azs = []
    zes = []

    for z in hrtfs.keys():
        for a in hrtfs[z]:
            zes.append(z)
            azs.append(a)

    azs = np.array(azs)
    zes = np.array(zes)

    args = dict(suptitle='HRTF', title='Head-related transfer function and the order of the phoenix')
    plt = plot_hrtfs(zes, azs, **args)
    plt.show()
    plt.close()


def plot_reflections(zeniths, azimuths, delays, amplitudes, delay_max=None, filepath=None, **kwargs):
    """Plot reflections."""
    import numpy as np
    from data_loader import list_hrtf_data
    from pathlib import Path
    import os

    zenith_min = min(list_hrtf_data().keys())
    zenith_max = max(list_hrtf_data().keys())
    if delay_max is None:
        delay_max = max(delays)

    plt, fig, ax = polar_figure('refs', **kwargs)

    theta = np.radians(azimuths)
    r = np.array(delays)
    area = ((np.array(zeniths) - zenith_min) / (zenith_max - zenith_min)) * (1500 - 200) + 200
    color = np.array(amplitudes)

    norm = plt.Normalize(0, 1)

    ax.set_rmax(delay_max)

    ax.text(np.radians(135),
            delay_max + 15,
            "Azimuth",
            ha='center',
            va='center',
            fontsize=20,
            rotation=45)
    ax.text(np.radians(95),
            delay_max / 2.,
            'Time/Delay, ms',
            ha='center',
            va='center',
            fontsize=20)

    h1 = ax.scatter([], [], s=150, c='k', alpha=0.5)
    h2 = ax.scatter([], [], s=600, c='k', alpha=0.5)
    h3 = ax.scatter([], [], s=1050, c='k', alpha=0.5)
    h4 = ax.scatter([], [], s=1500, c='k', alpha=0.5)

    handles = (h1, h2, h3, h4)
    labels = ('-45', '0', '45', '90')

    ax.legend(handles,
              labels,
              scatterpoints=1,
              loc='upper left',
              title='Zenith',
              title_fontsize=20,
              frameon=True,
              fancybox=True,
              fontsize=16,
              bbox_to_anchor=(0., -0.2, 1., .15),
              ncol=4,
              mode='expand',
              borderaxespad=.1,
              borderpad=1)

    ax.scatter(theta, r, c=color, s=area, cmap=color_palette, norm=norm, alpha=0.9)

    sns.set_style("darkgrid", {'axes.grid': False})

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_palette), ax=ax, orientation='horizontal')
    cb.set_label('Amplitude', size=20)
    cb.ax.tick_params(labelsize=16)

    if filepath is not None:
        path = Path(filepath)
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])
        plt.savefig(filepath)
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

        return filepath

    return plt, fig


def plot_sample(zenith,
                azimuth,
                zeniths,
                azimuths,
                delays,
                amplitudes,
                amplitude,
                delay,
                time,
                delay_max=None,
                time_max=None,
                filepath=None, **kwargs):
    """Plot sample."""
    import numpy as np
    from data_loader import list_hrtf_data
    import os
    from pathlib import Path

    if delay_max is None:
        delay_max = max(delays)
    if time_max is None:
        time_max = time

    zenith_min = min(list_hrtf_data().keys())
    zenith_max = max(list_hrtf_data().keys())

    area = ((zenith - zenith_min) / (zenith_max - zenith_min)) * (0.5 - 0.1) + 0.1
    err = (time / time_max) * (0.5 - 0.1) + 0.1

    plt, fig = plot_reflections(zeniths, azimuths, delays, amplitudes, **kwargs)

    cmap = plt.get_cmap(color_palette)

    ax = plt.gca()
    ax.bar(np.deg2rad(azimuth), delay_max - delay, xerr=err, bottom=delay, width=area, alpha=0.5, color=cmap(amplitude),
           ecolor=cmap(amplitude))

    if filepath is not None:
        path = Path(filepath)
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])
        plt.savefig(filepath)
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

        return filepath

    return plt


def interpolator(x, y, z):
    """Interpolate additional x and y values for z."""
    import numpy as np
    from scipy.interpolate import interp2d

    f = interp2d(x, y, z, kind='cubic')

    x1 = np.linspace(min(x), max(x), int(x.size * 1.5))
    y1 = np.linspace(min(y), max(y), int(y.size * 1.5))
    z1 = f(x1, y1)

    x2, y2 = np.meshgrid(x1, y1)

    return x2, y2, z1


def plot_binaural_activity_map_2d(z, filepath=None, **kwargs):
    """Plot a 2-dimensional binaural activity map"""
    import numpy as np
    from pathlib import Path
    import os

    z = z.transpose()
    x = np.linspace(-1, 1, z.shape[1])
    y = np.linspace(0, 50, z.shape[0])

    x, y, z = interpolator(x, y, z)

    plt, fig, ax = basic_figure('square', **kwargs)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 50)

    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['xlabel'], **label_font)
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'], **label_font)

    surf = ax.contourf(x, y, z, cmap=color_palette)

    sns.set_style("darkgrid", {"axes.grid": False})
    cb = plt.colorbar(surf, ax=ax)
    cb.set_label('Correlation', size=20)
    cb.ax.tick_params(labelsize=16)

    if filepath is not None:
        path = Path(filepath)
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])
        plt.savefig(filepath)
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

        return filepath

    return plt


def plot_binaural_activity_map_3d(z, filepath=None, **kwargs):
    """Plot a 3-dimensional binaural activity map"""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import os

    z = z.transpose()
    x = np.linspace(-1, 1, z.shape[1])
    y = np.linspace(0, 50, z.shape[0])

    x, y, z = interpolator(x, y, z)

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions['square'])

    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 50)
    ax.view_init(elev=80)

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)

    set_parameters(ax, **kwargs)

    sns.set_style("darkgrid", {"axes.grid": False})
    surf = ax.plot_surface(x, y, z, cmap=color_palette, rstride=1, cstride=1, edgecolor='none', aa=False)

    cb = plt.colorbar(surf, ax=ax)
    cb.set_label('Correlation', size=20)
    cb.ax.tick_params(labelsize=16)

    if filepath is not None:
        path = Path(filepath)
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])
        plt.savefig(filepath)
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

        return filepath

    return plt


def plot_zenith_range(z_min, z_max, filepath=None, **kwargs):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import os
    """Generates a polar figure with the specified shape."""

    cmap = mpl.cm.get_cmap('viridis')(0.5)

    z_min = np.deg2rad(z_min)
    z_max = np.deg2rad(z_max)
    theta = np.linspace(z_min, z_max)

    fig = plt.figure(layout='constrained')
    fig.set_size_inches((5, 5))

    ax = fig.add_subplot(projection='polar')

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)

    ax.set_rticks([])
    ax.tick_params(labelsize=20)

    # Arrow
    ax.annotate('',
                xy=(1, 0.5),
                xytext=(0.5, 0.5),
                xycoords="axes fraction",
                arrowprops=dict(facecolor='red',
                                width=6,
                                headwidth=12,
                                alpha=0.75))
    # Arrow text
    ax.annotate('Face Forward',
                xy=(0.5, 0.5),
                xytext=(0.75, 0.5),
                xycoords="axes fraction",
                ha='center',
                va='center',
                fontsize=20)

    ax.grid(True)
    area = plt.fill_between(theta, 0, 1, alpha=0.75, label='area', color=cmap)

    if filepath is not None:
        path = Path(filepath)
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])
        plt.savefig(filepath)
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

        return filepath

    return plt, fig, ax, area


def demo_plot_zenith_range():
    z_min = -20
    z_max = 45

    degree_text = r'$^{\circ}$'
    args = dict(suptitle='Zenith range',
                title=f'{z_min}{degree_text}  -  {z_max}{degree_text}')

    plt, _, _, _ = plot_zenith_range(z_min, z_max, **args)
    plt.show()
    plt.close()


def plot_azimuth_range(a_min, a_max, filepath=None, **kwargs):
    import matplotlib as mpl
    import numpy as np
    from pathlib import Path
    import os
    """Generates a polar figure with the specified shape."""

    cmap = mpl.cm.get_cmap('viridis')(0.5)

    a_min = np.deg2rad(a_min)
    a_max = np.deg2rad(a_max)
    theta = np.linspace(a_min, a_max)

    plt, fig, ax = polar_figure('square', **kwargs)
    fig.set_size_inches((5, 5))
    ax.set_rticks([])
    area = plt.fill_between(theta, 0, 1, alpha=0.75, label='area', color=cmap)

    if filepath is not None:
        path = Path(filepath)
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])
        plt.savefig(filepath)
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

        return filepath

    return plt, fig, ax, area


def demo_plot_azimuth_range():
    a_min = -15
    a_max = 60

    degree_text = r'$^{\circ}$'
    args = dict(suptitle='Azimuth range',
                title=f'{a_min}{degree_text}  -  {a_max}{degree_text}')

    plt, _, _, _ = plot_azimuth_range(a_min, a_max, **args)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # demo_plot_wave()
    # demo_plot_waves()
    # demo_plot_spectrogram()
    # demo_plot_spectrum()
    # demo_plot_cepstrum()
    # demo_plot_hrtfs()
    # demo_plot_reflections()
    # demo_plot_sample()
    # demo_plot_binaural_activity_map_2d()
    # demo_plot_binaural_activity_map_3d()
    demo_plot_zenith_range()
    demo_plot_azimuth_range()
