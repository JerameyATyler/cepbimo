""" Module for producing consistent figures """
def is_notebook():
    """ Returns True if we are running inside a Jupyter shell"""
    try:
        shell = get_ipython().__class__.__name__
        # Jupyter shell
        if shell == 'ZMQInteractiveShell':
            return True
        # IPython shell
        elif shell == 'TerminalInteractiveShell':
            return False

        shell = get_ipython().__class__.__module__
        # Google Colab shell
        if shell == 'google.colab.__shell':
            return True

        return False
    except NameError:
        return False

if is_notebook():
    # If we are running in a Jupyter shell run the magic command
    import magic

# Use seaborn to manage theme
import seaborn as sns

color_palette = 'viridis'

sns.set_theme(palette=color_palette)

figure_dimensions = dict(square=(10,10), wrect=(12, 5), hrect=(5, 12), refs=(10, 12))

label_font = dict(size=14, name='Courier', family='serif', style='italic', weight='bold')
suptitle_font = dict(size=20, name='Courier', family='serif', style='normal', weight='bold')
title_font = dict(size=16, name='Courier', family='sans', style='italic', weight='normal')

def plot_wave(x, **kwargs):
    """Plot waveform(s)"""
    import matplotlib.pyplot as plt
    import numpy as np

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

    return plt

def demo_plot_wave():
    """Demonstrate plot_wave usage"""
    import numpy as np

    t = np.linspace(0, 2*np.pi, 500)
    x1 = np.sin(t**2)
    x2 = np.cos(t**2)
    x = np.array([x1, x2]).transpose()

    args = dict(
        t = [t, t],
        suptitle='plot_wave demo',
        title=['Raiders of the lost wave plot', 'The wave plot of doom'],
        xlim=[(0, 2*np.pi), (0, 2*np.pi)],
        ylim=[(-1, 1), (-1, 1)],
        xlabel=[r'$2*\pi$', r'$2*\pi$'],
        ylabel=[r'$\sin(x^2)$', r'$\cos(x^2)$']
    )

    plt = plot_wave(x, **args)
    plt.show()
    plt.close()

def plot_spectrogram(x, **kwargs):
    """Plot spectrogram"""
    import matplotlib.pyplot as plt

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

    return plt

def demo_plot_spectrogram():
    from scipy.signal import chirp
    import numpy as np

    fs = 48000
    T = 4
    t = np.arange(0, int(T*fs))/fs
    f0 = 1
    f1 = 20000
    w = chirp(t, f0=f0, f1=f1, t1=T, method='logarithmic')

    args = dict(fs=fs, title='The last wave plot', suptitle='plot_spectrogram demo')

    plt = plot_spectrogram(w, **args)

    plt.show()
    plt.close()

def plot_spectrum(x, fs, spectrum='amplitude', **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.fft import fftfreq
    from spectrum import amplitude_spectrum, power_spectrum, phase_spectrum, log_spectrum

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

    step = 1/fs
    freqs = fftfreq(x.size, step)
    idx = np.argsort(freqs)

    n = int(np.ceil(x.size / 2))

    s = s[idx][n:]
    freqs = freqs[idx][n:]

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions['wrect'])

    ax = fig.add_subplot()

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)

    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)

    ax.plot(freqs, s)
    return plt

def demo_plot_spectrum():
    import numpy as np

    x = np.random.rand(301) - 0.5
    fs = 30

    plt = plot_spectrum(x, fs, **dict(suptitle='Amplitude spectrum', title='The spectral plot'))
    plt.show()
    plt.close()

    plt = plot_spectrum(x, fs, spectrum='power', **dict(suptitle='Power spectrum', title='The spectral plot reloaded'))
    plt.show()
    plt.close()

    plt = plot_spectrum(x, fs, spectrum='phase', **dict(suptitle='Phase spectrum', title='The spectral plot revolutions'))
    plt.show()
    plt.close()

    plt = plot_spectrum(x, fs, spectrum='log', **dict(suptitle='Logarithm of spectrum', title='The spectral plot resurrections'))
    plt.show()
    plt.close()

def plot_cepstrum(x, fs, offset, window_length, **kwargs):
    import matplotlib.pyplot as plt
    from spectrum import cepstrum

    C, q = cepstrum(x, fs, offset, window_length)

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions['wrect'])

    ax = fig.add_subplot()

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)
    if 'xlim' in kwargs.keys():
        ax.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs.keys():
        ax.set_ylim(kwargs['ylim'])
    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['xlabel'], **label_font)
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'], **label_font)

    ax.plot(q * 1000, C)
    return plt

def demo_plot_cepstrum():
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

def plot_hrtfs(zeniths, azimuths, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions['square'])

    ax = fig.add_subplot(projection='polar')

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)

    ax.set_rmax(90)
    ax.set_rticks(np.linspace(-30, 75, 8))
    ax.set_rlim(bottom=90, top=-40)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    ax.annotate('',
                xy=(0, -40),
                xytext=(0, 90),
                arrowprops=dict(
                    facecolor='red',
                    alpha=0.5, width=3,
                    headwidth=6),
                ha='center',
                va='top',
                fontsize=20)
    ax.annotate('Face forward',
                xy=(0, -40),
                xytext=(0, -40),
                ha='center',
                va='center',
                fontsize=20)

    ax.scatter(np.radians(azimuths), zeniths)
    return plt

def demo_plot_hrtfs():
    import numpy as np
    from dataLoader import list_hrtf_data

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

def plot_reflections(zeniths, azimuths, delays, amplitudes, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    from dataLoader import list_hrtf_data
    from RNG import RNG

    rng = RNG()

    zenith_min = min(list_hrtf_data().keys())
    zenith_max = max(list_hrtf_data().keys())
    delay_max = rng.delay_limits[1]

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions['refs'])

    ax = fig.add_subplot(projection='polar')

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)

    theta = np.radians(azimuths)
    r = np.array(delays)
    area = ((np.array(zeniths) - zenith_min) / (zenith_max - zenith_min)) * (1500 - 200) + 200
    color = np.array(amplitudes)

    norm = plt.Normalize(0, 1)

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_rmax(delay_max)
    ax.set_rlabel_position(90)

    ax.tick_params(labelsize=20)

    ax.annotate('',
                xy=(0, delay_max),
                xytext=(0, 0),
                arrowprops=dict(facecolor='red',
                                width=6,
                                headwidth=12,
                                alpha=0.75))
    ax.annotate('Face Forward',
                xy=(0, delay_max),
                xytext=(0, delay_max),
                ha='center',
                va='center',
                fontsize=20)

    ax.text(np.radians(135),
            delay_max + 15,
            'Azimuth$\degree$',
            ha='center',
            va='center',
            fontsize=20,
            rotation=45)
    ax.text(np.radians(95),
            delay_max/2.,
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
              title='Zenith$\degree$',
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

    return plt

def demo_plot_reflections():
    from RNG import RNG

    rng = RNG()

    count = 8
    amplitudes = [rng.get_amplitude() for _ in range(count)]
    delays = [rng.get_delay() for _ in range(count)]
    zeniths = [rng.get_zenith() for _ in range(count)]
    azimuths = [rng.get_azimuth(zenith=z) for z in zeniths]

    args = dict(suptitle='Reflections', title='Reflections plot: the dragon story')

    plt = plot_reflections(zeniths, azimuths, delays, amplitudes, **args)
    plt.show()
    plt.close()

def plot_sample(zenith, azimuth, zeniths, azimuths, delays, amplitudes, amplitude, delay, time, **kwargs):
    import numpy as np
    from RNG import RNG
    from dataLoader import list_hrtf_data

    rng = RNG()

    delay_max = rng.delay_limits[1]
    time_max = rng.time_limits[1]

    zenith_min = min(list_hrtf_data().keys())
    zenith_max = max(list_hrtf_data().keys())

    area = ((zenith - zenith_min) / (zenith_max - zenith_min)) * (0.5 - 0.1) + 0.1
    err = (time / time_max) * (0.5 - 0.1) + 0.1

    plt = plot_reflections(zeniths, azimuths, delays, amplitudes, **kwargs)

    cmap = plt.get_cmap(color_palette)

    ax = plt.gca()
    ax.bar(np.deg2rad(azimuth), delay_max - delay, xerr=err, bottom=delay, width=area, alpha=0.5, color=cmap(amplitude), ecolor=cmap(amplitude))

    return plt

def demo_plot_sample():
    from RNG import RNG

    rng = RNG()

    zenith = rng.get_zenith()
    azimuth = rng.get_azimuth(zenith=zenith)

    count = 8
    amplitudes = [rng.get_amplitude() for _ in range(count)]
    delays = [rng.get_delay() for _ in range(count)]
    zeniths = [rng.get_zenith() for _ in range(count)]
    azimuths = [rng.get_azimuth(zenith=z) for z in zeniths]

    amplitude = rng.rng.uniform(0, 0.05)
    delay = rng.get_delay()
    time = rng.get_time()

    args = dict(suptitle='Reflections', title='Reflections plot: the dragon story')

    plt = plot_sample(zenith, azimuth, zeniths, azimuths, delays, amplitudes, amplitude, delay, time, **args)
    plt.show()
    plt.close()

def plot_binaural_activity_map_2d(z, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp2d

    z = z.transpose()
    x1 = np.linspace(-1, 1, z.shape[1])
    y1 = np.linspace(0, 50, z.shape[0])

    f = interp2d(x1, y1, z, kind='cubic')
    x2 = np.linspace(-1, 1, int(z.shape[1] * 1.5))
    y2 = np.linspace(0, 50, int(z.shape[0] * 1.5))
    Z = f(x2, y2)

    X2, Y2 = np.meshgrid(x2, y2)


    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions['square'])

    ax = fig.add_subplot()
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 50)

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)

    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['xlabel'], **label_font)
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'], **label_font)

    surf = ax.contourf(X2, Y2, Z, cmap=color_palette)
    fig.colorbar(surf, ax=ax)

    return plt

def demo_plot_binaural_activity_map_2d():
    pass

def plot_binaural_activity_map_3d(z, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp2d

    z = z.transpose()
    x1 = np.linspace(-1, 1, z.shape[1])
    y1 = np.linspace(0, 50, z.shape[0])

    f = interp2d(x1, y1, z, kind='cubic')
    x2 = np.linspace(-1, 1, int(z.shape[1] * 1.5))
    y2 = np.linspace(0, 50, int(z.shape[0] * 1.5))
    Z = f(x2, y2)

    X2, Y2 = np.meshgrid(x2, y2)


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

    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['xlabel'], **label_font)
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'], **label_font)
    if 'zlabel' in kwargs.keys():
        ax.set_zlabel(kwargs['zlabel'], **label_font)

    surf = ax.plot_surface(X2, Y2, Z, cmap=color_palette, rstride=1, cstride=1, edgecolor='none', aa=False)
    fig.colorbar(surf, ax=ax)

    return plt


def demo_plot_binaural_activity_map_3d():
    pass

if __name__ == '__main__':
    # demo_plot_wave()
    # demo_plot_spectrogram()
    # demo_plot_spectrum()
    # demo_plot_cepstrum()
    # demo_plot_hrtfs()
    # demo_plot_reflections()
    demo_plot_sample()