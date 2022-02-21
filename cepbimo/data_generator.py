def generate_dataset_recipe(count):
    from RNG import RNG
    import pandas as pd
    from dataLoader import list_anechoic_data
    from pydub import AudioSegment
    rng = RNG()

    get_reflection_amplitudes = lambda x: [rng.get_amplitude() for _ in range(x)]
    get_reflection_delays = lambda x: [rng.get_delay() for _ in range(x)]
    get_reflection_zeniths = lambda x: [rng.get_zenith() for _ in range(x)]
    get_reflection_azimuths = lambda x: [rng.get_azimuth(zenith=z) for z in x]

    ls = {k: len(AudioSegment.from_mp3(list_anechoic_data()[k][0])) for k in list_anechoic_data().keys()}

    composers = [rng.get_composer() for _ in range(count)]
    part_counts = [rng.get_part_count(composers[i]) for i in range(count)]
    parts = [rng.get_parts(composer=composers[i], part_count=part_counts[i]) for i in range(count)]
    offsets = [rng.get_offset(ls[composers[i]]) for i in range(count)]
    zeniths = [rng.get_zenith() for _ in range(count)]
    azimuths = [rng.get_azimuth(zenith=zeniths[i]) for i in range(count)]
    reverb_times = [rng.get_time() for _ in range(count)]
    reverb_delays = [rng.get_delay() for _ in range(count)]
    reverb_amplitudes = [rng.rng.uniform(0, 0.05) for _ in range(count)]
    reflection_counts = [rng.get_reflection_count() for _ in range(count)]
    reflection_amplitudes = [get_reflection_amplitudes(reflection_counts[i]) for i in range(count)]
    reflection_delays = [get_reflection_delays(reflection_counts[i]) for i in range(count)]
    reflection_zeniths = [get_reflection_zeniths(reflection_counts[i]) for i in range(count)]
    reflection_azimuths = [get_reflection_azimuths(reflection_zeniths[i]) for i in range(count)]

    file_paths = [generate_filepath(composers[i], part_counts[i], zeniths[i], azimuths[i]) for i in range(count)]

    df = pd.DataFrame({
        'zenith': zeniths,
        'azimuth': azimuths,
        'composer': composers,
        'part_count': part_counts,
        'parts': parts,
        'offset': offsets,
        'duration': [rng.duration for _ in range(count)],
        'reverb_time': reverb_times,
        'reverb_delay': reverb_delays,
        'reverb_amplitude': reverb_amplitudes,
        'reflection_count': reflection_counts,
        'reflection_amplitude': reflection_amplitudes,
        'reflection_delay': reflection_delays,
        'reflection_zenith': reflection_zeniths,
        'reflection_azimuth': reflection_azimuths,
        'filepath': file_paths
    })

    return df


def generate_filepath(composer, part_count, zenith, azimuth):
    from pathlib import Path

    data_path = Path('data/reflections/')

    file_dir = f'{composer}_p{part_count:02d}_a{azimuth:03d}_e{zenith:03d}'

    df = data_path / file_dir / file_dir

    return df.__str__()


def generate_signal_raw(parts, filepath):
    import os
    from pathlib import Path
    from pydub import AudioSegment
    from transforms import mix_parts

    filepath = Path(f'{filepath}_raw.wav')

    if os.path.isfile(filepath):
        return AudioSegment.from_wav(filepath)

    return write_audiosegment(mix_parts(parts), filepath.__str__())


def generate_signal_hrtf(x, zenith, azimuth, filepath):
    import os
    from pathlib import Path
    from pydub import AudioSegment
    from transforms import apply_hrtf

    filepath = Path(f'{filepath}_hrtf.wav')

    if os.path.isfile(filepath):
        return AudioSegment.from_wav(filepath)

    return write_audiosegment(apply_hrtf(x, zenith, azimuth), filepath.__str__())


def generate_signal_reflection(x, count, amplitudes, delays, zeniths, azimuths, filepath):
    import os
    from pathlib import Path
    from pydub import AudioSegment
    from transforms import mix_reflections

    filepath = Path(f'{filepath}_reflections.wav')

    if os.path.isfile(filepath):
        return AudioSegment.from_wav(filepath)

    return write_audiosegment(mix_reflections(x, count, amplitudes, delays, zeniths, azimuths), filepath.__str__())


def generate_signal_reverberation(x, amplitude, delay, time, filepath):
    import os
    from pathlib import Path
    from pydub import AudioSegment
    from transforms import apply_reverberation

    filepath = Path(f'{filepath}_reverberation.wav')

    if os.path.isfile(filepath):
        return AudioSegment.from_wav(filepath)

    return write_audiosegment(apply_reverberation(x, amplitude, delay, time), filepath.__str__())


def generate_signal_summation(rf, rv, filepath):
    import os
    from pathlib import Path
    from pydub import AudioSegment
    from transforms import sum_signals

    filepath = Path(f'{filepath}_summation.wav')

    if os.path.isfile(filepath):
        return AudioSegment.from_wav(filepath)

    return write_audiosegment(sum_signals(rf, rv), filepath.__str__())


def generate_signal_noise(x, filepath):
    import os
    from pathlib import Path
    from pydub import AudioSegment
    from transforms import adjust_signal_to_noise

    filepath = Path(f'{filepath}_noise.wav')

    if os.path.isfile(filepath):
        return AudioSegment.from_wav(filepath)

    return write_audiosegment(adjust_signal_to_noise(x, -30), filepath.__str__())


def generate_signal_trimmed(x, offset, duration, filepath):
    import os
    from pathlib import Path
    from pydub import AudioSegment

    filepath = Path(f'{filepath}.wav')

    if os.path.isfile(filepath):
        return AudioSegment.from_wav(filepath)

    return write_audiosegment(x[offset:offset + duration * 1000], filepath.__str__())


def generate_room_impulse(zenith, azimuth, reflection_count, zeniths, azimuths, delays, amplitudes, amplitude, delay,
                          time, duration, filepath):
    import os
    from pathlib import Path
    from pydub import AudioSegment
    import numpy as np
    from transforms import apply_hrtf, mix_reflections, apply_reverberation, sum_signals
    from figures import plot_wave
    from utils import audiosegment_to_array, array_to_audiosegment

    f = Path(f'{filepath}_rir.wav')

    if os.path.isfile(f):
        summation = AudioSegment.from_wav(f)
    else:
        fs = 44100
        click = np.ones(1)
        click = np.concatenate((click, np.zeros(fs * duration - 1)))

        signal = array_to_audiosegment(click, fs)
        signal = apply_hrtf(signal, zenith, azimuth)

        reflections = mix_reflections(signal, reflection_count, amplitudes, delays, zeniths, azimuths)
        reverberation = apply_reverberation(signal, amplitude, delay, time)

        summation = sum_signals(reflections, reverberation)

        write_audiosegment(summation, f.__str__())

    f = Path(f'{filepath}_rir.png')

    if not os.path.isfile(f):
        t = np.linspace(0, int(np.ceil(summation.duration_seconds)), int(summation.frame_count()))

        if summation.channels == 1:
            summation = summation.set_channels(2)

        channels = summation.split_to_mono()
        left = audiosegment_to_array(channels[0])
        right = audiosegment_to_array(channels[1])
        summation = np.array([left, right]).transpose()

        args = dict(
            t=[t, t],
            suptitle='RIR, click=[1., 0., ..., 0.]',
            title=['Left channel', 'Right channel'],
            xlabel=['Time, s', 'Time, s'],
            ylabel=['Amplitude', 'Amplitude']
        )

        plt = plot_wave(summation, **args)
        write_figure(plt, f.__str__())
        plt.close()


def generate_plot_sample(zenith, azimuth, zeniths, azimuths, delays, amplitudes, amplitude, delay, time, filepath):
    import os
    from pathlib import Path
    from figures import plot_sample

    filepath = Path(f'{filepath}_sample.png')

    if not os.path.isfile(filepath):
        args = dict(
            suptitle='Sample',
            title=f'{len(zeniths)} reflections'
        )

        plt = plot_sample(zenith, azimuth, zeniths, azimuths, delays, amplitudes, amplitude, delay, time, **args)
        write_figure(plt, filepath.__str__())
        plt.close()


def generate_plot_signal(x, filepath):
    import os
    from pathlib import Path
    import numpy as np
    from figures import plot_wave
    from utils import audiosegment_to_array

    filepath = Path(f'{filepath}_raw_wave.png')

    if not os.path.isfile(filepath):
        t = np.linspace(0, int(np.ceil(x.duration_seconds)), int(x.frame_count()))

        if x.channels == 1:
            x = x.set_channels(2)

        channels = x.split_to_mono()
        left = audiosegment_to_array(channels[0])
        right = audiosegment_to_array(channels[1])
        x = np.array([left, right]).transpose()

        args = dict(
            t=[t, t],
            suptitle='Raw direct signal',
            title=['Left channel', 'Right channel'],
            xlabel=['Time, s', 'Time, s'],
            ylabel=['Amplitude', 'Amplitude']
        )

        plt = plot_wave(x, **args)
        write_figure(plt, filepath.__str__())
        plt.close()


def generate_plot_hrtf(x, filepath):
    import os
    from pathlib import Path
    import numpy as np
    from figures import plot_wave
    from utils import audiosegment_to_array

    filepath = Path(f'{filepath}_hrtf_wave.png')

    if not os.path.isfile(filepath):
        t = np.linspace(0, int(np.ceil(x.duration_seconds)), int(x.frame_count()))

        if x.channels == 1:
            x = x.set_channels(2)

        channels = x.split_to_mono()
        left = audiosegment_to_array(channels[0])
        right = audiosegment_to_array(channels[1])
        x = np.array([left, right]).transpose()

        args = dict(
            t=[t, t],
            suptitle='HRTF applied direct signal',
            title=['Left channel', 'Right channel'],
            xlabel=['Time, s', 'Time, s'],
            ylabel=['Amplitude', 'Amplitude']
        )

        plt = plot_wave(x, **args)
        write_figure(plt, filepath.__str__())
        plt.close()


def generate_plot_reflections(x, filepath):
    import os
    from pathlib import Path
    import numpy as np
    from figures import plot_wave
    from utils import audiosegment_to_array

    filepath = Path(f'{filepath}_reflections_wave.png')

    if not os.path.isfile(filepath):
        t = np.linspace(0, int(np.ceil(x.duration_seconds)), int(x.frame_count()))

        if x.channels == 1:
            x = x.set_channels(2)

        channels = x.split_to_mono()
        left = audiosegment_to_array(channels[0])
        right = audiosegment_to_array(channels[1])
        x = np.array([left, right]).transpose()

        args = dict(
            t=[t, t],
            suptitle='Reflections applied direct signal',
            title=['Left channel', 'Right channel'],
            xlabel=['Time, s', 'Time, s'],
            ylabel=['Amplitude', 'Amplitude']
        )

        plt = plot_wave(x, **args)
        write_figure(plt, filepath.__str__())
        plt.close()


def generate_plot_reverberation(x, filepath):
    import os
    from pathlib import Path
    import numpy as np
    from figures import plot_wave
    from utils import audiosegment_to_array

    filepath = Path(f'{filepath}_reverberation_wave.png')

    if not os.path.isfile(filepath):
        t = np.linspace(0, int(np.ceil(x.duration_seconds)), int(x.frame_count()))

        if x.channels == 1:
            x = x.set_channels(2)

        channels = x.split_to_mono()
        left = audiosegment_to_array(channels[0])
        right = audiosegment_to_array(channels[1])
        x = np.array([left, right]).transpose()

        args = dict(
            t=[t, t],
            suptitle='Reverberation applied direct signal',
            title=['Left channel', 'Right channel'],
            xlabel=['Time, s', 'Time, s'],
            ylabel=['Amplitude', 'Amplitude']
        )

        plt = plot_wave(x, **args)
        write_figure(plt, filepath.__str__())
        plt.close()


def generate_plot_summation(x, filepath):
    import os
    from pathlib import Path
    import numpy as np
    from figures import plot_wave
    from utils import audiosegment_to_array

    filepath = Path(f'{filepath}_summation_wave.png')

    if not os.path.isfile(filepath):
        t = np.linspace(0, int(np.ceil(x.duration_seconds)), int(x.frame_count()))

        if x.channels == 1:
            x = x.set_channels(2)

        channels = x.split_to_mono()
        left = audiosegment_to_array(channels[0])
        right = audiosegment_to_array(channels[1])
        x = np.array([left, right]).transpose()

        args = dict(
            t=[t, t],
            suptitle='Reflections and reverberation applied direct signal',
            title=['Left channel', 'Right channel'],
            xlabel=['Time, s', 'Time, s'],
            ylabel=['Amplitude', 'Amplitude']
        )

        plt = plot_wave(x, **args)
        write_figure(plt, filepath.__str__())
        plt.close()


def generate_plot_noise(x, filepath):
    import os
    from pathlib import Path
    import numpy as np
    from figures import plot_wave
    from utils import audiosegment_to_array

    filepath = Path(f'{filepath}_noise_wave.png')

    if not os.path.isfile(filepath):
        t = np.linspace(0, int(np.ceil(x.duration_seconds)), int(x.frame_count()))

        if x.channels == 1:
            x = x.set_channels(2)

        channels = x.split_to_mono()
        left = audiosegment_to_array(channels[0])
        right = audiosegment_to_array(channels[1])
        x = np.array([left, right]).transpose()

        args = dict(
            t=[t, t],
            suptitle='-30 dB of white noise added to direct signal',
            title=['Left channel', 'Right channel'],
            xlabel=['Time, s', 'Time, s'],
            ylabel=['Amplitude', 'Amplitude']
        )

        plt = plot_wave(x, **args)
        write_figure(plt, filepath.__str__())
        plt.close()


def generate_plot_trimmed(x, filepath):
    import os
    from pathlib import Path
    import numpy as np
    from figures import plot_wave
    from utils import audiosegment_to_array

    filepath = Path(f'{filepath}_wave.png')

    if not os.path.isfile(filepath):
        duration = int(np.ceil(x.duration_seconds))
        t = np.linspace(0, duration, int(x.frame_count()))

        if x.channels == 1:
            x = x.set_channels(2)

        channels = x.split_to_mono()
        left = audiosegment_to_array(channels[0])
        right = audiosegment_to_array(channels[1])
        x = np.array([left, right]).transpose()

        args = dict(
            t=[t, t],
            suptitle=f'Direct signal trimmed to {duration} s',
            title=['Left channel', 'Right channel'],
            xlabel=['Time, s', 'Time, s'],
            ylabel=['Amplitude', 'Amplitude']
        )

        plt = plot_wave(x, **args)
        write_figure(plt, filepath.__str__())
        plt.close()


def generate_plot_ceptstrum(x, filepath):
    import os
    from pathlib import Path
    from figures import plot_cepstrum
    from utils import audiosegment_to_array

    offset = 1024
    window_length = offset * 64 * 4

    fs = x.frame_rate

    channels = x.split_to_mono()
    left = audiosegment_to_array(channels[0])
    right = audiosegment_to_array(channels[1])

    f = Path(f'{filepath}_left_cepstrum.png')

    if not os.path.isfile(f):
        args = dict(
            suptitle='Amplitude cepstrum',
            title='Left channel',
            xlabel='Quefrency, ms',
            ylabel='Amplitude'
        )

        plt = plot_cepstrum(left, fs, offset, window_length, **args)
        write_figure(plt, f.__str__())
        plt.close()

    f = Path(f'{filepath}_right_cepstrum.png')

    if not os.path.isfile(f):
        args = dict(
            suptitle='Amplitude cepstrum',
            title='Right channel',
            xlabel='Quefrency, ms',
            ylabel='Amplitude'
        )

        plt = plot_cepstrum(right, fs, offset, window_length, **args)
        write_figure(plt, f.__str__())
        plt.close()

    f = Path(f'{filepath}_left_cepstrum_20.png')

    if not os.path.isfile(f):
        args = dict(
            suptitle='Amplitude cepstrum, 1-20 ms',
            title='Left channel',
            xlabel='Quefrency, ms',
            ylabel='Amplitude',
            xlim=(1, 20)
        )

        plt = plot_cepstrum(left, fs, offset, window_length, **args)
        write_figure(plt, f.__str__())
        plt.close()

    f = Path(f'{filepath}_right_cepstrum_20.png')

    if not os.path.isfile(f):
        args = dict(
            suptitle='Amplitude cepstrum, 1-20 ms',
            title='Right channel',
            xlabel='Quefrency, ms',
            ylabel='Amplitude',
            xlim=(1, 20)
        )

        plt = plot_cepstrum(right, fs, offset, window_length, **args)
        write_figure(plt, f.__str__())
        plt.close()


def generate_plot_cepbimo(x, filepath):
    import os
    from pathlib import Path
    import numpy as np
    from figures import plot_wave, plot_binaural_activity_map_3d, plot_binaural_activity_map_2d

    f = Path(f'{filepath}_cepbimo_wave.png')

    if not os.path.isfile(f):
        X = np.array([x.Xd / 200, x.XR / 200]).transpose()
        t = np.linspace(-1., 1., (x.lag * 2) + 1)

        args = dict(
            t=[t, t],
            suptitle=f'Cepbimo',
            title=['Left channel', 'Right channel', 'Interaural cross-correlation'],
            xlabel=['Quefrency, ms', 'Quefrency, ms', 'ITD, ms'],
            ylabel=['Amplitude', 'Amplitude', 'Correlation']
        )

        plt = plot_wave(X, **args)

        write_figure(plt, f.__str__())
        plt.close()

    f = Path(f'{filepath}_binaural_activity_map_3d.png')

    if not os.path.isfile(f):
        plt = plot_binaural_activity_map_3d(x.z)
        write_figure(plt, f.__str__())
        plt.close()



    f = Path(f'{filepath}_binaural_activity_map_2d.png')

    if not os.path.isfile(f):
        plt = plot_binaural_activity_map_2d(x.z)
        write_figure(plt, f.__str__())
        plt.close()

def write_recipe(recipe, path='data/', file_type='csv'):
    from pathlib import Path

    path = Path(path) / f'recipe.{file_type}'

    if file_type == 'csv':
        recipe.to_csv(path, index=False)
    if file_type == 'json':
        recipe.to_json(path, orient='records', lines=True)

    return path.__str__()


def read_recipe(path):
    import pandas as pd

    print(path)

    if path.endswith('.csv'):
        return pd.read_csv(path, converters={'parts': lambda x: x.strip('[]').replace("'", "").split(', ')})
    if path.endswith('.json'):
        return pd.read_json(path, orient='records', lines=True)


def make_recipe(recipe):
    from Cepbimo import Cepbimo

    print('Start plotting samples:')
    recipe.apply(lambda row: generate_plot_sample(row['zenith'],
                                                  row['azimuth'],
                                                  row['reflection_zenith'],
                                                  row['reflection_azimuth'],
                                                  row['reflection_delay'],
                                                  row['reflection_amplitude'],
                                                  row['reverb_amplitude'],
                                                  row['reverb_delay'],
                                                  row['reverb_time'],
                                                  row['filepath']),
                 axis=1)
    print('Finished plotting samples:\n')

    print('Start mixing parts into signals:')
    recipe['signal'] = recipe.apply(lambda row: generate_signal_raw(row['parts'],
                                                                    row['filepath']),
                                    axis=1)
    print('Finished mixing parts into signals:\n')

    print('Start plotting signals:')
    recipe.apply(lambda row: generate_plot_signal(row['signal'], row['filepath']), axis=1)
    print('Finished plotting signals:\n')

    print('Start applying HRTFs:')
    recipe['signal'] = recipe.apply(lambda row: generate_signal_hrtf(row['signal'],
                                                                     row['zenith'],
                                                                     row['azimuth'],
                                                                     row['filepath']),
                                    axis=1)
    print('Finished applying HRTFs:\n')

    print('Start plotting HRTFs:')
    recipe.apply(lambda row: generate_plot_hrtf(row['signal'], row['filepath']), axis=1)
    print('Finished plotting HRTFs:\n')

    print('Start applying reflections:')
    recipe['reflections'] = recipe.apply(lambda row: generate_signal_reflection(row['signal'],
                                                                                row['reflection_count'],
                                                                                row['reflection_amplitude'],
                                                                                row['reflection_delay'],
                                                                                row['reflection_zenith'],
                                                                                row['reflection_azimuth'],
                                                                                row['filepath']),
                                         axis=1)
    print('Finished applying reflections:\n')

    print('Start plotting reflections:')
    recipe.apply(lambda row: generate_plot_reflections(row['reflections'], row['filepath']), axis=1)
    print('Finished plotting reflections:\n')

    print('Start applying reverberation:')
    recipe['reverberation'] = recipe.apply(lambda row: generate_signal_reverberation(row['signal'],
                                                                                     row['reverb_amplitude'],
                                                                                     row['reverb_delay'],
                                                                                     row['reverb_time'],
                                                                                     row['filepath']),
                                           axis=1)
    print('Finished applying reverberation:\n')

    print('Start plotting reverberation:')
    recipe.apply(lambda row: generate_plot_reverberation(row['reverberation'], row['filepath']), axis=1)
    print('Finished plotting reverberation:\n')

    print('Start summing signals:')
    recipe['summation'] = recipe.apply(lambda row: generate_signal_summation(row['reflections'],
                                                                             row['reverberation'],
                                                                             row['filepath']),
                                       axis=1)
    print('Finished summing signals:\n')

    print('Start plotting summation:')
    recipe.apply(lambda row: generate_plot_summation(row['summation'], row['filepath']), axis=1)
    print('Finished plotting summation:\n')

    print('Start adjusting signal-to-noise ratio:')
    recipe['noise'] = recipe.apply(lambda row: generate_signal_noise(row['summation'], row['filepath']), axis=1)
    print('Finished adjusting signal-to-noise ratio:\n')

    print('Start plotting noise:')
    recipe.apply(lambda row: generate_plot_noise(row['noise'], row['filepath']), axis=1)
    print('Finished plotting noise:\n')

    print('Start trimming samples:')
    recipe['sample'] = recipe.apply(lambda row: generate_signal_trimmed(row['summation'],
                                                                        row['offset'],
                                                                        row['duration'],
                                                                        row['filepath']), axis=1)
    print('Finished trimming samples:\n')

    print('Start plotting trimmed:')
    recipe.apply(lambda row: generate_plot_trimmed(row['sample'], row['filepath']), axis=1)
    print('Finished plotting trimmed:\n')

    print('Start plotting cepstrums:')
    recipe.apply(lambda row: generate_plot_ceptstrum(row['sample'], row['filepath']), axis=1)
    print('Finished plotting cepstrums:\n')

    print('Start plotting RIR:')
    recipe.apply(lambda row: generate_room_impulse(row['zenith'],
                                                   row['azimuth'],
                                                   row['reflection_count'],
                                                   row['reflection_zenith'],
                                                   row['reflection_azimuth'],
                                                   row['reflection_delay'],
                                                   row['reflection_amplitude'],
                                                   row['reverb_amplitude'],
                                                   row['reverb_delay'],
                                                   row['reverb_time'],
                                                   row['duration'],
                                                   row['filepath']), axis=1)
    print('Finished plotting RIR:\n')

    print('Start generating Cepbimo:')
    recipe['cepbimo'] = recipe.apply(lambda row: Cepbimo(row['sample']), axis=1)
    print('Finished generating Cepbimo:\n')

    print('Start plotting Cepbimo:')
    recipe.apply(lambda row: generate_plot_cepbimo(row['cepbimo'], row['filepath']), axis=1)
    print('Finished plotting Cepbimo:\n')


def write_audiosegment(a, filepath):
    import os
    from pathlib import Path

    filepath = Path(filepath)
    if not os.path.isdir(filepath.parents[1]):
        print(f'\tMaking directory {filepath.parents[1]}')
        os.mkdir(Path(filepath.parents[1]))

    if not os.path.isdir(filepath.parents[0]):
        print(f'\tMaking directory {filepath.parents[0]}')
        os.mkdir(Path(filepath.parents[0]))

    if not os.path.isfile(filepath):
        print(f'\tMaking file {filepath}')
        a.export(filepath, format='wav')

    return a


def write_figure(f, filepath):
    import os
    from pathlib import Path

    filepath = Path(filepath)
    if not os.path.isdir(filepath.parents[1]):
        print(f'\tMaking directory {filepath.parents[1]}')
        os.mkdir(Path(filepath.parents[1]))

    if not os.path.isdir(filepath.parents[0]):
        print(f'\tMaking directory {filepath.parents[0]}')
        os.mkdir(Path(filepath.parents[0]))

    if not os.path.isfile(filepath):
        print(f'\tMaking file {filepath}')
        f.savefig(filepath, format='png')

    return f


if __name__ == '__main__':
    from pathlib import Path

    r_path = 'data/'
    r = generate_dataset_recipe(10)
    r_path = write_recipe(r, path=r_path, file_type='json')
    # r_path = (Path(r_path) / 'recipe.json').__str__()
    r = read_recipe(r_path)
    make_recipe(r)
