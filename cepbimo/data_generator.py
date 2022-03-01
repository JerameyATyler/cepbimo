"""Module for generating a reflection dataset."""


class DataGenerator:
    """Dataset generator

    Methods:

    get_reflections()
    generate_dataset_recipe()
    generate()

    Constructors:

    __init__()

    Properties (readonly):
    sample_count, rng
    """

    def __init__(self, sample_count, output_directory, rng=None):
        """Constructor.
        
        Arguments:
        
        sample_count, rng
        """
        import os
        from pathlib import Path

        if rng is None:
            from RNG import RNG
            rng = RNG()
        self.rng = rng
        self.sample_count = sample_count
        self.output_directory = output_directory
        self.recipe = None
        self.chunk_size = 10

        path = Path(output_directory)
        if not os.path.isdir(path):
            os.mkdir(path)

    def get_reflections(self, count):
        rng = self.rng
        """Generate the specified number of random reflections."""
        amplitudes = [rng.get_amplitude() for _ in range(count)]
        delays = [rng.get_delay() for _ in range(count)]
        zeniths = [rng.get_zenith() for _ in range(count)]
        azimuths = [rng.get_azimuth(zenith=zeniths[i]) for i in range(count)]
        return zeniths, azimuths, amplitudes, delays

    def generate_dataset_ingredients(self):
        import pandas as pd
        from pathlib import Path
        import os

        filepath = Path(self.output_directory) / 'ingredients.json'

        rng = self.rng

        seed = rng.seed
        duration = rng.duration
        delay_limits = rng.delay_limits
        time_limits = rng.time_limits
        reflection_limits = rng.reflection_limits
        zenith_limits = rng.zenith_limits
        azimuth_limits = rng.azimuth_limits
        sample_count = self.sample_count

        df = pd.DataFrame([dict(
            seed=seed,
            duration=duration,
            delay_limits=delay_limits,
            time_limits=time_limits,
            reflection_limits=reflection_limits,
            zenith_limits=zenith_limits,
            azimuth_limits=azimuth_limits,
            sample_count=sample_count
        )])

        if os.path.isfile(filepath):
            dfi = self.read_ingredients()
            if dfi.equals(df):
                return

        df.to_json(filepath)

    def generate_dataset_recipe(self):
        """Generate a random dataset of the specified size."""
        import numpy as np
        import pandas as pd
        from data_loader import list_anechoic_data
        from pydub import AudioSegment

        rng = self.rng
        sample_count = self.sample_count
        chunk_size = self.chunk_size

        ls = {k: len(AudioSegment.from_mp3(list_anechoic_data()[k][0])) for k in list_anechoic_data().keys()}

        chunks = []

        for chunk in range(int(np.ceil(sample_count / chunk_size))):
            if ((chunk + 1) * chunk_size) > sample_count:
                size = sample_count % chunk_size
            else:
                size = chunk_size
            composers = [rng.get_composer() for _ in range(size)]
            part_counts = [rng.get_part_count(composers[i]) for i in range(size)]
            parts = [rng.get_parts(composer=composers[i], part_count=part_counts[i]) for i in range(size)]
            offsets = [rng.get_offset(ls[composers[i]]) for i in range(size)]
            zeniths = [rng.get_zenith() for _ in range(size)]
            azimuths = [rng.get_azimuth(zenith=zeniths[i]) for i in range(size)]
            reverb_times = [rng.get_time() for _ in range(size)]
            reverb_delays = [rng.get_delay() for _ in range(size)]
            reverb_amplitudes = [rng.rng.uniform(0, 0.05) for _ in range(size)]
            reflection_counts = [rng.get_reflection_count() for _ in range(size)]
            reflection_zeniths = []
            reflection_azimuths = []
            reflection_amplitudes = []
            reflection_delays = []

            for i in range(size):
                zenith, azimuth, amplitude, delay = self.get_reflections(reflection_counts[i])
                reflection_zeniths.append(zenith)
                reflection_azimuths.append(azimuth)
                reflection_amplitudes.append(amplitude)
                reflection_delays.append(delay)

            file_paths = [self.generate_filepath(composers[i], part_counts[i], zeniths[i], azimuths[i]) for i in
                          range(size)]

            df = pd.DataFrame({
                'zenith': zeniths,
                'azimuth': azimuths,
                'composer': composers,
                'part_count': part_counts,
                'parts': parts,
                'offset': offsets,
                'duration': [rng.duration for _ in range(size)],
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

            chunks.append(df)
        return pd.concat(chunks)

    def generate(self, verbose=False):
        """Start generator."""
        self.generate_dataset_ingredients()

        recipe = self.generate_dataset_recipe()

        self.write_recipe(recipe, file_type="json")
        dataset = self.read_recipe(verbose=verbose)
        ingredients = self.read_ingredients()
        return dataset, ingredients

    def write_recipe(self, recipe, file_type='csv'):
        """Write the recipe to a file."""
        import os
        from pathlib import Path

        path = Path(self.output_directory) / f'recipe.{file_type}'

        if not os.path.isdir(path.parents[1]):
            os.mkdir(path.parents[1])
        if not os.path.isdir(path.parents[0]):
            os.mkdir(path.parents[0])

        if file_type == 'csv':
            recipe.to_csv(path, index=False)
        if file_type == 'json':
            recipe.to_json(path, orient='records', lines=True)
        if file_type == 'h5':
            recipe.to_hdf(path, 'recipe')

        return path.__str__()

    def make_recipe(self, recipe, verbose):
        """Make the recipe specified."""
        from Cepbimo import Cepbimo

        print('Start mixing parts into signals:')
        recipe['signal'] = recipe.apply(lambda row: self.generate_signal_raw(row['parts'],
                                                                             row['filepath'],
                                                                             row['offset'], row['duration'], verbose),
                                        axis=1)
        print('Finished mixing parts into signals:\n')

        print('Start applying HRTFs:')
        recipe['signal'] = recipe.apply(lambda row: self.generate_signal_hrtf(row['signal'],
                                                                              row['zenith'],
                                                                              row['azimuth'],
                                                                              row['filepath'], verbose),
                                        axis=1)
        print('Finished applying HRTFs:\n')

        print('Start applying reflections:')
        recipe['reflections'] = recipe.apply(lambda row: self.generate_signal_reflection(row['signal'],
                                                                                         row['reflection_count'],
                                                                                         row['reflection_amplitude'],
                                                                                         row['reflection_delay'],
                                                                                         row['reflection_zenith'],
                                                                                         row['reflection_azimuth'],
                                                                                         row['filepath'], verbose),
                                             axis=1)
        print('Finished applying reflections:\n')

        print('Start applying reverberation:')
        recipe['reverberation'] = recipe.apply(lambda row: self.generate_signal_reverberation(row['signal'],
                                                                                              row['reverb_amplitude'],
                                                                                              row['reverb_delay'],
                                                                                              row['reverb_time'],
                                                                                              row['filepath'], verbose),
                                               axis=1)
        print('Finished applying reverberation:\n')

        print('Start summing signals:')
        recipe['summation'] = recipe.apply(lambda row: self.generate_signal_summation(row['reflections'],
                                                                                      row['reverberation'],
                                                                                      row['filepath'], verbose),
                                           axis=1)
        print('Finished summing signals:\n')

        print('Start adjusting signal-to-noise ratio:')
        recipe['noise'] = recipe.apply(
            lambda row: self.generate_signal_noise(row['summation'], row['filepath'], verbose),
            axis=1)
        print('Finished adjusting signal-to-noise ratio:\n')

        print('Start trimming samples:')
        recipe['sample'] = recipe.apply(lambda row: self.generate_signal_trimmed(row['summation'],
                                                                                 row['duration'],
                                                                                 row['filepath']), axis=1)
        print('Finished trimming samples:\n')

        if verbose:
            print('Start plotting samples:')
            recipe.apply(lambda row: self.generate_plot_sample(row['zenith'],
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
            print('Start plotting signals:')
            recipe.apply(lambda row: self.generate_plot_signal(row['signal'], row['filepath']), axis=1)
            print('Finished plotting signals:\n')

            print('Start plotting HRTFs:')
            recipe.apply(lambda row: self.generate_plot_hrtf(row['signal'], row['filepath']), axis=1)
            print('Finished plotting HRTFs:\n')

            print('Start plotting reflections:')
            recipe.apply(lambda row: self.generate_plot_reflections(row['reflections'], row['filepath']), axis=1)
            print('Finished plotting reflections:\n')

            print('Start plotting reverberation:')
            recipe.apply(lambda row: self.generate_plot_reverberation(row['reverberation'], row['filepath']), axis=1)
            print('Finished plotting reverberation:\n')

            print('Start plotting summation:')
            recipe.apply(lambda row: self.generate_plot_summation(row['summation'], row['filepath']), axis=1)
            print('Finished plotting summation:\n')

            print('Start plotting noise:')
            recipe.apply(lambda row: self.generate_plot_noise(row['noise'], row['filepath']), axis=1)
            print('Finished plotting noise:\n')

            print('Start plotting trimmed:')
            recipe.apply(lambda row: self.generate_plot_trimmed(row['sample'], row['filepath']), axis=1)
            print('Finished plotting trimmed:\n')

            print('Start plotting cepstrums:')
            recipe.apply(lambda row: self.generate_plot_ceptstrum(row['sample'], row['filepath']), axis=1)
            print('Finished plotting cepstrums:\n')

            print('Start plotting RIR:')
            recipe.apply(lambda row: self.generate_room_impulse(row['zenith'],
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
            recipe.apply(lambda row: self.generate_plot_cepbimo(row['cepbimo'], row['filepath']), axis=1)
            print('Finished plotting Cepbimo:\n')
        return recipe

    def read_recipe(self, verbose=False):
        """Read the recipe from a file."""
        from pathlib import Path
        import pandas as pd
        import glob

        path = (Path(self.output_directory) / 'recipe.*').__str__()
        files = glob.glob(path)
        if len(files) < 1:
            return pd.DataFrame()
        path = files[0]
        chunks = []

        if path.endswith('.csv'):
            with pd.read_csv(path, converters={'parts': lambda x: x.strip('[]').replace("'", "").split(', ')},
                             chunksize=self.chunk_size) as reader:
                for chunk in reader:
                    chunks.append(self.make_recipe(chunk, verbose))
        if path.endswith('.json'):
            with pd.read_json(path, orient='records', lines=True, chunksize=self.chunk_size) as reader:
                for chunk in reader:
                    chunks.append(self.make_recipe(chunk, verbose))
        if path.endswith('.h5'):
            chunks.append(pd.read_hdf(path))

        df = pd.concat(chunks)

        return df

    def generate_filepath(self, composer, part_count, zenith, azimuth):
        """Generate the base filepath for the sample with the specified parameters."""

        return f'{composer}_p{part_count:02d}_a{azimuth:03d}_e{zenith:03d}'

    def read_ingredients(self):
        import pandas as pd
        from pathlib import Path
        import glob

        path = (Path(self.output_directory) / 'ingredients.*').__str__()
        files = glob.glob(path)
        if len(files) < 1:
            return pd.DataFrame()
        path = files[0]

        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.json'):
            return pd.read_json(path, orient='records')
        elif path.endswith('.h5'):
            return pd.read_hdf(path)
        else:
            return pd.read_pickle(path)

    def write_file(self, a, filepath):
        """Write a file to disk."""
        import os
        from pathlib import Path

        filepath = Path(filepath)
        if not os.path.isdir(filepath.parents[2]):
            print(f"\tMaking directory {filepath.parents[2]}")

        if not os.path.isdir(filepath.parents[1]):
            print(f'\tMaking directory {filepath.parents[1]}')
            os.mkdir(Path(filepath.parents[1]))

        if not os.path.isdir(filepath.parents[0]):
            print(f'\tMaking directory {filepath.parents[0]}')
            os.mkdir(Path(filepath.parents[0]))

        if not os.path.isfile(filepath):
            print(f'\tMaking file {filepath}')
            if ".wav" == filepath.suffix:
                a = a.set_frame_rate(24000)
                a.export(filepath, format=filepath.suffix.strip("."))
            elif ".mp3" == filepath.suffix:
                a.export(filepath, format=filepath.suffix.strip("."))
            elif ".png" == filepath.suffix:
                a.savefig(filepath, format=filepath.suffix.strip("."))

        return a

    def generate_signal_hrtf(self, x, zenith, azimuth, filepath, verbose):
        """Apply the HRTF to the signal and save the result."""
        import os
        from pathlib import Path
        from pydub import AudioSegment
        from transforms import apply_hrtf

        filepath = Path(f"{self.output_directory}/reflections/hrtf") / f'{filepath}_hrtf.wav'

        if os.path.isfile(filepath):
            return AudioSegment.from_wav(filepath)

        if verbose:
            return self.write_file(apply_hrtf(x, zenith, azimuth), filepath.__str__())
        return apply_hrtf(x, zenith, azimuth)

    def generate_signal_raw(self, parts, filepath, offset, duration, verbose):
        """Generate a raw direct signal by mixing the parts."""
        import os
        from pathlib import Path
        from pydub import AudioSegment
        from transforms import mix_parts

        f = Path(f"{self.output_directory}/reflections/raw") / f'{filepath}_raw.wav'

        if os.path.isfile(filepath):
            return AudioSegment.from_wav(filepath)

        if verbose:
            return self.write_file(mix_parts(parts, offset, duration), f.__str__())
        return mix_parts(parts, offset, duration)

    def generate_signal_reflection(self, x, count, amplitudes, delays, zeniths, azimuths, filepath, verbose):
        """Apply reflections to the signal and save the result."""
        import os
        from pathlib import Path
        from pydub import AudioSegment
        from transforms import mix_reflections

        filepath = Path(f"{self.output_directory}/reflections/reflections") / f'{filepath}_reflections.wav'

        if os.path.isfile(filepath):
            return AudioSegment.from_wav(filepath)

        if verbose:
            return self.write_file(mix_reflections(x, count, amplitudes, delays, zeniths, azimuths), filepath.__str__())
        return mix_reflections(x, count, amplitudes, delays, zeniths, azimuths)

    def generate_signal_reverberation(self, x, amplitude, delay, time, filepath, verbose):
        """Apply reverberation to the signal and save the result."""
        import os
        from pathlib import Path
        from pydub import AudioSegment
        from transforms import apply_reverberation

        filepath = Path(f"{self.output_directory}/reflections/reverberations") / f'{filepath}_reverberation.wav'

        if os.path.isfile(filepath):
            return AudioSegment.from_wav(filepath)

        if verbose:
            return self.write_file(apply_reverberation(x, amplitude, delay, time), filepath.__str__())
        return apply_reverberation(x, amplitude, delay, time)

    def generate_signal_summation(self, rf, rv, filepath, verbose):
        """Mix the reflection applied signal with the reverberation applied signal."""
        import os
        from pathlib import Path
        from pydub import AudioSegment
        from transforms import sum_signals

        filepath = Path(f"{self.output_directory}/reflections/summation") / f'{filepath}_summation.wav'

        if os.path.isfile(filepath):
            return AudioSegment.from_wav(filepath)

        if verbose:
            return self.write_file(sum_signals(rf, rv), filepath.__str__())
        return sum_signals(rf, rv)

    def generate_signal_noise(self, x, filepath, verbose):
        """Add white noise to the signal"""
        import os
        from pathlib import Path
        from pydub import AudioSegment
        from transforms import adjust_signal_to_noise

        filepath = Path(f"{self.output_directory}/reflections/noise") / f'{filepath}_noise.wav'

        if os.path.isfile(filepath):
            return AudioSegment.from_wav(filepath)

        if verbose:
            return self.write_file(adjust_signal_to_noise(x, -60), filepath.__str__())
        return adjust_signal_to_noise(x, -60)

    def generate_signal_trimmed(self, x, duration, filepath):
        """Trim the signal to the specified duration at the specified offset."""
        import os
        from pathlib import Path
        from pydub import AudioSegment

        filepath = Path(f"{self.output_directory}/reflections/samples") / f'{filepath}.wav'

        if os.path.isfile(filepath):
            return AudioSegment.from_wav(filepath)

        return self.write_file(x[:duration * 1000], filepath.__str__())

    def generate_room_impulse(self, zenith, azimuth, reflection_count, zeniths, azimuths, delays, amplitudes, amplitude,
                              delay,
                              time, duration, filepath):
        """
        Generate the room impulse response by applying an HRTF, reflections, and reverberation to a 'click'
        [1., 0., ..., 0.]
        """
        import os
        from pathlib import Path
        from pydub import AudioSegment
        import numpy as np
        from transforms import apply_hrtf, mix_reflections, apply_reverberation, sum_signals
        from figures import plot_wave
        from utils import audiosegment_to_array, array_to_audiosegment

        f = Path(f"{self.output_directory}/reflections/rir") / f'{filepath}_rir.wav'

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

            self.write_file(summation, f.__str__())

        f = Path(f"{self.output_directory}/reflections/rir") / f'{filepath}_rir.png'

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
            self.write_file(plt, f.__str__())
            plt.close()

    def generate_plot_sample(self, zenith, azimuth, zeniths, azimuths, delays, amplitudes, amplitude, delay, time,
                             filepath):
        """Plot the sample."""
        import os
        from pathlib import Path
        from figures import plot_sample

        filepath = Path(f"{self.output_directory}/reflections/samples") / f'{filepath}_sample.png'

        if not os.path.isfile(filepath):
            args = dict(
                suptitle='Sample',
                title=f'{len(zeniths)} reflections'
            )

            plt = plot_sample(zenith, azimuth, zeniths, azimuths, delays, amplitudes, amplitude, delay, time, **args)
            self.write_file(plt, filepath.__str__())
            plt.close()

    def generate_plot_signal(self, x, filepath):
        """Plot the signal."""
        import os
        from pathlib import Path
        import numpy as np
        from figures import plot_wave
        from utils import split_channels

        filepath = Path(f"{self.output_directory}/reflections/raw") / f'{filepath}_raw_wave.png'

        if not os.path.isfile(filepath):
            t = np.linspace(0, int(np.ceil(x.duration_seconds)), int(x.frame_count()))

            left, right = split_channels(x)

            x = np.array([left, right]).transpose()

            args = dict(
                t=[t, t],
                suptitle='Raw direct signal',
                title=['Left channel', 'Right channel'],
                xlabel=['Time, s', 'Time, s'],
                ylabel=['Amplitude', 'Amplitude']
            )

            plt = plot_wave(x, **args)
            self.write_file(plt, filepath.__str__())
            plt.close()

    def generate_plot_hrtf(self, x, filepath):
        """Plot the HRTF"""
        import os
        from pathlib import Path
        import numpy as np
        from figures import plot_wave
        from utils import split_channels

        filepath = Path(f"{self.output_directory}/reflections/hrtf") / f'{filepath}_hrtf_wave.png'

        if not os.path.isfile(filepath):
            t = np.linspace(0, int(np.ceil(x.duration_seconds)), int(x.frame_count()))

            left, right = split_channels(x)
            x = np.array([left, right]).transpose()

            args = dict(
                t=[t, t],
                suptitle='HRTF applied direct signal',
                title=['Left channel', 'Right channel'],
                xlabel=['Time, s', 'Time, s'],
                ylabel=['Amplitude', 'Amplitude']
            )

            plt = plot_wave(x, **args)
            self.write_file(plt, filepath.__str__())
            plt.close()

    def generate_plot_reflections(self, x, filepath):
        """Plot the reflections."""
        import os
        from pathlib import Path
        import numpy as np
        from figures import plot_wave
        from utils import split_channels

        filepath = Path(f"{self.output_directory}/reflections/reflections") / f'{filepath}_reflections_wave.png'

        if not os.path.isfile(filepath):
            t = np.linspace(0, int(np.ceil(x.duration_seconds)), int(x.frame_count()))

            left, right = split_channels(x)
            x = np.array([left, right]).transpose()

            args = dict(
                t=[t, t],
                suptitle='Reflections applied direct signal',
                title=['Left channel', 'Right channel'],
                xlabel=['Time, s', 'Time, s'],
                ylabel=['Amplitude', 'Amplitude']
            )

            plt = plot_wave(x, **args)
            self.write_file(plt, filepath.__str__())
            plt.close()

    def generate_plot_reverberation(self, x, filepath):
        """Plot the reverberation."""
        import os
        from pathlib import Path
        import numpy as np
        from figures import plot_wave
        from utils import split_channels

        filepath = Path(f"{self.output_directory}/reflections/reverberations") / f'{filepath}_reverberation_wave.png'

        if not os.path.isfile(filepath):
            t = np.linspace(0, int(np.ceil(x.duration_seconds)), int(x.frame_count()))

            left, right = split_channels(x)

            x = np.array([left, right]).transpose()

            args = dict(
                t=[t, t],
                suptitle='Reverberation applied direct signal',
                title=['Left channel', 'Right channel'],
                xlabel=['Time, s', 'Time, s'],
                ylabel=['Amplitude', 'Amplitude']
            )

            plt = plot_wave(x, **args)
            self.write_file(plt, filepath.__str__())
            plt.close()

    def generate_plot_summation(self, x, filepath):
        """Plot the summed signal."""
        import os
        from pathlib import Path
        import numpy as np
        from figures import plot_wave
        from utils import split_channels

        filepath = Path(f"{self.output_directory}/reflections/summation") / f'{filepath}_summation_wave.png'

        if not os.path.isfile(filepath):
            t = np.linspace(0, int(np.ceil(x.duration_seconds)), int(x.frame_count()))

            left, right = split_channels(x)
            x = np.array([left, right]).transpose()

            args = dict(
                t=[t, t],
                suptitle='Reflections and reverberation applied direct signal',
                title=['Left channel', 'Right channel'],
                xlabel=['Time, s', 'Time, s'],
                ylabel=['Amplitude', 'Amplitude']
            )

            plt = plot_wave(x, **args)
            self.write_file(plt, filepath.__str__())
            plt.close()

    def generate_plot_noise(self, x, filepath):
        """Plot the noise added signal."""
        import os
        from pathlib import Path
        import numpy as np
        from figures import plot_wave
        from utils import split_channels

        filepath = Path(f"{self.output_directory}/reflections/noise") / f'{filepath}_noise_wave.png'

        if not os.path.isfile(filepath):
            t = np.linspace(0, int(np.ceil(x.duration_seconds)), int(x.frame_count()))

            left, right = split_channels(x)

            x = np.array([left, right]).transpose()

            args = dict(
                t=[t, t],
                suptitle='-30 dB of white noise added to direct signal',
                title=['Left channel', 'Right channel'],
                xlabel=['Time, s', 'Time, s'],
                ylabel=['Amplitude', 'Amplitude']
            )

            plt = plot_wave(x, **args)
            self.write_file(plt, filepath.__str__())
            plt.close()

    def generate_plot_trimmed(self, x, filepath):
        """Plot the trimmed sample."""
        import os
        from pathlib import Path
        import numpy as np
        from figures import plot_wave
        from utils import split_channels

        filepath = Path(f"{self.output_directory}/reflections/samples") / f'{filepath}_wave.png'

        if not os.path.isfile(filepath):
            duration = int(np.ceil(x.duration_seconds))
            t = np.linspace(0, duration, int(x.frame_count()))

            left, right = split_channels(x)
            x = np.array([left, right]).transpose()

            args = dict(
                t=[t, t],
                suptitle=f'Direct signal trimmed to {duration} s',
                title=['Left channel', 'Right channel'],
                xlabel=['Time, s', 'Time, s'],
                ylabel=['Amplitude', 'Amplitude']
            )

            plt = plot_wave(x, **args)
            self.write_file(plt, filepath.__str__())
            plt.close()

    def generate_plot_ceptstrum(self, x, filepath):
        """Plot the cepstrum."""
        import os
        from pathlib import Path
        from figures import plot_cepstrum
        from utils import audiosegment_to_array

        offset = 1024
        window_length = offset * 64 * 2

        fs = x.frame_rate

        channels = x.split_to_mono()
        left = audiosegment_to_array(channels[0])
        right = audiosegment_to_array(channels[1])

        f = Path(f"{self.output_directory}/reflections/cepstrums") / f'{filepath}_left_cepstrum.png'

        if not os.path.isfile(f):
            args = dict(
                suptitle='Amplitude cepstrum',
                title='Left channel',
                xlabel='Quefrency, ms',
                ylabel='Amplitude'
            )

            plt = plot_cepstrum(left, fs, offset, window_length, **args)
            self.write_file(plt, f.__str__())
            plt.close()

        f = Path(f"{self.output_directory}/reflections/cepstrums") / f'{filepath}_right_cepstrum.png'

        if not os.path.isfile(f):
            args = dict(
                suptitle='Amplitude cepstrum',
                title='Right channel',
                xlabel='Quefrency, ms',
                ylabel='Amplitude'
            )

            plt = plot_cepstrum(right, fs, offset, window_length, **args)
            self.write_file(plt, f.__str__())
            plt.close()

        f = Path(f"{self.output_directory}/reflections/cepstrums") / f'{filepath}_left_cepstrum_20.png'

        if not os.path.isfile(f):
            args = dict(
                suptitle='Amplitude cepstrum, 0-20 ms',
                title='Left channel',
                xlabel='Quefrency, ms',
                ylabel='Amplitude',
                xlim=(0, 20)
            )

            plt = plot_cepstrum(left, fs, offset, window_length, **args)
            self.write_file(plt, f.__str__())
            plt.close()

        f = Path(f"{self.output_directory}/reflections/cepstrums") / f'{filepath}_right_cepstrum_20.png'

        if not os.path.isfile(f):
            args = dict(
                suptitle='Amplitude cepstrum, 0-20 ms',
                title='Right channel',
                xlabel='Quefrency, ms',
                ylabel='Amplitude',
                xlim=(0, 20)
            )

            plt = plot_cepstrum(right, fs, offset, window_length, **args)
            self.write_file(plt, f.__str__())
            plt.close()

    def generate_plot_cepbimo(self, x, filepath):
        """Plot the Cepbimo."""
        import os
        from pathlib import Path
        import numpy as np
        from figures import plot_waves, plot_binaural_activity_map_3d, plot_binaural_activity_map_2d

        f = Path(f"{self.output_directory}/reflections/cepbimo") / f'{filepath}_cepbimo_wave.png'

        if not os.path.isfile(f):
            X = np.array([x.Xd / 200, x.XR / 200]).transpose()
            t = np.linspace(-1., 1., (x.lag * 2) + 1)

            args = dict(
                t=t,
                suptitle=f'Cepbimo',
                title='Interaural cross-correlation',
                xlabel='ITD, ms',
                ylabel='Correlation',
                legend=True,
                legend_labels=['2nd-layer cross-correlation', 'Cross-correlation']
            )

            plt = plot_waves(X, **args)

            self.write_file(plt, f.__str__())
            plt.close()

        f = Path(f"{self.output_directory}/reflections/cepbimo") / f'{filepath}_binaural_activity_map_3d.png'

        if not os.path.isfile(f):
            plt = plot_binaural_activity_map_3d(x.z)
            self.write_file(plt, f.__str__())
            plt.close()

        f = Path(f"{self.output_directory}/reflections/cepbimo") / f'{filepath}_binaural_activity_map_2d.png'

        if not os.path.isfile(f):
            plt = plot_binaural_activity_map_2d(x.z)
            self.write_file(plt, f.__str__())
            plt.close()


def demo_data_generator():
    """Demonstrate DataGenerator usage."""
    from RNG import RNG
    rng = RNG()
    dg, ig = DataGenerator(1, 'data/sample', rng).generate(verbose=True)
    print(dg, ig)


if __name__ == '__main__':
    demo_data_generator()
