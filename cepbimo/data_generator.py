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

    def __init__(self, sample_count, output_directory, rng=None, fs=24000, verbose=False):
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
        self.recipe = None
        self.chunk_size = 10
        self.fs = fs
        self.verbose = verbose

        path = Path(output_directory)
        if not os.path.isdir(path):
            os.mkdir(path)
        path = path / 'reflections'
        if not os.path.isdir(path):
            os.mkdir(path)
        self.output_directory = path.__str__()

    def get_reflections(self, count):
        rng = self.rng
        """Generate the specified number of random reflections."""
        amplitudes = [rng.get_amplitude() for _ in range(count)]
        delays = [rng.get_delay() for _ in range(count)]
        zeniths = [rng.get_zenith() for _ in range(count)]
        azimuths = [rng.get_azimuth(zenith=zeniths[i]) for i in range(count)]
        return zeniths, azimuths, amplitudes, delays

    def generate_ingredients_list(self):
        import pandas as pd
        from pathlib import Path

        filepath = Path(self.output_directory) / 'ingredients.json'

        rng = self.rng

        df = pd.DataFrame(dict(
            seed=rng.seed,
            duration=rng.duration,
            delay_limits=[rng.delay_limits],
            time_limits=[rng.time_limits],
            reflection_limits=[rng.reflection_limits],
            zenith_limits=[rng.zenith_limits],
            azimuth_limits=[rng.azimuth_limits],
            sample_count=self.sample_count
        ))

        df.to_json(filepath, orient='records', lines=False)
        return df

    def generate_sample_recipe(self):
        from data_loader import list_anechoic_lengths
        import hashlib

        lengths = list_anechoic_lengths()
        rng = self.rng

        composer = rng.get_composer()
        part_count = rng.get_part_count(composer=composer)
        parts = rng.get_parts(composer=composer, part_count=part_count)
        offset = rng.get_offset(lengths[composer])
        duration = rng.duration
        zenith = rng.get_zenith()
        azimuth = rng.get_azimuth(zenith=zenith)
        reverb_time = rng.get_time()
        reverb_delay = rng.get_delay()
        reverb_amplitude = rng.rng.uniform(0, 0.05)
        reflection_count = rng.get_reflection_count()
        reflection_zenith, reflection_azimuth, reflection_amplitude, reflection_delay = self.get_reflections(
            reflection_count)
        s = f'{part_count:02d}' \
            f'{"".join(parts)}' \
            f'{offset}' \
            f'{zenith}' \
            f'{azimuth}' \
            f'{reflection_count}' \
            f'{"".join(str(x) for x in reflection_zenith)}' \
            f'{"".join(str(x) for x in reflection_azimuth)}' \
            f'{"".join(str(x) for x in reflection_amplitude)}' \
            f'{"".join(str(x) for x in reflection_delay)}' \
            f'{reverb_amplitude}' \
            f'{reverb_delay}' \
            f'{reverb_time}'

        s = str(int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8)
        filepath = f'{composer}_{s}'

        return dict(
            composer=composer,
            part_count=part_count,
            parts=parts,
            zenith=zenith,
            azimuth=azimuth,
            offset=offset,
            duration=duration,
            reverb_time=reverb_time,
            reverb_delay=reverb_delay,
            reverb_amplitude=reverb_amplitude,
            reflection_count=reflection_count,
            reflection_amplitude=reflection_amplitude,
            reflection_delay=reflection_delay,
            reflection_zenith=reflection_zenith,
            reflection_azimuth=reflection_azimuth,
            filepath=filepath,
            name=''
        )

    def generate_recipe(self):
        from pathlib import Path
        import pandas as pd
        import dask

        filepath = Path(self.output_directory) / 'recipe.json'

        sample_count = self.sample_count

        lazy_results = []
        for i in range(sample_count):
            lazy_result = dask.delayed(self.generate_sample_recipe)()
            lazy_results.append(lazy_result)

        df = pd.DataFrame(dask.compute(*lazy_results))

        df.to_json(filepath, orient='records', lines=True)
        return df

    def generate(self):
        """Start generator."""
        import dask.dataframe as dd
        import numpy as np

        ingredients = self.generate_ingredients_list()
        recipe = self.generate_recipe()
        ddf = dd.from_pandas(recipe, npartitions=int(np.sqrt(self.sample_count)))
        s = ddf.map_partitions(self.generate_samples, meta=ddf)
        s.compute()
        return ingredients, recipe

    def generate_samples(self, recipe):
        return recipe.apply(self.generate_sample, axis=1)

    def generate_sample(self, recipe):
        from transforms import mix_parts, apply_hrtf, mix_reflections, apply_reverberation, sum_signals, \
            adjust_signal_to_noise
        from figures import plot_sample, plot_wave, plot_waves, plot_binaural_activity_map_3d, \
            plot_binaural_activity_map_2d
        import numpy as np
        from utils import split_channels
        from spectrum import cepstrum
        from Cepbimo import Cepbimo
        print(f'Generating sample: {recipe["filepath"]}')

        print(f'\tMixing parts: {recipe["filepath"]}')
        signal = mix_parts(recipe['parts'], recipe['offset'], recipe['duration'])

        print(f'\tApplying HRTF: {recipe["filepath"]}')
        hrtf = apply_hrtf(signal, recipe['zenith'], recipe['azimuth'])

        print(f'\tApplying reflections: {recipe["filepath"]}')
        reflections = mix_reflections(hrtf, recipe['reflection_count'], recipe['reflection_amplitude'],
                                      recipe['reflection_delay'], recipe['reflection_zenith'],
                                      recipe['reflection_azimuth'])

        print(f'\tApplying reverberation: {recipe["filepath"]}')
        reverberation = apply_reverberation(hrtf, recipe['reverb_amplitude'], recipe['reverb_delay'],
                                            recipe['reverb_time'])

        print(f'\tSumming signals: {recipe["filepath"]}')
        summation = sum_signals(reflections, reverberation)

        print(f'\tAdjusting signal-to-noise ratio: {recipe["filepath"]}')
        noise = adjust_signal_to_noise(summation, -60)

        print(f'\tTrimming sample: {recipe["filepath"]}')
        sample = noise[:recipe['duration'] * 1000]
        self.write(sample, 'samples', f'{recipe["filepath"]}.wav')

        if self.verbose:
            self.write(signal, 'raw', f'{recipe["filepath"]}_raw.wav')
            self.write(hrtf, 'hrtf', f'{recipe["filepath"]}_hrtf.wav')
            self.write(reflections, 'reflections', f'{recipe["filepath"]}_reflections.wav')
            self.write(reverberation, 'reverberation', f'{recipe["filepath"]}_reverberation.wav')
            self.write(summation, 'summation', f'{recipe["filepath"]}_summation.wav')
            self.write(noise, 'noise', f'{recipe["filepath"]}_noise.wav')

            print(f'\tGenerating figures: {recipe["filepath"]}')

            args = dict(suptitle='Sample', title=f'{recipe["part_count"]}')
            plt = plot_sample(recipe['zenith'], recipe['azimuth'], recipe['reflection_zenith'],
                              recipe['reflection_azimuth'], recipe['reflection_delay'], recipe['reflection_amplitude'],
                              recipe['reverb_amplitude'], recipe['reverb_delay'], recipe['reverb_time'], **args)
            self.write(plt, 'samples', f'{recipe["filepath"]}.png')
            plt.ioff()
            plt.close()

            t = np.linspace(0, int(np.ceil(signal.duration_seconds)), int(signal.frame_count()))

            x = np.array(split_channels(signal)).transpose()
            args = dict(suptitle='Raw direct signal', title=['Left channel', 'Right channel'],
                        xlabel=['Time, s', 'Time, s'], ylabel=['Amplitude', 'Amplitude'], t=[t, t])
            plt = plot_wave(x, **args)
            self.write(plt, 'raw', f'{recipe["filepath"]}.png')
            plt.close()

            t = np.linspace(0, int(np.ceil(hrtf.duration_seconds)), int(hrtf.frame_count()))
            x = np.array(split_channels(hrtf)).transpose()
            args['suptitle'] = 'HRTF applied to direct signal'
            args['t'] = [t, t]
            plt = plot_wave(x, **args)
            self.write(plt, 'hrtf', f'{recipe["filepath"]}_hrtf.png')
            plt.close()

            t = np.linspace(0, int(np.ceil(reflections.duration_seconds)), int(reflections.frame_count()))
            x = np.array(split_channels(reflections)).transpose()
            args['t'] = [t, t]
            args['suptitle'] = 'Reflections applied to direct signal'
            plt = plot_wave(x, **args)
            self.write(plt, 'reflections', f'{recipe["filepath"]}_reflections.png')
            plt.close()

            t = np.linspace(0, int(np.ceil(reverberation.duration_seconds)), int(reverberation.frame_count()))
            x = np.array(split_channels(reverberation)).transpose()
            args['t'] = [t, t]
            args['suptitle'] = 'Reverberation applied to direct signal'
            plt = plot_wave(x, **args)
            self.write(plt, 'reverberation', f'{recipe["filepath"]}_reverberation.png')
            plt.close()

            t = np.linspace(0, int(np.ceil(summation.duration_seconds)), int(summation.frame_count()))
            x = np.array(split_channels(summation)).transpose()
            args['t'] = [t, t]
            args['suptitle'] = 'Reflections and reverberation applied to direct signal'
            plt = plot_wave(x, **args)
            self.write(plt, 'summation', f'{recipe["filepath"]}_summation.png')
            plt.close()

            t = np.linspace(0, int(np.ceil(noise.duration_seconds)), int(noise.frame_count()))
            x = np.array(split_channels(noise)).transpose()
            args['t'] = [t, t]
            args['suptitle'] = '-60dB of white noise added to direct signal'
            plt = plot_wave(x, **args)
            self.write(plt, 'noise', f'{recipe["filepath"]}_noise.png')
            plt.close()

            t = np.linspace(0, int(np.ceil(sample.duration_seconds)), int(sample.frame_count()))
            left, right = split_channels(sample)
            x = np.array([left, right]).transpose()
            args['t'] = [t, t]
            args['suptitle'] = 'Direct signal'
            plt = plot_wave(x, **args)
            self.write(plt, 'samples', f'{recipe["filepath"]}.png')
            plt.close()

            args = dict()
            offset = 1024
            window_length = offset * 64 * 2
            fs = sample.frame_rate
            Cl, ql = cepstrum(left, fs, offset, window_length)
            Cr, qr = cepstrum(right, fs, offset, window_length)
            x = np.array([Cl, Cr]).transpose()
            args['suptitle'] = 'Amplitude cepstrum'
            args['title'] = ['Left channel', 'Right channel']
            args['xlabel'] = ['Quefrency, ms', 'Quefrency, ms']
            args['ylabel'] = ['Amplitude', 'Amplitude']
            args['t'] = [ql, qr]
            plt = plot_wave(x, **args)
            self.write(plt, 'cepstrums', f'{recipe["filepath"]}_cepstrum.png')
            plt.close()

            args['suptitle'] = 'Amplitude cepstrum, 0-20 ms'
            args['xlim'] = [(0, 0.02), (0, 0.02)]
            plt = plot_wave(x, **args)
            self.write(plt, 'cepstrums', f'{recipe["filepath"]}_cepstrum20.png')
            plt.close()

            args = dict()
            cepbimo = Cepbimo(sample)
            t = np.linspace(-1, 1, (cepbimo.lag * 2) + 1)
            x = np.array([cepbimo.Xd / 200, cepbimo.XR / 200]).transpose()
            args['t'] = t
            args['suptitle'] = 'Cepbimo'
            args['title'] = 'Interaural cross-correlation (ITD)'
            args['xlabel'] = 'ITD, ms'
            args['ylabel'] = 'Correlation'
            args['legend'] = True
            args['legend_labels'] = ['2nd-layer cross-correlation', 'Cross-correlation']
            plt = plot_waves(x, **args)
            self.write(plt, 'cepbimo', f'{recipe["filepath"]}_cepbimo.png')
            plt.close()

            plt = plot_binaural_activity_map_2d(cepbimo.z)
            self.write(plt, 'cepbimo', f'{recipe["filepath"]}_binaural_activity_map_2d.png')
            plt.close()

            plt = plot_binaural_activity_map_3d(cepbimo.z)
            self.write(plt, 'cepbimo', f'{recipe["filepath"]}_binaural_activity_map_3d.png')
            plt.close()

            self.generate_room_impulse(recipe['zenith'], recipe['azimuth'], recipe['reflection_count'],
                                       recipe['reflection_zenith'], recipe['reflection_azimuth'],
                                       recipe['reflection_delay'], recipe['reflection_amplitude'],
                                       recipe['reverb_amplitude'], recipe['reverb_delay'], recipe['reverb_time'],
                                       recipe['duration'], recipe['filepath'])

    def write(self, file, directory, filename):
        from pathlib import Path
        import os

        path = Path(self.output_directory) / directory
        if not os.path.isdir(path):
            os.mkdir(path)

        path = path / filename
        if not os.path.isfile(path):
            print(f'\tWriting file: {filename}')
            file_format = path.suffix.strip('.')
            if file_format == 'wav':
                if self.fs != file.frame_rate:
                    file = file.set_frame_rate(self.fs)
                file.export(path, format=file_format)
            if file_format == 'png':
                file.savefig(path, format=file_format)

        return path

    def read_recipe(self):
        """Read the recipe from a file."""
        from pathlib import Path
        import pandas as pd

        path = Path(self.output_directory) / 'recipe.json'

        chunks = []

        with pd.read_json(path, orient='records', lines=True, chunksize=self.chunk_size) as reader:
            for chunk in reader:
                chunks.append(chunk)

        return pd.concat(chunks)

    def read_ingredients(self):
        import pandas as pd
        from pathlib import Path

        path = Path(self.output_directory) / 'ingredients.json'

        return pd.read_json(path, orient='records')

    def generate_room_impulse(self, zenith, azimuth, reflection_count, zeniths, azimuths, delays, amplitudes, amplitude,
                              delay,
                              time, duration, filepath):
        """
        Generate the room impulse response by applying an HRTF, reflections, and reverberation to a 'click'
        [1., 0., ..., 0.]
        """
        import numpy as np
        from transforms import apply_hrtf, mix_reflections, apply_reverberation, sum_signals
        from figures import plot_wave
        from utils import array_to_audiosegment, split_channels

        fs = 48000
        click = np.ones(1)
        click = np.concatenate((click, np.zeros(fs * duration - 1)))

        signal = array_to_audiosegment(click, fs)
        signal = apply_hrtf(signal, zenith, azimuth)

        reflections = mix_reflections(signal, reflection_count, amplitudes, delays, zeniths, azimuths)
        reverberation = apply_reverberation(signal, amplitude, delay, time)

        summation = sum_signals(reflections, reverberation)

        self.write(summation, 'rir', f'{filepath}_rir.wav')

        t = np.linspace(0, int(np.ceil(summation.duration_seconds)), int(summation.frame_count()))
        x = np.array(split_channels(summation)).transpose()
        args = dict(
            t=[t, t],
            suptitle='RIR, click=[1., 0., ..., 0.]',
            title=['Left channel', 'Right channel'],
            xlabel=['Time, s', 'Time, s'],
            ylabel=['Amplitude', 'Amplitude']
        )
        plt = plot_wave(x, **args)
        self.write(plt, 'rir', f'{filepath}_rir.png')
        plt.close()


def demo_data_generator():
    """Demonstrate DataGenerator usage."""
    from RNG import RNG
    rng = RNG()
    dg = DataGenerator(1, 'data/sample', rng, verbose=False)
    dg.generate()


if __name__ == '__main__':
    demo_data_generator()
