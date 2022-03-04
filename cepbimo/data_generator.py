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
        import hashlib
        from pathlib import Path

        if rng is None:
            from RNG import RNG
            rng = RNG()
        self.rng = rng
        self.sample_count = sample_count
        self.recipe = None
        self.chunk_size = 50
        self.fs = fs
        self.verbose = verbose

        path = Path(output_directory)
        if not os.path.isdir(path):
            os.mkdir(path)
        path = path / 'reflections'
        if not os.path.isdir(path):
            os.mkdir(path)
        self.output_directory = path.__str__()

        s = f'{rng.seed}{rng.duration}{rng.delay_limits}{rng.time_limits}{rng.reflection_limits}{rng.zenith_limits}' \
            f'{rng.azimuth_limits}{sample_count}{verbose}'
        self.hash = str(int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8)

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
        import os

        print('Generating ingredients list')

        filepath = Path(self.output_directory) / f'ingredients_{self.hash}.json'

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

        if not os.path.isfile(filepath):
            df.to_json(filepath, orient='records', lines=True)
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

        s = str(int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8)
        filepath = f'{composer}_{s}'
        print(f'Generating recipe {filepath}\n')

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

    def generate_recipe(self, count):
        import pandas as pd
        import dask
        print('Generating recipes')

        lazy_results = []
        for i in range(count):
            lazy_result = dask.delayed(self.generate_sample_recipe)()
            lazy_results.append(lazy_result)

        df = pd.DataFrame(dask.compute(*lazy_results))

        return df

    def generate(self):
        """Start generator."""
        import pandas as pd
        import dask.dataframe as dd
        import dask
        from pathlib import Path
        import numpy as np
        import os

        print('Data generator started')

        dfi = self.generate_ingredients_list()

        filepath = Path(self.output_directory) / f'recipe_{self.hash}'

        if not os.path.isdir(filepath):
            sample_count = self.sample_count
            chunk_size = self.chunk_size

            batches = int(np.ceil(sample_count / chunk_size))
            results = []
            print('Generating recipe batches')
            for i in range(batches):
                if (i + 1) * chunk_size > sample_count:
                    chunk = sample_count % chunk_size
                else:
                    chunk = chunk_size
                result = dask.delayed(self.generate_recipe)(chunk)
                results.append(result)

            df = pd.concat(dask.compute(*results))
            ddf = dd.from_pandas(df, chunksize=chunk_size)
            print('Writing recipes')
            ddf.to_parquet(filepath, engine='pyarrow')
            print('Generating samples')
            s = ddf.map_partitions(self.generate_samples, meta=ddf)
            s.compute()
            df = ddf.compute()
        else:
            df = self.read_recipe()
        return dfi, df

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
        from utils import generate_impulse
        from pathlib import Path
        from pydub import AudioSegment
        import os

        impulse = generate_impulse(recipe['duration'])

        print(f'Generating sample: {recipe["filepath"]}\n')

        print(f'\tMixing parts: {recipe["filepath"]}')
        filepath = Path(f"raw/{recipe['filepath']}_raw.wav")
        if os.path.isfile(filepath):
            signal = AudioSegment.from_wav(filepath)
        else:
            signal = mix_parts(recipe['parts'], recipe['offset'], recipe['duration'])

        print(f'\tApplying HRTF: {recipe["filepath"]}')
        filepath = Path(f"hrtf/{recipe['filepath']}_hrtf.wav")
        if os.path.isfile(filepath):
            hrtf = AudioSegment.from_wav(filepath)
        else:
            hrtf = apply_hrtf(signal, recipe['zenith'], recipe['azimuth'])

        print(f'\tApplying reflections: {recipe["filepath"]}')
        filepath = Path(f"reflections/{recipe['filepath']}_reflections.wav")
        if os.path.isfile(filepath):
            reflections = AudioSegment.from_wav(filepath)
        else:
            reflections = mix_reflections(hrtf, recipe['reflection_count'], recipe['reflection_amplitude'],
                                          recipe['reflection_delay'], recipe['reflection_zenith'],
                                          recipe['reflection_azimuth'])

        print(f'\tApplying reverberation: {recipe["filepath"]}')
        filepath = Path(f"reverberation/{recipe['filepath']}_reverberation.wav")
        if os.path.isfile(filepath):
            reverberation = AudioSegment.from_wav(filepath)
        else:
            reverberation = apply_reverberation(hrtf, recipe['reverb_amplitude'], recipe['reverb_delay'],
                                                recipe['reverb_time'])

        print(f'\tSumming signals: {recipe["filepath"]}')
        filepath = Path(f"summation/{recipe['filepath']}_summation.wav")
        if os.path.isfile(filepath):
            summation = AudioSegment.from_wav(filepath)
        else:
            summation = sum_signals(reflections, reverberation)

        print(f'\tAdjusting signal-to-noise ratio: {recipe["filepath"]}')
        filepath = Path(f"noise/{recipe['filepath']}_noise.wav")
        if os.path.isfile(filepath):
            noise = AudioSegment.from_wav(filepath)
        else:
            noise = adjust_signal_to_noise(summation, -60)

        print(f'\tTrimming sample: {recipe["filepath"]}')
        filepath = Path(f"samples/{recipe['filepath']}.wav")
        if os.path.isfile(filepath):
            sample = AudioSegment.from_wav(filepath)
        else:
            sample = noise[:recipe['duration'] * 1000]
        self.write(sample, 'samples', f'{recipe["filepath"]}.wav')

        if self.verbose:
            impulse_hrtf = apply_hrtf(impulse, recipe['zenith'], recipe['azimuth'])
            impulse_reflections = mix_reflections(impulse_hrtf, recipe['reflection_count'],
                                                  recipe['reflection_amplitude'], recipe['reflection_delay'],
                                                  recipe['reflection_zenith'], recipe['reflection_azimuth'])
            impulse_reverberation = apply_reverberation(impulse_hrtf, recipe['reverb_amplitude'],
                                                        recipe['reverb_delay'], recipe['reverb_time'])
            impulse_summation = sum_signals(impulse_reflections, impulse_reverberation)
            impulse_noise = adjust_signal_to_noise(impulse_summation, -60)
            impulse_sample = impulse_noise[:recipe['duration'] * 1000]

            self.write(signal, 'raw', f'{recipe["filepath"]}_raw.wav')
            self.write(hrtf, 'hrtf', f'{recipe["filepath"]}_hrtf.wav')
            self.write(reflections, 'reflections', f'{recipe["filepath"]}_reflections.wav')
            self.write(reverberation, 'reverberation', f'{recipe["filepath"]}_reverberation.wav')
            self.write(summation, 'summation', f'{recipe["filepath"]}_summation.wav')
            self.write(noise, 'noise', f'{recipe["filepath"]}_noise.wav')
            self.write(impulse_sample, 'rir', f'{recipe["filepath"]}_rir.wav')

            print(f'\tGenerating figures: {recipe["filepath"]}')

            print(f'\tGenerating figure: samples/{recipe["filepath"]}_sample.png')
            file_directory = Path(f'{self.output_directory}')

            print(f'\tGenerating figure: rir/{recipe["filepath"]}_rir.png')
            t = np.linspace(0, int(np.ceil(signal.duration_seconds)), int(signal.frame_count()))
            x = np.array(split_channels(impulse_sample)).transpose()
            args = dict(suptitle='Room impulse response', title=['Left channel', 'Right channel'],
                        xlabel=['Time, s', 'Time, s'], ylabel=['Amplitude', 'Amplitude'], t=[t, t])
            plot_wave(x, filepath=f'{(file_directory / "rir" / recipe["filepath"]).__str__()}'
                                  f'_rir.png', **args)

            args = dict(suptitle='Sample', title=f'{recipe["part_count"]}')
            plot_sample(recipe['zenith'], recipe['azimuth'], recipe['reflection_zenith'],
                        recipe['reflection_azimuth'], recipe['reflection_delay'], recipe['reflection_amplitude'],
                        recipe['reverb_amplitude'], recipe['reverb_delay'], recipe['reverb_time'],
                        filepath=f'{(file_directory / "samples" / recipe["filepath"]).__str__()}'
                                 f'_sample.png', **args)

            print(f'\tGenerating figure: raw/{recipe["filepath"]}_raw.png')
            t = np.linspace(0, int(np.ceil(signal.duration_seconds)), int(signal.frame_count()))
            x = np.array(split_channels(signal)).transpose()
            args = dict(suptitle='Raw direct signal', title=['Left channel', 'Right channel'],
                        xlabel=['Time, s', 'Time, s'], ylabel=['Amplitude', 'Amplitude'], t=[t, t])
            plot_wave(x, filepath=f'{(file_directory / "raw" / recipe["filepath"]).__str__()}'
                                  f'_raw.png', **args)

            print(f'\tGenerating figure: hrtf/{recipe["filepath"]}_hrtf.png')
            t = np.linspace(0, int(np.ceil(hrtf.duration_seconds)), int(hrtf.frame_count()))
            x = np.array(split_channels(hrtf)).transpose()
            args['suptitle'] = 'HRTF applied to direct signal'
            args['t'] = [t, t]
            plot_wave(x, filepath=f'{(file_directory / "hrtf" / recipe["filepath"]).__str__()}'
                                  f'_hrtf.png', **args)

            print(f'\tGenerating figure: reflections/{recipe["filepath"]}_reflections.png')
            t = np.linspace(0, int(np.ceil(reflections.duration_seconds)), int(reflections.frame_count()))
            x = np.array(split_channels(reflections)).transpose()
            args['t'] = [t, t]
            args['suptitle'] = 'Reflections applied to direct signal'
            plot_wave(x, filepath=f'{(file_directory / "reflections" / recipe["filepath"]).__str__()}'
                                  f'_reflections.png', **args)

            print(f'\tGenerating figure: reverberation/{recipe["filepath"]}_reverberation.png')
            t = np.linspace(0, int(np.ceil(reverberation.duration_seconds)), int(reverberation.frame_count()))
            x = np.array(split_channels(reverberation)).transpose()
            args['t'] = [t, t]
            args['suptitle'] = 'Reverberation applied to direct signal'
            plot_wave(x,
                      filepath=f'{(file_directory / "reverberation" / recipe["filepath"]).__str__()}'
                               f'_reverberation.png', **args)

            print(f'\tGenerating figure: summation/{recipe["filepath"]}_summation.png')
            t = np.linspace(0, int(np.ceil(summation.duration_seconds)), int(summation.frame_count()))
            x = np.array(split_channels(summation)).transpose()
            args['t'] = [t, t]
            args['suptitle'] = 'Reflections and reverberation applied to direct signal'
            plot_wave(x, filepath=f'{(file_directory / "summation" / recipe["filepath"]).__str__()}'
                                  f'_summation.png',
                      **args)

            print(f'\tGenerating figure: noise/{recipe["filepath"]}_noise.png')
            t = np.linspace(0, int(np.ceil(noise.duration_seconds)), int(noise.frame_count()))
            x = np.array(split_channels(noise)).transpose()
            args['t'] = [t, t]
            args['suptitle'] = '-60dB of white noise added to direct signal'
            plot_wave(x, filepath=f'{(file_directory / "noise" / recipe["filepath"]).__str__()}'
                                  f'_noise.png', **args)

            print(f'\tGenerating figure: samples/{recipe["filepath"]}.png')
            t = np.linspace(0, int(np.ceil(sample.duration_seconds)), int(sample.frame_count()))
            left, right = split_channels(sample)
            x = np.array([left, right]).transpose()
            args['t'] = [t, t]
            args['suptitle'] = 'Direct signal'
            plot_wave(x, filepath=f'{(file_directory / "samples" / recipe["filepath"]).__str__()}'
                                  f'.png', **args)

            print(f'\tGenerating figure: cepstrums/{recipe["filepath"]}_cepstrum.png')
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
            plot_wave(x, filepath=f'{(file_directory / "cepstrum" / recipe["filepath"]).__str__()}'
                                  f'_cepstrum.png',
                      **args)

            print(f'\tGenerating figure: cepstrums/{recipe["filepath"]}_cepstrum20.png')
            args['suptitle'] = 'Amplitude cepstrum, 0-20 ms'
            args['xlim'] = [(0, 0.02), (0, 0.02)]
            plot_wave(x, filepath=f'{(file_directory / "cepstrum" / recipe["filepath"]).__str__()}'
                                  f'_cepstrum20.png',
                      **args)

            print(f'\tGenerating figure: cepbimo/{recipe["filepath"]}_cepbimo.png')
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
            plot_waves(x, filepath=f'{(file_directory / "cepbimo" / recipe["filepath"]).__str__()}'
                                   f'_cepbimo.png', **args)

            print(f'\tGenerating figure: cepbimo/{recipe["filepath"]}_binaural_activity_map2d.png')
            plot_binaural_activity_map_2d(cepbimo.z,
                                          filepath=f'{(file_directory / "cepbimo" / recipe["filepath"]).__str__()}'
                                                   f'_binaural_activity_map_2d.png')

            print(f'\tGenerating figure: cepbimo/{recipe["filepath"]}_binaural_activity_map3d.png')
            plot_binaural_activity_map_3d(cepbimo.z,
                                          filepath=f'{(file_directory / "cepbimo" / recipe["filepath"]).__str__()}'
                                                   f'_binaural_activity_map_3d.png')

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
                file.figure().clear()
                file.close()
                file.cla()
                file.clf()

        return path

    def read_recipe(self):
        """Read the recipe from a file."""
        from pathlib import Path
        import dask.dataframe as dd

        path = Path(self.output_directory) / f'recipe_{self.hash}'
        path = path.__str__()

        return dd.read_parquet(path, engine='pyarrow').compute()

    def read_ingredients(self):
        import pandas as pd
        from pathlib import Path

        path = Path(self.output_directory) / f'ingredients_{self.hash}.json'

        return pd.read_json(path, orient='records', lines=True)


def demo_data_generator():
    """Demonstrate DataGenerator usage."""
    from RNG import RNG
    rng = RNG()
    dg = DataGenerator(10, 'data/sample', rng, verbose=True)
    ingredients, recipe = dg.generate()
    print(ingredients, recipe)


if __name__ == '__main__':
    demo_data_generator()
