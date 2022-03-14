class DataPicker:

    def __init__(self):
        from data_generator.RNG import RNG
        from data_generator.data_generator import DataGenerator

        rng = RNG()
        seed = rng.seed
        duration = rng.duration
        delay_limits = rng.delay_limits
        time_limits = rng.time_limits
        reflection_limits = rng.reflection_limits
        zenith_limits = rng.zenith_limits
        azimuth_limits = rng.azimuth_limits
        sample_count = 1
        verbose = True

        dg = DataGenerator(sample_count, 'data/sample_data', rng=rng, verbose=verbose)
        ingredients, recipe = dg.generate()

        sample_count = 10
        verbose = False
        self.props = dict(
            seed=seed,
            duration=duration,
            delay_limits=delay_limits,
            time_limits=time_limits,
            reflection_limits=reflection_limits,
            zenith_limits=zenith_limits,
            azimuth_limits=azimuth_limits,
            sample_count=sample_count,
            verbose=verbose
        )
        self.sample_data = recipe
        self.sample_parameters = ingredients

        self.params = dict(
            seed=seed,
            duration=duration,
            delay_limits=delay_limits,
            time_limits=time_limits,
            reflection_limits=reflection_limits,
            zenith_limits=zenith_limits,
            azimuth_limits=azimuth_limits,
            sample_count=sample_count,
            verbose=verbose
        )

    def sample_viewer(self):
        from ipywidgets import widgets
        from utils.utils import play_audio
        from pydub import AudioSegment
        from pathlib import Path

        def format_html(title, text):
            return widgets.HTML(value=f'<h1>{title}</h1><p>{text}</p>')

        def format_audio(path):
            output = widgets.Output()
            audio = AudioSegment.from_wav(path)
            output.append_display_data(play_audio(audio))
            return output

        def format_image(path):
            with open(path, 'rb') as file:
                return widgets.Image(value=file.read(), format='png', layout=dict(max_width='30%'))

        df = self.sample_data.iloc[0]
        directory = Path('../data/sample_data/reflections/')
        filename = df["filepath"]

        raw_html = format_html('Raw signal', 'An anechoic signal.')
        raw_audio = format_audio(str(directory / f'raw/{filename}_raw.wav'))
        raw_plot = format_image(str(directory / f'raw/{filename}_raw.png'))

        hrtf_html = format_html('HRTF', 'A head-related transfer function (HRTF) is applied to the signal.')
        hrtf_audio = format_audio(str(directory / f'hrtf/{filename}_hrtf.wav'))
        hrtf_plot = format_image(str(directory / f'hrtf/{filename}_hrtf.png'))

        reflection_html = format_html('Reflections', 'Reflections are generated with amplitude at zenith elevation,'
                                                     ' azimuth rotation, and delay distance.')
        reflection_audio = format_audio(str(directory / f'reflections/{filename}_reflections.wav'))
        reflection_plot = format_image(str(directory / f'reflections/{filename}_reflections.png'))

        reverb_html = format_html('Reverberation', 'Reverberation is applied with amplitude, delay, and time.')
        reverb_audio = format_audio(str(directory / f'reverberation/{filename}_reverberation.wav'))
        reverb_plot = format_image(str(directory / f'reverberation/{filename}_reverberation.png'))

        summation_html = format_html('Summation', 'The signal with reflections and reverberation are summed together.')
        summation_audio = format_audio(str(directory / f'summation/{filename}_summation.wav'))
        summation_plot = format_image(str(directory / f'summation/{filename}_summation.png'))

        noise_html = format_html('Noise', 'The signal-to-noise ratio is adjusted by -60dB.')
        noise_audio = format_audio(str(directory / f'noise/{filename}_noise.wav'))
        noise_plot = format_image(str(directory / f'noise/{filename}_noise.png'))

        sample_html = format_html('Sample', 'A data point in the dataset.')
        sample_audio = format_audio(str(directory / f'samples/{filename}.wav'))
        sample_plot = format_image(str(directory / f'samples/{filename}.png'))

        cepstrum_html = format_html('Cepstrum', 'The cepstrum of the sample signal.')
        cepstrum_plot1 = format_image(str(directory / f'cepstrum/{filename}_cepstrum.png'))
        cepstrum_plot2 = format_image(str(directory / f'cepstrum/{filename}_cepstrum20.png'))

        cepbimo_html = format_html('Cepbimo', 'The cepstral-based binaural model.')
        cepbimo_plot1 = format_image(str(directory / f'cepbimo/{filename}_cepbimo.png'))
        cepbimo_plot2 = format_image(str(directory / f'cepbimo/{filename}_binaural_activity_map_2d.png'))
        cepbimo_plot3 = format_image(str(directory / f'cepbimo/{filename}_binaural_activity_map_3d.png'))

        rir_html = format_html('Room impulse response', 'The response to a stimulus generated by an acoustic space.')
        rir_audio = format_audio(str(directory / f'rir/{filename}_rir.wav'))
        rir_plot = format_image(str(directory / f'rir/{filename}_rir.png'))
        return widgets.Box(
            [raw_html, raw_audio, raw_plot, hrtf_html, hrtf_audio, hrtf_plot, reflection_html, reflection_audio,
             reflection_plot, reverb_html, reverb_audio, reverb_plot, summation_html, summation_audio, summation_plot,
             noise_html, noise_audio, noise_plot, sample_html, sample_audio, sample_plot, cepstrum_html, cepstrum_plot1,
             cepstrum_plot2, cepbimo_html, cepbimo_plot1, cepbimo_plot2, cepbimo_plot3, rir_html, rir_audio, rir_plot],
            layout=dict(width='100%', height='auto', display='flex', flex_flow='column',
                        justify_content='space-between', overflow_y='scroll'))

    def picker_viewer(self):
        from utils.data_loader import get_hrtfs
        from ipywidgets import widgets

        def format_label(text):
            return widgets.Label(value=text)

        def format_selector(widget):
            layout = widgets.Layout(width='100%', height='auto', display='flex', flex_flow='row', align_items='center')
            return widgets.Box([widget], layout=layout)

        seed = self.props['seed']
        delay_limits = self.props['delay_limits']
        time_limits = self.props['time_limits']
        reflection_limits = self.props['reflection_limits']
        zenith_limits = self.props['zenith_limits']
        azimuth_limits = self.props['azimuth_limits']
        sample_count = self.props['sample_count']
        verbose = self.props['verbose']
        zeniths, azimuths = get_hrtfs(amin=azimuth_limits[0], amax=azimuth_limits[1], zmin=zenith_limits[0],
                                      zmax=zenith_limits[1])

        def on_change(prop, value):
            self.props[prop] = value

        seed_label = format_label('RNG seed, hex')
        seed_selector = widgets.IntSlider(value=int(seed, 0), min=0, max=int('0xffffff', 0), readout_format='x',
                                          continuous_update=False)
        seed_selector.observe((lambda value: on_change('seed', hex(value['new']))), 'value')

        delay_label = format_label('Reflection/reverb delay limits, ms')
        delay_selector = format_selector(
            widgets.IntRangeSlider(value=delay_limits, min=delay_limits[0], max=delay_limits[1],
                                   continuous_update=False))
        delay_selector.observe((lambda value: on_change('delay_limits', value['new'])), 'value')

        time_label = format_label('Reverb time limits, s')
        time_selector = widgets.IntRangeSlider(value=time_limits, min=time_limits[0], max=time_limits[1],
                                               continuous_update=False)
        time_selector.observe((lambda value: on_change('time_limits', value['new'])), 'value')

        reflection_label = format_label('Reflection count')
        reflection_selector = widgets.IntRangeSlider(value=reflection_limits, min=reflection_limits[0],
                                                     max=reflection_limits[1], continuous_update=False)
        reflection_selector.observe((lambda value: on_change('reflection_limits', value['new'])), 'value')

        zenith_label = format_label('Zenith limits')
        zenith_selector = widgets.SelectionRangeSlider(options=sorted(zeniths), index=(0, len(zeniths) - 1),
                                                       continuous_update=False)
        zenith_selector.observe((lambda value: on_change('zenith_limits', value['new'])), 'value')

        azimuth_label = format_label('Azimuth limits')
        azimuth_selector = widgets.SelectionRangeSlider(options=sorted(azimuths), index=(0, len(azimuths) - 1),
                                                        continuous_update=False)
        azimuth_selector.observe((lambda value: on_change('azimuth_limits', value['new'])), 'value')

        sample_label = format_label('Sample count')
        sample_selector = widgets.BoundedIntText(value=sample_count, min=100, max=10000, step=100,
                                                 continuous_update=False)
        sample_selector.observe((lambda value: on_change('sample_count', value['new'])), 'value')

        verbose_label = format_label('Verbose output')
        verbose_selector = widgets.Checkbox(value=verbose)
        verbose_selector.observe((lambda value: on_change('verbose', value['new'])), 'value')

        return widgets.Box(
            [seed_label, seed_selector, delay_label, delay_selector, time_label, time_selector, reflection_label,
             reflection_selector, zenith_label, zenith_selector, azimuth_label, azimuth_selector, sample_label,
             sample_selector, verbose_label, verbose_selector],
            layout=widgets.Layout(width='100%', height='auto', display="flex", flex_flow='column',
                                  justify_content='center'))

    def review_viewer(self):
        from ipywidgets import widgets

        seed = self.props['seed']
        seed_label = widgets.Label(value=f'seed = {seed}')
        delay_limits = self.props['delay_limits']
        delay_label = widgets.Label(value=f'delay_limits = {delay_limits}')
        time_limits = self.props['time_limits']
        time_label = widgets.Label(value=f'time_limits = {time_limits}')
        reflection_limits = self.props['reflection_limits']
        reflection_label = widgets.Label(value=f'reflection_limits = {reflection_limits}')
        zenith_limits = self.props['zenith_limits']
        zenith_label = widgets.Label(value=f'zenith_limits = {zenith_limits}')
        azimuth_limits = self.props['azimuth_limits']
        azimuth_label = widgets.Label(value=f'azimuth_limits = {azimuth_limits}')
        sample_count = self.props['sample_count']
        sample_label = widgets.Label(value=f'sample_count = {sample_count}')
        verbose = self.props['verbose']
        verbose_label = widgets.Label(value=f'verbose = {verbose}')

        def on_click(b):
            b.disabled = True
            self.params['seed'] = seed
            self.params['delay_limits'] = delay_limits
            self.params['time_limits'] = time_limits
            self.params['reflection_limits'] = reflection_limits
            self.params['zenith_limits'] = zenith_limits
            self.params['azimuth_limits'] = azimuth_limits
            self.params['sample_count'] = sample_count
            self.params['verbose'] = verbose
            b.disabled = False

        confirm_button = widgets.Button(description='Confirm', icon='check')
        confirm_button.on_click(on_click)

        return widgets.Box(
            [seed_label, delay_label, time_label, reflection_label, zenith_label, azimuth_label, sample_label,
             verbose_label, confirm_button], layout=widgets.Layout(display='flex', flex_flow='column'))

    def generate_viewer(self):
        import shutil
        from ipywidgets import widgets
        from IPython.display import display, HTML

        seed = self.params['seed']
        duration = self.params['duration']
        delay_limits = self.params['delay_limits']
        time_limits = self.params['time_limits']
        reflection_limits = self.params['reflection_limits']
        zenith_limits = self.params['zenith_limits']
        azimuth_limits = self.params['azimuth_limits']
        sample_count = self.params['sample_count']
        verbose = self.params['verbose']

        output = widgets.Output()

        def on_click(b):
            from pathlib import Path
            from data_generator.data_generator import DataGenerator
            from data_generator.RNG import RNG

            with output:
                b.disabled = True
                rng = RNG(seed=seed, duration=duration, delay_limits=delay_limits, time_limits=time_limits,
                          reflection_limits=reflection_limits, zenith_limits=zenith_limits,
                          azimuth_limits=azimuth_limits)
                dg = DataGenerator(sample_count, 'data/', rng=rng, verbose=verbose)

                ingredients, recipe = dg.generate()
                display(ingredients.head())
                display(recipe.head())
                filepath = Path('../data/')
                shutil.make_archive(f'{filepath.__str__()}', 'zip', filepath, 'reflections')
                html = f'''<a download="reflections.zip" href="{filepath.__str__()}.zip"
                 download>Download</a>'''
                display(HTML(html))
                b.disabled = False

        generate_button = widgets.Button(description='Generate')
        generate_button.on_click(on_click)

        return widgets.Box([generate_button, output],
                           layout=widgets.Layout(display='flex', flex_flow='column', max_height='400px'))
