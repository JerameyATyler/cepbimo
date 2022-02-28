class DataPicker:

    def __init__(self, seed='0xec0ec0', duration=20, delay_limits=(1, 60), time_limits=(1, 8),
                 reflections_limits=(4, 8), zenith_limits=(-40, 90), azimuth_limits=(-180, 180), sample_count=1,
                 verbose=True):
        from ipywidgets import widgets
        self.degree_text = r'$^{\circ}$'
        self.layouts = dict(
            row=widgets.Layout(display='flex', flex_flow='row', justify_content='space-between',
                               border='thin solid white'),
            column=widgets.Layout(display='flex', flex_flow='column', align_items='stretch', border='thin solid white'),
            container_row=widgets.Layout(display='flex', flex_flow='row', justify_content='space-between',
                                         border='thick solid white'),
            container_column=widgets.Layout(display='flex', flex_flow='column', align_items='stretch',
                                            border='thick solid white'),
            item_row=widgets.Layout(display='flex', flex_flow='row', justify_content='center', width='100%',
                                    max_width='90%',
                                    height='100%', border='thin solid white'),
            item_column=widgets.Layout(display='flex', flex_flow='column', align_items='center', width='100%',
                                       height='auto', border='thin solid white')
        )
        self.props = dict(
            seed=seed,
            duration=duration,
            delay_limits=delay_limits,
            time_limits=time_limits,
            reflection_limits=reflections_limits,
            zenith_limits=zenith_limits,
            azimuth_limits=azimuth_limits,
            sample_count=sample_count,
            verbose=verbose
        )

        self.df = None
        self.ingredients = None
        self.download_link = ''

    def on_change(self, prop, value):
        self.props[prop] = value

    def make_player(self, s):
        from IPython.display import display
        from utils import play_audio
        from ipywidgets import widgets
        output = widgets.Output()
        with output:
            display(play_audio(s))
        return output

    def make_picker(self, label, widget):
        from ipywidgets import widgets
        label = self.make_label(label)
        return widgets.Box([label, widget], layout=self.layouts['column'])

    def make_label(self, text):
        from ipywidgets import widgets
        return widgets.Label(text, layout=self.layouts['item_row'])

    def make_widget(self, selector, prop):
        from ipywidgets import widgets
        self.make_observer(selector, prop)
        return widgets.Box([selector], layout=self.layouts['item_row'])

    def make_observer(self, selector, prop):
        selector.observe((lambda value: self.on_change(prop, value['new'])), 'value')

    def on_click(self, b):
        from pathlib import Path
        from ipywidgets import widgets
        import shutil

        b.disabled = True
        self.make_dataset()

        filepath = Path('data/reflections')

        shutil.make_archive(f'{filepath.__str__()}', 'zip', filepath)

        return widgets.HTML(value='poop')

    def make_sample_data(self):
        from RNG import RNG
        from ipywidgets import widgets
        from data_generator import DataGenerator

        seed = self.props['seed']
        duration = self.props['duration']
        delay_limits = self.props['delay_limits']
        time_limits = self.props['time_limits']
        reflection_limits = self.props['reflection_limits']
        zenith_limits = self.props['zenith_limits']
        azimuth_limits = self.props['azimuth_limits']

        rng = RNG(seed=seed, duration=duration, delay_limits=delay_limits, time_limits=time_limits,
                  reflection_limits=reflection_limits, zenith_limits=zenith_limits, azimuth_limits=azimuth_limits)
        dg = DataGenerator(1, 'data/sample_data/', rng)

        output = widgets.Output()

        with output:
            df, ingredients = dg.generate(verbose=True)
            self.df = df
            self.ingredients = ingredients


        return widgets.Box([output], layout=widgets.Layout(height='15%', overflow_y='auto', border='thin solid white',
                                                           max_height='100px'))

    def make_sidebar(self):
        from ipywidgets import widgets
        from data_loader import get_zeniths, get_azimuths
        seed_picker = self.make_picker('RNG seed', self.make_widget(
            widgets.IntSlider(min=0, max=int('0xffffff', 0), value=(int(self.props['seed'], 0)), description='Seed',
                              continuous_update=False, readout_format='x'), 'seed'))
        sample_picker = self.make_picker('Number of samples', self.make_widget(
            widgets.IntSlider(min=1, max=100, value=self.props['sample_count'], description='Samples',
                              continuous_update=False), 'sample_count'))
        duration_picker = self.make_picker('Sample duration in seconds', self.make_widget(
            widgets.IntSlider(min=10, max=30, value=self.props['duration'], description='Duration',
                              continuous_update=False), 'duration'))
        zmin, zmax = self.props['zenith_limits']
        zeniths = get_zeniths(zmin, zmax)
        zenith_picker = self.make_picker(f'Zenith range in {self.degree_text}', self.make_widget(
            widgets.SelectionRangeSlider(options=zeniths, index=(0, len(zeniths) - 1),
                                         description=f'Zenith{self.degree_text}', continuous_update=False),
            'zenith_limits'))
        amin, amax = self.props['azimuth_limits']
        azimuths = get_azimuths(amin, amax)
        azimuth_picker = self.make_picker(f'Azimuth range in {self.degree_text}', self.make_widget(
            widgets.SelectionRangeSlider(options=azimuths, index=(0, len(azimuths) - 1),
                                         description=f'Azimuth{self.degree_text}', continuous_update=False),
            'azimuth_limits'))
        reflection_picker = self.make_picker('Number of reflections', self.make_widget(
            widgets.IntRangeSlider(min=self.props['reflection_limits'][0], max=self.props['reflection_limits'][1],
                                   value=(self.props['reflection_limits'][0], self.props['reflection_limits'][1]),
                                   description='Reflections', continuous_update=False), 'reflection_limits'))
        delay_picker = self.make_picker('Reflection/reverberation delay range in ms', self.make_widget(
            widgets.IntRangeSlider(mins=self.props['delay_limits'][0], max=self.props['delay_limits'][1],
                                   value=(self.props['delay_limits'][0], self.props['delay_limits'][1]),
                                   description='Delay', contiuous_update=False), 'delay_limits'))
        time_picker = self.make_picker('Reverberation time range in s', self.make_widget(
            widgets.IntRangeSlider(min=self.props['time_limits'][0], max=self.props['time_limits'][1],
                                   value=(self.props['time_limits'][0], self.props['time_limits'][1]),
                                   description='Time', continuous_update=False), 'time_limits'))
        verbose_picker = self.make_picker('Verbose output **Warning:**', self.make_widget(
            widgets.RadioButtons(options=[False, True], value=self.props['verbose'], description='Verbose'),
            'verbose'))

        return widgets.Box(
            [seed_picker, sample_picker, duration_picker, zenith_picker, azimuth_picker, reflection_picker,
             delay_picker, time_picker, verbose_picker],
            layout=widgets.Layout(display='flex', flex_flow='column', align_items='center', height='auto',
                                  border='thin solid white'))

    def make_confirm(self):
        import base64
        from pathlib import Path
        from ipywidgets import widgets
        from IPython.display import display, HTML
        import shutil

        confirm = widgets.Button(description='Confirm and generate', icon='check', tooltip='Confirm and generate')
        output = widgets.Output()

        def on_click(b):
            with output:
                b.disabled = True
                display(self.make_dataset())

                filepath = Path('data/reflections')
                shutil.make_archive(f'{(filepath / "reflections").__str__()}', 'zip', filepath)

                html = f'<a download="reflections.zip" href="{(filepath / "reflections.zip").__str__()}" download>Download dataset</a>'
                display(HTML(html))

                b.disabled = False

        confirm.on_click(on_click)
        confirm = widgets.Box([confirm], layout=self.layouts['item_row'])

        return widgets.Box([confirm, output], layout=self.layouts['column'])

    def make_dataset(self):
        from RNG import RNG
        from ipywidgets import widgets
        from data_generator import DataGenerator

        seed = self.props['seed']
        duration = self.props['duration']
        delay_limits = self.props['delay_limits']
        time_limits = self.props['time_limits']
        reflection_limits = self.props['reflection_limits']
        zenith_limits = self.props['zenith_limits']
        azimuth_limits = self.props['azimuth_limits']
        sample_count = self.props['sample_count']

        rng = RNG(seed=seed, duration=duration, delay_limits=delay_limits, time_limits=time_limits,
                  reflection_limits=reflection_limits, zenith_limits=zenith_limits, azimuth_limits=azimuth_limits)
        dg = DataGenerator(sample_count, 'data/', rng)

        output = widgets.Output()

        with output:
            self.df, self.ingredients = dg.generate(verbose=True)

        return widgets.Box([output], layout=widgets.Layout(overflow_y='auto', border='thin solid white',
                                                           max_height='100px'))

    def ui(self):
        from ipywidgets import widgets

        stdout = self.make_sample_data()
        sidebar = self.make_sidebar()
        confirm = self.make_confirm()

        tabs = self.make_tabs()

        content = widgets.Box([sidebar, tabs], layout=self.layouts['container_row'])

        return widgets.Box([stdout, content, confirm], layout=self.layouts['container_column'])

    def make_parameters_tab(self):
        from ipywidgets import widgets
        if self.ingredients is None:
            ingredients = self.props
        else:
            ingredients = self.ingredients.iloc[0]
        html = '''
        <h1>Dataset parameters</h1>
        <p>Here you will find a list of the parameters used to generate the dataset. The data generator can be 
        customized by adjusting its input arguments. The input arguments can be adjusted using the sliders. Sample
        output produced by the data generator can be found in the other tabs.</p>
        ''' + f'''
        <h3>RNG seed</h3>
        <p>The <em>RNG seed</em> is a hexidecimal value that is used to consistently generate random numbers. Two 
        datasets generated with all of the same parameters will be identical however two datasets generated with all of 
        the same parameters <em>except</em> RNG seed will produce different datasets. The RNG seed used to generate this 
        sample data is <em><b>{ingredients['seed']}</b></em>.
        </p>
        <h3>Number of samples</h3>
        <p>The <em>number of samples</em> indicates how many data points to generate, currently set to 
        {ingredients['sample_count']}. </p>
        <h3>Sample duration</h3>
        <p>The <em>sample duration in seconds</em> indicates how long each sample data should be seconds, currently set
        to {ingredients['duration']} seconds. </p>''' + '''
        <h3>Zenith range in $^{\circ}$</h3>  
        <p>The <em>zenith range in $^{\circ}$</em> indicates the minimum and maximum zenith (elevation) angles in 
        degrees that head-related transfer functions (HRTF) generated for the direct signal and reflections.''' + f'''
        Zeniths are currently set to occur in the range {ingredients['zenith_limits']}</p>''' + '''
        <h3>Azimuth range in $^{\circ}$</h3>
        <p>The <em>azimuth range in $^{\circ}$</em> indicates the minimum and maximum azimuth (horizontal rotation) 
        angle in degrees that HRTFs generated for the direct signal and reflections.''' + f'''
        Azimuths are currently set to occur in the range {ingredients['azimuth_limits']}</p>
        </p>
        <h3>Number of reflections</h3>
        <p><em>Number of reflections</em> is the limit on the number of reflections to apply to a sample.
        Currently set to generate between {ingredients['reflection_limits']} reflections.</p>
        <h3>Reflection/Reverberation delay range in ms</h3>
        <p><em>Reflection/Reverberation delay range in ms</em> is the length of time that the reflection or 
        reverberation will lag behind the direct signal.
        Currently set to generate reflections and reverberations between {ingredients['delay_limits']} milliseconds.</p>
        <h3>Reverberation time range in s</h3>
        <p><Reverberation time range in s</em> is the length of time that reverberation will last in seconds.
        Currently set to generate reverberation times between {ingredients['time_limits']} seconds.</p>
        <h3>Verbose output</h3>
        <p><em>Verbose</em> indicates whether or not to produce all possible output (e.g. figures and audio files)
        or a limited set. 
        Verbose is currently set to {self.props['verbose']}.</p>
        '''

        return widgets.HTMLMath(value=html)

    def make_raw_signal_tab(self):
        from pathlib import Path
        import os
        from ipywidgets import widgets
        from pydub import AudioSegment

        directory = Path('data/sample_data/reflections/raw')
        filepath = directory / Path(f"{self.df['filepath'][0]}_raw.wav")

        if os.path.isfile(filepath):
            audio = AudioSegment.from_wav(filepath)
            player = self.make_player(audio)
        else:
            player = widgets.HTML(value='')
        player = widgets.Box([player], layout=self.layouts['item_row'])

        filepath = directory / Path(f'{self.df["filepath"][0]}_raw_wave.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image = widgets.Box([image], layout=self.layouts['item_row'])
        return widgets.Box([player, image], layout=self.layouts['container_column'])

    def make_hrtf_tab(self):
        from pathlib import Path
        import os
        from ipywidgets import widgets
        from pydub import AudioSegment

        directory = Path('data/sample_data/reflections/hrtf')
        filepath = directory / Path(f"{self.df['filepath'][0]}_hrtf.wav")

        if os.path.isfile(filepath):
            audio = AudioSegment.from_wav(filepath)
            player = self.make_player(audio)
        else:
            player = widgets.HTML(value='')
        player = widgets.Box([player], layout=self.layouts['item_row'])

        filepath = directory / Path(f'{self.df["filepath"][0]}_hrtf_wave.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image = widgets.Box([image], layout=self.layouts['item_row'])
        return widgets.Box([player, image], layout=self.layouts['container_column'])

    def make_reflections_tab(self):
        from pathlib import Path
        import os
        from ipywidgets import widgets
        from pydub import AudioSegment

        directory = Path('data/sample_data/reflections/reflections')
        filepath = directory / Path(f"{self.df['filepath'][0]}_reflections.wav")

        if os.path.isfile(filepath):
            audio = AudioSegment.from_wav(filepath)
            player = self.make_player(audio)
        else:
            player = widgets.HTML(value='')
        player = widgets.Box([player], layout=self.layouts['item_row'])

        filepath = directory / Path(f'{self.df["filepath"][0]}_reflections_wave.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image = widgets.Box([image], layout=self.layouts['item_row'])
        return widgets.Box([player, image], layout=self.layouts['container_column'])

    def make_reverberations_tab(self):
        from pathlib import Path
        import os
        from ipywidgets import widgets
        from pydub import AudioSegment

        directory = Path('data/sample_data/reflections/reverberations')
        filepath = directory / Path(f"{self.df['filepath'][0]}_reverberation.wav")

        if os.path.isfile(filepath):
            audio = AudioSegment.from_wav(filepath)
            player = self.make_player(audio)
        else:
            player = widgets.HTML(value='')
        player = widgets.Box([player], layout=self.layouts['item_row'])

        filepath = directory / Path(f'{self.df["filepath"][0]}_reverberation_wave.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image = widgets.Box([image], layout=self.layouts['item_row'])
        return widgets.Box([player, image], layout=self.layouts['container_column'])

    def make_output_tab(self):
        from pathlib import Path
        import os
        from ipywidgets import widgets
        from pydub import AudioSegment

        directory = Path('data/sample_data/reflections/samples')
        filepath = directory / Path(f"{self.df['filepath'][0]}.wav")

        if os.path.isfile(filepath):
            audio = AudioSegment.from_wav(filepath)
            player = self.make_player(audio)
        else:
            player = widgets.HTML(value='')
        player = widgets.Box([player], layout=self.layouts['item_row'])

        filepath = directory / Path(f'{self.df["filepath"][0]}_wave.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image_wave = widgets.Box([image], layout=self.layouts['item_row'])

        filepath = directory / Path(f'{self.df["filepath"][0]}_sample.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image_sample = widgets.Box([image], layout=self.layouts['item_row'])

        images = widgets.Box([image_wave, image_sample], layout=self.layouts['item_row'])

        return widgets.Box([player, images], layout=self.layouts['container_column'])

    def make_rir_tab(self):
        from pathlib import Path
        import os
        from ipywidgets import widgets
        from pydub import AudioSegment

        directory = Path('data/sample_data/reflections/rir')
        filepath = directory / Path(f"{self.df['filepath'][0]}_rir.wav")

        if os.path.isfile(filepath):
            audio = AudioSegment.from_wav(filepath)
            player = self.make_player(audio)
        else:
            player = widgets.HTML(value='')
        player = widgets.Box([player], layout=self.layouts['item_row'])

        filepath = directory / Path(f'{self.df["filepath"][0]}_rir.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image = widgets.Box([image], layout=self.layouts['item_row'])
        return widgets.Box([player, image], layout=self.layouts['container_column'])

    def make_cepstral_tab(self):
        from pathlib import Path
        import os
        from ipywidgets import widgets

        directory = Path('data/sample_data/reflections/cepstrums')
        filepath = directory / Path(f'{self.df["filepath"][0]}_left_cepstrum.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image_l = widgets.Box([image], layout=self.layouts['item_row'])

        filepath = directory / Path(f'{self.df["filepath"][0]}_left_cepstrum_20.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image_l_20 = widgets.Box([image], layout=self.layouts['item_row'])

        filepath = directory / Path(f'{self.df["filepath"][0]}_right_cepstrum.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image_r = widgets.Box([image], layout=self.layouts['item_row'])

        filepath = directory / Path(f'{self.df["filepath"][0]}_right_cepstrum_20.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image_r_20 = widgets.Box([image], layout=self.layouts['item_row'])

        directory = Path('data/sample_data/reflections/cepbimo')
        filepath = directory / Path(f'{self.df["filepath"][0]}_binaural_activity_map_2d.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image_c2d = widgets.Box([image], layout=self.layouts['item_row'])

        filepath = directory / Path(f'{self.df["filepath"][0]}_binaural_activity_map_3d.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image_c3d = widgets.Box([image], layout=self.layouts['item_row'])

        filepath = directory / Path(f'{self.df["filepath"][0]}_cepbimo_wave.png')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                image = f.read()
                image = widgets.Image(value=image, format='png')
        else:
            image = widgets.HTML(value='')
        image_cwave = widgets.Box([image], layout=self.layouts['item_row'])

        l_box = widgets.Box([image_l, image_l_20], layout=self.layouts['container_row'])
        r_box = widgets.Box([image_r, image_r_20], layout=self.layouts['container_row'])
        cwave = widgets.Box([image_cwave], layout=self.layouts['container_row'])
        bamap = widgets.Box([image_c2d, image_c3d], layout=self.layouts['container_row'])

        return widgets.Box([l_box, r_box, cwave, bamap], layout=self.layouts['container_column'])

    def make_tabs(self):
        from ipywidgets import widgets

        titles = ['Parameters', 'Raw signal', 'HRTF', 'Reflections', 'Reverberation', 'Output signal',
                  'Room impulse response', 'Cepstral analysis']
        children = [self.make_parameters_tab(), self.make_raw_signal_tab(), self.make_hrtf_tab(),
                    self.make_reflections_tab(), self.make_reverberations_tab(), self.make_output_tab(),
                    self.make_rir_tab(), self.make_cepstral_tab()]
        tab = widgets.Tab(layout=widgets.Layout(width='100%', max_width='80%'))
        [tab.set_title(i, title=t) for i, t in enumerate(titles)]
        tab.children = children
        return tab
