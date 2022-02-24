degree_text = r'$^{\circ}$'
dataset_parameters = dict(seed="0xec0ec0", duration=20, delay_limits=(1, 60), time_limits=(1, 8),
                          reflection_limits=(4, 8))


def make_label(text):
    from ipywidgets import widgets

    layout = widgets.Layout(display='flex',
                            justify_content='space-between',
                            align_items='center',
                            border='solid thin white')

    label = widgets.Label(value=text)
    return widgets.HBox([label], width='100%', height='100%', layout=layout)


def make_player(zenith, azimuth):
    from utils import play_audio
    from IPython.display import display
    from ipywidgets import widgets
    from data_loader import list_anechoic_data
    from transforms import mix_parts, apply_hrtf

    signal = apply_hrtf(mix_parts(list_anechoic_data()['beethoven']), zenith, azimuth)
    layout = widgets.Layout(display='flex',
                            justify_content='space-between',
                            align_items='center')

    output = widgets.Output()
    with output:
        display(play_audio(signal))
    return widgets.HBox([output], width='auto', height='100%', layout=layout)


def make_slider(arr, description):
    from ipywidgets import widgets
    layout = widgets.Layout(display='flex',
                            justify_content='space-between',
                            align_items='center')
    return widgets.SelectionRangeSlider(options=arr, index=(0, len(arr) - 1), description=description, width='auto',
                                        height='auto', layout=layout)


def make_hbox(contents):
    from ipywidgets import widgets
    layout = widgets.Layout(display='flex',
                            justify_content='space-between',
                            align_items='center')
    return widgets.HBox(contents, width='auto', height='100%', layout=layout)


def make_vbox(contents):
    from ipywidgets import widgets
    layout = widgets.Layout(display='flex',
                            justify_content='space-between',
                            align_items='center',
                            border='solid thin white')
    return widgets.VBox(contents, width='100%', height='auto', layout=layout)


def update_hrtf(zenith_limits, azimuth_limits):
    import numpy as np
    from ipywidgets import widgets
    from data_loader import get_hrtfs

    dataset_parameters['zenith_limits'] = zenith_limits
    dataset_parameters['azimuth_limits'] = azimuth_limits

    a_min, a_max = azimuth_limits
    z_min, z_max = zenith_limits

    zeniths, azimuths = get_hrtfs(amin=a_min, amax=a_max, zmin=z_min, zmax=z_max)
    azimuths = np.array(azimuths)
    zeniths = np.array(zeniths)

    out = widgets.Output()
    with out:
        make_figure(z_min, z_max, zeniths, a_min, a_max, azimuths)

    return make_vbox([out])


def make_figure(z_min, z_max, zeniths, a_min, a_max, azimuths):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from figures import suptitle_font, title_font

    a_min = np.deg2rad(a_min)
    a_max = np.deg2rad(a_max)
    azimuths = np.radians(azimuths)
    theta_a = np.linspace(a_min, a_max)

    z_min = np.deg2rad(z_min)
    z_max = np.deg2rad(z_max)
    theta_z = np.linspace(z_min, z_max)

    cmap = mpl.cm.get_cmap('viridis')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True), layout='constrained')

    plt.suptitle('Zenith and azimuth ranges', **suptitle_font)

    ax1.set_title('Zenith range', **title_font)
    ax2.set_title('Azimuth range', **title_font)
    ax3.set_title('HRTFs', **title_font)

    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    ax3.tick_params(labelsize=20)

    ax1.set_yticklabels([])
    ax2.set_yticklabels([])

    ax2.set_theta_direction(-1)
    ax3.set_theta_direction(-1)

    ax2.set_theta_zero_location('N')
    ax3.set_theta_zero_location('N')

    ax3.set_rlabel_position(90)
    ax3.set_rmax(90)
    ax3.set_rticks(np.linspace(-45, 75, 4))
    ax3.set_rlim(bottom=90, top=-50)

    ax1.annotate('', xy=(1, 0.5), xytext=(0.5, 0.5), xycoords='axes fraction',
                 arrowprops=dict(facecolor='red', width=6, headwidth=12, alpha=0.75))
    ax2.annotate('', xy=(0.5, 1), xytext=(0.5, 0.5), xycoords='axes fraction',
                 arrowprops=dict(facecolor='red', width=6, headwidth=12, alpha=0.75))
    ax3.annotate('', xy=(0.5, 1), xytext=(0.5, 0.5), xycoords='axes fraction',
                 arrowprops=dict(facecolor='red', width=6, headwidth=12, alpha=0.75))

    ax1.annotate('Face Forward', xy=(0.5, .5), xytext=(0.75, .5), xycoords="axes fraction", ha='center', va='center',
                 fontsize=20)
    ax2.annotate('Face Forward', xy=(0.5, .75), xytext=(0.5, .75), xycoords="axes fraction", ha='center', va='center',
                 fontsize=20, rotation=90)
    ax3.annotate('Face Forward', xy=(0.5, .75), xytext=(0.5, .75), xycoords="axes fraction", ha='center', va='center',
                 fontsize=20, rotation=90)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    ax1.fill_between(theta_z, 0, 1, alpha=0.4, label='area', color=cmap(0.5))
    ax2.fill_between(theta_a, 0, 1, alpha=0.4, label='area', color=cmap(0.5))
    ax3.scatter(azimuths, zeniths, c=cmap(0.5), alpha=0.4)


def make_header(zenith_slider, azimuth_slider):
    zenith_label = make_label('Set zenith range')
    azimuth_label = make_label('Set azimuth range')

    zenith_vbox = make_vbox([zenith_label, make_hbox([zenith_slider])])
    azimuth_vbox = make_vbox([azimuth_label, make_hbox([azimuth_slider])])

    return make_hbox([zenith_vbox, azimuth_vbox])


def make_hrtf_picker():
    from data_loader import get_azimuths, get_zeniths
    from ipywidgets import widgets

    zeniths = get_zeniths()
    azimuths = get_azimuths()

    zenith_slider = make_slider(zeniths, description='Zenith')
    azimuth_slider = make_slider(azimuths, description='Azimuth')

    header = make_header(zenith_slider, azimuth_slider)
    figures = widgets.interactive_output(update_hrtf,
                                         {'zenith_limits': zenith_slider, 'azimuth_limits': azimuth_slider})
    return make_vbox([header, figures])


def on_seed_change(seed):
    from RNG import RNG

    dataset_parameters['seed'] = hex(seed)

    r1 = RNG(hex(seed - 1)).rng.random(5)
    r2 = RNG(hex(seed)).rng.random(5)
    r3 = RNG(hex(seed + 1)).rng.random(5)

    print('First 5 random floats for seed:')
    print(f'\t{hex(seed - 1)}:[{r1[0]:0.2f}, {r1[1]:0.2f}, {r1[2]:0.2f}, {r1[3]:0.2f}, {r1[4]:0.2f}]')
    print(f'\t{hex(seed)}:[{r2[0]:0.2f}, {r2[1]:0.2f}, {r2[2]:0.2f}, {r2[3]:0.2f}, {r2[4]:0.2f}]')
    print(f'\t{hex(seed + 1)}:[{r3[0]:0.2f}, {r3[1]:0.2f}, {r3[2]:0.2f}, {r3[3]:0.2f}, {r3[4]:0.2f}]')


def make_seed_picker():
    from ipywidgets import widgets

    seed = dataset_parameters['seed']
    seed = int(seed, 0)

    seed_label = make_label('Set seed value')
    seed_slider = widgets.IntSlider(value=seed, min=0, max=int('0xffffff', 0), description='Seed', readout_format='x')
    seed_picker = widgets.interactive(on_seed_change, seed=seed_slider)
    return make_vbox([seed_label, make_hbox([seed_picker])])


def on_duration_change(duration):
    dataset_parameters['duration'] = duration
    print(duration)


def make_duration_picker():
    from ipywidgets import widgets

    duration = dataset_parameters['duration']

    duration_label = make_label('Set sample duration in seconds')
    duration_slider = widgets.IntSlider(value=duration, min=10, max=30, description='Duration')
    duration_picker = widgets.interactive(on_duration_change, duration=duration_slider)
    return make_vbox([duration_label, make_hbox([duration_picker])])


def on_reflection_change(reflection_limits, delay_limits):
    dataset_parameters['reflection_limits'] = reflection_limits
    dataset_parameters['delay_limits'] = delay_limits
    print(reflection_limits, delay_limits)


def make_reflection_picker():
    from ipywidgets import widgets

    reflection_limits = dataset_parameters['reflection_limits']
    delay_limits = dataset_parameters['delay_limits']

    reflections_label = make_label('Set reflection count and delay limits')

    reflections_slider = widgets.IntRangeSlider(value=reflection_limits, min=4, max=8, step=1, continuous_update=False,
                                                description='Reflections')
    delay_slider = widgets.IntRangeSlider(value=delay_limits, min=1, max=60, step=1, continuous_update=False,
                                          description='Delay')

    reflections_picker = widgets.interactive(on_reflection_change,
                                             reflection_limits=reflections_slider, delay_limits=delay_slider)
    return make_vbox([reflections_label, make_hbox([reflections_picker])])


def on_reverberation_change(time):
    dataset_parameters['time_limits'] = time
    print(time)


def make_reverberation_picker():
    from ipywidgets import widgets

    time = dataset_parameters['time_limits']
    time_label = make_label('Set reverberation time in seconds')
    time_slider = widgets.IntRangeSlider(value=time, min=1, max=8, continuous_update=False, descrpition='Time')
    time_picker = widgets.interactive(on_reverberation_change, time=time_slider)
    return make_vbox([time_label, make_hbox([time_picker])])


def on_count_change(count):
    dataset_parameters['count'] = count


def make_count_picker():
    from ipywidgets import widgets

    count = 1
    count_label = make_label('Set the number of samples')
    count_slider = widgets.IntSlider(value=count, min=1, max=10, continuous_update=False, description='Count')
    count_picker = widgets.interactive(on_count_change, count=count_slider)
    return make_vbox([count_label, make_hbox([count_picker])])


def make_reviewer():
    seed = dataset_parameters['seed']
    duration = dataset_parameters['duration']
    zenith_limits = dataset_parameters['zenith_limits']
    azimuth_limits = dataset_parameters['azimuth_limits']
    reflection_limits = dataset_parameters['reflection_limits']
    delay = dataset_parameters['delay_limits']
    time = dataset_parameters['time_limits']
    sample_count = dataset_parameters['count']

    seed_label = make_label(f'Seed value: {seed}')
    duration_label = make_label(f'Sample duration: {duration}s')
    zenith_label = make_label(f'Zenith range: {zenith_limits[0]}{degree_text}  -  {zenith_limits[1]}{degree_text}')
    azimuth_label = make_label(f'Azimuth range: {azimuth_limits[0]}{degree_text}  -  {azimuth_limits[1]}{degree_text}')
    reflection_label = make_label(f'Reflection limits: {reflection_limits[0]} - {reflection_limits[1]}')
    delay_label = make_label(f'Delay range: {delay[0]}ms - {delay[1]}ms')
    time_label = make_label(f'Reverb time limits: {time[0]}s - {time[1]}s')
    sample_label = make_label(f'Number of samples: {sample_count}')

    hb1 = make_hbox([seed_label, duration_label])
    hb2 = make_hbox([zenith_label, azimuth_label])
    hb3 = make_hbox([reflection_label, delay_label])
    hb4 = make_hbox([time_label, sample_label])

    vb = make_vbox([hb1, hb2, hb3, hb4])
    return vb
