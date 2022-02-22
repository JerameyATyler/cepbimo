def list_composers(console=False):
    """Lists composers available in the anechoic data."""
    import os
    from pathlib import Path

    path = Path('data/anechoic')

    composers = os.listdir(path)

    if console:
        [print(c) for c in composers]

    return composers


def list_anechoic_data(console=False):
    """Lists all the anechoic data available."""
    import os
    from pathlib import Path

    path = Path('data/anechoic')

    composers = list_composers()

    a = {c: None for c in composers}

    for c in a.keys():
        a[c] = [(path / c / f).__str__() for f in os.listdir(path / c) if
                not f.endswith('5.mp3') and not f.endswith('8.mp3')]

    if console:
        for c in a.keys():
            print(f'{c.upper()}: \n{a[c]}')

    return a


def list_hrtf_data(console=False):
    """Lists all the HRTFs available."""
    from pathlib import Path
    import os

    path = Path('data/hrtf/')

    zeniths = {}

    for z in os.listdir(path):
        if z.startswith('elev'):
            zenith = int(z.split('elev')[1].strip())
            if zenith not in zeniths:
                zeniths[zenith] = {}

            for a in os.listdir(path / z):
                azimuth = int(a.split(f'{zenith}e')[1].split('a.wav')[0].strip())
                if azimuth not in zeniths[zenith]:
                    zeniths[zenith][azimuth] = set()
                zeniths[zenith][azimuth].add(path / z / a)

    if console:
        for z in sorted(list(zeniths.keys())):
            print(f'Zenith {z} : Azimuths {sorted(list(zeniths[z].keys()))}')

    return zeniths
