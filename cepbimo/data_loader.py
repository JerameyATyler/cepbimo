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
                not f.endswith('5.mp3') and not f.endswith('8.mp3') and f.endswith('.mp3')]

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


def get_zeniths(zmin=None, zmax=None):
    zeniths = sorted(list(list_hrtf_data().keys()))
    if zmin is None:
        zmin = min(zeniths)
    if zmax is None:
        zmax = max(zeniths)
    zs = []
    for z in zeniths:
        if zmin <= z <= zmax:
            zs.append(z)
    return sorted(zs)


def get_azimuths(amin=None, amax=None):
    from itertools import chain

    hrtfs = list_hrtf_data()
    azimuths = [[a for a in hrtfs[z].keys()] for z in hrtfs.keys()]
    azimuths = set(chain.from_iterable(azimuths))
    if amin is None:
        amin = min(azimuths)
    if amax is None:
        amax = max(azimuths)
    return sorted(list([a for a in azimuths if amin <= a <= amax]))


def get_hrtfs(amin=None, amax=None, zmin=None, zmax=None):
    hrtfs = list_hrtf_data()
    zes = []
    azi = []

    if amin is None:
        amin = 0
    if amax is None:
        amax = 360
    if zmin is None:
        zmin = min(hrtfs.keys())
    if zmax is None:
        zmax = max(hrtfs.keys())

    for z in sorted(list(hrtfs.keys())):
        if zmin <= z <= zmax:
            for a in sorted(list(hrtfs[z].keys())):
                if amin <= a <= amax:
                    zes.append(z)
                    azi.append(a)
    return zes, azi


if __name__ == '__main__':
    print(get_azimuths(0, 30))
