def list_composers(console=False):
    """Lists composers available in the anechoic data."""
    import os
    from pathlib import Path

    path = Path('../data/anechoic')

    composers = os.listdir(path)

    if console:
        [print(c) for c in composers]

    return composers


def list_anechoic_data(console=False):
    """Lists all the anechoic data available."""
    import os
    from pathlib import Path

    path = Path('../data/anechoic')

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

    path = Path('../data/hrtf/')

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


def list_anechoic_lengths(composer=None):
    from pydub import AudioSegment

    if composer is not None:
        composers = [composer]
    else:
        composers = sorted(list(list_anechoic_data().keys()))
    return {c: len(AudioSegment.from_mp3(list_anechoic_data()[c][0])) for c in composers}


def read_recipe(path):
    import dask.dataframe as dd
    return dd.read_parquet(path, engine='pyarrow').compute()


def read_ingredients(path):
    import pandas as pd
    return pd.read_json(path, orient='records', lines=True)


def train_test_validation_split(filepaths):
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(filepaths, test_size=0.2, random_state=1)
    train, validate = train_test_split(train, test_size=0.25, random_state=1)
    return train, test, validate


def split_data(recipe_directory):
    from zipfile import ZipFile
    from pathlib import Path
    import os
    import dask.dataframe as dd
    import shutil

    recipe = read_recipe(recipe_directory)
    train, test, validate = train_test_validation_split(recipe[['filepath']])

    train_labels = recipe[(recipe.filepath.isin(train['filepath'].tolist()))]
    test_labels = recipe[(recipe.filepath.isin(test['filepath'].tolist()))]
    validate_labels = recipe[(recipe.filepath.isin(validate['filepath'].tolist()))]

    root_path = Path(recipe_directory).parents[0]
    directories = ['cepbimo', 'cepstrum', 'hrtf', 'noise', 'raw', 'reflections', 'reverberation', 'rir', 'samples',
                   'summation']

    with ZipFile('train.zip', 'w') as z:
        ddf = dd.from_pandas(train_labels, chunksize=50)
        ddf.to_parquet('train_recipe', engine='pyarrow')
        shutil.make_archive('train_recipe', 'zip', 'train_recipe')
        z.write('train_recipe.zip')
        for file in train['filepath'].tolist():
            for d in directories:
                directory_path = root_path / d
                if os.path.isdir(directory_path):
                    for f in os.listdir(directory_path):
                        if file in f:
                            z.write(directory_path / f'{f}')

    with ZipFile('test.zip', 'w') as z:
        ddf = dd.from_pandas(test_labels, chunksize=50)
        ddf.to_parquet('test_recipe', engine='pyarrow')
        shutil.make_archive('test_recipe', 'zip', 'test_recipe')
        z.write('test_recipe.zip')
        for file in test['filepath'].tolist():
            for d in directories:
                directory_path = root_path / d
                if os.path.isdir(directory_path):
                    for f in os.listdir(directory_path):
                        if file in f:
                            z.write(directory_path / f'{f}')

    with ZipFile('validate.zip', 'w') as z:
        ddf = dd.from_pandas(validate_labels, chunksize=50)
        ddf.to_parquet('validate_recipe', engine='pyarrow')
        shutil.make_archive('validate_recipe', 'zip', 'validate_recipe')
        z.write('validate_recipe.zip')
        for file in validate['filepath'].tolist():
            for d in directories:
                directory_path = root_path / d
                if os.path.isdir(directory_path):
                    for f in os.listdir(directory_path):
                        if file in f:
                            z.write(directory_path / f'{f}')

    return train_labels, test_labels, validate_labels


def demo_ttv_split():
    from pathlib import Path

    path = Path('data/sample/reflections/recipe_50921501').__str__()
    split_data(path)


def s3_upload():
    import boto3
    from pathlib import Path

    session = boto3.session.Session()
    client = session.client('s3', endpoint_url='https://sfo3.digitaloceanspaces.com', region_name='sfo3',
                            aws_access_key_id='2SYC4YBEBTTZTY37MO4E',
                            aws_secret_access_key='G7rnSe/ZCVi+CScIU++2Dp/IIWRNb4aaWPzBCswGyF0')

    path = 'C:/Users/Jeram/Desktop/datasets'
    path = Path(path)

    # client.upload_file((path / 'train.zip').__str__(), 'reflections', 'train.zip')
    client.upload_file((path / 'full.zip').__str__(), 'reflections', 'full.zip')


if __name__ == '__main__':
    # demo_ttv_split()
    s3_upload()
