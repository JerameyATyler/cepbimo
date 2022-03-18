from torch.utils.data import Dataset


class Anechoic(Dataset):
    """Full anechoic dataset with all labels."""

    def __init__(self, root, ttv, download=False, transform=None, target_transform=None, columns=None):
        """
        Args:
            root (string): Root path to the data directory.
            ttv (string): {"train"|"test"|"validate"} Which subset of the data to use.
            download (bool, optional): If True, data will be downloaded if it is not available in the root directory.
            transform (callable, optional): Optional transform to be applied to the samples.
            target_transform (callable, optional): Optional transform to be applied to the labels.
        """
        from pathlib import Path

        ttvs = ['train', 'test', 'validate']
        assert ttv in ttvs, f'Acceptable values for ttv are {", ".join(ttvs)}'

        self.ttv = ttv
        self.transform = transform
        self.target_transform = target_transform
        self.root = Path(root).__str__()
        self.data_path = None
        self.label_path = None

        if download:
            self.download()
        else:
            self.check_directories()

        self.labels = self.set_labels(columns)

    def download(self):
        from pathlib import Path
        import requests
        import zipfile
        import io
        import shutil
        import os

        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        _download_url = 'https://reflections.speakeasy.services'

        print(f'Downloading dataset at {_download_url}/{self.ttv}.zip')
        r = requests.get(f'{_download_url}/{self.ttv}.zip', stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        print(f'Finished downloading')

        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.isdir(self.label_path):
            os.mkdir(self.label_path)

        print('Extracting dataset')
        for f in z.namelist():
            filename = Path(f).name
            if not filename:
                continue
            source = z.open(f)
            if filename.endswith('.zip'):
                target = open((Path(self.root) / filename).__str__(), 'wb')
            else:
                target = open((Path(self.root) / self.ttv / filename).__str__(), 'wb')
            print(f'\tExtracting file: {filename}')
            with source, target:
                shutil.copyfileobj(source, target)

        recipe_path = f"{(Path(self.root) / self.ttv).__str__()}_recipe.zip"
        assert os.path.isfile(recipe_path), f"{recipe_path} missing"
        z = zipfile.ZipFile(recipe_path)
        z.extractall(recipe_path)

    def check_directories(self):
        from pathlib import Path
        import os

        assert os.path.isdir(self.root), f"Root directory {self.root} must exist if download=False"

        data_path = (Path(self.root) / self.ttv).__str__()
        label_path = f"{data_path}_recipe"
        assert os.path.isdir(data_path), f"Data directory {data_path} must exist if download=False"
        assert os.path.isdir(label_path), f"Label directory {label_path} must exist if download=False"
        self.data_path = data_path
        self.label_path = label_path

    def set_labels(self, columns):
        from utils.data_loader import read_recipe

        if columns is not None:
            return read_recipe(self.label_path)[columns]

        return read_recipe(self.label_path)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        from pathlib import Path
        from pydub import AudioSegment
        import numpy as np
        from utils.utils import split_channels

        labels = self.labels.iloc[item]
        audio = AudioSegment.from_wav((Path(self.data_path) / f"{labels['filepath']}.wav").__str__())

        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            labels = self.target_transform(labels)

        audio = np.array(split_channels(audio)).transpose()

        return audio, labels

    def play_sample(self, item):
        from pathlib import Path
        from pydub import AudioSegment
        from utils.utils import play_audio
        from IPython.display import display
        import os

        filepath = f"{(Path(self.data_path) / self.labels.iloc[item]['filepath']).__str__()}.wav"
        assert os.path.isfile(filepath), f"{filepath} does not exist"
        audio = AudioSegment.from_wav(filepath)
        return display(play_audio(audio))


def demo_anechoic():
    ani = Anechoic('../data/ani', 'train', download=True, columns='reflection_count')
    print(len(ani))

    for i in range(5):
        print(ani[i])


if __name__ == '__main__':
    demo_anechoic()
