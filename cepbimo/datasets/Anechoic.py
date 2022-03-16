from torch.utils.data import Dataset


class Anechoic(Dataset):
    """Full anechoic dataset with all labels."""

    _download_url = 'https://reflections.speakeasy.services'

    def __init__(self, root, ttv, download=False, transform=None, target_transform=None):
        """
        Args:
            root (string): Root path to the data directory.
            ttv (string): {"train"|"test"|"validate"} Which subset of the data to use.
            download (bool, optional): If True, data will be downloaded if it is not available in the root directory.
            transform (callable, optional): Optional transform to be applied to the samples.
            target_transform (callable, optional): Optional transform to be applied to the labels.
        """
        from pathlib import Path
        import os
        import requests
        import zipfile
        import io
        import shutil

        assert ttv in ['train', 'test', 'validate'], 'Acceptable values for ttv are {"train"|"test"|"validate"}'

        self.dataset_path = (Path(root) / ttv).__str__()
        self.recipe_path = f'{self.dataset_path}_recipe'

        self.transform = transform
        self.target_transform = target_transform

        if not download:
            assert os.path.isdir(root), 'root directory must exist if download=False'
            assert os.path.isdir(self.dataset_path), f'data directory {self.dataset_path} must exist if download=False'
            assert os.path.isdir(self.recipe_path), f'label directory {self.recipe_path} must exist if download=False'
        else:
            if not os.path.isdir(root):
                os.mkdir(root)
            if not os.path.isdir(self.dataset_path):
                os.mkdir(self.dataset_path)
            if not os.path.isdir(self.recipe_path):
                os.mkdir(self.recipe_path)

            print(f'Downloading dataset at {self._download_url}/{ttv}.zip')
            r = requests.get(f'{self._download_url}/{ttv}.zip', stream=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))

            print('Extracting dataset')
            for f in z.namelist():
                filename = Path(f).name
                if not filename:
                    continue
                source = z.open(f)
                if filename.endswith('.zip'):
                    target = open((Path(root) / filename).__str__(), 'wb')
                else:
                    target = open((Path(self.dataset_path) / filename).__str__(), 'wb')
                print(f'\tExtracting file: {filename}')
                with source, target:
                    shutil.copyfileobj(source, target)

            recipe_path = f"{Path(self.dataset_path).__str__()}_recipe.zip"
            assert os.path.isfile(recipe_path), f"{recipe_path} missing"
            z = zipfile.ZipFile(recipe_path)
            z.extractall(self.recipe_path)

        self.labels = self.set_labels()

    def set_labels(self):
        from utils.data_loader import read_recipe

        return read_recipe(self.recipe_path)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        from pathlib import Path
        from pydub import AudioSegment
        import numpy as np
        from utils.utils import split_channels

        labels = self.labels.iloc[item]
        audio = AudioSegment.from_wav((Path(self.dataset_path) / f"{labels['filepath']}.wav").__str__())

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

        filepath = f"{(Path(self.dataset_path) / self.labels.iloc[item]['filepath']).__str__()}.wav"
        assert os.path.isfile(filepath), f"{filepath} does not exist"
        audio = AudioSegment.from_wav(filepath)
        return display(play_audio(audio))


def demo_anechoic():
    ani = Anechoic('../data/ani', 'test', download=False)
    print(len(ani))

    for i in range(5):
        print(ani[i])


if __name__ == '__main__':
    demo_anechoic()
