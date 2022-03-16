from datasets.Anechoic import Anechoic


class Reverberation(Anechoic):
    """Reverberations dataset."""

    def __init__(self, root, ttv, download=False, transform=None, target_transform=None):
        """
        Pass all arguments to super().__init__(...).

        Args:
            root (string): Root path to the data directory.
            ttv (string): {"train"|"test"|"validate"} Which subset of the data to use.
            download (bool, optional): If True, data will be downloaded if it is not available in the root directory.
            transform (callable, optional): Optional transform to be applied to the samples.
            target_transform (callable, optional): Optional transform to be applied to the labels.
        """
        super().__init__(root, ttv, download=download, transform=transform, target_transform=target_transform)

    def set_labels(self):
        """Overrides super().set_labels to limit labels to those related to reverberation."""
        from utils.data_loader import read_recipe

        labels = read_recipe(self.recipe_path)[['reverb_time', 'reverb_delay', 'reverb_amplitude', 'filepath']]
        labels.rename(columns={'reverb_time': 'time', 'reverb_delay': 'delay', 'reverb_amplitude': 'amplitude'},
                      inplace=True)
        return labels


def demo_reverberation():
    reverb = Reverberation('../data/ani', 'test', download=False)
    print(len(reverb))

    for i in range(5):
        print(reverb[i])


class Time(Reverberation):
    """Reverberation time dataset."""

    def __init__(self, root, ttv, download=False, transform=None, target_transform=None):
        """
        Pass all arguments to super().__init__(...).

        Args:
            root (string): Root path to the data directory.
            ttv (string): {"train"|"test"|"validate"} Which subset of the data to use.
            download (bool, optional): If True, data will be downloaded if it is not available in the root directory.
            transform (callable, optional): Optional transform to be applied to the samples.
            target_transform (callable, optional): Optional transform to be applied to the labels.
        """
        super().__init__(root, ttv, download=download, transform=transform, target_transform=target_transform)

    def __getitem__(self, item):
        audio, labels = super().__getitem__(item)
        labels = labels.iloc[0]
        return audio, labels

    def set_labels(self):
        """Overrides super().set_labels to limit labels to those related to reverberation."""
        from utils.data_loader import read_recipe

        labels = read_recipe(self.recipe_path)[['reverb_time', 'filepath']]
        labels.rename(columns={'reverb_time': 'time'}, inplace=True)
        return labels


class Delay(Reverberation):
    """Reverberation delay dataset."""

    def __init__(self, root, ttv, download=False, transform=None, target_transform=None):
        """
        Pass all arguments to super().__init__(...).

        Args:
            root (string): Root path to the data directory.
            ttv (string): {"train"|"test"|"validate"} Which subset of the data to use.
            download (bool, optional): If True, data will be downloaded if it is not available in the root directory.
            transform (callable, optional): Optional transform to be applied to the samples.
            target_transform (callable, optional): Optional transform to be applied to the labels.
        """
        super().__init__(root, ttv, download=download, transform=transform, target_transform=target_transform)

    def set_labels(self):
        """Overrides super().set_labels to limit labels to those related to reverberation."""
        from utils.data_loader import read_recipe

        labels = read_recipe(self.recipe_path)[['reverb_delay', 'filepath']]
        labels.rename(columns={'reverb_delay': 'delay'}, inplace=True)
        return labels


class Amplitude(Reverberation):
    """Reverberation amplitude dataset."""

    def __init__(self, root, ttv, download=False, transform=None, target_transform=None):
        """
        Pass all arguments to super().__init__(...).

        Args:
            root (string): Root path to the data directory.
            ttv (string): {"train"|"test"|"validate"} Which subset of the data to use.
            download (bool, optional): If True, data will be downloaded if it is not available in the root directory.
            transform (callable, optional): Optional transform to be applied to the samples.
            target_transform (callable, optional): Optional transform to be applied to the labels.
        """
        super().__init__(root, ttv, download=download, transform=transform, target_transform=target_transform)

    def set_labels(self):
        """Overrides super().set_labels to limit labels to those related to reverberation."""
        from utils.data_loader import read_recipe

        labels = read_recipe(self.recipe_path)[['reverb_amplitude', 'filepath']]
        labels.rename(columns={'reverb_amplitude': 'amplitude'}, inplace=True)
        return labels


if __name__ == '__main__':
    demo_reverberation()
