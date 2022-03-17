from datasets.Anechoic import Anechoic


class Reflections(Anechoic):
    """Reflections dataset."""

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
        """Overrides super().set_labels to limit labels to those related to reflections."""
        from utils.data_loader import read_recipe

        labels = read_recipe(self.recipe_path)[
            ['reflection_count', 'reflection_amplitude', 'reflection_delay', 'reflection_zenith', 'reflection_azimuth',
             'filepath']]
        labels.rename(
            columns={'reflection_count': 'count', 'reflection_amplitude': 'amplitude', 'reflection_delay': 'delay',
                     'reflection_zenith': 'zenith', 'reflection_azimuth': 'azimuth'}, inplace=True)
        return labels


def demo_reflections():
    reflections = Reflections('../data/ani', 'test', download=False)
    print(len(reflections))

    for i in range(5):
        print(reflections[i])


class Count(Reflections):
    """Reflection count dataset."""

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

        labels = read_recipe(self.recipe_path)[['reflection_count', 'filepath']]
        labels.rename(columns={'reflection_count': 'count'}, inplace=True)
        return labels


if __name__ == '__main__':
    demo_reflections()
