def get_ttv_paths():
    from pathlib import Path

    train = Path('data/train').__str__()
    test = Path('data/test').__str__()
    validate = Path('data/validate').__str__()
    return train, test, validate
