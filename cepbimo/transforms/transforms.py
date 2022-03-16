def cepstrum(x):
    from utils.utils import split_channels, arrays_to_audiosegment
    from transforms.spectrum import cepstrum

    fs = x.frame_rate

    offset = 1024
    window_length = offset * 64 * 2

    left, right = split_channels(x[offset: offset + window_length])

    cl, _ = cepstrum(left, fs, offset, window_length)
    cr, _ = cepstrum(right, fs, offset, window_length)

    return arrays_to_audiosegment(left, right, fs)


def mfcc(x):
    import librosa
    from utils.utils import split_channels, arrays_to_audiosegment

    left, right = split_channels(x)

    mfcc_l = librosa.feature.mfcc(y=left, sr=x.frame_rate)
    mfcc_r = librosa.feature.mfcc(y=right, sr=x.frame_rate)
    return arrays_to_audiosegment(mfcc_l, mfcc_r, x.frame_rate)
