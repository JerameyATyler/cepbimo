def audiosegment_to_array(audiosegment):
    import numpy as np

    y = np.array(audiosegment.get_array_of_samples())
    if audiosegment.channels == 2:
        y = y.reshape((-1, 2))

    y = np.float32(y) / 2 ** 15
    return y


def array_to_audiosegment(arr, fs):
    from pydub import AudioSegment
    import numpy as np

    y = np.int16(arr * 2 ** 15)
    s = AudioSegment(y.tobytes(), frame_rate=fs, sample_width=2, channels=1)

    return s


def arrays_to_audiosegment(left, right, fs):
    from pydub import AudioSegment

    l = array_to_audiosegment(left, fs)
    r = array_to_audiosegment(right, fs)
    return AudioSegment.from_mono_audiosegments(l, r)
