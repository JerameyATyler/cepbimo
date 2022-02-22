def apply_hrtf(x, zenith, azimuth):
    """Apply the HRTF to the signal"""
    from utils import audiosegment_to_array, arrays_to_audiosegment
    from pydub import AudioSegment
    from data_loader import list_hrtf_data
    from scipy.signal import convolve

    left, right = sorted(list(list_hrtf_data()[zenith][azimuth]))
    left = AudioSegment.from_wav(left)
    right = AudioSegment.from_wav(right)
    h = AudioSegment.from_mono_audiosegments(left, right)

    fs = max(h.frame_rate, x.frame_rate)

    if fs > h.frame_rate:
        h = h.set_frame_rate(fs)
    if fs > x.frame_rate:
        x = x.set_frame_rate(fs)

    if x.channels == 1:
        x = x.set_channels(2)

    channels_h = h.split_to_mono()
    channels_x = x.split_to_mono()
    left_h = audiosegment_to_array(channels_h[0])
    left_x = audiosegment_to_array(channels_x[0])
    right_h = audiosegment_to_array(channels_h[1])
    right_x = audiosegment_to_array(channels_x[1])

    left = convolve(left_h, left_x)
    right = convolve(right_h, right_x)

    return arrays_to_audiosegment(left, right, fs)


def apply_reflection(x, amplitude, delay, zenith, azimuth):
    """Apply the reflection to the signal."""
    from pydub import AudioSegment
    from utils import audiosegment_to_array, arrays_to_audiosegment

    x = apply_hrtf(x, zenith, azimuth)
    fs = x.frame_rate

    channels = x.split_to_mono()
    left = audiosegment_to_array(channels[0]) * amplitude
    right = audiosegment_to_array(channels[1]) * amplitude
    y = AudioSegment.silent(delay) + arrays_to_audiosegment(left, right, fs)

    return y


def apply_reverberation(x, amplitude, delay, time):
    """Apply the reverberation to the signal."""
    from pydub import AudioSegment
    import numpy as np
    from scipy.signal import fftconvolve
    from utils import audiosegment_to_array, arrays_to_audiosegment

    fs = x.frame_rate

    length = fs * time * 2
    t = np.linspace(0, int(np.ceil(length / fs)), int(length + 1))
    envelope = np.exp(-1 * (t / time) * (60 / 20) * np.log(10)).transpose()

    left_reverb = np.random.randn(t.shape[0], ) * envelope
    right_reverb = np.random.randn(t.shape[0], ) * envelope

    channels = x.split_to_mono()
    left = audiosegment_to_array(channels[0])
    right = audiosegment_to_array(channels[1])

    left = fftconvolve(left_reverb, left) * amplitude
    right = fftconvolve(right_reverb, right) * amplitude
    y = AudioSegment.silent(delay) + arrays_to_audiosegment(left, right, fs)

    return y


def mix_parts(parts):
    """Mix the parts into a signal."""
    from pydub import AudioSegment

    sounds = [AudioSegment.from_mp3(p) for p in parts]
    t = max([len(p) for p in sounds])

    s = AudioSegment.silent(duration=t)

    for p in sounds:
        s = s.overlay(p)

    return s


def mix_reflections(x, count, amplitudes, delays, zeniths, azimuths):
    """Mix the reflections into a signal."""
    from pydub import AudioSegment

    reflections = [apply_reflection(x, amplitudes[r], delays[r], zeniths[r], azimuths[r]) for r in range(count)]
    t = max([len(r) for r in reflections])

    s = AudioSegment.silent(duration=t)
    s = s.overlay(x)

    for r in reflections:
        s = s.overlay(r)
    return s


def sum_signals(x, y):
    """Sum two signal."""
    from pydub import AudioSegment

    t = max(len(x), len(y))

    s = AudioSegment.silent(duration=t)
    s = s.overlay(x)
    s = s.overlay(y)

    return s


def adjust_signal_to_noise(x, dB):
    """Add white noise to a signal."""
    from pydub.generators import WhiteNoise

    noise = WhiteNoise().to_audio_segment(duration=len(x))

    return noise.overlay(x, gain_during_overlay=dB)
