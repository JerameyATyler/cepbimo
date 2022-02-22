def amplitude_spectrum(x):
    """Calculate the amplitude spectrum for the signal."""
    from numpy.fft import fft
    import numpy as np
    return np.abs(fft(x))


def power_spectrum(x):
    """Calculate the power spectrum for the signal."""
    return amplitude_spectrum(x) ** 2


def phase_spectrum(x):
    """Calculate the phase spectrum for the signal."""
    from numpy.fft import fft
    import numpy as np
    return np.angle(fft(x))


def log_spectrum(x):
    """Calculate the log of the spectrum for the signal."""
    import numpy as np
    return np.log(amplitude_spectrum(x))


def cepstrum(x, fs, offset, window_length):
    """Calculate the cepstrum for the signal."""
    from scipy.signal.windows import hamming
    from numpy.fft import ifft
    import numpy as np

    w = hamming(window_length, False)

    x = x[offset:offset + window_length] * w

    number_unique_points = int(np.ceil((window_length + 1) / 2))

    C = np.real(ifft(log_spectrum(x)))[0:number_unique_points]
    q = np.arange(0, number_unique_points) / fs

    return C, q
