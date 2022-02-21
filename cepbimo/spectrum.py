def amplitude_spectrum(x):
    from numpy.fft import fft
    import numpy as np
    return np.abs(fft(x))

def power_spectrum(x):
    return amplitude_spectrum(x)**2

def phase_spectrum(x):
    from numpy.fft import fft
    import numpy as np
    return np.angle(fft(x))

def log_spectrum(x):
    import numpy as np
    return np.log(amplitude_spectrum(x))

def cepstrum(x, fs, offset, window_length):
    from scipy.signal.windows import hamming
    from numpy.fft import ifft
    import numpy as np

    w = hamming(window_length, False)

    x = x[offset:offset + window_length] * w

    number_unique_points = int(np.ceil((window_length + 1) / 2))

    C = np.real(ifft(log_spectrum(x)))[0:number_unique_points]
    q = np.arange(0, number_unique_points) / fs

    return C, q