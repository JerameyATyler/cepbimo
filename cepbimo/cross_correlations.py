def xcorr(x, y, lag):
    from scipy.signal import correlate

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must have the same length')

    c = correlate(x, y)

    if lag is None:
        lag = Nx - 1

    if lag >= Nx or lag < 1:
        raise ValueError

    return c[Nx - 1 - lag: Nx + lag]

def ncorr(x, y, lag):
    import numpy as np

    c = xcorr(x, y, lag)

    s = np.sqrt(sum(x ** 2) * sum(y ** 2))

    if s == 0:
        c *= 0
    else:
        c /= s
    return c

