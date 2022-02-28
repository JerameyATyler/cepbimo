class Cepbimo:
    """Cepstrum based binaural model

    Methods:
    align()

    Constructors:

    __init__()

    Properties (readonly):
    fs, lag, XR, C, q, Xd, XNd, z
    """
    def __init__(self, x):
        import numpy as np
        from scipy.signal import convolve
        from utils import audiosegment_to_array, arrays_to_audiosegment
        from spectrum import cepstrum
        from cross_correlations import xcorr, ncorr

        offset = 1024
        window_length = offset * 64 * 2

        self.fs = x.frame_rate
        self.lag = (1 * int(self.fs / 1000.))

        x_channels = x[offset: offset + window_length].split_to_mono()
        x_left = audiosegment_to_array(x_channels[0])
        x_right = audiosegment_to_array(x_channels[1])

        self.XR = xcorr(x_left, x_right, self.lag)

        cl, ql = cepstrum(x_left, x.frame_rate, offset, window_length)
        cr, qr = cepstrum(x_right, x.frame_rate, offset, window_length)

        self.C = arrays_to_audiosegment(cl, cr, x.frame_rate)
        self.q = ql

        hl = np.concatenate((np.ones((1,)), np.zeros((298,)), -1 * cl[300:4000]))
        hr = np.concatenate((np.ones((1,)), np.zeros((298,)), -1 * cr[300:4000]))

        yl = convolve(x_left, hl)
        yr = convolve(x_right, hr)

        yl = convolve(yl, hl)
        yr = convolve(yr, hr)

        self.Xd = xcorr(yl, yr, self.lag)
        self.XNd = ncorr(yl, yr, self.lag)

        self.z = self.align(cl, cr)

    def align(self, x, y):
        import numpy as np
        from scipy.signal.windows import hann
        from scipy.signal import convolve
        early_zone = 4000
        other_zone = 700

        x = x[0:early_zone]
        maxi = max(x[other_zone:])
        maxi_dir = max(x[0:other_zone])

        x[0:other_zone] = x[0:other_zone] * (maxi / maxi_dir)

        y = y[0:early_zone]
        maxi = max(y[other_zone:])
        maxi_dir = max(y[0:other_zone])

        y[0:other_zone] = y[0:other_zone] * (maxi / maxi_dir)

        taps = self.lag

        XNd = self.XNd

        ncorr_max_index = XNd.argmax() - taps + 1

        itd_offset = np.zeros((taps,))
        ncorr_max_offset = np.zeros((abs(ncorr_max_index),))

        if ncorr_max_index == 0:
            xx = np.concatenate((itd_offset, x.transpose()))
            yy = np.concatenate((itd_offset, y.transpose()))
        else:
            xx = np.concatenate((itd_offset, x.transpose(), ncorr_max_offset))
            yy = np.concatenate((itd_offset, ncorr_max_offset, y.transpose()))

        sig_l = convolve(xx, hann(5, sym=False))
        sig_r = convolve(yy, hann(5, sym=False))

        return np.array([np.sqrt(abs(sig_l[taps + 1: taps * 50] * sig_r[taps + 1 + i:taps * 50 + i])) for i in
                         np.arange(-taps, taps)])
