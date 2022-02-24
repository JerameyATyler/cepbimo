class RNG:
    """Random number generator

    Methods:

    get_composer()
    get_part_count()
    get_parts()
    get_zenith()
    get_azimuth()
    get_delay()
    get_amplitude()
    get_time()
    get_reflection_count()
    get_offset()

    Constructors:

    __init__()

    Properties (readonly):
    seed, part_limits, duration, delay_limits, time_limits, reflection_limits, rng, composers, anechoic_data, hrtf_data
    """

    def __init__(self,
                 seed='0xecc0',
                 duration=20,
                 delay_limits=(0, 60),
                 time_limits=(1, 8),
                 reflection_limits=(4, 8),
                 zenith_limits=None,
                 azimuth_limits=None):
        """Constructor.

        Arguments:

        seed, duration, delay_limits, time_limits, reflection_limits
        """
        import numpy as np
        from data_loader import list_hrtf_data, list_anechoic_data, list_composers, get_hrtfs

        self.seed = seed
        self.duration = duration
        self.delay_limits = delay_limits
        self.time_limits = time_limits
        self.reflection_limits = reflection_limits
        self.zenith_limits = zenith_limits
        self.azimuth_limits = azimuth_limits

        self.rng = np.random.default_rng(int(self.seed, 0))

        self.composers = list_composers()
        self.anechoic_data = list_anechoic_data()

        if zenith_limits is not None:
            zmin, zmax = zenith_limits
        else:
            zmin, zmax = None, None
        if azimuth_limits is not None:
            amin, amax = azimuth_limits
        else:
            amin, amax = None, None

        hrtf_data = list_hrtf_data()
        zeniths, azimuths = get_hrtfs(amin=amin, amax=amax, zmin=zmin, zmax=zmax)
        hrtfs = {z: {} for z in zeniths}
        for z in zeniths:
            for a in azimuths:
                if a in hrtf_data[z].keys():
                    hrtfs[z][a] = hrtf_data[z][a]

        self.hrtf_data = hrtfs

    def get_composer(self):
        """Select a composer from those available in the data and return as string"""
        return self.rng.choice(self.composers)

    def get_part_count(self, composer):
        """Return a random integer number of parts to mix for the provided composer"""
        parts = self.anechoic_data[composer]
        part_limits = (2, len(parts))
        return self.rng.integers(part_limits[0], part_limits[1])

    def get_parts(self, composer=None, part_count=None):
        """Select parts to be mixed. If composer or part_count are not provided they will be generated."""
        if composer is None:
            composer = self.get_composer()
        if part_count is None:
            part_count = self.get_part_count(composer)

        return self.rng.choice(self.anechoic_data[composer], part_count, replace=False)

    def get_zenith(self, azimuth=None):
        """Select an integer zenith for HRTF"""
        zeniths = sorted(list(self.hrtf_data.keys()))
        if azimuth is not None:
            zeniths = [z for z in zeniths if azimuth in self.hrtf_data[z]]

        return self.rng.choice(zeniths)

    def get_azimuth(self, zenith=None):
        """Select an integer azimuth for HRTF."""
        zeniths = []
        if zenith is not None:
            zeniths.append(zenith)
        else:
            zeniths = sorted(list(self.hrtf_data.keys()))

        azimuths = set()
        for z in zeniths:
            for a in self.hrtf_data[z]:
                azimuths.add(a)

        return self.rng.choice(list(azimuths))

    def get_delay(self):
        """Select an integer number of milliseconds to delay."""
        return self.rng.integers(low=self.delay_limits[0], high=self.delay_limits[1] + 1)

    def get_amplitude(self):
        """Select a float amplitude between 0.0-1.0."""
        return self.rng.random()

    def get_time(self):
        """Select an integer reverberation time in seconds."""
        return self.rng.integers(low=self.time_limits[0], high=self.time_limits[1] + 1)

    def get_reflection_count(self):
        """Select the integer number of reflections to apply."""
        return self.rng.integers(low=self.reflection_limits[0], high=self.reflection_limits[1] + 1)

    def get_offset(self, length):
        """
        Select the integer offset in milliseconds for a sample of (duration) seconds from a clip of (length)
        milliseconds.
        """
        length = length - self.duration * 1000
        return self.rng.integers(low=0, high=length)


def demo():
    import numpy as np

    """Demonstrate RNG class usage."""
    # Instantiate RNG object
    rng = RNG()

    # Generate a direct signal of (p_count) parts for (composer) composer at (az) azimuth degrees
    # and (ze) zenith degrees.
    c = rng.get_composer()
    p_count = rng.get_part_count(c)
    ps = rng.get_parts(c, p_count)
    # Azimuth and zenith also used for reflections.
    ze = rng.get_zenith(azimuth=90)
    az = rng.get_azimuth(ze)

    print('Source signal')
    print(f'Azimuth: {az}, Zenith: {ze}')
    print(f'Composer: {c.upper()}, Parts: {p_count}')
    [print(f'\tPart: {p}') for p in ps]

    # Generate (de) delay time and (am) amplitude for reflections/reverberation.
    # ** Note: Reverberation amplitudes need to be much smaller than reflection amplitudes or clipping occurs. **
    de = rng.get_delay()
    am = rng.get_amplitude() / 10

    # Generate (t) time for reverberation
    t = rng.get_time()

    print('Reverberation')
    print(f'Amplitude: {am:1.3}, Delay: {de} ms, Time: {t} s')

    # Generate the number of reflections
    r_count = rng.get_reflection_count()

    print(f'Reflection count: {r_count}')

    # Generate the offset for a (rng.duration) second clip from a 60,000 millisecond source signal
    off = int(np.ceil(rng.get_offset(60000) / 1000))

    print(f'Offset: {off} s')


if __name__ == '__main__':
    demo()
