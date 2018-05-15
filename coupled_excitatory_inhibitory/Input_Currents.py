"""
These functions generate either square wave or sin wave as input to any model.
In this case, we used it as inputs to Coupled excitatory-inhibitory model
"""
def Sin_current(Duration, Ml, Amp, Fs, Fc, Nois):

    import numpy as np
    # Time specifications:
    dt = 1 / Fs                                                    # Seconds per sample
    t =  np.array(np.linspace(0, Duration, Duration//dt))    # Seconds
    I = (Amp * -np.sin(2 * np.pi * Fc * t)) / 2 + Ml               # Signal
    I = I + np.array(Nois * np.random.randn(1, len(I)) )                     # Adding some noise to the signal
    return I, t

def Sqr_current(Duration, Ml, Amp, Fs, Fc, Nois):

    import numpy as np
    from scipy import signal
    # Time specifications:
    dt = 1 / Fs                                                    # Seconds per sample
    t =  np.array(np.linspace(0, Duration, Duration//dt))    # Seconds
    I = (Amp * -signal.square(2 * np.pi * Fc * t)) / 2 + Ml        # Signal
    I = I + np.array(Nois * np.random.randn(1, len(I)))  # Adding some noise to the signal
    return I, t




