#import numpy as np
#from scipy import signal
#import matplotlib.pyplot as plt
#t = np.linspace(-1, 1, 201)
#x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) + 0.1*np.sin(2*np.pi*1.25*t + 1)
#     + 0.18*np.cos(2*np.pi*3.85*t))
#xn = x + np.random.randn(len(t)) * 0.08
#
#b, a = signal.butter(3, 0.05)
#
#z0 = signal.lfilter(b, a, xn)
#
#zi = signal.lfilter_zi(b, a)
#z1, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
#z2, _ = signal.lfilter(b, a, z1, zi=zi*z1[0])
#
#y = signal.filtfilt(b, a, xn)
#
#
#plt.figure
#plt.plot(t, xn, 'b', alpha=0.75)
#plt.plot(t, z0, 'r',
#         t, z1, 'g--',
#         t, z2, 'g',
#         t, y, 'k')
#plt.legend(('noisy signal',
#            'lfilter, simple',
#            'lfilter, once',
#            'lfilter, twice',
#            'filtfilt'), loc='best')
#plt.grid(True)
#plt.show()
#
#from numpy import array, ones
#from scipy.signal import lfilter, lfilter_zi, butter
#
#b, a = butter(5, 0.25)
#zi = lfilter_zi(b, a)
#y0, zo = lfilter(b, a, ones(10), zi=zi*0.5)
#print(y0)
#
#x = array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
#y1, zf = lfilter(b, a, x, zi=zi*x[0])
#print(y1)
###############################################################################
from numpy import sin, cos, pi, linspace
from numpy.random import randn
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
from matplotlib.pyplot import plot, legend, show, hold, grid, figure, savefig

t = linspace(-1, 1, 201)
x = (sin(2 * pi * 0.75 * t*(1-t) + 2.1) + 0.1*sin(2 * pi * 1.25 * t + 1) + 0.18*cos(2 * pi * 3.85 * t))
xn = x + randn(len(t)) * 0.08

b, a = butter(3, 0.05)

zi = lfilter_zi(b, a)
z, _ = lfilter(b, a, xn, zi=zi*xn[0])

z2, _ = lfilter(b, a, z, zi=zi*z[0])

y = filtfilt(b, a, xn)

figure(figsize=(10,5))
hold(True)
plot(t, xn, 'b', linewidth=1.75, alpha=0.75)
plot(t, z, 'r--', linewidth=1.75)
plot(t, z2, 'r', linewidth=1.75)
plot(t, y, 'k', linewidth=1.75)
legend(('noisy signal',
        'lfilter, once',
        'lfilter, twice',
        'filtfilt'),
        loc='best')
hold(False)
grid(True)
show()
###############################################################################
#from scipy.signal import butter, lfilter
#
#def butter_bandpass(lowcut, highcut, fs, order=5):
#    nyq = 0.5 * fs
#    low = lowcut / nyq
#    high = highcut / nyq
#    b, a = butter(order, [low, high], btype='band')
#    return b, a
#
#
#def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#    y = lfilter(b, a, data)
#    return y
#
#
#def run():
#    import numpy as np
#    import matplotlib.pyplot as plt
#    from scipy.signal import freqz
#
#    # Sample rate and desired cutoff frequencies (in Hz).
#    fs = 5000.0
#    lowcut = 500.0
#    highcut = 1250.0
#
#    # Plot the frequency response for a few different orders.
#    plt.figure(1)
#    plt.clf()
#    for order in [3, 6, 9]:
#        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#        w, h = freqz(b, a, worN=2000)
#        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
#
#    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
#             '--', label='sqrt(0.5)')
#    plt.xlabel('Frequency (Hz)')
#    plt.ylabel('Gain')
#    plt.grid(True)
#    plt.legend(loc='best')
#
#    # Filter a noisy signal.
#    T = 0.05
#    nsamples = T * fs
#    t = np.linspace(0, T, nsamples, endpoint=False)
#    a = 0.02
#    f0 = 600.0
#    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
#    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
#    x += a * np.cos(2 * np.pi * f0 * t + .11)
#    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
#    plt.figure(2)
#    plt.clf()
#    plt.plot(t, x, label='Noisy signal')
#
#    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
#    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
#    plt.xlabel('time (seconds)')
#    plt.hlines([-a, a], 0, T, linestyles='--')
#    plt.grid(True)
#    plt.axis('tight')
#    plt.legend(loc='upper left')
#
#    plt.show()
#
#run()
###############################################################################