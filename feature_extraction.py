import pathlib
import scipy.fft
from scipy.io import wavfile
import soundfile
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy.signal import correlate

def load_data_wavfile(path):
    """
    returns data loaded with wavfile and samplerate
    """
    samplerate, data = wavfile.read(path)
    return data, samplerate

def load_data_soundifle(path):
    """
    returns data loaded with soundfile and samplerate
    """
    data, samplerate = soundfile.read(path)
    return data, samplerate

def save_as_wav(data, path, samplerate=48000):
    """
    saves data as wav in specified path
    there's almost the same saving method in test3.py
    """
    soundfile.write(path, data, samplerate, "PCM_24", format="WAV")

def draw(data, title, range, ylabel, xlabel, ylim=None, xlim=None):
    """
    one line function for drawing via plt
    """
    plt.plot(range, data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

def calculateILD(channels: np.ndarray):
    """
    simple ILD calculation, takes mean of the channel and divides it
    :return: scalar of left_ch / right_ch
    """
    left = np.mean(channels[:,0])
    right = np.mean(channels[:,1])
    # print(left, right)
    return np.divide(left, right)


def calculateILDdb(channels: np.ndarray):
    """
    ILD calculation in dB, takes the mean of the left and right channels,
    and computes the ILD in decibels (dB).
    :return: scalar ILD in dB
    """
    # print(channels.shape)
    left = np.mean(channels[:, 0])
    right = np.mean(channels[:, 1])

    # ILD w dB
    ild_db = 20 * np.log10(left/right)

    return ild_db

def calculateILDrmsdb(channels):
    """
    calculates ILD by dividing rms of left channel by right channel but to match with sadie chart right/left
    :return: scalar ILD in dB
    """
    rms_left = np.sqrt(np.mean(np.square(channels[:, 0])))
    rms_right = np.sqrt(np.mean(np.square(channels[:, 1])))
    # min_value = 1e-10
    # if rms_right < min_value:
    #     rms_right = min_value
    # if rms_left < min_value:
    #     rms_left = min_value
    # print(rms_right)
    # return 20 * np.log10(rms_left/rms_right)
    return 20 * np.log10(rms_right/rms_left)

def calculateITD(channels: np.ndarray, sample_rate: int):
    """
    calculates ITD by using cross correlation lag
    :return: interaural time difference
    """
    left = channels[:, 0]
    right = channels[:, 1]
    # left = channels[0]
    # right = channels[1]
    correlation = signal.correlate(left, right, mode="full")
    lags = signal.correlation_lags(left.size, right.size, mode="full")
    # lag = lags[np.argmax(correlation)]
    max_index = np.argmax(correlation)

    # if max_index > 0 and max_index < len(correlation) - 1:
    #     # Maksimum i jego sąsiedzi
    #     y0 = correlation[max_index - 1]
    #     y1 = correlation[max_index]
    #     y2 = correlation[max_index + 1]
    #
    #     # Interpolacja paraboliczna, by znaleźć lepsze przybliżenie , zwraca overflow
    #     if (2 * y1 - y2 - y0) != 0:
    #         delta = (y2 - y0) / (2 * (2 * y1 - y2 - y0))
    #         interpolated_lag = lags[max_index] + delta
    #     else:
    #         interpolated_lag = lags[max_index]
    # else:
    #     # Jeśli interpolacja się nie uda, korzystamy z klasycznego maksimum
    interpolated_lag = lags[max_index]

    return interpolated_lag / sample_rate


def calculateITD_windowed(channels: np.ndarray, sampling_rate: float, window_size: float = 0.02, overlap: float = 0.5):
    """
    Calculates ITD using a windowed cross-correlation approach.

    :param channels: 2D numpy array where the first column is the left channel and the second column is the right channel.
    :param sampling_rate: Sampling rate of the audio signal in Hz.
    :param window_size: Size of each window in seconds (default is 20 ms).
    :param overlap: Overlap between windows as a fraction (default is 50%).
    :return: Average ITD in seconds across all windows.
    """
    # Oddzielenie lewego i prawego kanału
    left = channels[:, 0]
    right = channels[:, 1]

    # Przeliczenie rozmiaru okna i kroku na próbki
    window_length = int(window_size * sampling_rate)  # długość okna w próbkach
    step = int(window_length * (1 - overlap))  # krok między oknami w próbkach

    itd_values = []

    # Przechodzenie przez sygnał w oknach czasowych
    for start in range(0, len(left) - window_length + 1, step):
        left_window = left[start:start + window_length]
        right_window = right[start:start + window_length]

        # Korelacja wzajemna dla okna
        cross_corr = correlate(left_window, right_window, mode='full')
        max_corr_index = np.argmax(cross_corr)

        # Obliczenie opóźnienia w próbkach
        delay_in_samples = max_corr_index - (len(right_window) - 1)

        # Przeliczenie na sekundy
        itd_seconds = delay_in_samples / sampling_rate
        itd_values.append(itd_seconds)

    # Zwracamy średnią ITD z całego sygnału
    return np.mean(itd_values)

def checkITD(ear_distance: float = 0.2, sound_speed: int = 343, alpha: int = 0):
    ear_distance = 0.18
    return (ear_distance * np.sin(np.radians(alpha)) / sound_speed) #* np.cos(alpha)

def calculate_ITD_ch(data, sample_rate):
    # Znajdź przesunięcie odpowiadające maksymalnej korelacji
    # delay_samples = np.argmax(correlation) - len(right_channel) + 1
    # Przekształć próbki na czas (sekundy)
    # itd = delay_samples / sample_rate
    # return itd
    pass


def calculate_features(data, samplerate=48000):
    """
    function to calculate ITD and ILD on given data
    :return: ITD, ILD
    """
    data_low_freq_left, data_low_freq_right, data_high_freq_left, data_high_freq_right = filter_pass(samplerate, data)
    # print(data_low_freq.shape)
    # plt.plot(data_low_freq_left)
    # plt.plot(data_low_freq_right)
    # plt.show()
    data_low_freq = np.stack([data_low_freq_left, data_low_freq_right]).T
    data_high_freq = np.stack([data_high_freq_left, data_high_freq_right]).T
    # data_low_freq = np.array([data_low_freq_left.T, data_low_freq_left.T])
    # data_high_freq = np.array([data_high_freq_left, data_high_freq_right])
    # print(data_low_freq.shape)
    # if data_low_freq.shape[1] < 3:
    #     plt.plot(data_low_freq)
    # plt.show()
    return calculateITD(data_low_freq, samplerate), calculateILDrmsdb(data_high_freq)

def calculate_features_on_directory(directory, plot=False):
    """
    function to calculate ITD and ILD for whole dataset
    :return: ITD_sorted, ILD_sorted
    """
    ITD = []
    ILD = []
    for path in directory:
        # print(path)
        data, samplerate = load_data_soundifle(pathlib.PureWindowsPath(path)) #shape (samples, channels)
        current_ITD, current_ILD = calculate_features(data, samplerate)
        ITD.append(current_ITD)
        ILD.append(current_ILD)
        # return ITD, ILD
    if plot:
        # print(ITD)

        draw(ITD, f"ITD chart on directory {path.parent}", range(-90,91), "time in sec", "angle")
        # plt.plot(ILD)
        # plt.show()
        draw(ILD, f"ILD chart on directory {path.parent}", range(-90,91), "ILD in db", "angle")
    print("done")
    return ITD, ILD


def filter_pass(samplerate, data, plot_signal=False, plot_filter=False):
    """
    filters signal below and above cutoff frequency
    :return: filtered signal below cutoff frequency left and right, filtered signal above cutoff frequency left and right
    """
    cutoff_freq_hz = 1500
    # nyq = 0.5 * samplerate
    # znormalizowana f nyquista, ją chcemy analizować lub filtrować , potrzebna do signal.butter jeśli nie podajemy fs
    # normal_cutoff = cutoff_freq_hz / nyq
    # order czyli rząd - ok 4

    # print(data.shape)
    # Create low-pass filter
    sos_low = signal.butter(4, cutoff_freq_hz, 'lowpass', fs=samplerate, output='sos', analog=False)
    # filtered_low = signal.sosfilt(sos_low, data)
    # filtered_low_left = signal.sosfiltfilt(sos_low, data[:, 0])
    # filtered_low_right = signal.sosfiltfilt(sos_low, data[:, 1])
    filtered_low_left = signal.sosfilt(sos_low, data[:, 0])
    filtered_low_right = signal.sosfilt(sos_low, data[:, 1])

    # Create high-pass filter
    sos_high = signal.butter(4, cutoff_freq_hz, 'high', fs=samplerate, output='sos', analog=False)
    # filtered_high_left = signal.sosfiltfilt(sos_high, data[:, 0])
    # filtered_high_right = signal.sosfiltfilt(sos_high, data[:, 1])
    filtered_high_left = signal.sosfilt(sos_high, data[:, 0])
    filtered_high_right = signal.sosfilt(sos_high, data[:, 1])

    # Plot the filtered signals
    if plot_signal:
        plt.figure(figsize=(10, 6))
        plt.plot(data[:,0], label='Original left Signal', alpha=0.5)
        plt.plot(data[:,1], label='Original right Signal', alpha=0.5)
        plt.plot(filtered_low_left, label='Filtered Low-Pass left Signal', linestyle='--')
        plt.plot(filtered_low_right, label='Filtered low-Pass right Signal', linestyle='--')
        plt.plot(filtered_high_left, label='Filtered High-Pass left Signal', linestyle='-.')
        plt.plot(filtered_high_right, label='Filtered High-Pass right Signal', linestyle='-.')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title('Signal Filtering')
        plt.show()

    if plot_filter:
        w, h = signal.sosfreqz(sos_low, worN=2000, fs=samplerate)
        plt.plot(w, abs(h))
        plt.title('Charakterystyka filtra dolnoprzepustowego')
        plt.xlabel('Częstotliwość (Hz)')
        plt.ylabel('Amplituda')
        plt.xlim(0,3000)
        plt.show()

        w, h = signal.sosfreqz(sos_high, worN=2000, fs=samplerate)
        plt.plot(w, abs(h))
        plt.title('Charakterystyka filtra górnoprzepustowego')
        plt.xlabel('Częstotliwość (Hz)')
        plt.ylabel('Amplituda')
        plt.xlim(0, 3000)
        plt.show()

    return filtered_low_left, filtered_low_right, filtered_high_left, filtered_high_right

# do wizualizacji , czatowe
def plot_frequency_spectrum(data, samplerate):
    # Wykonanie FFT na sygnale
    N = len(data)
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(N, 1 / samplerate)

    # Zatrzymaj się na połowie (częstotliwości dodatnie)
    idx = np.where(xf >= 0)
    xf = xf[idx]
    yf = np.abs(yf[idx])

    plt.figure()
    plt.plot(xf, yf)
    plt.title('Widmo częstotliwościowe sygnału wejściowego')
    plt.xlabel('Częstotliwość (Hz)')
    plt.ylabel('Amplituda')
    plt.xlim(0, 5000)  # Ogranicz wykres do 5000 Hz, aby zobaczyć więcej szczegółów poniżej 1500 Hz
    plt.show()

# czatowe
def smooth_itd(itd_values, window_len=5):
    """Smoothing ITD values using a simple moving average."""
    return np.convolve(itd_values, np.ones(window_len)/window_len, mode='valid')

if __name__ == "__main__":
    print("in main of feature_extraction")
    # directory = pathlib.Path("C:/Users/uzytek/PycharmProjects/inzynierka/files/ignitionDB/").iterdir()
    # path = pathlib.Path("D:/inzynierka_baza/wynikowy/542740__papajelly__old-file-cabinet-opening-andclosing")
    # directory = path.iterdir()
    # directory = sorted(directory, key=lambda x: int(x.stem.replace(path.stem,"")))
    # path = "./files/ignitionDB/44.wav"
    # samplerate, data = load_data(path)
    # data, samplerate = load_data_soundifle(path)
    # data_left = data.T[0]
    # data_right = data.T[1]
    # filtered_low_left, filtered_low_right, filtered_high_left, filtered_high_right = filter_pass(samplerate, data)
    # data_low_freq = np.stack([filtered_low_left, filtered_low_right]).T
    # data_high_freq = np.stack([filtered_high_left, filtered_high_right]).T
    # print(data_low_freq.shape)
    # save_as_wav(data_low_freq, "C:/Users/uzytek/PycharmProjects/inzynierka/files/filtered/sosfilt_ignition_low_freq.wav", samplerate)
    # save_as_wav(data_high_freq, "C:/Users/uzytek/PycharmProjects/inzynierka/files/filtered/sosfilt_ignition_high_freq.wav", samplerate)
    # plot_frequency_spectrum(data_left, samplerate)
    # frequencies, times, spectrogram = signal.spectrogram(data_left, samplerate)
    # plt.pcolormesh(times, frequencies, np.log(spectrogram))
    # plt.imshow(spectrogram)
    # plt.ylabel("Frequency [Hz]")
    # plt.xlabel("Time [sec]")
    # plt.show()
    # print(data[:50,0], data[:50,1])
    # ILD = calculateILD(data)
    # print(ILD)
    # ITD = calculateITD(data, samplerate)
    # print("ITD: ", ITD)
    # print("ITD check: ", checkITD(alpha=44))

    # itd_theoretical = []
    # for angle in range(-90, 91):
    #     itd_theoretical.append(checkITD(alpha=angle))

    # plot all charts
    # itd_all = []
    # ild_all = []
    # parent_directory = list(pathlib.Path("D:/inzynierka_baza/wynikowy").iterdir())
    # # i=0
    # for directory in parent_directory:
    #     stem = pathlib.Path(directory).stem
    #     directory = pathlib.Path(directory).iterdir()
    #     directory = sorted(directory, key=lambda x: int(x.stem.replace(stem,"")))
    #     itd_real, _ = calculate_features_on_directory(directory, plot=False)
    #     # itd_all.append(itd_real)
    #     itd_all.append(itd_real)
    #     # i+=1
    #     # if i>5:
    #     #     break
    # plt.plot(np.array(itd_all).T)
    # plt.show()
    # with open("obliczone-ild.txt", "w")as f:
    #     for directory, ild in zip(parent_directory, ild_all):
    #         f.write(f"{directory.stem}: {ild}\n")
    # print(ild_all)

    # itd_real, ild_real = calculate_features_on_directory(directory, plot=True)


    # plt.plot(rng, itd_theoretical)
    # plt.plot(smooth_itd(itd_real), rng)
    # plt.show()

# check difference between two signals -75 and -90 channels
# path = "D:/inzynierka_baza/wynikowy/759264__lil_slugger__cartoon-slurp"
# data, samplerate = load_data_soundifle(path+"/-75.wav")
# plt.plot(data[:,0], label="-75")
# data, samplerate = load_data_soundifle(path+"/-90.wav")
# plt.plot(data[:,0], label=-90)
# plt.legend()
# plt.show()