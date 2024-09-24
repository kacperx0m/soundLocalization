import pathlib

import numpy as np
from scipy.io import wavfile
import soundfile
import matplotlib.pyplot as plt
from scipy import signal

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

    :param channels: ndarray with shape (n_samples, 2), where the first column
                     is the left channel and the second is the right channel.
    :return: scalar ILD in dB
    """
    left = np.mean(channels[:, 0])
    right = np.mean(channels[:, 1])

    # Unikamy dzielenia przez zero
    # if right == 0:
    #     right = 1e-10  # Bardzo mała wartość, aby uniknąć dzielenia przez zero

    # ILD w dB
    ild_db = 10 * np.log10(right / left)

    return ild_db

def calculateILDrms(channels):
    rms_left = np.sqrt(np.mean(channels[0]**2))
    rms_right = np.sqrt(np.mean(channels[1]**2))
    return rms_left/rms_right

def calculateITD(channels: np.ndarray, sample_rate: int):
    """
    calculates ITD by using cross correlation lag
    :return: interaural time difference
    """
    left = channels[:, 0]
    right = channels[:, 1]
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


import numpy as np
from scipy.signal import correlate


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


# Przykładowe użycie:
# channels = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # Przykładowa tablica sygnałów
# sampling_rate = 44100  # Częstotliwość próbkowania 44.1 kHz
# itd = calculateITD_windowed(channels, sampling_rate)
# print(f"Average Interaural Time Difference: {itd} seconds")


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

def extractFeatures():
    pass

def draw_ITD_chart(directory):
    """
    function to calculate ITD and ILD for whole dataset
    :return: ITD_sorted, ILD_sorted
    """
    ITD = []
    ILD = []
    for path in directory:
        # print(path)
        data, samplerate = load_data_soundifle(pathlib.PureWindowsPath(path))
        ITD.append(calculateITD(data, samplerate))
        ILD.append(calculateILDdb(data))
    # print(ITD_sorted)
    # plt.plot(ITD_sorted)
    # plt.show()
    # plt.plot(ILD_sorted)
    # plt.show()
    print("done")
    return ITD, ILD

def draw_ITD_chart_original(directory):
    """
    function to calculate ITD and ILD for whole dataset
    :return: ITD_sorted, ILD_sorted
    """
    ITD = []
    ILD = []
    for path in directory:
        # print(path)
        data, samplerate = load_data_soundifle(pathlib.PureWindowsPath(path))
        ITD.append(calculateITD(data, samplerate))
        ILD.append(calculateILDdb(data))
    # print(ITD_sorted)
    plt.plot(ITD)
    plt.show()
    plt.plot(ILD)
    plt.show()
    return ITD, ILD

def filter_pass():
    pass

def smooth_itd(itd_values, window_len=5):
    """Smoothing ITD values using a simple moving average."""
    return np.convolve(itd_values, np.ones(window_len)/window_len, mode='valid')

if __name__ == "__main__":
    directory = pathlib.Path("C:/Users/uzytek/PycharmProjects/inzynierka/files/frogsDB/").iterdir()
    directory = sorted(directory, key=lambda x: int(x.stem))
    # path = "./files/rms_ignitionDB/44.wav"
    # samplerate, data = load_data(path)
    # data, samplerate = load_data_soundifle(path)
    # data_left = data.T[0]
    # data_right = data.T[1]
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

    itd_theoretical = []
    for angle in range(-90, 91):
        itd_theoretical.append(checkITD(alpha=angle))
    itd_real, ild_real = draw_ITD_chart(directory)
    rng = range(-90, 91)
    # print(len(rng), len(itd_real))
    # plt.plot(rng, itd_real)
    # plt.plot(rng, itd_theoretical)
    # # plt.plot(smooth_itd(itd_real), rng)
    # plt.title("ITD chart on full database")
    # plt.ylabel("time in sec")
    # plt.xlabel("angle")
    # plt.show()

    plt.plot(rng, ild_real)
    plt.title("ILD chart on full database")
    plt.ylabel("ILD (dB)")
    plt.xlabel("angle")
    plt.show()

# fig, axs = plt.subplots(3,1)
# axs[0].plot(ILD)
# axs[0].set_title("ILD")
# axs[1].plot(data[0])
# axs[1].set_title("data0")
# axs[2].plot(data[1])
# axs[2].set_title("data1")
# plt.show()

# plt.plot(data)
# plt.show()