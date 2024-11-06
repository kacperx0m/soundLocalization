import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import sofar as sf
import soundfile
from pydub import AudioSegment

sampleRate = 48000
#bits = 24
sofaPath = "C:/Users/uzytek/Desktop/studia/inzynierka/D1_HRIR_SOFA/D1_48K_24bit_256tap_FIR_SOFA.sofa"
#wavPath = "C:/Users/uzytek/Desktop/studia/inzynierka/D1_HRIR_WAV/48K_24bit"

def draw_plot(signal):
    plt.plot(signal)
    plt.grid()
    plt.show()

def draw_plots(signal1, signal2, signal3=None, signal4=None):
    fig, axs = plt.subplots(2,1)
    axs[0].plot(signal1)
    axs[0].grid()
    axs[1].plot(signal2)
    axs[1].grid()
    plt.xlabel("time in secs")
    plt.show()

def generate_white_noise():
    mean = 0
    std = 0.1
    durationInSecs = 3
    silence = np.zeros(int(sampleRate*0.02))
    whiteNoise = np.random.normal(mean, std, sampleRate * durationInSecs)
    outputNoise = np.concatenate((silence, whiteNoise, silence))

    #draw_plots(whiteNoise, outputNoise)
    return whiteNoise

def generate_pink_noise():
    pass

# problem lekki
def convert_to_24bit(signal):
    # outputSignal = np.ndarray(np.abs(signal))
    # outputSignal = outputSignal.astype(np.int32)
    outputSignal = signal / np.max(np.abs(signal))
    outputSignal = (outputSignal * 2**23).astype(np.int32)

    return outputSignal

def write_to_file(filename, signal):
    # signal = frames x channels
    soundfile.write(filename, signal, sampleRate, format="WAV", subtype="PCM_24")

def load_sound(path):
    data = soundfile.read(path)
    return data

# juz w normalnym zakresie od -90 do 90 stopni
def rotate_angle(angle):
    if angle <= 180:
        realAngle = 180 - angle
    else:
        realAngle = angle + 90
    return realAngle

def get_random_angle():
    # zakres -90 do 90 stopni ale dolem
    angle = np.random.randint(90,269)
    # zakres od -90 do 90 stopni
    return rotate_angle(angle)


def set_range(angle):
    if angle <= 90:
        realAngle = -angle
    else:
        realAngle = 360-angle
    return realAngle

def normalize(signal1, signal2):
    # signal1 = map(signal1, signal1*2)
    # signal2 = map(signal2, signal2*2)
    signal1 *= 4
    signal2 *= 4
    return signal1, signal2

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def iterate_convolute_save(monoSignal, sofa, path):
    # zakres od -90 do 90 stopni
    for angle in range(90, 269):
        angle = rotate_angle(angle)
        realAngle = set_range(angle)
        sourceIndexes = np.where(sofa.SourcePosition == angle)
        hrtfResponse = sofa.Data_IR[sourceIndexes[0][0]]
        convolvedLeft = np.convolve(monoSignal, hrtfResponse[0])
        convolvedRight = np.convolve(monoSignal, hrtfResponse[1])
        convolvedLeft, convolvedRight = normalize(convolvedLeft, convolvedRight)
        convolved = np.array([convolvedLeft, convolvedRight])

        if convolved.shape[0] == 2:
            convolved = convolved.T

        write_to_file(path + str(realAngle) + ".wav", convolved)


import wave
def check_channel_number(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        channels = wav_file.getnchannels()
        print(f"Channels: {channels}")
        if channels == 1:
            print("File is mono.")
        elif channels == 2:
            print("File is stereo.")
        else:
            print("More than 2 channels.")


#check_channel_number('./files/applause-mono-24bit-48khz.wav')

whiteNoiseSignal = generate_white_noise()
#print(whiteNoiseSignal.shape)
#whiteNoiseSignal = convert_to_24bit(whiteNoiseSignal)
#write_to_file(whiteNoiseSignal, "./files/white_noise.wav")


#whiteNoiseSignal = load_sound("./files/applause-mono-24bit-48khz.wav")
applauseSignal = load_sound("./files/applause-mono-24bit-48khz.wav")[0]
# print(applauseSignal[0].shape, whiteNoiseSignal.shape, applauseSignal[0].dtype, whiteNoiseSignal.dtype)
draw_plots(whiteNoiseSignal, applauseSignal)
#draw_plot(whiteNoiseSignal)

#sofa = sf.Sofa("SimpleFreeFieldHRIR")
#sofa.list_dimensions
sofa = sf.read_sofa(sofaPath)
#sofa.inspect()
#sofa.list_dimensions

#
# iterate_convolute_save(whiteNoiseSignal, sofa, "./files/noiseDB/")
# iterate_convolute_save(applauseSignal, sofa, "./files/applauseDB/")
# kluczowe dane
#print(sofa.Data_IR[0])

# wykres
# draw_plots(sofa.Data_IR[12][0], sofa.Data_IR[12][1])

# [ azymut, kat do horyzontu, elewacja w m]
# co 24, a potem 22 nowy kat?
# 2200 zaczyna sie 90
# print(sofa.SourcePosition[0])
#print(sofa.Data_IR.where())

# zapis kata potrzebny do SADIE
randAngle = get_random_angle()
# zapis ladny
realAngle = set_range(randAngle)
#print(randAngle)
#print(realAngle)
sourceIndexes = np.where(sofa.SourcePosition == randAngle)
#print(sofa.SourcePosition[sourceIndexes[0]])

hrtfResponse = sofa.Data_IR[sourceIndexes[0][0]]
# print(hrtfResponse)
# convolvedLeft = np.convolve(whiteNoiseSignal[0], hrtfResponse[0])
# convolvedRight = np.convolve(whiteNoiseSignal[0], hrtfResponse[1])

#normalized_sound = match_target_amplitude(whiteNoiseSignal, -20.0)

convolvedLeft = np.convolve(whiteNoiseSignal, hrtfResponse[0])
convolvedRight = np.convolve(whiteNoiseSignal, hrtfResponse[1])

convolvedLeft, convolvedRight = normalize(convolvedLeft, convolvedRight, target_dBFS=1.0)

#convolvedLeft = convert_to_24bit(convolvedLeft)
#convolvedRight = convert_to_24bit(convolvedRight)

convolved = np.array([convolvedLeft, convolvedRight])
#print(convolved.shape)
#print(convolved.dtype)

if convolved.shape[0] == 2:
    convolved = convolved.T

# print(convolved.shape)
draw_plots(whiteNoiseSignal, convolved)

# write_to_file("./files/audio"+str(realAngle)+".wav", convolved)
#soundfile.write("./files/audio"+str(realAngle)+".wav", convolved, sampleRate, format="WAV", subtype="PCM_24")