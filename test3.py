import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import sofar as sf
import soundfile

sampleRate = 48000
sofaPath = "C:/Users/uzytek/Desktop/studia/inzynierka/D1_HRIR_SOFA/D1_48K_24bit_256tap_FIR_SOFA.sofa"
#wavPath = "C:/Users/uzytek/Desktop/studia/inzynierka/D1_HRIR_WAV/48K_24bit"

def draw_plot(signal):
    plt.plot(signal)
    plt.grid()
    plt.show()

def draw_plots(signal1, signal2, signal3=None, signal4=None, title=None):
    fig, axs = plt.subplots(2,1)
    axs[0].plot(signal1)
    axs[0].grid()
    axs[0].set_ylim(-1,1)
    axs[1].plot(signal2)
    axs[1].grid()
    axs[1].set_ylim(-1,1)
    plt.xlabel("time in secs")
    if title is not None:
        fig.suptitle(title)
    plt.show()

def generate_white_noise():
    mean = 0
    std = 0.1
    durationInSecs = 0.1
    silence = np.zeros(int(sampleRate*0.05))
    whiteNoise = np.random.normal(mean, std, int(sampleRate * durationInSecs))
    outputNoise = np.concatenate((silence, whiteNoise, silence))
    for i in range(10):
        outputNoise = np.concatenate((outputNoise, silence, whiteNoise, silence))

    # draw_plots(whiteNoise, outputNoise)
    return outputNoise

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
    return data[0]

def load_sound_mono(path: str):
    """
    load data from path and if it's stereo, convert to mono by taking mean of two signals
    :return: data of type soundfile.read
    """
    data = soundfile.read(path)[0]
    if len(data.shape) > 1 and data.shape[1] == 2:
        data = np.mean(data, axis=1)
    return data

# w zakresie od -90 do 90 stopni ale pod SADIE (90 do 0, 270 do 359)
def rotate_angle(angle):
    if angle <= 180:
        realAngle = 180 - angle
    else:
        realAngle = 360 - (angle - 180)
    return realAngle

def get_random_angle(userAngle=None):
    if userAngle is not None:
        return rotate_angle(userAngle)
    # zakres -90 do 90 stopni ale dolem
    angle = np.random.randint(90,271)
    # zakres od -90 do 90 stopni
    return rotate_angle(angle)

# zakres czytelny, finalny
def set_real_range(angle):
    if angle <= 90:
        realAngle = -angle
    else:
        realAngle = 360 - angle
    return realAngle

import pyloudnorm as pyln
def normalize(signal1, signal2, method, targetLufs=-23.0):
    if method.lower() == "rms":
        rms1 = np.sqrt(np.mean(np.square(signal1)))
        rms2 = np.sqrt(np.mean(np.square(signal2)))
        max_rms = max(rms1, rms2)
        signal1 = signal1 / max_rms
        signal2 = signal2 / max_rms
        # 0.1 means fraction of rms
        return signal1 * 0.1, signal2 * 0.1
    elif method.lower() == "lufs":
        meter = pyln.Meter(sampleRate)
        combinedSignal = np.vstack((signal1, signal2)).T
        loudness = meter.integrated_loudness(combinedSignal)

        normalizedCombinedSignal = pyln.normalize.loudness(combinedSignal, loudness, targetLufs)
        signal1, signal2 = normalizedCombinedSignal.T
        return signal1, signal2
    else:
        sig1 = np.max(np.abs(signal1))
        sig2 = np.max(np.abs(signal2))

    # draw_plots(signal1, signal2, title="before")

    # reduce to 0.9 of max value
    scalar = max(sig1, sig2) * 1.1

    signal1 = signal1 / scalar
    signal2 = signal2 / scalar

    # draw_plots(signal1, signal2, title="after")

    return signal1, signal2

def repeat(monoSignal, times):
    return np.tile(monoSignal, times)

def iterate_convolute_save(monoSignal, sofa, path, norm="peak"):
    # zakres od -90 do 90 stopni
    # 89, 271 zakres normalnie
    # result = []
    for angle in range(90, 271):
        rotatedAngle = rotate_angle(angle)
        realAngle = set_real_range(rotatedAngle)
        sourceIndexes = np.where(sofa.SourcePosition[:,0] == rotatedAngle)[0]
        zeroIndex = np.where(sofa.SourcePosition[sourceIndexes][:,1] == 0)[0]
        hrtfResponse = sofa.Data_IR[sourceIndexes[zeroIndex]][0]
        convolvedLeft = np.convolve(monoSignal, hrtfResponse[0])
        convolvedRight = np.convolve(monoSignal, hrtfResponse[1])
        # print(f"this normalization is for angle {realAngle}")
        convolvedLeft, convolvedRight = normalize(convolvedLeft, convolvedRight, method=norm)
        # if angle==90 or angle ==180 or angle == 270:
        #     draw_plots(convolvedLeft, convolvedRight)
        # print(np.mean(convolvedLeft)/np.mean(convolvedRight))
        convolved = np.array([convolvedLeft, convolvedRight])

        if convolved.shape[0] == 2:
            convolved = convolved.T

        write_to_file(path + str(realAngle) + ".wav", convolved)
    #     result.append(np.mean(convolvedLeft)/ np.mean(convolvedRight))
    # return result



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

# whiteNoiseSignal = generate_white_noise()
#print(whiteNoiseSignal.shape)

# applauseSignal = load_sound_mono("./files/applause-mono-24bit-48khz.wav")
# buttonSignal = load_sound_mono("./files/button-clicking-1.wav")
# buttonSignal = repeat(buttonSignal, 3)
ignitionSignal = load_sound_mono("./files/car-ignition.wav")
# frogsSignal = load_sound_mono("./files/frogs.wav")
# musicSignal = load_sound_mono("./files/music.wav")
# horseSignal = load_sound_mono("./files/horse.wav")
# print(horseSignal)
# musicSignal = load_sound_mono("./files/music.wav")
# sfxSignal = load_sound_mono("./files/sfx.wav")
# tillSignal = load_sound_mono("./files/till-with-bell.wav")
# print(applauseSignal[0].shape, whiteNoiseSignal.shape, applauseSignal[0].dtype, whiteNoiseSignal.dtype)
# draw_plots(buttonSignal, buttonSignalTest)
# draw_plot(frogsSignal)

#sofa = sf.Sofa("SimpleFreeFieldHRIR")
#sofa.list_dimensions
sofa = sf.read_sofa(sofaPath)
# sofa.inspect()
# print(sofa.list_dimensions)

# =========
# currSignal = load_sound("./files/buttonDB/90.wav")
# left, right = np.mean(horseSignal[:, 1]), np.mean(horseSignal[:, 0])
# test =(left/right)
# print(test)
# draw_plot(horseSignal)
# ==========

#
# iterate_convolute_save(whiteNoiseSignal, sofa, "./files/normalized_noiseDB/")
# iterate_convolute_save(buttonSignal, sofa, "./files/buttonDB/")
# iterate_convolute_save(horseSignal, sofa, "./files/lufs_horseDB/", norm="lufs")
# iterate_convolute_save(applauseSignal, sofa, "./files/normalized_applauseDB/")
# iterate_convolute_save(applauseSignal, sofa, "./files/normalized_applauseDB/")
# iterate_convolute_save(frogsSignal, sofa, "./files/frogsDB/")
iterate_convolute_save(ignitionSignal, sofa, "./files/ignitionDB/", norm="peak")
iterate_convolute_save(ignitionSignal, sofa, "./files/rms_ignitionDB/", norm="rms")
iterate_convolute_save(ignitionSignal, sofa, "./files/lufs_ignitionDB/", norm="lufs")
# sig = np.array(sig).T
# plt.plot(sig, sig2)
# plt.show()
# div = np.divide(sig[0], sig[1])
# draw_plot(div[91])
# draw_plots(sig[0], sig[1])
# kluczowe dane
# print(sofa.Data_IR[0])

# wykres
# draw_plots(sofa.Data_IR[12][0], sofa.Data_IR[12][1])

# [ azymut, kat do horyzontu, elewacja w m]
# co 24, a potem 22 nowy kat?
# 2200 zaczyna sie 90
# print("source position: ", sofa.SourcePosition[0])
# print(sofa.SourcePosition)

# zapis kata potrzebny do SADIE (0,360)
# randAngle = get_random_angle(240)
# zapis ladny
# realAngle = set_real_range(randAngle)
# print("random angle: ", randAngle)
# print("real angle: ", realAngle)
# sourceIndexes = np.where(sofa.SourcePosition[:,0] == randAngle)[0]
# sourceIndexesT = np.where(sofa.SourcePosition == randAngle)
# print("source position index nr exact: ", sofa.SourcePosition[sourceIndexes[0]][11])
# print("source position index nr exact: ", sofa.SourcePosition[sourceIndexes[0]][10])
# print(sourceIndexes)
# print(np.where(sofa.SourcePosition[sourceIndexes[0]] == 0))
# print(sofa.SourcePosition[sourceIndexes[0]])
# print(sourceIndexes)
# zeroIndex = np.where(sofa.SourcePosition[sourceIndexes][:,1] == 0)[0]
# print("zero index: ",zeroIndex)
# print(sofa.SourcePosition[0].__repr__())
# print(sofa.SourcePosition[sourceIndexes[zeroIndex]])

# hrtfResponse = sofa.Data_IR[sourceIndexes[zeroIndex[0]]]
# draw_plot(hrtfResponse)
# print(hrtfResponse)
# draw_plots(hrtfResponse, hrtfResponseTest)

#normalized_sound = match_target_amplitude(whiteNoiseSignal, -20.0)

# to sprawie ze jezdzi mi dookola glowy, source index
'''
for index in range(len(sourceIndexes[0])):
    hrtfResponseTest = sofa.Data_IR[sourceIndexes[0][index]]
    # draw_plots(hrtfResponseTest[0], hrtfResponseTest[1], title="response "+str(index))
    print(sofa.SourcePosition[sourceIndexes[0][index]])
    convolvedLeft = np.convolve(applauseSignal, hrtfResponseTest[0])
    convolvedRight = np.convolve(applauseSignal, hrtfResponseTest[1])

    convolvedLeft, convolvedRight = normalize(convolvedLeft, convolvedRight)

    convolved = np.array([convolvedLeft, convolvedRight])
    #print(convolved.shape)
    #print(convolved.dtype)

    if convolved.shape[0] == 2:
        convolved = convolved.T      
'''

# convolvedLeft = np.convolve(applauseSignal, hrtfResponse[0])
# convolvedRight = np.convolve(applauseSignal, hrtfResponse[1])
#
# convolvedLeft, convolvedRight = normalize(convolvedLeft, convolvedRight)
#
# convolved = np.array([convolvedLeft, convolvedRight])
#print(convolved.shape)
#print(convolved.dtype)

# if convolved.shape[0] == 2:
#     convolved = convolved.T

# print(convolved.shape)
# draw_plots(hrtfResponse[0], hrtfResponse[1])
# draw_plots(whiteNoiseSignal, convolved)

# write_to_file("./files/audio"+str(realAngle)+".wav", convolved)
#soundfile.write("./files/audio"+str(realAngle)+".wav", convolved, sampleRate, format="WAV", subtype="PCM_24")