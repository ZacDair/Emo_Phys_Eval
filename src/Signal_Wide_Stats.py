import os
import pandas as pd
import pickle
import heartpy as hp
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import signal as sci_sig
import neurokit2 as nk
from src import Configs

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

"""CASE Specific Functions"""


# Low - Neutral - High from numbers 0-10
def lowNeutralHigh(value):
    if value <= 3.5:
        return "L"
    elif 3.5 <= value <= 7:
        return "N"
    else:
        return "H"


# String label to number
def stringLabelToNumber(labelString):
    res = {"L-L": 0, "L-N": 1, "L-H": 2, "N-L": 3, "N-N": 4, "N-H": 5, "H-L": 6, "H-N": 7, "H-H": 8}
    return res[labelString]


# Converts Arousal/Valence values into a single value
def convertArousalValence(arousalLabels, valenceLabels):
    discreteLabels = []
    for a, v in zip(arousalLabels, valenceLabels):
        discreteLabels.append(stringLabelToNumber(lowNeutralHigh(a)+"-"+lowNeutralHigh(v)))
    return discreteLabels


# Get CASE signals
def CASE_getSignals(signalFile, labelFile, signalPath, labelPath):
    # Get subject from path
    subject = signalFile.split("_")[0].replace('\\', '')
    print("Running on Subject:", subject)

    # Load signals
    sFile = os.path.join(signalPath, signalFile)
    sigData = pd.read_csv(sFile, sep="\t", header=None)
    sigData.columns = ["DaqTime", "ECG", "BVP", "GSR", "RSP", "SKT", "emg_zygo", "emg_coru", "emg_trap"]
    sigData = sigData[["DaqTime", "ECG", "BVP"]]

    # Load labels
    lFile = os.path.join(labelPath, labelFile)
    annoData = pd.read_csv(lFile, sep="\t", header=None)
    annoData.columns = ["JsTime", "X", "Y"]

    # CASE specific data alterations - ECG and Annotations
    # Convert Time from seconds to ms with 3 decimal rounding
    sigData["DaqTime"] = sigData["DaqTime"].apply(lambda x: x * 1000).round(decimals=3)

    # Convert ECG (usually measured in millivolts (sensor I/P range +-40 mV))
    # And convert volts to milliVolts (mV) with rounding to three decimal places
    sigData["ECG"] = sigData["ECG"].apply(lambda x: ((x - 2.8) / 50) * 1000).round(decimals=3)

    # Convert Joystick values Arousal(x) and Valence(y) axis values to range [0.5 9.5]
    annoData["Valence"] = annoData["Y"].apply(lambda x: 0.5 + 9 * (x + 26225) / 52450)
    annoData["Arousal"] = annoData["X"].apply(lambda x: 0.5 + 9 * (x + 26225) / 52450)

    # Convert ECG, BVP and Label to same format as WESAD
    ecg = sigData["ECG"].to_numpy()
    ppg = sigData["BVP"].to_numpy()
    label_a = annoData["Arousal"].to_numpy()
    label_v = annoData["Valence"].to_numpy()

    # Convert continous X/Y axis arousal into discrete values
    label = convertArousalValence(label_a, label_v)
    return ecg, ppg, label


"""WESAD Specific Functions"""


# Get WESAD signals
def WESAD_getSignals(subject):
    with open(os.path.join(datasetPath, subject, str(subject + ".pkl")), 'rb') as f:
        # Extract relevant signals
        data = pickle.load(f, encoding="latin1")
        ecg = data['signal']['chest']['ECG'].flatten()
        ppg = data['signal']['wrist']['BVP'].flatten()
        label = data['label'].flatten()
    return ecg, ppg, label


"""Stats and Running Loops"""

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


# Stat Computation - Signal Wide Stats
def getStats(signal, sampleRate, dataType):
    signal = hp.filter_signal(signal, cutoff=[0.8, 2.5], sample_rate=sampleRate, filtertype='bandpass')

    # Signal duration
    dur = int(len(signal)/sampleRate)

    # Signal Measures
    workingData, measures = hp.process(signal, sampleRate)

    # Heart Rate
    hr = measures["bpm"]

    # Breathing Rate
    br = measures["breathingrate"]

    # Min, Max, Mean Amplitude Value
    minVal = np.min(signal)
    maxVal = np.max(signal)
    meanVal = np.mean(signal)

    # SNR
    snr = signaltonoise(signal)

    # # Simulate same signal type
    # if dataType == "ecg":
    #     synthSig = nk.ecg_simulate(duration=dur, sampling_rate=sampleRate, noise=0)
    # else:
    #     synthSig = nk.ppg_simulate(duration=dur, sampling_rate=sampleRate)
    #
    # synthSig = synthSig.flatten()
    #
    # snrSynth = signaltonoise(synthSig)
    # diff = abs(signal) - abs(synthSig)
    #
    # print(f"Our SNR: {snr}, Clean SNR: {snrSynth}")
    # print(f"Mean Amplitude Diff: {np.mean(diff)}")


    # Correlation between raw and clea
    # correlate_result = np.correlate(signal, synthSig, 'full')
    # print("len(a)", len(signal), "len(b)", len(synthSig), "len(correlate_result)", len(correlate_result))


    print(f"Beats Per Min: {hr}, Breathing Rate: {br}")
    print(f"Min: {minVal}, Max: {maxVal}, Mean {meanVal}")

    # Peak/ Rejected Peaks

    # Similarity
    return hr, br, snr


def firstMiddleLast(signal, sampleRate, windowSize):
    length = sampleRate*windowSize
    first = signal[0:length]
    middleMark = int(len(signal) / 2)
    middle = signal[middleMark:middleMark + length]
    end = len(signal)
    last = signal[end - length:end]

    return first, middle, last


# Window the data
def getLabelledWindows(signalData, labelData, windowSize, sigSampleRate, labSampleRate, dataName):
    # Store the windows
    res = {}
    # Loop through signal data - using relative conversions to Hz values based on Signal and Label sample rates
    sigIter, labIter, i, windowPos = 0, 0, 0, 0
    while i < int(len(signalData)/(windowSize*sigSampleRate)):
        # Store the updated positions based on window size and sample rate
        sigIncr = windowSize*sigSampleRate
        labIncr = windowSize*labSampleRate
        # Store the associated window of data and it's coresponding labels
        #res[i] = {dataName: signalData[windowPos:windowPos + sigIncr], "Label": int(np.mean(labelData[labIter:labIter+labIncr]))}
        res[i] = {dataName: signalData[windowPos:windowPos + sigIncr]}
        # Update the position iteration values
        windowPos += sigIncr
        labIter += labIncr
        i += 1
    return res

# CASE Specific Variables
caseDatasetPath = Configs.dataset_path
signalPath = os.path.join(caseDatasetPath, "raw", "physiological")
labelsPath = os.path.join(caseDatasetPath, "raw", "annotations")
signalFiles = os.listdir(signalPath)
labelsFiles = os.listdir(labelsPath)

# Overaching dataframe
res = []
bpmDiff = []
breathingRate = []
ibiDiff = []
breathingDiff = []
ecgBR = 0
ppgBR = 0
# Loop Through CASE
with open("CASE_COMP.txt", "a+") as f:
    for sFile, lFile in zip(signalFiles, labelsFiles):
        if sFile.endswith(".txt"):
            subject = str(sFile.split("_")[0].replace('\\', ''))
            ecg, ppg, label = CASE_getSignals(sFile, lFile, signalPath, labelsPath)

            ecgWindows = getLabelledWindows(ecg, label, 60, 1000, 1000, "ECG")
            ppgWindows = getLabelledWindows(ppg, label, 60, 1000, 1000, "PPG")
#
#             # Portion of ecg to ease computations
#             first_ecg, middle_ecg, last_ecg = firstMiddleLast(ecg, 1000, 30)
#             first_ppg, middle_ppg, last_ppg = firstMiddleLast(ppg, 1000, 30)
#
#             hr_ecg, br_ecg, snr_ecg = getStats(first_ecg, 1000, "ecg")
#             hr_ppg, br_ppg, snr_ppg = getStats(first_ppg, 1000, "ppg")
#
#             f.write(subject+",first,")
#             f.write(str(f"{hr_ecg},{hr_ppg},"))
#             f.write(str(f"{br_ecg},{br_ppg},"))
#             f.write(str(f"{snr_ecg},{snr_ppg}\n"))
#
#             hr_ecg, br_ecg, snr_ecg = getStats(middle_ecg, 1000, "ecg")
#             hr_ppg, br_ppg, snr_ppg = getStats(middle_ppg, 1000, "ppg")
#
#             f.write(subject+",middle,")
#             f.write(str(f"{hr_ecg},{hr_ppg},"))
#             f.write(str(f"{br_ecg},{br_ppg},"))
#             f.write(str(f"{snr_ecg},{snr_ppg}\n"))
#
#             hr_ecg, br_ecg, snr_ecg = getStats(last_ecg, 1000, "ecg")
#             hr_ppg, br_ppg, snr_ppg = getStats(last_ppg, 1000, "ppg")
#
#             f.write(subject+",last,")
#             f.write(str(f"{hr_ecg},{hr_ppg},"))
#             f.write(str(f"{br_ecg},{br_ppg},"))
#             f.write(str(f"{snr_ecg},{snr_ppg}\n"))

            for eWin, pWin in zip(ecgWindows, ppgWindows):
                try:
                    signal = hp.filter_signal(ecgWindows[eWin]["ECG"], cutoff=[0.8, 2.5], sample_rate=1000, filtertype='bandpass')
                    eData, eMesures = hp.process(signal, sample_rate=1000)
                    signal = hp.filter_signal(ppgWindows[pWin]["PPG"], cutoff=[0.8, 2.5], sample_rate=1000, filtertype='bandpass')
                    pData, pMesures = hp.process(signal, sample_rate=1000)
                    bpmDiff.append(abs(eMesures["bpm"] - pMesures["bpm"]))
                    ibiDiff.append(abs(eMesures["ibi"] - pMesures["ibi"]))
                    breathingDiff.append(abs(eMesures["breathingrate"] - pMesures["breathingrate"]))
                    ecgBR += eMesures["breathingrate"]
                    ppgBR += pMesures["breathingrate"]
                except:
                    pass


print("Mean ECG BR: ", ecgBR/len(bpmDiff))
print("Mean PPG BR: ",ppgBR/len(bpmDiff))

exit()


fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(bpmDiff)
axs[0, 0].set(xlabel='Windows',ylabel='BPM')
axs[0, 0].set_ylim([0, 200])
axs[0, 0].set_title("BPM Diff ECG/PPG - CASE")


axs[1, 0].plot(ibiDiff)
axs[1, 0].set(xlabel='Windows',ylabel='Ms')
axs[1, 0].set_ylim([0, 1000])
axs[1, 0].set_title("IBI Diff ECG/PPG - CASE")


axs[2, 0].plot(breathingDiff)
axs[2, 0].set(xlabel='Windows',ylabel='Hz')
axs[2, 0].set_ylim([0, 0.4])
axs[2, 0].set_title("BR Diff ECG/PPG - CASE")


# WESAD Specific Variables
datasetPath = "Data/Datasets/WESAD/"
dataFiles = os.listdir(datasetPath)
data = {}

# Clean Sig
# ecgC = nk.ecg_simulate(duration=10, sampling_rate=700, noise=0.001)
# ppgC = nk.ppg_simulate(duration=10, sampling_rate=64)

# Loop Through WESAD
features = ["bpm", "ibi", "breathingrate"]
bpmDiff = []
ibiDiff = []
breathingDiff = []
# #print("clean_ecg, ecg, clean_ppg, ppg, subject")
with open("WESAD_COMP.txt", "a+") as f:
    for subject in dataFiles:
        ecg, ppg, label = WESAD_getSignals(subject)
#
#         # Portion of ecg to ease computations
#         # first_ecg, middle_ecg, last_ecg = firstMiddleLast(ecg, 700, 30)
#         # first_ppg, middle_ppg, last_ppg = firstMiddleLast(ppg, 64, 30)
#         #
#         # hr_ecg, br_ecg, snr_ecg = getStats(first_ecg, 700, "ecg")
#         # hr_ppg, br_ppg, snr_ppg = getStats(first_ppg, 64, "ppg")
#         #
#         # f.write(subject + ",first,")
#         # f.write(str(f"{hr_ecg},{hr_ppg},"))
#         # f.write(str(f"{br_ecg},{br_ppg},"))
#         # f.write(str(f"{snr_ecg},{snr_ppg},\n"))
#         #
#         # hr_ecg, br_ecg, snr_ecg = getStats(middle_ecg, 700, "ecg")
#         # hr_ppg, br_ppg, snr_ppg = getStats(middle_ppg, 64, "ppg")
#         #
#         # f.write(subject + ",middle,")
#         # f.write(str(f"{hr_ecg},{hr_ppg},"))
#         # f.write(str(f"{br_ecg},{br_ppg},"))
#         # f.write(str(f"{snr_ecg},{snr_ppg}\n"))
#         #
#         # hr_ecg, br_ecg, snr_ecg = getStats(last_ecg, 700, "ecg")
#         # hr_ppg, br_ppg, snr_ppg = getStats(last_ppg, 64, "ppg")
#         #
#         # f.write(subject + ",last,")
#         # f.write(str(f"{hr_ecg},{hr_ppg},"))
#         # f.write(str(f"{br_ecg},{br_ppg},"))
#         # f.write(str(f"{snr_ecg},{snr_ppg}\n"))
#
        # Window and Label the ECG and PPG data
        ecgWindows = getLabelledWindows(ecg, label, 60, 700, 700, "ECG")
        ppgWindows = getLabelledWindows(ppg, label, 60, 64, 700, "PPG")

        for eWin, pWin in zip(ecgWindows, ppgWindows):
            try:
                eData, eMesures = hp.process(ecgWindows[eWin]["ECG"], sample_rate=700)
                pData, pMesures = hp.process(ppgWindows[pWin]["PPG"], sample_rate=64)
                bpmDiff.append(abs(eMesures["bpm"] - pMesures["bpm"]))
                ibiDiff.append(abs(eMesures["ibi"] - pMesures["ibi"]))
                breathingDiff.append(abs(eMesures["breathingrate"] - pMesures["breathingrate"]))
            except:
                #bpmDiff.append(0)
                #ibiDiff.append(0)
                #breathingDiff.append(0)
                pass




        # count = 0
        #
        # for ewin, pwin in zip(ecgWindows, ppgWindows):
        #     if ecgWindows[ewin]["Label"] == 1 and ppgWindows[pwin]["Label"] == 1:
        #         eData = ecgWindows[ewin]["ECG"]
        #         pData = ppgWindows[pwin]["PPG"]
        #         esnr = signaltonoise(eData)
        #         psnr = signaltonoise(pData)
        #         cesnr = signaltonoise(ecgC)
        #         cpsnr = signaltonoise(ppgC)
        #         print(cesnr, ",", esnr, ",",cpsnr,",",psnr, ",", subject)
        #         count += 1
        #     if count == 5:
        #         break

axs[0, 1].plot(bpmDiff)
axs[0, 1].set_ylim([0, 200])
axs[0, 1].set(xlabel='Windows',ylabel='y-label')
axs[0, 1].set_title("BPM Diff ECG/PPG - WESAD")


axs[1, 1].plot(ibiDiff)
axs[1, 1].set_ylim([0, 1000])
axs[1, 1].set(xlabel='Windows',ylabel='Ms')
axs[1, 1].set_title("IBI Diff ECG/PPG - WESAD")


axs[2, 1].plot(breathingDiff)
axs[2, 1].set_ylim([0, 0.4])
axs[2, 1].set(xlabel='Windows', ylabel='Hz')
axs[2, 1].set_title("BR Diff ECG/PPG - WESAD")

plt.show()





