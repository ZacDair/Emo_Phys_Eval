import os
import pandas as pd
import pickle
import heartpy as hp
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import signal as sci_sig
import neurokit2 as nk

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)


# Gets all indexes where the label changes - Returns {'endIndex': int, 'label': label_value}
def getLabelIntervals(labelData):
    res = []
    # Find where the labels change
    labelChangeIndex = np.where(np.diff(labelData, prepend=np.nan))[0]
    # Append accordingly skipping the first value
    startIndex = 0
    for i in labelChangeIndex:
        if i != 0:
            res.append({"startIndex": startIndex, "endIndex": i-1, "label": labelData[i-1]})
            startIndex = i
    return res


# Window the Data
def getLabelledWindows(signalData, labelData, windowSize, sigSampleRate, labSampleRate, dataName):
    # Store the windows
    res = {}
    # Loop through signal Data - using relative conversions to Hz values based on Signal and Label sample rates
    sigIter, labIter, i, windowPos = 0, 0, 0, 0
    while i < int(len(signalData)/(windowSize*sigSampleRate)):
        try:
            # Store the updated positions based on window size and sample rate
            sigIncr = windowSize*sigSampleRate
            labIncr = windowSize*labSampleRate
            # Store the associated window of Data and it's corresponding labels
            # print(labelData[labIter:labIter + labIncr])
            res[i] = {dataName: signalData[windowPos:windowPos + sigIncr], "Label": int(np.mean(labelData[labIter:labIter+labIncr]))}
            # Update the position iteration values
        except ValueError:
            break
        windowPos += sigIncr
        labIter += labIncr
        i += 1
    return res


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


# Extract PPG features using Neurokit -2
def featureExtractionPPG(windows, dataKey, sampleRate, toFilter, sub, lib="HeartPy"):
    """Extracts HRV associated features from ECG or PPG windows using HeartPy or Neurokit
    Parameters:
        windows (dict): Dictionary returned from function: getLabelledWindows()
        dataKey (str): String used in function: getLabelledWindows() to determine between ECG and PPG
        sampleRate (int): Sample rate of the windowed signal
        toFilter (bool): True or False to conduct signal cleaning
        lib (str): Feature extraction library of choice - default: "HeartPy", other options: "Neurokit"

    """
    # Neurokit PPG feature extraction
    if lib == "Neurokit":
        for window in windows:
            print("Running on", window, "Label:", windows[window]["Label"])
            # ppg_cleaned = nk.ppg_clean(windows[window][dataKey], method='elgendi')
            signals, info = nk.ecg_process(windows[window][dataKey], sampling_rate=sampleRate)
            print(nk.hrv(info['ECG_R_Peaks'], sampling_rate=sampleRate))

    # HeartPy PPG feature extraction
    else:
        features = []
        for window in windows:
            windowData = windows[window][dataKey]
            try:
                # Conduct signal filtering if required
                if toFilter:
                    windowData = hp.filter_signal(windowData, cutoff=[0.8, 2.5], sample_rate=sampleRate, filtertype='bandpass')

                # Extract the signal features
                workingPPG, metrics = hp.process(windowData, sampleRate)

                # Add the index and label values, and store in our running list
                metrics["og_window_index"] = window
                metrics["label"] = windows[window]["Label"]
                features.append(metrics)
            except:
                failedWindows.append({"Signal": dataKey, "Subject": sub, "WindowNum": window, "WindowData": windowData,
                                      "SampleRate": sampleRate})
                print(f"Window-{window} Omitted")

        return pd.DataFrame(features)


# Load CASE Data files
failedWindows = []
datasetPath = "Datasets/case_dataset/Data"
signalPath = os.path.join(datasetPath, "raw", "physiological")
labelsPath = os.path.join(datasetPath, "raw", "annotations")
datasetOutputPath = "Datasets/Custom/CASE/Features/"
signalFiles = os.listdir(signalPath)
labelsFiles = os.listdir(labelsPath)
data = {}
for sFile, lFile in zip(signalFiles, labelsFiles):
    if sFile.endswith(".txt"):
        # Get subject from path
        subject = sFile.split("_")[0].replace('\\', '')
        print("Running on Subject:", subject)

        # Load signals
        sFile = os.path.join(signalPath, sFile)
        sigData = pd.read_csv(sFile, sep="\t", header=None)
        sigData.columns = ["DaqTime", "ECG", "BVP", "GSR", "RSP", "SKT", "emg_zygo", "emg_coru", "emg_trap"]
        sigData = sigData[["DaqTime", "ECG", "BVP"]]

        # Load labels
        lFile = os.path.join(labelsPath, lFile)
        annoData = pd.read_csv(lFile, sep="\t", header=None)
        annoData.columns = ["JsTime", "X", "Y"]

        # CASE specific Data alterations - ECG and Annotations
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

        # print some info
        print(f"ECG samples: {len(ecg)}, ECG length: {len(ecg)/1000} secs, Max Windows: {(len(ecg)/1000)/10}")
        print(f"PPG samples: {len(ppg)}, PPG length: {len(ecg) / 1000} secs, Max Windows: {(len(ecg) / 1000) / 10}")
        print(f"Labels samples: {len(label)}, ECG length: {len(label) / 20} secs, Max Windows: {(len(label) / 20) / 10}")

        # Window and Label the ECG and PPG Data
        ecgWindows = getLabelledWindows(ecg, label, 10, 1000, 20, "ECG")
        ppgWindows = getLabelledWindows(ppg, label, 10, 1000, 20, "PPG")

        # Feature extraction PPG - ECG
        ppgFeatureDf = featureExtractionPPG(ppgWindows, "PPG", 1000, True, subject)
        ecgFeatureDf = featureExtractionPPG(ecgWindows, "ECG", 1000, True, subject)

        # Save the extracted features to csv files
        # Created the subject directory if needed
        if not os.path.exists(os.path.join(datasetOutputPath, subject)):
            os.mkdir(os.path.join(os.path.join(datasetOutputPath, subject)))

        outputPath = os.path.join(datasetOutputPath, subject, subject)
        ppgFeatureDf.to_csv(str(outputPath+"_ppg.csv"))
        ecgFeatureDf.to_csv(str(outputPath + "_ecg.csv"))

df = pd.DataFrame(failedWindows)
df.to_csv("Results/CASE_Failed_Windows.csv", sep=";")