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
        # Store the updated positions based on window size and sample rate
        sigIncr = windowSize*sigSampleRate
        labIncr = windowSize*labSampleRate
        # Store the associated window of Data and it's coresponding labels
        res[i] = {dataName: signalData[windowPos:windowPos + sigIncr], "Label": int(np.mean(labelData[labIter:labIter+labIncr]))}
        # Update the position iteration values
        windowPos += sigIncr
        labIter += labIncr
        i += 1
    return res


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
            if windows[window]["Label"] not in [0, 5, 6, 7]:
                print("Running on", window, "Label:", windows[window]["Label"])
                #ppg_cleaned = nk.ppg_clean(windows[window][dataKey], method='elgendi')
                signals, info = nk.ecg_process(windows[window][dataKey], sampling_rate=sampleRate)
                print(nk.hrv(info['ECG_R_Peaks'], sampling_rate=sampleRate))

    # HeartPy PPG feature extraction
    else:
        features = []
        for window in windows:
            windowData = windows[window][dataKey]
            try:
                # Ignore windows with Label values 0, 5, 6, 7 - See WESAD docs
                if windows[window]["Label"] not in [0, 5, 6, 7]:
                    #features.append({"Signal":windowData, "label": windows[window]["Label"]})
                    print("Running on", window, "Label:", windows[window]["Label"])
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
                failedWindows.append({"Signal": dataKey, "Subject": sub, "WindowNum": window, "WindowData": windowData, "SampleRate": sampleRate})
                print(f"Window-{window} Omitted")

        return pd.DataFrame(features)


# Load WESAD Data files
ppgTotal = 0
ecgTotal = 0
failedWindows = []
datasetPath = "Datasets/WESAD/"
datasetOutputPath = "Datasets/Custom/WESAD/Features/"
dataFiles = os.listdir(datasetPath)
data = {}
for subject in dataFiles:
    with open(os.path.join(datasetPath, subject, str(subject + ".pkl")), 'rb') as f:
        # Extract relevant signals
        data = pickle.load(f, encoding="latin1")
        ecg = data['signal']['chest']['ECG'].flatten()

        ppg = data['signal']['wrist']['BVP'].flatten()
        label = data['label'].flatten()
        # labelIndexes = getLabelIntervals(label)

        print(f"ECG Duration: {len(ecg)/700}, PPG Duration: {len(ppg)/64}")

        # Window and Label the ECG and PPG Data
        ecgWindows = getLabelledWindows(ecg, label, 10, 700, 700,  "ECG")
        ppgWindows = getLabelledWindows(ppg, label, 10, 64, 700, "PPG")

        # Feature extraction PPG - ECG
        ppgFeatureDf = featureExtractionPPG(ppgWindows, "PPG", 64, True, subject)
        ecgFeatureDf = featureExtractionPPG(ecgWindows, "ECG", 700, True, subject)

        # Save the extracted features to csv files
        # Created the subject directory if needed
        if not os.path.exists(os.path.join(datasetOutputPath, subject)):
            os.mkdir(os.path.join(os.path.join(datasetOutputPath, subject)))

        outputPath = os.path.join(datasetOutputPath, subject, subject)
        ppgFeatureDf.to_csv(str(outputPath+"_ppg.csv"))
        ecgFeatureDf.to_csv(str(outputPath + "_ecg.csv"))


# df = pd.DataFrame(failedWindows)
# df.to_csv("Results/WESAD_NoFilter_Feature_Fails.csv", sep=";")


