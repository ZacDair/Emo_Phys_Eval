# Third Party Imports
import numpy as np


# Window the data
def getLabelledWindows(signalData, labelData, windowSize, sigSampleRate, labSampleRate, dataName):
    """
    Separates a physiological signal into several windows of a given duration YAML (windowing), using the sample rates
    takes the associated labels and computes the mean per window. If the operation completes with data remaining that
    will not fit into a window it is omitted, future work will see this final window padded.
    :param signalData: Numpy array of signal data values
    :param labelData: Numpy array of label data values
    :param windowSize: Integer value in seconds for the duration of each window
    :param sigSampleRate: Integer value representing the sample rate of the physiological signal
    :param labSampleRate: Integer value representing the sample rate of the label
    :param dataName: String key to distinguish between signal types ('ecg' or 'ppg' or custom) but the same key must be
    used in all subsequent stages.
    :return: Dictionary containing rows structures as index: signal data key: window data values, label: label value
    """
    # Store the windows
    res = {}
    # Loop through signal data - using relative conversions to Hz values based on Signal and Label sample rates
    sigIter, labIter, i, windowPos = 0, 0, 0, 0
    while i < int(len(signalData)/(windowSize*sigSampleRate)):
        try:
            # Store the updated positions based on window size and sample rate
            sigIncr = windowSize*sigSampleRate
            labIncr = windowSize*labSampleRate
            # Store the associated window of data and it's corresponding labels
            # print(labelData[labIter:labIter + labIncr])
            res[i] = {dataName: signalData[windowPos:windowPos + sigIncr], "Label": int(np.mean(labelData[labIter:labIter+labIncr]))}
            # Update the position iteration values
        except ValueError as e:
            print("Error", e)
            break
        windowPos += sigIncr
        labIter += labIncr
        i += 1
    return res