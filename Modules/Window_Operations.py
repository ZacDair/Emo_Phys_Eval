# Third Party Imports
import numpy as np
import statistics as st


# nils-werner
def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    """
    if axis >= data.ndim:
        raise ValueError("Axis value out of range")

    if stepsize < 1:
        raise ValueError("Stepsize may not be zero or negative")

    if size > data.shape[axis]:
        raise ValueError("Sliding window size may not exceed size of selected axis")

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    if copy:
        return strided.copy()
    else:
        return strided


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


def getWindows(signalData, labelData, windowSize, sigSampleRate, labSampleRate, dataName, slidingDuration=1):
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
        used in all subsequent stages
        :param slidingDuration: Default 1 Second (==sampleRate) sliding offset.
        :return: Dictionary containing rows structures as index: signal data key: window data values, label: label value
        """

    # Split signal and label into windows
    signalSlidingWindow = slidingDuration*sigSampleRate
    labelSlidingWindow = slidingDuration*labSampleRate
    signalWindowLength = windowSize*sigSampleRate
    labelWindowLength = windowSize*labSampleRate

    # Validate the data can be windowed
    if len(signalData) <= signalWindowLength:
        print(f"Error - Signal is smaller than the window size: {windowSize} seconds ")
        return {dataName: None, 'label': None}

    if len(labelData) <= labelWindowLength:
        print(f"Error - Signal is smaller than the window size: {windowSize} seconds ")
        return {dataName: None, 'label': None}

    # nils-werner methods
    signalWindows = sliding_window(signalData, signalWindowLength, stepsize=signalSlidingWindow)
    labelWindows = sliding_window(labelData, labelWindowLength, stepsize=labelSlidingWindow)

    res = {}
    for i, (sigWindow, labWindow) in enumerate(zip(signalWindows, labelWindows)):
        # Merge labels into single value
        mode = st.mode(labWindow)
        res[i] = {dataName: sigWindow, "Label": int(mode)}

    return res
