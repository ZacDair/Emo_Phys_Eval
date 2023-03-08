# Third Party Imports
import heartpy as hp
import neurokit2 as nk
import os
import pandas as pd
import warnings
import numpy as np

np.set_printoptions(threshold=5000)

# Store information about any windows that failed
failedWindows = []


def computeFeatures(windows, dataKey, sampleRate, filterParams, subject, dataset, ignoreLabels, mode="HeartPy"):
    """
    Processes each window of physiological signal data, extracting HRV features (by default using HeartPy lib),
    depending on YAML configs signal noise filtering may be applied, additionally stores any windows which fail the
    feature extraction process in a global variable: failed windows.
    :param windows: Dictionary of signal data split into segments
    :param dataKey: String value denoting which signal type ('ecg' or 'ppg')
    :param sampleRate: Dictionary containing keys: 'ecg', 'ppg', 'labels' and associated int sample rate values
    :param filterParams: Dictionary containing filtering method and parameters, defined in YAML (noise reduction)
    :param subject: String value representing the subject where the signals originate from
    :param dataset: String value representing the dataset name
    :param ignoreLabels: List of labels to ignore (WESAD: 5, 6, 7)
    :param mode: String value representing which feature extraction method to use defaults to "HeartPy"
    :return: Pandas DataFrame containing rows: Dataset(name), Signal, Subject, WindowNum, all features (individal cols)
    """
    # Add warning suppression for np.nan filtering warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Handle incorrect feature modes
        if mode.lower().strip() in ["neurokit", "neurokit2"]:
            print("ERROR - Neurokit feature extraction is currently in development, defaulting to HeartPy")
            mode = "heartpy"
        elif mode.lower().strip() != "heartpy":
            print(f"ERROR - Unsupported feature extraction mode: {mode}, defaulting to HeartPy")
            mode = "heartpy"

        result = []
        for window in windows:
            try:
                windowData = windows[window][dataKey]

                # Ignore windows with Label values 0, 5, 6, 7 - See WESAD docs
                if windows[window]["Label"] in ignoreLabels:
                    print("Skipping", window, "Label:", windows[window]["Label"])
                else:
                    # features.append({"Signal":windowData, "label": windows[window]["Label"]})
                    print("Running on", window, "Label:", windows[window]["Label"])

                    # Todo: flesh out Neurokit support and signal filtering
                    # if mode.lower().strip() == "neurokit":
                    #       signals, metrics = nk.ecg_process(windows[window][dataKey], sampling_rate=sampleRate)

                    # Conduct signal filtering if required
                    if filterParams['enabled']:
                        windowData = hp.filter_signal(windowData,
                                                      cutoff=[
                                                          filterParams['method']['min cutoff'],
                                                          filterParams['method']['max cutoff']
                                                      ],
                                                      sample_rate=sampleRate,
                                                      filtertype=filterParams['method']['filter type'])

                    # Extract the signal features
                    workingPPG, metrics = hp.process(windowData, sampleRate)

                    # Add the index and label values, and store in our running list
                    metrics['window_data'] = windowData
                    metrics["og_window_index"] = window
                    metrics["label"] = windows[window]["Label"]
                    metrics["subject"] = subject
                    result.append(metrics)
            except KeyError as e:
                print(f"ERROR - Attempted to use a non-existent key: {e}")
                return pd.DataFrame(result)
            except Exception as e:
                print(f"ERROR - Unexpected error occurred {e}")
                failedWindows.append({"Dataset": dataset, "Signal": dataKey, "Subject": subject, "WindowNum": window, "WindowData": windowData,
                                      "SampleRate": sampleRate})
                print(f"Window-{window} Omitted (stored in result with 0 as feature values)")
                #metrics = {'bpm':0,'ibi':0,'sdnn':0,'sdsd':0,'rmssd':0,'pnn20':0,'pnn50':0,'hr_mad':0,'sd1':0,'sd2':0,'s':0,'sd1/sd2':0,'breathingrate':0,'window_data':windowData,'og_window_index':window,'label':windows[window]["Label"],'subject':subject}
                #result.append(metrics)
        return pd.DataFrame(result)
