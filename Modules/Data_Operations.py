# Third Party Imports
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize

# Project Level Imports
from Modules.Feature_Operations import computeFeatures
from Modules.Label_Operations import convertArousalValence
from Modules.Window_Operations import getLabelledWindows


# CASE specific data alterations - following instructions from dataset authors
def caseAlterations(dataFile, labelFile, sampleRates):
    """
    Takes in the data, label file paths and sample rates, reads the csv files given.
    Conducts conversions detailed by CASE dataset authors such as converting time to milliseconds, ECG to millivolts.
    Additionally converts the arousal valence values from joystick input to discrete values.
    :param dataFile: Physiological signal file path *.csv required
    :param labelFile: Annotation file path *.csv required
    :param sampleRates: Dictionary containing keys: 'ecg', 'ppg', 'labels' and associated int sample rate values
    :return: ECG data converted to np.array, PPG data converted to np.array, Labels converted to discrete np.array
    """
    # Load signals
    sigData = pd.read_csv(dataFile, sep="\t", header=None)
    sigData.columns = ["DaqTime", "ECG", "BVP", "GSR", "RSP", "SKT", "emg_zygo", "emg_coru", "emg_trap"]
    sigData = sigData[["DaqTime", "ECG", "BVP"]]

    # Load labels
    annoData = pd.read_csv(labelFile, sep="\t", header=None)
    annoData.columns = ["JsTime", "X", "Y"]

    # CASE specific data alterations - ECG and Annotations
    # Convert Time from seconds to ms with 3 decimal rounding
    sigData["DaqTime"] = sigData["DaqTime"].apply(lambda x: x * 1000).round(decimals=3)

    # Convert ECG (usually measured in millivolts (sensor I/P range +-40 mV))
    # And convert volts to milliVolts (mV) with rounding to three decimal places
    sigData["ECG"] = sigData["ECG"].apply(lambda x: ((x - 2.8) / 50) * sampleRates['ecg']).round(decimals=3)

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


def wesadAlterations(dataFile):
    """
    Takes in the singular data file path containing signals and labels, loads the pickle.
    Extracts the chest ECG, wrist BVP(PPG) and label as per author instructions
    :param dataFile: File path containing signals and labels *pkl required
    :return: ECG data converted to np.array, PPG data converted to np.array, Labels converted to discrete np.array
    """
    with open(dataFile, 'rb') as f:
        # TODO: Add support for other file types
        if dataFile.endswith(".pkl"):
            # Extract relevant signals
            try:
                data = pickle.load(f, encoding="latin1")
                ecg = data['signal']['chest']['ECG'].flatten()
                ppg = data['signal']['wrist']['BVP'].flatten()
                label = data['label'].flatten()
                return ecg, ppg, label
            except KeyError as e:
                print(f"ERROR - Please ensure the file {dataFile} includes the correct keys!")
                return None, None, None
        else:
            print(f"ERROR - Unsupported file {dataFile} type must be .pkl")
            return None, None, None


def processRawData(dataset, datasetPath, windowSize, sampleRates, signalCleaning, ignoreLabels, outputPath):
    """
    Function responsible for loading each of data file, windowing the data, computing features and altering labels if
    required. Once features computed they are stored in location given from YAML file(dataloading => output location)
    as *_ecg.csv and *_ppg.csv (replacing '*' with dataset name). Each file contains extracted features, labels and
    other window information such as subject, window number, etc.
    :param dataset: Dataset name, defined in YAML (dataloading)
    :param datasetPath: Dataset location on disk, defined in YAML (dataloading)
    :param windowSize: Length in seconds of each window of data, defined in YAML (dataloading)
    :param sampleRates: Dict {'ecg', 'ppg', 'label'} and associated sample rate values, defined in YAML (dataloading)
    :param signalCleaning: Dict containing filtering method and parameters, defined in YAML (noise reduction)
    :param ignoreLabels: List of labels or empty list to ignore, defined in YAML (windowing)
    :param outputPath: Location to store the computed feature csv files, defined in YAML (dataloading)
    :return: Status boolean - True if the code completed, False if an error occurred
    """
    if dataset.lower().strip() == "case":
        signalPath = os.path.join(datasetPath, "physiological")
        labelsPath = os.path.join(datasetPath, "annotations")
        signalFiles = os.listdir(signalPath)
        labelsFiles = os.listdir(labelsPath)
        for sFile, lFile in zip(signalFiles, labelsFiles):
            if sFile.endswith(".txt"):
                subject = sFile.split("_")[0].replace('\\', '')

                sFile = os.path.join(signalPath, sFile)
                lFile = os.path.join(labelsPath, lFile)

                ecg, ppg, label = caseAlterations(sFile, lFile, sampleRates)

                # Todo: encapsulate the following code
                # Windowing
                ecgWindows = getLabelledWindows(ecg, label, windowSize, sampleRates['ecg'], sampleRates['label'], "ECG")
                ppgWindows = getLabelledWindows(ppg, label, windowSize, sampleRates['ppg'], sampleRates['label'], "PPG")

                # Feature extraction ECG - PPG
                ecgFeatureDf = computeFeatures(ecgWindows, "ECG", sampleRates['ecg'], signalCleaning, subject, dataset,
                                               ignoreLabels)
                ppgFeatureDf = computeFeatures(ppgWindows, "PPG", sampleRates['ppg'], signalCleaning, subject, dataset,
                                               ignoreLabels)

                # Save the extracted features to csv files
                # Created the main and subject directory if needed
                if not os.path.exists(outputPath):
                    os.mkdir(outputPath)

                if not os.path.exists(os.path.join(outputPath, subject)):
                    os.mkdir(os.path.join(os.path.join(outputPath, subject)))

                outPath = os.path.join(outputPath, subject, subject)
                ppgFeatureDf.to_csv(str(outPath + "_ppg.csv"))
                ecgFeatureDf.to_csv(str(outPath + "_ecg.csv"))
        return True

    elif dataset.lower().strip() == "wesad":
        dataFiles = os.listdir(datasetPath)
        for subject in dataFiles:
            dataFile = os.path.join(datasetPath, subject, str(subject + ".pkl"))
            ecg, ppg, label = wesadAlterations(dataFile)

            # Todo: encapsulate the following code
            # Windowing
            ecgWindows = getLabelledWindows(ecg, label, windowSize, sampleRates['ecg'], sampleRates['label'], "ECG")
            ppgWindows = getLabelledWindows(ppg, label, windowSize, sampleRates['ppg'], sampleRates['label'], "PPG")

            # Feature extraction ECG - PPG
            ecgFeatureDf = computeFeatures(ecgWindows, "ECG", sampleRates['ecg'], signalCleaning, subject, dataset,
                                           ignoreLabels)
            ppgFeatureDf = computeFeatures(ppgWindows, "PPG", sampleRates['ppg'], signalCleaning, subject, dataset,
                                           ignoreLabels)

            # Save the extracted features to csv files
            # Created the subject directory if needed
            if not os.path.exists(outputPath):
                os.mkdir(outputPath)

            if not os.path.exists(os.path.join(outputPath, subject)):
                os.mkdir(os.path.join(os.path.join(outputPath, subject)))

            outPath = os.path.join(outputPath, subject, subject)
            ppgFeatureDf.to_csv(str(outPath + "_ppg.csv"))
            ecgFeatureDf.to_csv(str(outPath + "_ecg.csv"))
        return True

    else:
        print(f"ERROR - Unsupported dataset name {dataset}!")
        return False


# Returns a list of file paths from the features dir of a dataset
def findAllFiles(location):
    """
    Locates a files in a directory, and any subdirectories
    :param location: String file path to search
    :return: a list of file paths found within the location
    """
    try:
        contents = os.listdir(location)
        files = []
        for subFile in contents:
            if os.path.isdir(os.path.join(location, subFile)):
                for f in os.listdir(os.path.join(location, subFile)):
                    files.append(os.path.join(location, subFile, f))
            else:
                files.append(os.path.join(location, subFile))
        return files
    except FileNotFoundError as e:
        print(f"ERROR - No directory '{location}' found!")
        return []


# Combine DataFrames from each subject
def combineDataFrames(filePaths, dataKey):
    """
    Loads all per subject labelled feature dataframes from .csv files, sets the subject column value and merges into a
     single dataframe
    :param filePaths: List of feature *.csv to process (must end with either *_ecg.csv or *_ppg.csv)
    :param dataKey: String value to distinguish between ECG and PPG signals by default ('ecg' and 'ppg')
    :return: Pandas DataFrame containing features labelled from all subjects per signal
    """
    # Overarching dataFrame
    resDf = None

    # For every file that ends in *dataKey*.csv
    for f in filePaths:
        if f.endswith(str(dataKey)+".csv"):
            # Get subject name from path
            f = f.replace("\\", "/")
            subject = f.split("/")[4].replace("/", "")

            # Load dataframe and add subject col
            try:
                df = pd.read_csv(f)
                df["subject"] = subject

                # Append to overarching Df
                if resDf is None:
                    resDf = df
                else:
                    resDf = resDf.append(df, ignore_index=True)

            except FileNotFoundError as e:
                print(f"ERROR - File not found {f}")
                return None
    return resDf


# Clean np.nan and other values from a dataframe
def cleanDataFrame(df):
    """
    Removes any rows containing np.nan, np.inf or -np.inf
    :param df: Pandas DataFrame to clean
    :return: Cleaned Pandas DataFrame, Indices of the rows to retain
    """
    try:
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64), indices_to_keep
    except AssertionError as e:
        print("ERROR - Function requires as Pandas DataFrame")
        return None, None


# Remove columns and return the dataframe
def dropColumns(df, columns):
    """
    Removes all columns from the Pandas Dataframe provided in from a list
    :param df: Pandas DataFrame to operate on
    :param columns: List of column names to remove
    :return: Pandas DataFrame with specified columns removed
    """
    try:
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df = df.drop(columns=columns)
        df.dropna(inplace=True)
        return df
    except KeyError as e:
        print(f"ERROR - Please ensure all columns {columns} exist in the dataframe")
        return df
    except AssertionError as e:
        print(f"ERROR - Please ensure all the df parameter is a Pandas DataFrame")
        return None


# Return dataframe of seleced columns
def selectColumns(df, columns):
    """
    Returns a dataframe containing only the columns given in a list
    :param df:
    :param columns:
    :return:
    """
    try:
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        if len(columns) != 0:
            return df[columns]
        else:
            return df

    except KeyError:
        print(f"ERROR - Please ensure the column(s) {columns} exist in the dataframe")
        return None
    except AssertionError:
        print(f"ERROR - Please ensure the df parameter is a Pandas DataFrame")
        return None


# Split into features and labels
def getFeaturesAndLabels(data, subjects, labels, removeCols, featureList, subjectCol='subject', labelCol='label'):
    """
    Extracts features (can be specified using a list) and labels from an overarching Pandas DataFrame, additionally
    removing any subjects, labels or columns passed in the associated arguments.
    :param data: Pandas DataFrame containing feature data, subjects and labels
    :param subjects: List of subjects to keep or String 'all' to keep all subjects
    :param labels: List of labels to keep or String 'all' to keep all subjects
    :param removeCols: List of strings: column names, to remove from the DataFrame
    :param featureList: List of strings: feature names to select from the DataFrame
    :param subjectCol: String name of the DataFrame column containing the subjects
    :param labelCol: String name of the DataFrame column containing the labels
    :return: Two DataFrames one of feature data another of label data
    """

    if subjects != 'all':
        data = data[data[subjectCol].isin(subjects)]

    if labels != 'all':
        data = data[data[labelCol].isin(labels)]

    if dropColumns:
        data = dropColumns(data, removeCols)

    features = selectColumns(data, featureList)

    # Clean Features - removes an unsuable rows
    features, indexes = cleanDataFrame(features)

    labels = selectColumns(data, labelCol)
    labels = labels[indexes].astype(np.float64)

    return features, labels


# Binarize labels
def binarizeLabels(labelData, labels):
    return label_binarize(labelData, classes=labels)
