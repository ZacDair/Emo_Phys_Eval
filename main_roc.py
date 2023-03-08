# Third Party Imports
from itertools import cycle

import yaml
from matplotlib import pyplot as plt
import pandas as pd
# Project Level Imports
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

from Modules.Data_Operations import processRawData, findAllFiles, combineDataFrames, getFeaturesAndLabels
from Modules.Experiments import featureDifference, featureImportance, modelSelection, getTrainTestSplit
from Modules.Yaml_Operations import parseFeatureKeys, parseDataKeys, parseModelKeys, getYamlFiles
from Configs import config
import matplotlib

matplotlib.use('TkAgg')
plt.style.use('seaborn')

pd.set_option('display.max_columns', None) #prevents trailing elipses
pd.set_option('display.max_rows', None)

# Plot ROC
def plotRoc(y_test, y_score, y_test_1, y_score_1):
    """
    Plots a binary OVR ROC Curve for ECG and PPG
    Args:
        y_test: Test split labels for ECG
        y_score: Test split predictions for ECG
        y_test_1: Test split labels for PPG
        y_score_1: Test split predictions for PPG

    Returns:

    """
    lw = 2

    # classes = ["LA-LV", "LA-HV", "HA-LV", "HA-HV"]

    classes = ["LA-LV","LA-HV","HA-LV","HA-HV"]
    #classes = ["Neutral","Stress","Amusement", "Meditation"]

    n_classes = len(classes)

    # ECG ROC
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # Plot all ROC curves
    plt.figure()

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "lightcoral"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ECG-{0} (area = {1:0.2f})".format(classes[i], roc_auc[i]),
        )
    # PPG ROC
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_1[:, i], y_score_1[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            linestyle=":",
            label="PPG-{0} (area = {1:0.2f})".format(classes[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve for ExtraTrees OVR on ECG and PPG")
    plt.legend(loc="lower right")
    plt.show()


def run():
    # Find all yaml files
    files = getYamlFiles(config.yaml_directory)
    for file in files:
        with open(file, 'r') as f:
            # Read YAML file and split into sub-dictionaries
            experimentParams = yaml.load(f, Loader=yaml.FullLoader)
            dataloading = experimentParams['dataloading']
            windowing = experimentParams['windowing']
            noiseReduction = experimentParams['noise reduction']
            featureExtraction = experimentParams['feature extraction']
            experiments = experimentParams['experiments']

            # Todo: Validate each in terms of values, and structure

            # Store ECG and PPG data
            ecg, ppg = 0, 0

            # Load data ready for experiments
            # TODO: error message if features is used but the correct data is not present (autodetect if feature extraction is required)
            if dataloading['method'] == "features":
                dataFiles = findAllFiles(dataloading['data location'])
                ecg = combineDataFrames(dataFiles, 'ecg')
                ppg = combineDataFrames(dataFiles, 'ppg')

            # Todo: Enable loading data from windows
            elif dataloading['method'] == "windows":
                # featureExtraction()
                # combineFeatureData()
                print(f"Unsupported dataloading method {dataloading['method']}")
                exit()
                pass

            elif dataloading['method'] == "raw":
                processRawData(
                    dataset=dataloading['dataset'],
                    datasetPath=dataloading['data location'],
                    windowSize=windowing['window size'],
                    sampleRates=dataloading['sample rates'],
                    signalCleaning=noiseReduction,
                    ignoreLabels=windowing['drop labels'],
                    outputPath=dataloading['output location']
                )
                dataFiles = findAllFiles(dataloading['output location'])
                ecg = combineDataFrames(dataFiles, 'ecg')
                ppg = combineDataFrames(dataFiles, 'ppg')

            else:
                print(f"Sorry invalid dataloading method used in {file}")
                exit()

            # Run experiments
            for exp in experiments:
                # TODO: Check running configs for train, test and holdout data for this dataset - in this YAML FILE

                features = parseFeatureKeys(experiments[exp]['feature keys'])


                ecgData, ecgLabels = getFeaturesAndLabels(ecg,
                                                          subjects=experiments[exp]['subjects'],
                                                          labels=experiments[exp]['labels'],
                                                          removeCols=['Unnamed: 0', 'subject'],
                                                          featureList=features
                                                          )
                #ecgLabels = label_binarize(ecgLabels, classes=[1, 2, 3, 4])
                ecgLabels = label_binarize(ecgLabels, classes=[0, 2, 6, 8])

                ppgData, ppgLabels = getFeaturesAndLabels(ppg,
                                                          subjects=experiments[exp]['subjects'],
                                                          labels=experiments[exp]['labels'],
                                                          removeCols=['Unnamed: 0', 'subject'],
                                                          featureList=features
                                                          )

                #ppgLabels = label_binarize(ppgLabels, classes=[1, 2, 3, 4])
                ppgLabels = label_binarize(ppgLabels, classes=[0, 2, 6, 8])

                # Get ECG holdout and train test splits
                xRem_ECG, xHoldout_ECG, yRem_ECG, yHoldout_ECG = getTrainTestSplit(ecgData, ecgLabels, size=0.20, randomState=21)

                classifier = OneVsRestClassifier(
                    ExtraTreesClassifier(n_estimators=500),
                )
                y_score_1 = classifier.fit(xRem_ECG, yRem_ECG).predict_proba(xHoldout_ECG)

                # Get PPG holdout and train test splits
                xRem_PPG, xHoldout_PPG, yRem_PPG, yHoldout_PPG = getTrainTestSplit(ppgData, ppgLabels, size=0.20,randomState=21)

                classifier = OneVsRestClassifier(
                    ExtraTreesClassifier(n_estimators=500),
                )
                y_score_2 = classifier.fit(xRem_PPG, yRem_PPG).predict_proba(xHoldout_PPG)

                # TODO: OVR conversion to enable ROC - maybe split model selection and ROC
                plotRoc(yHoldout_ECG, y_score_1, yHoldout_PPG, y_score_2)


run()
