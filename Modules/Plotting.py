# Third Party Imports
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


# Plot ROC
def plotRoc(labels, predictions, signals, classes):
    lw = 2
    n_classes = len(classes)

    for y_test, y_score, signalName in zip(labels, predictions, signals):

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
                label=signalName+"-{0} (area = {1:0.2f})".format(classes[i], roc_auc[i]),
            )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve for RandomForest OVR on ECG and PPG")
    plt.legend(loc="lower right")
    plt.show()


# Plot feature difference
def plotFeatureDiff(differenceDict, dataset):
    featureCount = len(differenceDict)
    plotRow = 0
    fig, axs = plt.subplots(featureCount)
    fig.set_title("Feature Differnce - " + dataset)
    for key in differenceDict:
        # title = key + " Diff ECG/PPG - " + dataset
        axs[plotRow].plot(differenceDict[key])
        axs[plotRow].set(xlabel='Windows', ylabel=key)
        # axs[plotRow].set_title(title)

        plotRow += 1

    plt.show()
