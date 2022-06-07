import os
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron, BayesianRidge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.model_selection import cross_val_score

# Pandas config changes
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Returns a list of file paths from the features dir of a dataset
def findAllFeatureFiles(location):
    contents = os.listdir(location)
    files = []
    for subdir in contents:
        for f in os.listdir(os.path.join(location, subdir)):
            files.append(os.path.join(location, subdir, f))
    return files


# Combine DataFrames from each subject
def combineDataFrames(filePaths, dataKey):
    # Overarching dataFrame
    resDf = ""

    # For every file that ends in *dataKey*.csv
    for f in filePaths:
        if f.endswith(dataKey+".csv"):
            # Get subject name from path
            subject = f.split("\\")[4].replace("\\", "")

            # Load dataframe and add subject col
            df = pd.read_csv(f)
            df["subject"] = subject

            # Append to overarching Df
            if type(resDf) == str:
                resDf = df
            else:
                resDf = resDf.append(df, ignore_index=True)

    return resDf


# Clean dataset
def clean_dataset(df):
    """
    Removes any rows of data containing np.nan, np.inf, -np.inf values
    Args:
        df: Pandas DataFrame to be cleaned

    Returns:
        Cleaned Pandas DataFrame, Indices of remaining data

    """
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64), indices_to_keep


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

    classes = ["LA-LV", "LA-HV", "HA-LV", "HA-HV"]
    n_classes = len(classes)
    # classes = ["LA-LV","LA-HV","HA-LV","HA-HV"]
    #classes = ["Neutral","Stress","Amusement", "Meditation"]

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
    plt.title("ROC curve for RandomForest OVR on ECG and PPG")
    plt.legend(loc="lower right")
    plt.show()

# Find file paths for all extracted feature csv files
wesadFiles = findAllFeatureFiles("Datasets\\Custom\\WESAD\\Features")

# Load and combine all features into a signal DF
combinedDf = combineDataFrames(wesadFiles, "ppg")
# combinedPpgDf = combineDataFrames(wesadFiles, "ppg")

# Isolate neutral samples for a subject
# samplesECG = combinedDf[(combinedDf['label'] == 1) & (combinedDf['subject'] == "S10")].head(5)
# samplesPPG = combinedPpgDf[(combinedPpgDf['label'] == 1) & (combinedPpgDf['subject'] == "S10")].head(5)

# Evenly Distribute
# combinedDf = combinedDf[(combinedDf['label'] == 1) | (combinedDf['label'] == 4) | (combinedDf['label'] == 7)]
# combinedDf = combinedDf[(combinedDf['label'] < 4)]
# n = combinedDf['label'].value_counts().min()
# combinedDf = combinedDf.groupby('label').head(n)


# combinedDf = combinedDf[combinedDf.subject == 'S10']

# Drop individual DF index col and subject
combinedDf = combinedDf.drop(columns=['Unnamed: 0', 'subject'])
combinedDf.dropna(inplace=True)


# WESAD: transient = 0, baseline = 1, stress= 2, amusement = 3, meditation = 4, ignore = 5, 6, 7
# CASE: "L-L": 0, "L-N": 1, "L-H": 2, "N-L": 3, "N-N": 4, "N-H": 5, "H-L": 6, "H-N": 7, "H-H": 8 (Arousal-Valence)
# Simplifying the classification: combinedDf[1 < combinedDf['label']] = Amusment, Stress, Meditation
# combinedDf = combinedDf[(combinedDf['label'] == 3) | (combinedDf['label'] == 5)] - case
# combinedDf = combinedDf[(combinedDf['label'] == 3) | (combinedDf['label'] == 5)]
#combinedDf = combinedDf[(combinedDf['label'] == 0) | (combinedDf['label'] == 2) | (combinedDf['label'] == 6) | (combinedDf['label'] == 8)]

# Extract features and labels
# print(combinedDf.columns) - ignoring sdsd (non numerics), subject and label cols
desiredFeatures = ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate']

features = combinedDf[desiredFeatures]
features, indexes = clean_dataset(features)
labels = combinedDf['label']
labels = labels[indexes].astype(np.float64)
# labels = label_binarize(labels, classes=[0, 2, 6, 8])

# OVR ROC curve - ECG
# Store holdout Data
xRem, xHoldout, yRem, yHoldout = train_test_split(features, labels, test_size=0.20, random_state=21)
# # Train/Test split
# xTrain, xTest, yTrain, yTest = train_test_split(xRem, yRem, test_size=0.20, random_state=21)
# classifier = OneVsRestClassifier(
#     RandomForestClassifier()
# )
# y_score_1 = classifier.fit(xRem, yRem).predict_proba(xHoldout)

# OVR ROC curve - PPG
# combinedDf = combineDataFrames(wesadFiles, "ppg")
# combinedDf = combinedDf.drop(columns=['Unnamed: 0', 'subject'])
# combinedDf.dropna(inplace=True)
# # # combinedDf = combinedDf[(combinedDf['label'] < 4)]
# #combinedDf = combinedDf[(combinedDf['label'] == 0) | (combinedDf['label'] == 2) | (combinedDf['label'] == 6) | (combinedDf['label'] == 8)]
# features = combinedDf[desiredFeatures]
# features, indexes = clean_dataset(features)
# labels = combinedDf['label']
# labels = labels[indexes].astype(np.float64)
# #labels = label_binarize(labels, classes=[0, 2, 6, 8])
# # Store holdout Data
# xRem_2, xHoldout_2, yRem_2, yHoldout_2 = train_test_split(features, labels, test_size=0.20, random_state=21)
# # Train/Test split
# xTrain_2, xTest_2, yTrain_2, yTest_2 = train_test_split(xRem_2, yRem_2, test_size=0.20, random_state=21)
# # OVR ROC curve - PPG
# classifier = OneVsRestClassifier(
#     RandomForestClassifier()
# )
# y_score_2 = classifier.fit(xRem_2, yRem_2).predict_proba(xHoldout_2)

# plotRoc(yHoldout, y_score_1, yHoldout_2, y_score_2)


# Feature Importance
forest = RandomForestClassifier(random_state=0)
forest.fit(xRem, yRem)

print("Feature Importance based on mean decrease in impurity")
start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances, index=desiredFeatures)

# Plotting impurity-based importance
# fig, ax = plt.subplots()
# # forest_importances.plot.bar(yerr=std, ax=ax)
# # ax.set_title("Feature importances using MDI")
# # ax.set_ylabel("Mean decrease in impurity")
# # fig.tight_layout()
# # plt.show()
#
# # print("Feature Importance based on Feature Permutation")
# # start_time = time.time()
# # result = permutation_importance(
# #     forest, xHoldout, yHoldout, n_repeats=10, random_state=42, n_jobs=2
# # )
# # elapsed_time = time.time() - start_time
# # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
#
# # forest_importances = pd.Series(result.importances_mean, index=desiredFeatures)
#
# # fig, ax = plt.subplots()
# # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
# # ax.set_title("Feature importances using permutation on full model")
# # ax.set_ylabel("Mean accuracy decrease")
# # fig.tight_layout()
# # plt.show()

# rf = RandomForestClassifier()
# rf.fit(xTest, yTest)
explainer = shap.TreeExplainer(forest)
shap_values = explainer.shap_values(xHoldout)
shap.summary_plot(shap_values, xHoldout,
                  class_inds='original',
                  class_names=["Neutral", "Stress", "Amusement", "Meditation"],
                  feature_names= ['BPM', 'IBI', 'SDNN', 'SDSD', 'RMSSD', 'pNN20', 'pNN50', 'MAD','SD1', 'SD2', 'S', 'SD1/SD2', 'BR']
                  )

exit()


# Model Selection
models = [
    {'name': 'LogReg', 'model': LogisticRegression(max_iter=1000)},
    {'name': 'Preceptron', 'model': Perceptron()},
    {'name': 'RandForest', 'model': RandomForestClassifier()},
    {'name': 'LinearSVC', 'model': LinearSVC()},
    {'name': 'SVC-RBF', 'model': SVC(kernel='rbf')},
    {'name': 'ExtraTrees', 'model': ExtraTreesClassifier()},
    {'name': 'AdaBoost', 'model': AdaBoostClassifier()},
    {'name': 'LDA', 'model': LinearDiscriminantAnalysis()},
    {'name': 'KNN', 'model': KNeighborsClassifier()}
]

modelPerformance = {}
bestAcc = -1
bestModel = "None"
timestamp = time.time()

for m in models:
    # Get model name and function
    name = m['name']
    classifier = m['model']

    print(f"Running Model: {name}")

    bestInterModel = ""
    bestInterAcc = -1
    bestConf = ""
    bestReport = ""
    interAccs = []

    # Run 5 iterations

    for i in range(0, 5):
        # Train model
        classifier.fit(xTrain, yTrain)

        # Test model
        preds = classifier.predict(xTest)

        # Evaluation Metrics
        acc = accuracy_score(yTest, preds)
        conf = multilabel_confusion_matrix(yTest, preds)
        report = classification_report(yTest, preds)

        interAccs.append(acc)

        # Update best inter model
        if acc > bestInterAcc:
            bestInterAcc = acc
            bestInterModel = classifier
            bestReport = report
            bestConf = conf

    # Display model results
    print(f"{name} Accuracy:", bestInterAcc)
    print(f"{name} Confusion Matrix:")
    print(bestConf)
    print(f"{name} Report:")
    print(bestReport, "\n\n")

    # Store model results
    modelPerformance[name] = {"accuracy": bestInterAcc,
                              "conf_matrix": bestConf,
                              "report": bestReport,
                              "time": timestamp,
                              "model": bestInterModel
                              }

    # Update best model
    if bestInterAcc > bestAcc:
        bestAcc = bestInterAcc
        bestModel = bestInterModel

print(f"Best model was {bestModel} with {bestAcc} accuracy.")

# Test on holdout
preds = bestModel.predict(xHoldout)

# Evaluation Metrics
acc = accuracy_score(yHoldout, preds)
conf = multilabel_confusion_matrix(yHoldout, preds)
report = classification_report(yHoldout, preds)

# Display model results
print(f"Best Model Accuracy:", acc)
print(f"Best Model Confusion Matrix:")
print(conf)
print(f"Best Model Report:")
print(report, "\n\n")

print("ROC CURVE")
plotRoc(xHoldout, preds)

# Create a directory to store this run's results
resultsPath = os.path.join("Results", str(timestamp))
os.mkdir(resultsPath)

# Save the model performance, and the train-test splits
variables = [modelPerformance, xTrain,xTest,yTrain,yTest]
paths = [os.path.join(resultsPath, str("performance_" + str(timestamp) + ".pkl")),
         os.path.join(resultsPath, str("xTrain" + str(timestamp) + ".pkl")),
         os.path.join(resultsPath, str("xTest" + str(timestamp) + ".pkl")),
         os.path.join(resultsPath, str("yTrain" + str(timestamp) + ".pkl")),
         os.path.join(resultsPath, str("yTest" + str(timestamp) + ".pkl")),
         ]
for var, file in zip(variables, paths):
    with open(file, 'wb+') as f:
        pickle.dump(var, f)


runNotes = [
    "This run is all models, 20% Holdout, 20% Test\n",
    "Classification: all, Evenly split\n",
    "Using ppg Data from WESAD and HeartPy Features\n"
]
with open(os.path.join(resultsPath, "notes.txt"), "w+") as f:
    f.writelines(runNotes)
