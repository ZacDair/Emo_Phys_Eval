# Third Party Imports
import os
from datetime import datetime

import matplotlib
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report, f1_score, \
    precision_recall_fscore_support, precision_score, recall_score
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Project Level Imports
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

import Configs.config as cfg
from lssvm import LSSVC
from utils.encoding import dummie2multilabel

pd.set_option('display.max_rows', None)
matplotlib.use('TkAgg')
plt.style.use('seaborn')

# Splits data into features and labels for training and test, also used for holdout data
def getTrainTestSplit(xData, yData, size, randomState):
    return train_test_split(xData, yData, test_size=size, random_state=randomState)


# Identify correct shap explainer from model type and return shap values
def getShapValues(model, xData):
    modelKey = str(model).split('(')[0]  # Split by '(' - use item at index 0
    try:
        shapType = cfg.modelSpecificShapExplainers[modelKey]
        print(f"Selected {shapType} SHAP explainer for {model} model")
        if shapType == 'Linear':
            masker = shap.maskers.Independent(xData)
            explainer = shap.LinearExplainer(model, masker=masker)
        elif shapType == 'Tree':
            explainer = shap.TreeExplainer(model)
        elif shapType == 'Kernel':
            explainer = shap.KernelExplainer(model.predict_proba, xData)
            # print(f"WARNING - {str(model)} is currently unsupported for SHAP feature importance, returning blank shap values...")
            # return ''
        elif shapType == 'Permutation':
            explainer = shap.PermutationExplainer(model, masker=shap.maskers.Independent(xData))
        else:
            print(f"ERROR - Unsupported ShapType from Configs {shapType}, returning blank shap values...")
            return ''

        shapValues = explainer.shap_values(xData)
        return shapValues
    except KeyError as e:
        print(f"ERROR - Model key {modelKey} not found within configs...\n{e}")


# Feature Importance
def featureImportance(model, xData, yData, featureList, classNames, identifier, dataset, saveLocation, method='SHAP'):
    print(f"Feature Importance: Fitting Model {model}...")
    model.fit(xData, yData)
    if method.lower().strip() == "shap":

        # Discern model type
        shap_values = getShapValues(model, xData)

        # If shap values were not computed - exit the function
        if shap_values == '':
            print("ERROR - No Shap Values Computed Returning...")
            return

        shap.summary_plot(shap_values, xData,
                          class_inds='original',
                          class_names=classNames,
                          feature_names=featureList,
                          show=False
                          )

        experimentName = "featureImportance_new_" + dataset
        saveLoc = os.path.join(saveLocation, experimentName)
        if not os.path.exists(saveLoc):
            os.mkdir(saveLoc)

        plt.savefig(os.path.join(saveLoc, str(model) + "_" + identifier + "_featImportance"))
        plt.close()

        # Store shap values
        # savename = f"Results/{identifier}.save"
        # joblib.dump(shap_values, filename=savename)
        # savename = f"Results/exp_{identifier}.save"
        # joblib.dump(explainer, filename=savename)

    elif method.lower().strip() == "impurity":
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

        forest_importances = pd.Series(importances, index=featureList)

        # Plotting impurity-based importance
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()

        result = permutation_importance(
            model, xData, yData, n_repeats=10, random_state=42, n_jobs=2
        )

        forest_importances = pd.Series(result.importances_mean, index=featureList)

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.show()
    else:
        print(f"Incorrect method {method} given, check YAML file.")


# Run Model
def runModel(model, xTrain, xTest, yTrain, yTest):
    # Train model

    # if model == 'LS-SVC':
    #     model = LSSVC(gamma=1, kernel='rbf', sigma=.5)  # Class instantiation
    #     model.fit(xTrain.to_numpy(), yTrain.to_numpy())  # Fitting the model
    #     preds = model.predict(xTest)  # Making predictions with the trained model
    #     preds = dummie2multilabel(preds)
    #
    #
    # else:
    #     model.fit(xTrain, yTrain)
    #
    #     # Test model
    #     preds = model.predict(xTest)

    if model == 'LS-SVC':
        model = LSSVC(kernel='rbf')  # Class instantiation
        #model = OneVsRestClassifier(model)
        model.fit(xTrain.to_numpy(), yTrain.to_numpy())  # Fitting the model
        preds = model.predict(xTest)  # Making predictions with the trained model
        preds = dummie2multilabel(preds)

    else:
        ovr = OneVsRestClassifier(model)
        #ovo = OneVsOneClassifier(model)
        ovr.fit(xTrain, yTrain)

        # Test model
        preds = ovr.predict(xTest)

    # Evaluation Metrics
    acc = accuracy_score(yTest, preds)
    conf = multilabel_confusion_matrix(yTest, preds)
    report = classification_report(yTest, preds, output_dict=True)

    return model, preds, acc, conf, report


# Model Selection
def modelSelection(models, xData, yData, xHoldout=None, yHoldout=None, signalType='None', datasetName='None'):
    modelPerformance = {}
    bestAcc = -1
    bestModel = "None"
    timestamp = time.time()

    crossValidationReport = {}

    for m in models:
        # Get model name and function
        name = m
        model = models[m]
        print(f"Running Model: {name}")

        crossValidationReport[name] = {}

        # Store metrics
        bestCVModel = ""
        bestCVAcc = -1
        CVAcc = []
        CVF1 = []
        CVPrec = []
        CVRecall = []
        CVModel = []
        CVConf = []
        CVReport = []

        # Cross-validation folds
        numFolds = 5
        kf = KFold(n_splits=numFolds, shuffle=True)

        # Run cross-validation
        for train, test in kf.split(X=xData, y=yData):
            print(f"Running K-Fold: {len(CVAcc)+1}")
            xTrain = xData.iloc[train]
            xTest = xData.iloc[test]
            yTrain = yData.iloc[train]
            yTest = yData.iloc[test]

        #xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.20, random_state=21)

            # Fit model and get basic reports
            trainedModel, preds, acc, conf, report = runModel(model, xTrain, xTest, yTrain, yTest)

            # Store CV metrics
            CVAcc.append(acc)
            CVF1.append(f1_score(yTest, preds, average='micro'))
            CVPrec.append(precision_score(yTest, preds, average='micro'))
            CVRecall.append(recall_score(yTest, preds, average='micro'))
            CVModel.append(trainedModel)
            CVConf.append(conf)
            CVReport.append(report)

            if model == 'LS-SVC':
                break

        # Store average CV metrics
        crossValidationReport[name] = {'MeanAcc': np.mean(CVAcc),
                                       'MeanF1': np.mean(CVF1),
                                       'MeanPrecision': np.mean(CVPrec),
                                       'MeanRecall': np.mean(CVRecall),
                                       'CVAcc': CVAcc,
                                       'CVF1': CVF1,
                                       'CVPrec': CVPrec,
                                       'CVRecall': CVRecall,
                                       'CVModel': CVModel,
                                       # 'CVConf': CVConf,
                                       'CVReport': CVReport,
                                       'Time':timestamp
                                       }

        # Pick best model
        if crossValidationReport[name]['MeanAcc'] > bestCVAcc:
            bestCVAcc = crossValidationReport[name]['MeanAcc']
            bestCVModel = model

        # Display model results
        filename = "Results/Models/"+name+"_"+signalType+"_"+str(timestamp)+".txt"
        with open(filename, 'w+') as f:
            f.write(f"{name} Accuracy: {bestCVAcc}\n")
            for key in crossValidationReport[name]:
                if key == 'CVReport' or key == 'CVConf':
                    f.write(f"{key}:\n")
                    reports = crossValidationReport[name][key]
                    for i, report in enumerate(reports):
                        f.write(f"Fold-{i}\n")
                        for val in report:
                            f.write(f"\t{val} {report[val]}\n")
                else:
                    f.write(f"{key}: {crossValidationReport[name][key]}\n")

        # Update best model
        if bestCVAcc > bestAcc:
            bestAcc = bestCVAcc
            bestModel = bestCVModel

    filename = "Results/Models/holdout" + "_" + signalType + "_" + str(timestamp) + ".txt"
    with open(filename, 'w+') as f:
        f.write(f"Best k-fold model was {bestModel} with {bestAcc} accuracy.\n")
        #bestModel = model

        if xHoldout is not None and yHoldout is not None:
            # Fit, Predict and get evaluation metircs
            model, predictions, accuracy, confMatrix, classReport = runModel(bestModel, xData, xHoldout, yData, yHoldout)

            # Display model results
            f.write(f"{bestModel} Accuracy on holdout data:{accuracy}\n")
            f.write("Best Model Confusion Matrix:\n")
            f.write(f"{confMatrix}\n")
            f.write("Best Model Report:\n")
            f.write(f"{classReport}\n")

            f.write(f"F1 Score: {f1_score(yHoldout, predictions, average='micro')}\n")
            f.write(f"Precision Score: {precision_score(yHoldout, predictions, average='micro')}\n")
            f.write(f"Recall Score: {recall_score(yHoldout, predictions, average='micro')}\n")

            return yHoldout, predictions

    return None, None


# Compute absolute difference of selected features
def featureDifference(features, ecg, ppg):
    differenceDict = {}
    ecg = ecg.replace(np.nan, 0.0)
    ecg = ecg.replace('--', 0.0)
    ecg = ecg.fillna(0.0)
    ppg = ppg.replace(np.nan, 0.0)
    ppg = ppg.replace('--', 0.0)
    ppg = ppg.fillna(0.0)
    # ecg = ecg.astype({'sdsd': 'float'})
    # ppg = ppg.astype({'sdsd': 'float'})
    for feature in features:
        try:
            diff = pd.Series.abs(ecg[feature] - ppg[feature])
            differenceDict[feature] = diff.values.tolist()
        except KeyError as e:
            print(f"{e} Not found in the dataframe, please ensure it's a valid feature")
        except Exception as e:
            print(f"An Error Occurred, {e}")

    return differenceDict
