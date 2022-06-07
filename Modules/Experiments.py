# Third Party Imports
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Splits data into features and labels for training and test, also used for holdout data
def getTrainTestSplit(xData, yData, size, randomState):
    return train_test_split(xData, yData, test_size=size, random_state=randomState)


# Feature Importance
def featureImportance(model, xData, yData, featureList, classNames, identifier, method='SHAP'):

    model.fit(xData, yData)
    print("Feature Importance Model Trained...")

    if method.lower().strip() == "shap":
        print("Starting SHAP Summary...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(xData)
        shap.summary_plot(shap_values, xData,
                          class_inds='original',
                          class_names=classNames,
                          feature_names=featureList
                          )
        # Store shap values
        savename = f"Results/{identifier}.save"
        joblib.dump(shap_values, filename=savename)
        savename = f"Results/exp_{identifier}.save"
        joblib.dump(explainer, filename=savename)

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
    model.fit(xTrain, yTrain)

    # Test model
    preds = model.predict(xTest)

    # Evaluation Metrics
    acc = accuracy_score(yTest, preds)
    conf = multilabel_confusion_matrix(yTest, preds)
    report = classification_report(yTest, preds)

    return model, preds, acc, conf, report


# Model Selection
def modelSelection(models, xTrain, xTest, yTrain, yTest, xHoldout=None, yHoldout=None):

    modelPerformance = {}
    bestAcc = -1
    bestModel = "None"
    timestamp = time.time()

    for m in models:
        # Get model name and function
        name = m
        model = models[m]

        print(f"Running Model: {name}")

        bestInterModel = ""
        bestInterAcc = -1
        bestConf = ""
        bestReport = ""
        interAccs = []

        # Run 5 iterations
        # TODO: Implement true cross-validation
        for i in range(0, 5):

            # Fit, Predict and get evaluation metircs
            model, predictions, accuracy, confMatrix, classReport = runModel(model, xTrain, xTest, yTrain, yTest)

            interAccs.append(accuracy)

            # Update best inter model
            if accuracy > bestInterAcc:
                bestInterAcc = accuracy
                bestInterModel = model
                bestReport = classReport
                bestConf = confMatrix

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

    if xHoldout is not None and yHoldout is not None:
        # Fit, Predict and get evaluation metircs
        model, predictions, accuracy, confMatrix, classReport = runModel(bestModel, xTrain, xHoldout, yTrain, yHoldout)

        # Display model results
        print(f"Best Model Accuracy:", accuracy)
        print(f"Best Model Confusion Matrix:")
        print(confMatrix)
        print(f"Best Model Report:")
        print(classReport, "\n\n")

        return xHoldout, predictions

    return xTest, predictions


# Compute absolute difference of selected features
def featureDifference(features, ecg, ppg):
    differenceDict = {}

    for feature in features:
        try:
            diff = pd.Series.abs(ecg[feature] - ppg[feature])
            differenceDict[feature] = diff.values.tolist()
        except KeyError as e:
            print(f"{e} Not found in the dataframe, please ensure it's a valid feature")
        except Exception as e:
            print(f"An Error Occurred, {e}")

    return differenceDict
