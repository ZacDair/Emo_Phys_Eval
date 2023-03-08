# Third Party Imports
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
import os

from sklearn.tree import DecisionTreeClassifier

from lssvm import LSSVC

yaml_directory = "Yaml_Experiments"


heartpyFeatures = ['bpm', 'ibi', 'sdnn', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate']  # NOTE: sdsd removed

sklearnBasicModels = {
    'LogisticRegression': LogisticRegression(max_iter=3000),
    'Perceptron': Perceptron(),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=500, random_state=21),
    # 'LinearSVC': SVC(kernel='linear'),
    'LS-SVC': 'LS-SVC',  # HSU
    'SVC': SVC(kernel='rbf', C=0.2, gamma=0.2),  # Muhkerjee - minus the AE and RBF
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=500),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=2),
}

modelSpecificShapExplainers = {
    'LogisticRegression': 'Linear',
    'Perceptron': 'Linear',
    'RandomForestClassifier': 'Tree',
    'ExtraTreesClassifier': 'Tree',
    'AdaBoostClassifier': 'Kernel',
    'LinearSVC': 'Kernel',
    'SVC': 'Kernel',
    'LinearDiscriminantAnalysis': 'Linear',
    'KNeighborsClassifier': 'Kernel',
}

# modelSpecificShapExplainers = {
#     'LogisticRegression': shap.LinearExplainer,
#     'Perceptron': shap.LinearExplainer,
#     'RandomForestClassifier': shap.TreeExplainer,
#     'ExtraTreesClassifier': shap.TreeExplainer,
#     'AdaBoostClassifier': shap.TreeExplainer,
#     'LinearSVC': shap.KernelExplainer,
#     'SVC': shap.KernelExplainer,
#     'LinearDiscriminantAnalysis': shap.LinearExplainer,
#     'KNeighborsClassifier': shap.KernelExplainer
# }