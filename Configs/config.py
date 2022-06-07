# Third Party Imports
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
import os

print(os.getcwd())
yaml_directory = "Yaml_Experiments"

heartpyFeatures = ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate']

sklearnBasicModels = {
    'LogisticRegression': LogisticRegression(max_iter=1500),
    'Preceptron': Perceptron(),
    'RandomForestClassifier': RandomForestClassifier(random_state=0),
    'LinearSVC': LinearSVC(),
    'SVC': SVC(kernel='rbf'),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'KNeighborsClassifier': KNeighborsClassifier()
}

