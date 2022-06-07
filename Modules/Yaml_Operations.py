# Third Part Imports
import os

# Project Level imports
from Configs import config


def parseFeatureKeys(featureKeys, featureMode="heartpy"):
    if featureMode.lower().strip() == "heartpy":
        if not isinstance(featureKeys, list):
            if featureKeys.lower().strip() == 'all':
                return config.heartpyFeatures
            else:
                print("Unsupported Feature Keys found! Check Feature sections of YAML")
                exit()
        else:
            return featureKeys


def parseDataKeys(dataKeys='all'):
    if not isinstance(dataKeys, list):
        if dataKeys.lower().strip() == 'both':
            return ['ecg', 'ppg']
        else:
            print("Unsupported Data Keys found! Check Data sections of YAML")
            exit()
    else:
        return dataKeys


def parseModelKeys(modelKeys):
    if not isinstance(modelKeys, dict):
        if modelKeys.lower().strip() == 'all':
            return config.sklearnBasicModels
        else:
            print("Unsupported Model Keys found! Check Model sections of YAML")
            exit()
    else:
        models = {}
        for key in modelKeys:
            if key in config.sklearnBasicModels:
                models[key] = config.sklearnBasicModels[key]
            else:
                print(f"Unsupported Model: {key}, See config.py for supported models")
        return models


def getYamlFiles(location):
    yamlFiles = []
    for root, dirs, files in os.walk(location):
        for file in files:
            if file.endswith(".yaml"):
                yamlFiles.append(os.path.join(root, file))
    return yamlFiles
