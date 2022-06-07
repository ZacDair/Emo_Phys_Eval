# Third Party Imports
import yaml

# Project Level Imports
from Modules.Data_Operations import processRawData, findAllFiles, combineDataFrames, getFeaturesAndLabels
from Modules.Experiments import featureDifference, featureImportance, modelSelection, getTrainTestSplit
from Modules.Plotting import plotRoc, plotFeatureDiff
from Modules.Yaml_Operations import parseFeatureKeys, parseDataKeys, parseModelKeys, getYamlFiles
from Configs import config


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
            if dataloading['method'] == "features":
                dataFiles = findAllFiles(dataloading['data location'])

                ecg = combineDataFrames(dataFiles, 'ecg')
                ppg = combineDataFrames(dataFiles, 'ppg')

            # Todo: Enable loading data from windows
            elif dataloading['method'] == "windows":
                # featureExtraction()
                # combineFeatureData()
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
                signals = parseDataKeys(experiments[exp]['data'])

                for signal in signals:

                    # Status Update
                    print(f"Running {exp} Experiement on {signal} from {dataloading['dataset']}")

                    # Select correct signal dataframe
                    if signal == "ecg":
                        data = ecg
                    elif signal == "ppg":
                        data = ppg
                    else:
                        print("Unsupported Signal Type")
                        exit()

                    # Split data into features and labels
                    xData, yData = getFeaturesAndLabels(data,
                                                        subjects=experiments[exp]['subjects'],
                                                        labels=experiments[exp]['labels'],
                                                        removeCols=['Unnamed: 0', 'subject'],
                                                        featureList=features
                                                        )
                    # Get holdout and train test splits
                    xRem, xHoldout, yRem, yHoldout = getTrainTestSplit(xData, yData, size=0.20, randomState=21)
                    xTrain, xTest, yTrain, yTest = getTrainTestSplit(xRem, yRem, size=0.20, randomState=21)

                    if exp == 'feature difference':

                        absoluteDiffs = featureDifference(
                            features=features,
                            ecg=ecg,
                            ppg=ppg
                        )
                        plotFeatureDiff(absoluteDiffs, dataloading['dataset'])

                    elif exp == 'feature importance':

                        # TODO: Check running configs for train, test and holdout data for this dataset - in this YAML FILE

                        models = parseModelKeys(experiments[exp]['models'])
                        for m in models:
                            featureImportance(model=models[m],
                                              xData=xData,
                                              yData=yData,
                                              featureList=[feature.upper() for feature in features],
                                              classNames=experiments[exp]['label names'],
                                              identifier=signal,
                                              method=experiments[exp]['method'])

                    elif exp == 'model selection':
                        models = parseModelKeys(experiments[exp]['models'])
                        trueLabels, predictions = modelSelection(models, xTrain, xTest, yTrain, yTest, xHoldout, yHoldout)
                        # TODO: OVR conversion to enable ROC - maybe split model selection and ROC
                        plotRoc(trueLabels, predictions, [signal], experiments[exp]['labels'])

                    else:
                        print(f"Sorry invalid experiment name used in {file}")


run()
