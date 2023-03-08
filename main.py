# Third Party Imports
import yaml
from matplotlib import pyplot as plt
import pandas as pd
# Project Level Imports


from Modules.Data_Operations import processRawData, findAllFiles, combineDataFrames, getFeaturesAndLabels
from Modules.Experiments import featureDifference, featureImportance, modelSelection, getTrainTestSplit
from Modules.Plotting import plotRoc, plotFeatureDiff
from Modules.Yaml_Operations import parseFeatureKeys, parseDataKeys, parseModelKeys, getYamlFiles
from Configs import config


pd.set_option('display.max_columns', None) #prevents trailing elipses
pd.set_option('display.max_rows', None)


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
                    #yData = yData.replace(4, 1)


                    '''
                        Print covariance matrix and plot correlations
                    '''
                    # print(f"{signal.upper()} Signal Covariance")
                    # print(xData.cov())
                    #
                    # corr = xData.corr()
                    # plt.matshow(corr)
                    # plt.yticks(ticks=range(len(xData.columns)), labels=xData.columns)
                    # plt.title("Pearson Correlation - " + dataloading['dataset'] + " - " + signal)
                    # cb = plt.colorbar()
                    # cb.ax.tick_params(labelsize=14)
                    # plt.savefig("Results/Pearson Correlation - " + dataloading['dataset'] + " - " + signal + ".png")
                    # plt.close()
                    #
                    # corr = xData.corr(method='spearman')
                    # plt.matshow(corr)
                    # plt.yticks(ticks=range(len(xData.columns)), labels=xData.columns)
                    # plt.title("Spearman Correlation - " + dataloading['dataset'] + " - " + signal)
                    # cb = plt.colorbar()
                    # cb.ax.tick_params(labelsize=14)
                    # plt.savefig("Results/Spearman Correlation - " + dataloading['dataset'] + " - " + signal + ".png")
                    # plt.close()

                    # # Merge CASE video labels
                    # yData = yData.replace(2, 1)  # Merge amusing
                    # yData = yData.replace(4, 3)  # Merge boring
                    # yData = yData.replace(6, 5)  # Merge relaxed
                    # yData = yData.replace(8, 7)  # Merge scary

                    # Get holdout and train test splits
                    xRem, xHoldout, yRem, yHoldout = getTrainTestSplit(xData, yData, size=0.20, randomState=21)

                    if exp == 'feature difference':
                        ecgData, ecgLabels = getFeaturesAndLabels(ecg,
                                                                  subjects=experiments[exp]['subjects'],
                                                                  labels=experiments[exp]['labels'],
                                                                  removeCols=['Unnamed: 0', 'subject'],
                                                                  featureList=features
                                                                  )
                        ppgData, ppgLabels = getFeaturesAndLabels(ppg,
                                                                  subjects=experiments[exp]['subjects'],
                                                                  labels=experiments[exp]['labels'],
                                                                  removeCols=['Unnamed: 0', 'subject'],
                                                                  featureList=features
                                                                  )
                        absoluteDiffs = featureDifference(
                            features=features,
                            ecg=ecgData,
                            ppg=ppgData
                        )
                        plotFeatureDiff(absoluteDiffs, dataloading['dataset'], experiments[exp]['save location'],  mode="individual") # mode= ('individual' or 'combined')

                    elif exp == 'feature importance':

                        # TODO: Check running configs for train, test and holdout data for this dataset - in this YAML FILE
                        models = parseModelKeys(experiments[exp]['models'])
                        trainingData = pd.concat([xRem, yRem], axis=1)
                        # Remove outliers
                        for feat in features:
                            # ECG Outliers
                            q_low = trainingData[feat].quantile(0.25)
                            q_hi = trainingData[feat].quantile(0.75)

                        trainingData_filtered = trainingData[(trainingData[feat] < q_hi) & (trainingData[feat] > q_low)]

                        # Split data into features and labels
                        xRem, yRem = getFeaturesAndLabels(trainingData_filtered,
                                                          subjects=experiments[exp]['subjects'],
                                                          labels=experiments[exp]['labels'],
                                                          removeCols=['Unnamed: 0', 'subject'],
                                                          featureList=features
                                                          )


                        for m in models:
                            featureImportance(model=models[m],
                                              xData=xRem,
                                              yData=yRem,
                                              featureList=[feature.upper() for feature in features],
                                              classNames=experiments[exp]['label names'],
                                              identifier=signal,
                                              dataset=dataloading['dataset'],
                                              saveLocation=experiments[exp]['save location'],
                                              method=experiments[exp]['method'])

                    elif exp == 'model selection':
                        models = parseModelKeys(experiments[exp]['models'])

                        # # Get last 20% as holdout data
                        # holdout = data.tail(int((len(data)/100)*20))
                        # remainder = data.head(int((len(data)/100)*80))

                        # print(holdout['label'].value_counts())
                        trainingData = pd.concat([xRem, yRem], axis=1)


                        # Remove outliers
                        for feat in features:
                            # ECG Outliers
                            q_low = trainingData[feat].quantile(0.25)
                            q_hi = trainingData[feat].quantile(0.75)

                        trainingData_filtered = trainingData[(trainingData[feat] < q_hi) & (trainingData[feat] > q_low)]

                        #Split data into features and labels
                        xRem, yRem = getFeaturesAndLabels(trainingData_filtered,
                                                            subjects=experiments[exp]['subjects'],
                                                            labels=experiments[exp]['labels'],
                                                            removeCols=['Unnamed: 0', 'subject'],
                                                            featureList=features
                                                            )

                        # Split data into features and labels
                        # xRem, yRem = getFeaturesAndLabels(remainder_filtered,
                        #                                     subjects=experiments[exp]['subjects'],
                        #                                     labels=experiments[exp]['labels'],
                        #                                     removeCols=['Unnamed: 0', 'subject'],
                        #                                     featureList=features
                        #                                     )
                        #
                        # # Split data into features and labels
                        # xHoldout, yHoldout = getFeaturesAndLabels(holdout,
                        #                                     subjects=experiments[exp]['subjects'],
                        #                                     labels=experiments[exp]['labels'],
                        #                                     removeCols=['Unnamed: 0', 'subject'],
                        #                                     featureList=features
                        #                                     )

                        # xRem, xHoldout, yRem, yHoldout = getTrainTestSplit(xData, yData, size=0.20, randomState=21)

                        trueLabels, predictions = modelSelection(models, xRem, yRem, xHoldout, yHoldout,
                                                                 signalType=signal,
                                                                 datasetName=dataloading['dataset'])
                        # TODO: OVR conversion to enable ROC - maybe split model selection and ROC
                        #plotRoc(trueLabels, predictions, [signal], experiments[exp]['labels'])

                    else:
                        print(f"Sorry invalid experiment name used in {file}")


run()
