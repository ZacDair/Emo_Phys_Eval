# Variance and Performance of ECG and PPG Signals in Classifying Affective State

Initial version of the physiological signal evaluation platform.

## Provided Functions
- ECG and PPG Feature Extraction - leveraging HeartPy
- ECG and PPG Windowing and labelling in accordance with dataset author instructions
- Cardiac feature difference analysis between ECG and PPG of the same dataset
- Cardiac feature importance from emotive ECG and PPG, using SHAP and a selected Machine Learning classifier
- Model selection including cross-validation (several base Sklearn models supported)

## Instructions
1. Download the supported datasets, unzip and place in the Data/Datasets directory
2. Ensure a correct installation of python is available
3. Load the datasets, window the signals, extract features and label accordingly using CASE_Features.py and WESAD_Features.py respectively
4. ML Operations including model selection, feature importance, and plotting ROC curves is handled in ML.py
5. Feature difference analysis is provided by Signal_Wide_Stats.py


## Supported Datasets
CASE - [The Continuously Annotated Signals of Emotion (CASE) dataset](https://www.nature.com/articles/s41597-019-0209-0)  
WESAD - [WESAD (Wearable Stress and Affect Detection) Data Set](https://archive.ics.uci.edu/ml/datasets/WESAD)

**_Note:_** _Datasets following the same structure as the above may also be supported._

 _The expected structure is a Root directory containing subdirectories per participant, which in turn include either individual files for ECG, PPG and Labels as in CASE, or a combined file as in WESAD
 Reading from .pkl and .csv supported._ 
