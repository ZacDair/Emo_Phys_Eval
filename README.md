# Variance and Performance of ECG and PPG Signals in Classifying Affective State

Initial version of the physiological signal evaluation platform.

## Provided Functions
- ECG and PPG Feature Extraction - leveraging HeartPy
- ECG and PPG Windowing and labelling in accordance with dataset author instructions
- Cardiac feature difference analysis between ECG and PPG of the same dataset
- Cardiac feature importance from emotive ECG and PPG, using SHAP and a selected Machine Learning classifier
- Model selection including cross-validation (several base Sklearn models supported)


## Supported Datasets
CASE - [The Continuously Annotated Signals of Emotion (CASE) dataset](https://www.nature.com/articles/s41597-019-0209-0)  
WESAD - [WESAD (Wearable Stress and Affect Detection) Data Set](https://archive.ics.uci.edu/ml/datasets/WESAD)

**_Note:_** _Datasets following the same structure as the above may also be supported._

 _The expected structure is a Root directory containing subdirectories per participant, which in turn include either individual files for ECG, PPG and Labels as in CASE, or a combined file as in WESAD
 Reading from .pkl and .csv supported._ 