# :warning: POST AICS 2022 -  Update In Progress :warning:  
Codebase will be updated shortly


# Variance and Performance of ECG and PPG Signals in Classifying Affective State

Second version of the physiological signal evaluation platform.
Now includes automated experiment running from YAML configurations.

## Provided Functions
- Automated/Compartmentalised experiment running via YAML files
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

## Documentation
For generated documentation [see here](https://github.com/ZacDair/Emo_Phys_Eval/tree/master/Docs/html/Modules/index.html)  
Each individual Module is documented as follows:
- [Data_Operations.py](https://github.com/ZacDair/Emo_Phys_Eval/tree/master/Docs/html/Modules/Data_Operations.html)
- [Experiments.py](https://github.com/ZacDair/Emo_Phys_Eval/tree/master/Docs/html/Modules/Experiments.html)
- [Feature_Operations.py](https://github.com/ZacDair/Emo_Phys_Eval/tree/master/Docs/html/Modules/Feature_Operations.html)
- [Label_Operations.py](https://github.com/ZacDair/Emo_Phys_Eval/tree/master/Docs/html/Modules/Label_Operations.html)
- [Plotting.py](https://github.com/ZacDair/Emo_Phys_Eval/tree/master/Docs/html/Modules/Plotting.html)
- [Window_Operations.py](https://github.com/ZacDair/Emo_Phys_Eval/tree/master/Docs/html/Modules/Window_Operations.html)
- [Yaml_Operations.py](https://github.com/ZacDair/Emo_Phys_Eval/tree/master/Docs/html/Modules/Yaml_Operations.html)

Runtime configurations are located in Configs.config.py - these include HeartPy feature names, Dictionary of Sklearn models, YAML Experiment directory

Main.py acts as the entry point, which runs all experiments located in the YAML Experiment directory one by one.


# Future Development
### V0.2 Feature List
#### Experiments:
- [ ] Isolate ROC functionality to it’s own experiment
#### General:
- [ ] Makefile for current dependencies
- [ ] YAML template (requirements and datatypes) + validator
- [X] Config to point to YAML locations, dataset locations etc
- [ ] Unit testing
#### Automation:
- [X] Encapsulate data loading functions
- [X] Encapsulate signal processing, windowing, feature extraction
- [X] Encapsulate individual experiment procedures
- [X] YAML file design to automate ISSC work
- [X] YAML main loop

### V0.3 Feature List
#### Experiments:
- [ ] Continuous arousal/valence value classification
- [ ] Semi-Supervised Approaches
- [ ] Intra/Inter personal variances of emotion - re-align and compare windows of emotion
- [ ] Expand feature importance to include Sklearn methods
#### Signal Processing:
- [ ] Add a signal processing stage with basic hearty filtering
- [ ] Add further methods based on literature
- [ ] Anomaly detection to identify electrode disconnection, or noise
####  Feature Extraction:
- [ ] Add support for NeuroKit features 
#### General:
- [ ] Implement starting from windowed data
- [ ] Summarise YAML giving a experiment description
- [ ] Physiological signal identification of meta data - take sample rate identify length of signal, compare to other, signals, expected length, label length etc
- [ ] Signal wide processing (might fit with anomaly detection) - take whole signal identify at what points emotion is showing
#### UI:
- [ ] CLI or dashboard UI creation
- [ ] Results logging through google sheets or alternative + YAML description
#### API:
- [ ] Send signal window - retrieve emotion label (requires phys data window, phys signal name, sample rate, signal processing mode, feature extraction mode)

