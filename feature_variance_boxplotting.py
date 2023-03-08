import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from Modules.Data_Operations import combineDataFrames, cleanDataFrame, findAllFiles, selectColumns
from Modules.Experiments import featureDifference
import numpy as np
import matplotlib
from scipy import stats
import matplotlib.patches as mpatches
matplotlib.use('TkAgg')
plt.style.use('seaborn')

dataFiles = findAllFiles("Data/Saves/WESAD_10sec_1secSliding_new_cleaning/Features")
ecgDf = combineDataFrames(dataFiles, 'ecg')
ppgDf = combineDataFrames(dataFiles, 'ppg')

# Get all subjects
subjects = ecgDf['subject'].unique()
labels = ecgDf['label'].unique()
features = ['bpm', 'ibi', 'sdnn', 'rmssd', 'pnn20', 'pnn50','hr_mad', 'sd1', 'sd2', 's', 'sd1/sd2', 'breathingrate']

# Remove nans etc
ecgData = ecgDf.copy()
ecgData = ecgData[features]
ppgData = ppgDf.copy()
ppgData = ppgData[features]
ecgData = ecgData.fillna(0.00)
ecgData = ecgData.replace([np.nan, np.inf, -np.inf], 0.00)
ppgData = ppgData.fillna(0.00)
ppgData = ppgData.replace([np.nan, np.inf, -np.inf], 0.00)

# Replace labels with plain text
ecgDf['label'] = ecgDf['label'].replace(1, "ECG-Neutral")
ecgDf['label'] = ecgDf['label'].replace(2, "ECG-Stress")
ecgDf['label'] = ecgDf['label'].replace(3, "ECG-Amusement")
ecgDf['label'] = ecgDf['label'].replace(4, "ECG-Meditation")
ppgDf['label'] = ppgDf['label'].replace(1, "PPG-Neutral")
ppgDf['label'] = ppgDf['label'].replace(2, "PPG-Stress")
ppgDf['label'] = ppgDf['label'].replace(3, "PPG-Amusement")
ppgDf['label'] = ppgDf['label'].replace(4, "PPG-Meditation")

# Outlier Removal
ogECGCount = len(ecgDf)
ogPPGCount = len(ppgDf)


for f in features:
    # ECG Outliers
    q_low = ecgDf[f].quantile(0.25)
    q_hi = ecgDf[f].quantile(0.75)

    ecgdf_filtered = ecgDf[(ecgDf[f] < q_hi) & (ecgDf[f] > q_low)]
    #ecgdf_filtered = ecgDf[(ecgDf[f] <= q_hi) | (ecgDf[f] >= q_low)]

    # PPG Outliers
    q_low = ppgDf[f].quantile(0.25)
    q_hi = ppgDf[f].quantile(0.75)

    ppgdf_filtered = ppgDf[(ppgDf[f] < q_hi) & (ppgDf[f] > q_low)]
    #ppgdf_filtered = ppgDf[(ppgDf[f] <= q_hi) | (ppgDf[f] >= q_low)]

print(f"Total Original ECG Samples: {ogECGCount}, Remaining Samples: {len(ecgdf_filtered)}, Removed Samples: {ogECGCount-len(ecgdf_filtered)}")
print(f"Total Original PPG Samples: {ogPPGCount}, Remaining Samples: {len(ppgdf_filtered)}, Removed Samples: {ogPPGCount-len(ppgdf_filtered)}")

# print(ecgDf.groupby(['label', 'subject'])['bpm'].mean())
# print(ppgDf.groupby(['label', 'subject'])['bpm'].mean())
#
# print(ecgDf.groupby(['label'])['bpm'].mean())
# print(ppgDf.groupby(['label'])['bpm'].mean())

# ecgDf.boxplot(column='bpm')
# plt.show()
#
# ppgDf.boxplot(column='bpm')
# plt.show()

# ecgdf_filtered.boxplot(column='bpm', by='label')
# ppgdf_filtered.boxplot(column='bpm', by='label')

columns_my_order = ['ECG-Neutral', 'PPG-Neutral', 'ECG-Amusement', 'PPG-Amusement', 'ECG-Meditation', 'PPG-Meditation', 'ECG-Stress', 'PPG-Stress']
signal_my_order = ['ECG', 'PPG', 'ECG', 'PPG', 'ECG', 'PPG', 'ECG', 'PPG']
colors = ["aqua", "aqua", "darkorange","darkorange", "cornflowerblue","cornflowerblue", "lightcoral", "lightcoral"]
hatchStyles = ['', '/', '', '/', '', '/', '', '/']
fig, ax = plt.subplots()
for position, column in enumerate(columns_my_order):
    if column.startswith('ECG'):
        bplot = ax.boxplot(ecgdf_filtered[ecgdf_filtered['label'] == column]['breathingrate'], positions=[position], patch_artist=True)
    else:
        bplot = ax.boxplot(ppgdf_filtered[ppgdf_filtered['label'] == column]['breathingrate'], positions=[position], patch_artist=True)

    for patch in bplot['boxes']:
        patch.set(facecolor=colors[position])
        # patch.set(hatch=hatchStyles[position])
    # bplot['boxes'].set_facecolor(colors[position])


legendPatches = []
for colr, label in zip(['aqua', 'darkorange', 'cornflowerblue', 'lightcoral'],['Neutral', 'Amusement', 'Meditation', 'Stress']):
    legendPatches.append(mpatches.Patch(color=colr, label=label))

ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
ax.legend(handles=legendPatches, fontsize=16)
ax.set_xticks(range(position+1))
ax.set_xticklabels(signal_my_order)
ax.set_xlim(xmin=-0.5)
plt.ylabel('Beats-Per-Minute', fontsize=16)
plt.show()

