import matplotlib
import numpy as np

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

plt.style.use('seaborn')
font = {'size': 22}

matplotlib.rc('font', **font)
labels = ['Log. Reg.', 'Perceptron', 'LDA', 'QDC', 'SVM', 'Extra Trees', 'Rand For.', 'Decision Tree', 'AdaBoost',
          'KNN','', 'ExtraTrees-Holdout']

wesadECG = [0.52, 0.34, 0.52, 0.44, 0.40, 0.68, 0.67, 0.54, 0.58, 0.45,0, 0.69]
wesadPPG = [0.30, 0.30, 0.50, 0.43, 0.41, 0.58, 0.58, 0.45, 0.51, 0.41,0, 0.59]
caseECG = [0.52, 0.42, 0.52, 0.44, 0.52, 0.57, 0.56, 0.36, 0.52, 0.30,0, 0.58]
casePPG = [0.52, 0.44, 0.52, 0.26, 0.51, 0.56, 0.55, 0.34, 0.52, 0.30,0, 0.57]

x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

r1 = np.arange(len(labels))
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]
r4 = [x + width for x in r3]

fig, ax = plt.subplots()
rects1 = ax.bar(r1, wesadECG, width, label='WESAD-ECG (4 Classes)')
rects2 = ax.bar(r2, wesadPPG, width, label='WESAD-PPG (4 Classes)')
rects3 = ax.bar(r3, caseECG, width, label='CASE-ECG (4 Classes)')
rects4 = ax.bar(r4, casePPG, width, label='CASE-PPG (4 Classes)')

plt.axvline(x=10.30, color='r', linestyle='--')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy', fontsize=16)
ax.set_title('Mean 5-Fold Cross-validation accuracy per classifier')
plt.xticks([r + (width*1.5) for r in range(len(r1))], labels)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
plt.setp(ax.get_xticklabels(), horizontalalignment='center')
ax.legend(fontsize=12)

# ax.bar_label(rects1)
# ax.bar_label(rects2)
# ax.bar_label(rects3)
# ax.bar_label(rects4)

plt.show()
