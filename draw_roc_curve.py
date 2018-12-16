from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import math
import numpy as np

score_list = np.loadtxt("scores.txt")
true_labels = np.loadtxt("true_labels.txt")


fpr, tpr, threshold = roc_curve(true_labels, score_list)
roc_auc = auc(fpr, tpr)

fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print("EER:",EER)

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
