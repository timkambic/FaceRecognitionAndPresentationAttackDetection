import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--scores", required=True, help="numpy array with scores")
ap.add_argument("-l", "--true_labels", required=True, help="true labels")
ap.add_argument("-t", "--threshold", required=True, help="classification threshold for score [-1,1]")
args = vars(ap.parse_args())

score_list = np.loadtxt(args["scores"])
true_labels = np.loadtxt(args["true_labels"])
THRESHOLD = float(args["threshold"])

NC = np.where(true_labels >= 0)[0].size  # n of true clients
NI = np.where(true_labels < 0)[0].size  # number of impostors
print(NC, NI)

FA = FR = 0
for i in range(score_list.size):
    true_label = true_labels[i]
    label = 1 if score_list[i] > THRESHOLD else -1
    if label == -1 and true_label == 1:
        FR += 1
    elif label == 1 and true_label == -1:
        FA += 1

print(FA, FR)
FAR = FA / NI
FRR = FR / NC
HTER = (FAR + FRR) / 2

print("FAR:", FAR)
print("FRR:", FRR)
print("HTER:", HTER)

# -------------------------------------------------------------------

fpr, tpr, threshold = roc_curve(true_labels, score_list)
roc_auc = auc(fpr, tpr)

fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print("EER:", EER)

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
