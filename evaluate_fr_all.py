import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def euclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

test_pairs = [
    [2,3],
    [4,5],
    [10,11],
    [19,23],
    [29,34],
    [35,44],
    [28,45],
    [46,51],
    [25,61],
    [55,67],
    [14,70],
    [24,72],
    [33,74],
    [105,2601],
    [2985,1360],
    [1368,2203],
    [2407,1080],
    [1098,258],
    [3197,3160],
    [3170,1903],
    [1555,504],
    [113,1456],
    [1472,2989],
    [3190,1850],
    [1599,846],
    [492,493],
    [509,105],
    [2549,2083],
    [2097,2054],
    [1308,1295],
    [974,982],
    [970,969],
    [1005,540],
    [530,651],
    [544,571],
    [301,332],
    [2700,2114],
    [1684,1685],
    [1690,1691],
    [1704,1705],
    [1688,1674],
    [980,1002]
]
def ROCcurve(scores,ids):
    distance_list = []
    label_list = []

    for x in test_pairs:
        idx1 = x[0]
        idx2 = x[1]
        f1 = scores[idx1]
        f2 = scores[idx2]
        distance = euclideanDistance(f1, f2)
        label = 0
        if ids[idx1] == ids[idx2]:
            label = 1
        distance_list.append(1 / distance)
        label_list.append(label)

    fpr, tpr, threshold = roc_curve(label_list, distance_list)
    return fpr,tpr,threshold

scores1 = np.loadtxt("scores/features_orgVggFace.txt")
ids_one_hot = np.loadtxt("scores/IDs_onehot")
ids1 = [np.argmax(x) for x in ids_one_hot]
# scores2 = np.loadtxt("scores/features_orgVggFace.txt") #TODO
# ids_one_hot = np.loadtxt("scores/IDs_onehot")
# ids2 = [np.argmax(x) for x in ids_one_hot]
# np.savetxt("scores/IDs",ids, fmt='%d')


fpr1, tpr1,_ = ROCcurve(scores1,ids1)
roc_auc1 = auc(fpr1, tpr1)
# fpr2, tpr2,_ = ROCcurve(scores2,ids2)
# roc_auc2 = auc(fpr2, tpr2)


plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='original (area = %0.2f)' % roc_auc1)
# plt.plot(fpr2, tpr2, color='green', lw=2, label='pad finetunning (area = %0.2f)' % roc_auc2)
plt.xlabel('False Positive Rate')q
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()