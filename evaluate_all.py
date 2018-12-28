import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def evaluate(score_file, true_labels_file):
    THRESHOLD = 0
    score_list = np.loadtxt(score_file)
    true_labels = np.loadtxt(true_labels_file)
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

    # print(FA, FR)
    FAR = FA / NI
    FRR = FR / NC
    HTER = (FAR + FRR) / 2

    # print("FAR:", FAR)
    # print("FRR:", FRR)
    # print("HTER:", HTER)

    fpr, tpr, threshold = roc_curve(true_labels, score_list)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # print("EER:", EER)

    return fpr, tpr, roc_auc, HTER, EER


fpr_replay,tpr_replay,auc_replay,HTER_replay,EER_replay = evaluate("scores/scores_model1_Replay_20122018.txt", "scores/truelabels_model1_Replay_20122018.txt")
fpr_replay2,tpr_replay2,auc_replay2,HTER_replay2,EER_replay2 = evaluate("scores/scores_model2_Replay_20122018.txt", "scores/truelabels_model2_Replay_20122018.txt")
fpr_replay_oulu2,tpr_replay_oulu2,auc_replay_oulu2,HTER_replay_oulu2,EER_replay_oulu2 = evaluate("scores/scores_model2_Replay_dataOuluP1_20122018.txt", "scores/truelabels_model2_Replay_dataOuluP1_20122018.txt")
fpr_replay3,tpr_replay3,auc_replay3,HTER_replay3,EER_replay3 = evaluate("scores/scores_model3_9epc_Replay_28122018.txt", "scores/truelabels_model3_Replay_28122018.txt")
# fpr_replay_oulu3,tpr_replay_oulu3,auc_replay_oulu3,HTER_replay_oulu3,EER_replay_oulu3 = evaluate("scores/scores_model3_Replay_Oulu_p1_28122018.txt", "scores/true_labels_model3_Replay_28122018.txt")


print("replay model1 - HTER: %0.3f , EER: %0.3f" % (HTER_replay,EER_replay))
print("replay model2 - HTER: %0.3f , EER: %0.3f" % (HTER_replay2,EER_replay2))
print("replay+oulu model2 - HTER: %0.3f , EER: %0.3f" % (HTER_replay_oulu2,EER_replay_oulu2))
print("replay model3 - HTER: %0.3f , EER: %0.3f" % (HTER_replay3,EER_replay3))
# print("replay+oulu model3 - HTER: %0.3f , EER: %0.3f" % (HTER_replay_oulu3,EER_replay_oulu3))


plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.plot(fpr_replay, tpr_replay, color='darkorange', lw=2, label='ReplayAttack model1 (area = %0.3f)' % auc_replay)
plt.plot(fpr_replay2, tpr_replay2, color='blue', lw=2, label='ReplayAttack model2 (area = %0.3f)' % auc_replay2)
plt.plot(fpr_replay_oulu2, tpr_replay_oulu2, color='green', lw=2, label='ReplayAttack+Oulu model2 (area = %0.3f)' % auc_replay_oulu2)
plt.plot(fpr_replay3, tpr_replay3, color='red', lw=2, label='ReplayAttack model3 (area = %0.3f)' % auc_replay3)
# plt.plot(fpr_replay_oulu3, tpr_replay_oulu3, color='green', lw=2, label='ReplayAttack+Oulu model3 (area = %0.3f)' % auc_replay_oulu3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
