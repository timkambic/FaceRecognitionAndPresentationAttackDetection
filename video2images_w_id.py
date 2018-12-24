import cv2, os
import re
SAVE_EVERY_Nth_FRAME = 25

img_idx = 0
def video2imgs(video_path, save_folder, client_id):
    global img_idx
    cap = cv2.VideoCapture(video_path)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if not i % SAVE_EVERY_Nth_FRAME:
            imgname = save_folder + "/img" + str(img_idx) + "_" + client_id + ".jpg"
            img_idx +=1
            cv2.imwrite(imgname, frame)
        i += 1
    cap.release()


def processFolder(path_to_videos, folder_out):
    print("Proccessing: ", path_to_videos)
    files = os.listdir(path_to_videos)
    total_length = len(files)
    for i, file in enumerate(files):
        if not i % 20:
            print(str(i) + "/" + str(total_length))

        match = re.search('client(\d+)',file)
        if match:
            # print(match.group(1), file)
            client_id = match.group(1)
            video2imgs(path_to_videos + "/" + file, folder_out,client_id )


# DIRECTORY FOR SAVING IMAGES MUST ALREADY EXIST !!!!!!!!!!!!!!!!!!!
# processFolder("datasets/IdiapReplayAttack/train/attack/fixed", "datasets/ReplayAttackIMGs_w_ID/attack_fixed")
processFolder("datasets/IdiapReplayAttack/train/attack/hand", "datasets/ReplayAttackIMGs_w_ID/attack_hand")
# img_idx = 0
processFolder("datasets/IdiapReplayAttack/train/real", "datasets/ReplayAttackIMGs_w_ID/real")
#
#
img_idx = 0
# processFolder("datasets/IdiapReplayAttack/test/attack/fixed", "datasets/ReplayAttackIMGs_w_ID_test/attack_fixed")
processFolder("datasets/IdiapReplayAttack/test/attack/hand", "datasets/ReplayAttackIMGs_w_ID_test/attack_hand")
processFolder("datasets/IdiapReplayAttack/test/real", "datasets/ReplayAttackIMGs_w_ID_test/real")
