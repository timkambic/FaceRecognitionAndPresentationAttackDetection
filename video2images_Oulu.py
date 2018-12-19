import cv2, os

SAVE_EVERY_Nth_FRAME = 60

img_idx = 0


def video2imgs(video_path, save_folder):
    global img_idx
    cap = cv2.VideoCapture(video_path)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if not i % SAVE_EVERY_Nth_FRAME:
            imgname = save_folder + "/img" + str(img_idx) + ".jpg"
            img_idx += 1
            cv2.imwrite(imgname, frame)
        i += 1
    cap.release()


def processFile(protocol_file, t, save_path):
    print("Proccessing: ", protocol_file)
    with open(protocol_file, 'r') as file:
        data = file.read().splitlines()

    total_length = len(data)
    for i, line in enumerate(data):
        if not i % 20:
            print(str(i) + "/" + str(total_length))

        label = line.split(',')[0]
        videofile = line.split(',')[1] + ".avi"
        if t == "test":
            videofile = "/opt/data/Oulu/Test_files/" + videofile
        if t == "train":
            videofile = "/opt/data/Oulu/Train_files/" + videofile
        # print(videofile)

        if label == "+1":
            if t == "test":
                video2imgs(videofile, save_path+"_test/real")
            if t == "train":
                video2imgs(videofile, save_path+"/real")
        else:
            if t == "test":
                video2imgs(videofile, save_path+"_test/attack")
            if t == "train":
                video2imgs(videofile, save_path+"/attack")


# processFile("/opt/data/Oulu/Protocols/Protocol_4/Train_1.txt", "train", "OuluIMGs")
# processFile("/opt/data/Oulu/Protocols/Protocol_4/Train_2.txt", "train", "OuluIMGs")
# processFile("/opt/data/Oulu/Protocols/Protocol_4/Train_3.txt", "train", "OuluIMGs")
# processFile("/opt/data/Oulu/Protocols/Protocol_4/Train_4.txt", "train", "OuluIMGs")
# processFile("/opt/data/Oulu/Protocols/Protocol_4/Train_5.txt", "train", "OuluIMGs")
# processFile("/opt/data/Oulu/Protocols/Protocol_4/Train_6.txt", "train", "OuluIMGs")

# img_idx = 0
# processFile("/opt/data/Oulu/Protocols/Protocol_4/Test_1.txt", "test","OuluIMGs")
# processFile("/opt/data/Oulu/Protocols/Protocol_4/Test_2.txt", "test","OuluIMGs")
# processFile("/opt/data/Oulu/Protocols/Protocol_4/Test_3.txt", "test","OuluIMGs")
# processFile("/opt/data/Oulu/Protocols/Protocol_4/Test_4.txt", "test","OuluIMGs")
# processFile("/opt/data/Oulu/Protocols/Protocol_4/Test_5.txt", "test","OuluIMGs")
# processFile("/opt/data/Oulu/Protocols/Protocol_4/Test_6.txt", "test","OuluIMGs")

# processFile("/opt/data/Oulu/Protocols/Protocol_1/Train.txt", "train", "datasets/Oulu_p1_IMGs")
#
# img_idx = 0
processFile("/opt/data/Oulu/Protocols/Protocol_1/Test.txt", "test", "datasets/Oulu_p1_IMGs")