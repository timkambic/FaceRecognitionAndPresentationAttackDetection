import os
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def getTrainingDataReplay():
    train_imgs = []
    labels = []
    for file in os.listdir("datasets/ReplayAttackIMGs/attack_hand"):
        train_imgs.append(preprocess_image("datasets/ReplayAttackIMGs/attack_hand/" + file))
        labels.append(-1)
    for file in os.listdir("datasets/ReplayAttackIMGs/real"):
        train_imgs.append(preprocess_image("datasets/ReplayAttackIMGs/real/" + file))
        labels.append(1)
    train_imgs = np.array(train_imgs)
    train_imgs = train_imgs[:, 0, :, :]
    labels = np.array(labels)
    return train_imgs, labels


def getTestDataReplay():
    test_imgs = []
    labels = []
    for file in os.listdir("datasets/ReplayAttackIMGs_test/attack_hand"):
        test_imgs.append(preprocess_image("datasets/ReplayAttackIMGs_test/attack_hand/" + file))
        labels.append(-1)
    for file in os.listdir("datasets/ReplayAttackIMGs_test/real"):
        test_imgs.append(preprocess_image("datasets/ReplayAttackIMGs_test/real/" + file))
        labels.append(1)
    test_imgs = np.array(test_imgs)
    test_imgs = test_imgs[:, 0, :, :]
    labels = np.array(labels)
    return test_imgs, labels


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

def getTrainingDataOulu():
    train_imgs = []
    labels = []
    for file in os.listdir("datasets/OuluIMGs/attack"):
        train_imgs.append(preprocess_image("datasets/OuluIMGs/attack/" + file))
        labels.append(-1)
    for file in os.listdir("datasets/OuluIMGs/real"):
        train_imgs.append(preprocess_image("datasets/OuluIMGs/real/" + file))
        labels.append(1)
    train_imgs = np.array(train_imgs)
    train_imgs = train_imgs[:, 0, :, :]
    labels = np.array(labels)
    return train_imgs, labels


def getTestDataOulu():
    test_imgs = []
    labels = []
    for file in os.listdir("datasets/OuluIMGs_test/attack"):
        test_imgs.append(preprocess_image("datasets/OuluIMGs_test/attack/" + file))
        labels.append(-1)
    for file in os.listdir("datasets/OuluIMGs_test/real"):
        test_imgs.append(preprocess_image("datasets/OuluIMGs_test/real/" + file))
        labels.append(1)
    test_imgs = np.array(test_imgs)
    test_imgs = test_imgs[:, 0, :, :]
    labels = np.array(labels)
    return test_imgs, labels


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

def getTrainingDataOuluP1():
    train_imgs = []
    labels = []
    for file in os.listdir("datasets/Oulu_p1_IMGs/attack"):
        train_imgs.append(preprocess_image("datasets/Oulu_p1_IMGs/attack/" + file))
        labels.append(-1)
    for file in os.listdir("datasets/Oulu_p1_IMGs/real"):
        train_imgs.append(preprocess_image("datasets/Oulu_p1_IMGs/real/" + file))
        labels.append(1)
    train_imgs = np.array(train_imgs)
    train_imgs = train_imgs[:, 0, :, :]
    labels = np.array(labels)
    return train_imgs, labels


def getTestDataOuluP1():
    test_imgs = []
    labels = []
    for file in os.listdir("datasets/Oulu_p1_IMGs_test/attack"):
        test_imgs.append(preprocess_image("datasets/Oulu_p1_IMGs_test/attack/" + file))
        labels.append(-1)
    for file in os.listdir("datasets/Oulu_p1_IMGs_test/real"):
        test_imgs.append(preprocess_image("datasets/Oulu_p1_IMGs_test/real/" + file))
        labels.append(1)
    test_imgs = np.array(test_imgs)
    test_imgs = test_imgs[:, 0, :, :]
    labels = np.array(labels)
    return test_imgs, labels
