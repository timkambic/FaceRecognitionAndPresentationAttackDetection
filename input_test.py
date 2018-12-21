import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# img = preprocess_image("datasets/Oulu_p1_IMGs/attack/img3.jpg")
img = load_img("datasets/Oulu_p1_IMGs/attack/img3.jpg", target_size=(224, 224))
img = img_to_array(img)
print(img.shape)
plt.imshow(img)
plt.show()