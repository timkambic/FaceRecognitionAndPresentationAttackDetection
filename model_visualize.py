import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, GlobalAveragePooling2D
from vggFace import vggFace

model = vggFace()


x = GlobalAveragePooling2D()(model.get_layer('last_feature_layer').output)
x = Dense(100,activation='relu')(x)
x = Dense(80,activation='relu')(x)
x = Dense(1,activation='tanh')(x)
pad_model = Model(inputs=model.layers[0].input, outputs=[x,model.layers[-2].output])


# for layer in pad_model.layers:
#     layer.trainable = False
# pad_model.layers[-1].trainable = True
# pad_model.layers[-2].trainable = True
# pad_model.layers[-3].trainable = True

pad_model.summary()

# model.compile(optimizer='adam', loss='mse', metrics=['accuracy',binary_accuracy])
pad_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


plot_model(pad_model, to_file='model.png',show_shapes=True)