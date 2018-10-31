from keras.models import load_model
import sys
import numpy as np
from keras.preprocessing import image as img
import argparse
import os
from keras import applications

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense



top_model_weights_path = "bottleneck_fc_model.h5"

def predict(image_path):
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)


    print("[INFO] loading and preprocessing image...")
    image = img.load_img(image_path, target_size=(150, 150))
    image = img.img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    # get the prediction label
    print("Image ID: {}, Label: {}".format(inID, label))


predict("test_image_1.JPG")
