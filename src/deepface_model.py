from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from deepface.basemodels import VGGFace
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation

def loadModel_emotion():
    num_classes = 7
    model = Sequential()
    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    # fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    # ----------------------------
    model.load_weights('network/facial_expression_model_weights.h5')

    return model

def loadModel_gender():
    model = VGGFace.baseModel()
    classes = 2
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    # --------------------------
    gender_model = Model(inputs=model.input, outputs=base_model_output)
    # --------------------------
    # load weights
    gender_model.load_weights('network/gender_model_weights.h5')

    return gender_model


def loadModel_age():
    model = VGGFace.baseModel()
    # --------------------------
    classes = 101
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)
    # --------------------------
    age_model = Model(inputs=model.input, outputs=base_model_output)
    # --------------------------
    # load weights
    age_model.load_weights('network/age_model_weights.h5')

    return age_model

def loadModel_race():
    model = VGGFace.baseModel()
    # --------------------------
    classes = 6
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)
    # --------------------------
    race_model = Model(inputs=model.input, outputs=base_model_output)
    # --------------------------
    # load weights
    race_model.load_weights('network/race_model_single_batch.h5')

    return race_model
