from deepface_model import *
import warnings
import cv2
from deepface import DeepFace




if __name__ == "__main__":
    pre_models = {}
    pre_models['emotion'] = loadModel_emotion()
    pre_models['age'] = loadModel_age()
    pre_models['gender'] = loadModel_gender()
    pre_models['race'] = loadModel_race()

    src = cv2.imread('res/test-img/FinalData/001.jpg')
    demography = DeepFace.analyze('res/test-img/FinalData/001.jpg',
                                  actions=['age', 'gender', 'race', 'emotion'],
                                  models=pre_models)
    # demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
    print("Age: ", demography["age"])
    print("Gender: ", demography["gender"])
    print("Emotion: ", demography["dominant_emotion"])
    print("Race: ", demography["dominant_race"])

