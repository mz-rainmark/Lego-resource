import numpy as np
import cv2
import os

def get_opencv_path():
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]

    path = folders[0]
    for folder in folders[1:]:
        path = path + "/" + folder

    face_detector_path = path + "/data/haarcascade_frontalface_default.xml"

    if os.path.isfile(face_detector_path) != True:
        raise ValueError("Confirm that opencv is installed on your environment! Expected path ", face_detector_path,
                         " violated.")

    return path + "/data/"

def max_length(contours):
    maxLength = 0
    ptr = 0
    for i in range(len(contours)):
        if len(contours[i]) > maxLength:
            maxLength = len(contours[i])
            ptr = i
    return ptr

def max_height(contours):
    y_record = 500
    ptr = 0
    for i in range(len(contours)):
        rect = cv2.boundingRect(contours[i])
        if rect[1] < y_record:
            y_record = rect[1]
            ptr = i
    return ptr

def min_height(contours):
    y_record = 0
    ptr = 0
    for i in range(len(contours)):
        rect = cv2.boundingRect(contours[i])
        if rect[1] > y_record:
            y_record = rect[1]
            ptr = i
    return ptr

def rotate_head(angle, human_contours, anchor_center, nose_center):
    out_1 = {}
    out_2 = {}
    for name, contours in human_contours.items():
        new_contours = []
        for contour in contours:
            new_contour = contour.copy()
            for i, points in enumerate(contour):
                x = points[0][0] - nose_center[0]
                y = points[0][1] - nose_center[1]
                new_contour[i][0][0] = x * np.cos(angle) + y * np.sin(angle) + nose_center[0]
                new_contour[i][0][1] = -x * np.sin(angle) + y * np.cos(angle) + nose_center[1]

                new_contour[i][0][0] = 0 if new_contour[i][0][0] < 0 else new_contour[i][0][0]
                new_contour[i][0][0] = 512 if new_contour[i][0][0] > 512 else new_contour[i][0][0]
                new_contour[i][0][1] = 0 if new_contour[i][0][1] < 0 else new_contour[i][0][1]
                new_contour[i][0][1] = 512 if new_contour[i][0][1] > 512 else new_contour[i][0][1]

            new_contours.append(new_contour)
        out_1[name] = new_contours

    for name in anchor_center:
        out_2[name] = [0, 0]

        x = anchor_center[name][0] - nose_center[0]
        y = anchor_center[name][1] - nose_center[1]

        out_2[name][0] = x * np.cos(angle) + y * np.sin(angle) + nose_center[0]
        out_2[name][1] = -x * np.sin(angle) + y * np.cos(angle) + nose_center[1]

    return out_1, out_2

# def rotate_head(angle, human_contours, nose_center):
#     out = {}
#     for name, contours in human_contours.items():
#         ptr = max_length(contours)
#         new_contour = contours[ptr].copy()
#
#         for i, points in enumerate(contours[ptr]):
#             x = points[0][0] - nose_center[0]
#             y = points[0][1] - nose_center[1]
#             new_contour[i][0][0] = x * np.cos(angle) + y * np.sin(angle) + nose_center[0]
#             new_contour[i][0][1] = -x * np.sin(angle) + y * np.cos(angle) + nose_center[1]
#
#         out[name] = new_contour
#     return out

def xy2polar(contour_xy, center):
    contour_polar = contour_xy.copy()
    for i, point in enumerate(contour_xy):
        x = point[0][0] - center[0]
        y = center[1] - point[0][1]
        contour_polar[i][0][0] = np.sqrt(np.square(x) + np.square(y))
        contour_polar[i][0][1] = np.rad2deg(np.arccos(y/contour_polar[i][0][0]))
    return contour_polar

def format(pic_num, dict):
    # dataframe = {}
    # for k, v in dict.items():
    #     if k == 'hair':
    #         tmpStr = str(v[0])
    #         dataframe['hair'] = v[1]
    #         dataframe['color'] = tmpStr[tmpStr.find('[')+1:tmpStr.find(']')]
    #         if v[2] == [0, 1] or v[2] == [0, 2]:
    #             dataframe['bangs'] = '01'
    #         elif v[2] == [1, 0] or v[2] == [2, 0]:
    #             dataframe['bangs'] = '10'
    #         elif v[2] == [1, 1] or v[2] == [2, 2]:
    #             dataframe['bangs'] = '11'
    #         else:
    #             dataframe['bangs'] = '00'
    #     else:
    #         dataframe[k] = v

    # print(dataframe['gender'], end='\t')
    # print(dataframe['hair'], end='\t')
    # print(dataframe['color'], end='\t')
    # print(dataframe['bangs'], end='\t')
    # print(dataframe['race'], end='\t')
    # print(dataframe['eye'], end='\t')
    # print(dataframe['chin'], end='\t')

    tmpStr = str(dict['haircolor'])
    colorStr = tmpStr[tmpStr.find('[')+1:tmpStr.find(']')]
    return [pic_num, dict['gender'], dict['haircut'], colorStr,
            dict['bangs'], dict['race'], dict['eye'], dict['chin']]



