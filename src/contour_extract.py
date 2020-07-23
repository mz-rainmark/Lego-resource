from model import BiSeNet
import torch
import json
import os
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import cv2

from deepface_model import *
from deepface import DeepFace

from utils import *

atts = {
        1:  'skin',
        2:  'l_brow',
        3:  'r_brow',
        4:  'l_eye',
        5:  'r_eye',
        6:  'eye_g',
        7:  'l_ear',
        8:  'r_ear',
        9:  'ear_r',
        10: 'nose',
        11: 'mouth',
        12: 'u_lip',
        13: 'l_lip',
        14: 'neck',
        15: 'neck_l',
        16: 'clothes',
        17: 'hair',
        18: 'hat'
}


description = {
    # value==-1 or empty:  该属性没有检测结果，
    'gender': -1, # 0 女  1 男
    'brow': -1, # 1，2，3 短 中 长
    'eye': -1,  # 1，2，3 小 中 大
    'ear': -1,  # 1，2，3 小 中 大
    'nose': -1, # 1，2，3 小 中 大
    'mouth': -1,# 1，2，3 小 中 大
    'hair': [[0, 0, 0], -1, [0, 0]], # [[RGB color], 0123 稀疏-短-中-长, [左（0,1,2 无 短 长）,右（0,1,2 无 短 长）]]
    'beard': -1, # 1，2，3 唇上小胡子  下颌有胡子  唇上至下颌一圈胡子
    'skin': [], # [RGB color]
    'face': -1, # 1，2，3 瓜子脸 适中 胖脸 (上三角)
    'glasses': -1, # -1，1，2 无眼镜 小框眼镜 大框眼镜
    'clothes': [[], []] # [[RGB color],[]]
}

face_contours = {}
nose_rect = [None, None, None, None]
mouth_rect = [None, None, None, None]
nose_center = [None, None]
face_angle = None

def evaluate(face_img, cp='79999_iter.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    # net.cuda()
    save_pth = osp.join('network/', cp)
    net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = to_tensor(face_img)
        img = torch.unsqueeze(img, 0)
        # img = img.cuda()
        out = net(img)[0]
        # print(out.shape)     # torch.Size([1, 19, 512, 512])
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing.astype(np.uint8)

# ----------------------------------------------------------------------------------------------------- #

def face_process(img_face, img_map):
    global filename
    global nose_center
    global face_angle

    gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
    origin_contours = {}
    # ------------------------ 预处理，轮廓旋转 ------------------------
    # --- 轮廓提取
    anchor_center = {}
    for it in range(1, 20):
        idx = np.where(img_map == it)
        if idx[0].size == 0:
            continue
        tmp_img = (np.zeros((img_map.shape[0], img_map.shape[1])) + 255).astype(np.uint8)
        tmp_img[idx[0], idx[1]] = gray[idx[0], idx[1]]
        ret, img_bin = cv2.threshold(tmp_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        if it != 2 and it != 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12)) if it == 1 else cv2.getStructuringElement(
                cv2.MORPH_RECT, (6, 6))
            img_bin = cv2.dilate(img_bin, kernel)
            img_bin = cv2.erode(img_bin, kernel)
            img_bin = cv2.dilate(img_bin, kernel)

        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        origin_contours[atts[it]] = contours
        # --- 关键点中心计算
        if it in [2, 3, 4, 5, 7, 8, 10, 11]:
            anchor_center[atts[it]] = [0, 0]
            if it == 10:
                for item in contours:
                    M = cv2.moments(item)
                    if 0 == M["m00"]:
                        continue
                    anchor_center[atts[it]][0] += M["m10"] / M["m00"]
                    anchor_center[atts[it]][1] += M["m01"] / M["m00"]
                nose_center[0] = int(anchor_center[atts[it]][0] / len(contours))
                nose_center[1] = int(anchor_center[atts[it]][1] / len(contours))
                del anchor_center[atts[it]] # 旋转anchor_center时不需旋转鼻子的中心点
            else:
                M = cv2.moments(contours[max_length(contours)])
                if 0 == M["m00"]:
                    continue
                anchor_center[atts[it]][0] = M["m10"] / M["m00"]
                anchor_center[atts[it]][1] = M["m01"] / M["m00"]

    # img_outline = (np.zeros((img_map.shape[0], img_map.shape[1])) + 255).astype(np.uint8)
    # for k in origin_contours:
    #     # img_outline = (np.zeros((img_map.shape[0], img_map.shape[1])) + 255).astype(np.uint8)
    #     cv2.drawContours(img_outline, origin_contours[k], -1, 20, 3)
    # cv2.imshow('lllllllllllllll',img_outline)
    if nose_center[0] is None:
        print('nose detected failed !!!! ' + filename)
        return {}

    # --- 轮廓旋转
    face_angle = 0.0
    if 'l_eye' in anchor_center.keys() and 'r_eye' in anchor_center.keys():
        point_1 = anchor_center['l_eye']
        point_2 = anchor_center['r_eye']
        vertical = point_1[1] - point_2[1] #右眼高，逆时针旋转矫正
        horizontal = point_1[0] - point_2[0]
        if horizontal < 0:
            vertical *= -1
            horizontal *= -1
        face_angle = np.arctan2(vertical, horizontal)

    elif 'l_brow' in anchor_center.keys() and 'r_brow' in anchor_center.keys():
        point_1 = anchor_center['l_brow']
        point_2 = anchor_center['r_brow']
        vertical = point_1[1] - point_2[1] #右眉高，逆时针旋转矫正
        horizontal = point_1[0] - point_2[0]
        if horizontal < 0:
            vertical *= -1
            horizontal *= -1
        face_angle = np.arctan2(vertical, horizontal)

    elif 'mouth' in anchor_center.keys():
        mouth_center = anchor_center['mouth']
        face_angle = np.arctan2(nose_center[0]-mouth_center[0], mouth_center[1]-nose_center[1])

    else:
        print('angle calculate failed !!! ' + filename)

    # print('face_angle :' + str(np.rad2deg(face_angle)))
    output, anchor_center = rotate_head(face_angle, origin_contours, anchor_center, nose_center)

    # --- 矫正左右眉眼name
    left2right = False
    for it in ['l_brow', 'l_eye', 'l_ear']:
        if it in anchor_center and anchor_center[it][0] < nose_center[0]: # 正常应该是左侧眉眼x > 鼻子x
            left2right = True
            break

    if left2right:
        for it in [2, 4, 7]: # ['l_brow', 'l_eye', 'l_ear']
            if atts[it] in output:
                if atts[it+1] not in output:
                    output[atts[it+1]] = output[atts[it]]
                    del output[atts[it]]
                else:
                    tmp = output[atts[it]]
                    output[atts[it]] = output[atts[it+1]]
                    output[atts[it+1]] = tmp
            elif atts[it+1] in output:
                output[atts[it]] = output[atts[it+1]]
                del output[atts[it+1]]

    # img_outline = (np.zeros((img_map.shape[0], img_map.shape[1])) + 255).astype(np.uint8)
    # for k in output:
    #     # img_outline = (np.zeros((img_map.shape[0], img_map.shape[1])) + 255).astype(np.uint8)
    #     cv2.drawContours(img_outline, output[k], -1, 20, 3)
    # cv2.imshow('lllllllllllllll',img_outline)

    return output

def body_process(img_body):
    gray = cv2.cvtColor(img_body, cv2.COLOR_BGR2GRAY)

    ret, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

    # 腐蚀
    process = cv2.erode(img_bin, kernel)
    process = cv2.erode(process, kernel)
    # 膨胀
    process = cv2.dilate(process, kernel)
    process = cv2.dilate(process, kernel)

    contours, hierarchy = cv2.findContours(process, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxLength = len(contours[0])
    ptr = 0
    for i in range(len(contours)):
        if len(contours[i]) > maxLength:
            maxLength = len(contours[i])
            ptr = i
    cv2.drawContours(img_body, contours[ptr], -1, (0, 0, 255), 3)

    return contours[ptr]

def all_contour(origin_img):
    global filename
    variables_init()

    gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    #/Users/rain/Downloads/opencv-master/data/haarcascades
    face_cascade = cv2.CascadeClassifier(get_opencv_path() + 'haarcascade_frontalface_default.xml')
    # 检测脸部
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    global face_contours
    # 标记位置， 计算提取区域
    if len(faces) != 0:
        (x, y, w, h) = faces[0]
        center_x = x + w/2
        center_y = y + h/2

        (x1, x2, y1, y2) = (x, x, y, y)
        alpha = 1.0
        while x2-x1 != y2-y1 or alpha == 1.0:
            x1 = int(center_x - alpha * h) if int(center_x - alpha * h) > 0 else 0
            y1 = int(center_y - alpha * h) if int(center_y - alpha * h) > 0 else 0
            x2 = int(center_x + alpha * h) if int(center_x + alpha * h) < gray.shape[1] else gray.shape[1]
            y2 = int(center_y + alpha * h) if int(center_y + alpha * h) < gray.shape[0] else gray.shape[0]
            alpha -= 0.1

        # cv2.rectangle(gray, (x1, y1), (x2, y2), (255, 0, 0), 1)
        face = origin_img[y1:y2, x1:x2, :].copy().astype(np.uint8)
        face = cv2.resize(face, (512, 512), cv2.INTER_LINEAR) # cv2.INTER_CUBIC
        body = origin_img[y2:, :, :].copy().astype(np.uint8)

        # cv2.imshow('face', face)

        parsing_map = evaluate(face)
        face_contours = face_process(face, parsing_map)

        if len(face_contours):
            describe(face)

        # if 'clothes' in face_contours.keys():
        #     face_contours['clothes'].append( body_process(body) )
        # else:
        #     face_contours['clothes'] = []
        #     face_contours['clothes'].append(body_process(body))
    else:
        print('face detected failed !!!!! ' + filename)

def describe(image):
    global face_contours, description
    global nose_center, nose_rect, mouth_rect

    # painting = (np.zeros((512, 512)) + 255).astype(np.uint8)
    cloth_color = np.array([0, 0, 0])
    face_size = [0, 0, 0]
    nose_size = [0, 0, 0]
    brow_rate, brow_record = 0.0, 1.0
    eye_area, ear_length, glass_area = 0, 0, 0
    eye_record, ear_record = 0, 0

    mouth_record = []

    for name, contours in face_contours.items():
        if name == 'skin':
            for item in contours:
                rect = cv2.boundingRect(item)
                face_size[0] += cv2.contourArea(item)
                face_size[1] = rect[2] if rect[2] > face_size[1] else face_size[1]
                face_size[2] += rect[3]

        elif name == 'l_brow' or name == 'r_brow':
            if len(contours[max_length(contours)]) < 6:
                continue
            ellipse = cv2.fitEllipse(contours[max_length(contours)])
            brow_rate = np.min(ellipse[1])/np.max(ellipse[1])
            brow_rate = brow_record if brow_record < brow_rate else brow_rate # 取最小的
            brow_record = brow_rate

        elif name == 'l_eye' or name == 'r_eye':
            for i in range(len(contours)):
                eye_area += cv2.contourArea(contours[i])
            eye_area = eye_record if eye_record > eye_area else eye_area
            eye_record = eye_area

        elif name == 'eye_g':
            for i in range(len(contours)):
                glass_area += cv2.contourArea(contours[i])

        elif name == 'l_ear' or name == 'r_ear':
            rect = cv2.minAreaRect(contours[max_length(contours)])
            ear_length = np.max(rect[1])
            ear_length = ear_record if ear_record > ear_length else ear_length
            ear_record = ear_length

        elif name == 'nose':
            nose_upperline, nose_lowerline = 500, 0
            nose_leftline, nose_rightline = 500, 0

            for item in contours:
                nose_size[0] += cv2.contourArea(item)
                rect = cv2.boundingRect(item)
                nose_leftline = rect[0] if rect[0] < nose_leftline else nose_leftline
                nose_rightline = rect[0]+rect[2] if rect[0]+rect[2] > nose_rightline else nose_rightline
                nose_upperline = rect[1] if rect[1] < nose_upperline else nose_upperline
                nose_lowerline = rect[1]+rect[3] if rect[1]+rect[3] > nose_lowerline else nose_lowerline

            nose_rect = [nose_leftline, nose_upperline, nose_rightline-nose_leftline, nose_lowerline-nose_upperline]
            # print('nose_lowerline :' + str(nose_lowerline))

        elif name == 'mouth' or name == 'u_lip' or name == 'l_lip':
            for item in contours:
                mouth_record.append(item)

        # elif name == 'clothes':
        #     M = cv2.moments(contours[max_length(contours)])  # 计算第一条轮廓的各阶矩,字典形式
        #     if 0 == M["m00"]:
        #         continue
        #     center_x = int(M["m10"] / M["m00"])
        #     center_y = int(M["m01"] / M["m00"])
        #     for i in range(center_x-6, center_x+6):
        #         for j in range(center_y-6, center_y+6):
        #             cloth_color += image[i,j,:]
        #     cloth_color[0] /= 144
        #     cloth_color[1] /= 144
        #     cloth_color[2] /= 144

    mouth_upperline, mouth_lowerline = 500, 0
    mouth_leftline, mouth_rightline = 500, 0
    for item in mouth_record:
        rect = cv2.boundingRect(item)
        mouth_leftline = rect[0] if rect[0] < mouth_leftline else mouth_leftline
        mouth_rightline = rect[0]+rect[2] if rect[0]+rect[2] > mouth_rightline else mouth_rightline
        mouth_upperline = rect[1] if rect[1] < mouth_upperline else mouth_upperline
        mouth_lowerline = rect[1]+rect[3] if rect[1]+rect[3] > mouth_lowerline else mouth_lowerline

    mouth_rect = [mouth_leftline, mouth_upperline, mouth_rightline-mouth_leftline, mouth_lowerline-mouth_upperline]

    if nose_rect[0] is None:
        return
    ref_area = 0.58 * nose_rect[2] * nose_rect[3]
    nose_size[0] = 1.2*ref_area if nose_size[0] < ref_area else nose_size[0]
    nose_size[1] = nose_rect[2]
    nose_size[2] = nose_rect[3]
    mouth_width = mouth_rect[2]

    # 眉毛
    if brow_rate > 0.23:
        description['brow'] = 3
    elif brow_rate > 0.125:
        description['brow'] = 2
    else:
        description['brow'] = 1
    # 眼镜
    if glass_area > 2.33 * nose_size[0]:
        description['glasses'] = 2
    elif glass_area > 0.66 * nose_size[0]:
        description['glasses'] = 1
    # 眼睛
    if eye_area > nose_size[0]/5:
        description['eye'] = 3
    elif eye_area > nose_size[0]/6.5:
        description['eye'] = 2
    else:
        description['eye'] = 1
    # 耳朵
    if ear_length > 0.38 * face_size[2]:
        description['ear'] = 3
    elif ear_length > 0.2333 * face_size[2]:
        description['ear'] = 2
    elif ear_length > 0.12 * face_size[2]:
        description['ear'] = 1
    # 鼻子
    if nose_size[2] > 0.2999 * face_size[2]:
        description['nose'] = 3
    elif nose_size[2] > 0.26 * face_size[2]:
        description['nose'] = 2
    else:
        description['nose'] = 1
    # 嘴
    if mouth_width > 0.389 * face_size[1]:
        description['mouth'] = 3
    elif mouth_width > 0.28 * face_size[1]:
        description['mouth'] = 2
    else:
        description['mouth'] = 1
    # 衣服
    description['clothes'][0] = cloth_color.tolist()

    # 头发 胡子
    hair_beard_process(image)

    # print('mouth_width :' + str(mouth_width))
    # print('face_width :'+ str(face_size[1]))

def hair_beard_process(image):
    global description, face_contours
    global face_angle, nose_center, nose_rect, mouth_rect

    if not ('hair' in face_contours.keys() and 'skin' in face_contours.keys()):
        description['hair'][2] = [0, 0]
        description['hair'][1] = -1
        return
    painting = (np.zeros((512, 512)) + 255).astype(np.uint8)

    hair_rect = cv2.boundingRect(face_contours['hair'][max_length(face_contours['hair'])])
    face_rect = cv2.boundingRect(face_contours['skin'][max_height(face_contours['skin'])])
    brow_rect = None
    if 'l_brow' in face_contours.keys():
        brow_rect = cv2.boundingRect(face_contours['l_brow'][max_height(face_contours['l_brow'])])
    elif 'r_brow' in face_contours.keys():
        brow_rect = cv2.boundingRect(face_contours['r_brow'][max_height(face_contours['r_brow'])])

    cv2.rectangle(painting, (hair_rect[0], hair_rect[1]), (hair_rect[0]+hair_rect[2], hair_rect[1]+hair_rect[3]), 100, 1)

    # ------------------------ 刘海 & 脸型 ------------------------
    if brow_rect is None:
        brow_rect = [face_rect[0]+10, face_rect[1]+0.2678*face_rect[3], 0, 0]

    description['hair'][2] = [0, 0]
    left_side_u, right_side_u = 500, 100
    left_side_l, right_side_l = 500, 100

    for points in face_contours['skin'][max_height(face_contours['skin'])]:
        if points[0][1] < nose_rect[1]:
            left_side_u = points[0][0] if points[0][0] < left_side_u else left_side_u
            right_side_u = points[0][0] if points[0][0] > right_side_u else right_side_u
        elif points[0][1] > nose_rect[1] + nose_rect[3] + 6:
            left_side_l = points[0][0] if points[0][0] < left_side_l else left_side_l
            right_side_l = points[0][0] if points[0][0] > right_side_l else right_side_l
    face_width_u = right_side_u - left_side_u
    face_width_l = right_side_l - left_side_l

    if face_width_l > 1.1 * face_width_u:
        description['face'] = 3 # 胖脸 上三角
    elif np.abs(face_width_l - face_width_u) < 20:
        description['face'] = 2 # 适中
    else:
        description['face'] = 1 # 瘦脸 瓜子脸

    left_ref = int(left_side_u + 0.75 * (right_side_u - left_side_u))
    right_ref = int(left_side_u + 0.25 * (right_side_u - left_side_u))
    lr_cnt = np.array([0, 0])

    ref_line1 = int(face_rect[1] + 0.3888*(brow_rect[1]-face_rect[1]))
    ref_line2 = int(face_rect[1] + 0.8666*(brow_rect[1]-face_rect[1]))
    for y in range(ref_line1, ref_line2):
        for x in range(right_ref-10, left_ref+10):
            res = cv2.pointPolygonTest(face_contours['hair'][max_length(face_contours['hair'])], (x, y), False)
            if res > 0: # inside
                painting[y, x] = 100
                if x < nose_center[0]: # right side
                    lr_cnt[1] += 1
                else:                  # left side
                    lr_cnt[0] += 1
    if np.max(lr_cnt) > 500:
        if lr_cnt[0] > 1.23*lr_cnt[1]:
            description['hair'][2] = [1, 0]
            if cv2.pointPolygonTest(face_contours['hair'][max_length(face_contours['hair'])],
                                    (left_ref, brow_rect[1]-6), False) >= 0:
                description['hair'][2] = [2, 0]
        elif lr_cnt[1] > 1.23*lr_cnt[0]:
            description['hair'][2] = [0, 1]
            if cv2.pointPolygonTest(face_contours['hair'][max_length(face_contours['hair'])],
                                    (right_ref, brow_rect[1]-6), False) >= 0:
                description['hair'][2] = [0, 2]
    # print(lr_cnt)
    # ------------------------ 长度 ------------------------
    hair_contour_polar = xy2polar(face_contours['hair'][max_length(face_contours['hair'])], nose_center)
    if face_rect[1] - hair_rect[1] > 0.54321 * nose_rect[3]:
        description['hair'][1] = 2 # 中长发
    else:
        description['hair'][1] = 1  # 短发
    for points in hair_contour_polar:
        if np.abs(points[0][1]) > 150: # 披肩长发
            description['hair'][1] = 3

    # ------------------------ 发色 ------------------------
    M = cv2.getRotationMatrix2D((nose_center[0], nose_center[1]), np.rad2deg(face_angle), 1)
    picture = cv2.warpAffine(image, M, (image.shape[0], image.shape[1]))
    hair_color = np.array([0, 0, 0])

    mask = (np.zeros((512, 512))).astype(np.uint8)
    cnt = 0
    for y in range(hair_rect[1], hair_rect[1]+hair_rect[3]):
        for x in range(hair_rect[0], hair_rect[0]+hair_rect[2]):
            res = cv2.pointPolygonTest(face_contours['hair'][max_length(face_contours['hair'])], (x, y), False)
            if res > 0:  # inside
                cnt += 1
                hair_color += picture[y, x]
                mask[y, x] = 255
    if cnt != 0:
        hair_color[0] /= cnt
        hair_color[1] /= cnt
        hair_color[2] /= cnt
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    mask = cv2.erode(mask, kernel)
    erode_contours, _xxx_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(erode_contours):
        cnt = 0
        hair_color = np.array([0, 0, 0])
        for y in range(hair_rect[1], hair_rect[1]+hair_rect[3]):
            for x in range(hair_rect[0], hair_rect[0]+hair_rect[2]):
                res = cv2.pointPolygonTest(erode_contours[max_length(erode_contours)], (x, y), False)
                if res > 0:  # inside
                    cnt += 1
                    hair_color += picture[y, x]
            if cnt > 2333:
                break
        if cnt != 0:
            hair_color[0] /= cnt
            hair_color[1] /= cnt
            hair_color[2] /= cnt
        hair_color = np.int0(hair_color)
    else: ### !!!! 说明头发非常稀疏，只有薄薄一层
        description['hair'][1] = 0

    description['hair'][0] = [hair_color[2], hair_color[1], hair_color[0]]

    # ------------------------ 胡子 ------------------------
    if 'u_lip' in face_contours.keys() and 'l_lip' in face_contours.keys():
        ulip_rect = cv2.boundingRect(face_contours['u_lip'][max_length(face_contours['u_lip'])])
        llip_rect = cv2.boundingRect(face_contours['l_lip'][max_length(face_contours['l_lip'])])

        nose_lowerline = nose_rect[1] + nose_rect[3]
        mouth_lowerline = llip_rect[1] + llip_rect[3]
        mouth_upperline = ulip_rect[1]

        tmp_color = np.array([0, 0, 0])
        skin_color = np.array([0, 0, 0])
        if hair_color is None:
            hair_color = skin_color

        skin_color[0] = picture[nose_center[1]-8:nose_center[1]+8, nose_center[0]-4:nose_center[0]+4, 0].mean()
        skin_color[1] = picture[nose_center[1]-8:nose_center[1]+8, nose_center[0]-4:nose_center[0]+4, 1].mean()
        skin_color[2] = picture[nose_center[1]-8:nose_center[1]+8, nose_center[0]-4:nose_center[0]+4, 2].mean()
        description['skin'] = [skin_color[2], skin_color[1], skin_color[0]]
        picture[nose_center[1] - 8:nose_center[1] + 8,nose_center[0] - 4:nose_center[0] + 4] = 255
        face_rect = cv2.boundingRect(face_contours['skin'][min_height(face_contours['skin'])])

        for i in range(int(nose_lowerline+12), int(mouth_upperline-6)):
            tmp_color[0] = picture[i-3:i+3, nose_center[0]-15:nose_center[0]+15, 0].mean()
            tmp_color[1] = picture[i-3:i+3, nose_center[0]-15:nose_center[0]+15, 1].mean()
            tmp_color[2] = picture[i-3:i+3, nose_center[0]-15:nose_center[0]+15, 2].mean()
            if np.sum(np.sqrt(np.square(tmp_color-hair_color))) < 188 or\
                np.max(np.sqrt(np.square(tmp_color-skin_color))) > 180:
                description['beard'] = 1
                picture[i-3:i+3, nose_center[0]-15:nose_center[0]+15] = 0
                break
        for i in range(int(mouth_lowerline+0.12*nose_rect[3]), face_rect[1]+face_rect[3]-12):
            tmp_color[0] = picture[i-6:i+6, nose_center[0]-15:nose_center[0]+15, 0].mean()
            tmp_color[1] = picture[i-6:i+6, nose_center[0]-15:nose_center[0]+15, 1].mean()
            tmp_color[2] = picture[i-6:i+6, nose_center[0]-15:nose_center[0]+15, 2].mean()
            if np.sum(np.sqrt(np.square(tmp_color-hair_color))) < 188 or\
                np.max(np.sqrt(np.square(tmp_color-skin_color))) > 180:
                description['beard'] = 3 if description['beard'] == 1 else 2
                picture[i-6:i+6, nose_center[0]-15:nose_center[0]+15] = 0
                break

        # cv2.circle(picture, (nose_center[0], int(mouth_lowerline+0.12*nose_rect[3])), 5, (0, 255, 255), 2)
        # cv2.circle(picture, (nose_center[0], face_rect[1]+face_rect[3]-12), 5, (0, 255, 255), 2)
    # -----------------------------------------------------——

    # cv2.circle(picture, (left_ref, nose_rect[1]), 5, (0, 255, 0), 2)
    # cv2.circle(picture, (right_ref, nose_rect[1]), 5, (255, 0, 0), 2)

    for k, v in face_contours.items():
        # if k == 'hair' or k == 'skin':
        cv2.drawContours(painting, v, -1, 20, 3)

    global pictureShow
    if pictureShow:
        cv2.imshow('rotate', picture)
        cv2.namedWindow('face_contour', 0)  # 可调大小
        cv2.imshow('face_contour', painting)

# ============================================================================================ #
def variables_init():
    global face_angle, face_contours
    global description
    global nose_center, nose_rect, mouth_rect

    description = {
        'gender': -1,
        'brow': -1,
        'eye': -1,
        'ear': -1,
        'nose': -1,
        'mouth': -1,
        'hair': [[0, 0, 0], -1, [0, 0]],
        'beard': -1,
        'skin': [],
        'face': -1,
        'glasses': -1,
        'clothes': [[], []]
    }

    face_contours = {}
    nose_rect = [None, None, None, None]
    mouth_rect = [None, None, None, None]
    nose_center = [None, None]
    face_angle = None

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


pictureShow = False
writeFile = True
filename = ''

if __name__ == "__main__":

    dspth = 'res/test-img/FinalData'

    pre_models = {}
    pre_models['emotion'] = loadModel_emotion()
    pre_models['age'] = loadModel_age()
    pre_models['gender'] = loadModel_gender()
    pre_models['race'] = loadModel_race()

    if not writeFile:
        src = cv2.imread(osp.join(dspth, '028.jpg'))
        all_contour(src)
        print(description)

        while 27 != cv2.waitKey(2):
            pass
        cv2.destroyAllWindows()
    else:
        for image_name in os.listdir(dspth):
            filename = image_name
            src = cv2.imread(osp.join(dspth, image_name))

            demography = DeepFace.analyze(osp.join(dspth, image_name),
                                          actions=['age', 'gender', 'race', 'emotion'],
                                          models=pre_models)

            all_contour(src)


            description['gender'] = 1 if demography["gender"] == 'Man' else 0
            if demography["gender"] == 'Woman':
                description['beard'] = -1

            # if writeFile:
            #     with open('log/' + image_name[:image_name.find('.')] + '.json', 'w') as f:
            #         json.dump(description, f, cls=NpEncoder)
            #         # f.write(str(description))

            print(image_name)
            print(description)

