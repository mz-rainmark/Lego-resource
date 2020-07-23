import cv2
import numpy as np


def boundingTest(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = cv2.boundingRect(contours[0])
    # print(cv2.boundingRect(contours[0]))

    painting = (np.zeros((img.shape[0], img.shape[1])) + 255).astype(np.uint8)
    cv2.drawContours(painting, contours[0], -1, 20, 3)
    cv2.rectangle(painting, (x,y), (x+w,y+h), 0, 2)

    cv2.namedWindow('lllll', 0)
    cv2.imshow('lllll', painting)

def test(img_body):
    gray = cv2.cvtColor(img_body, cv2.COLOR_BGR2GRAY)

    ret, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU) # cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))

    # 腐蚀
    process = cv2.erode(img_bin, kernel)
    process = cv2.erode(process, kernel)
    # 膨胀
    process = cv2.dilate(process, kernel)
    process = cv2.dilate(process, kernel)

    cv2.namedWindow('bin', 0)  # 可调大小
    cv2.imshow('bin', process)

    contours, hierarchy = cv2.findContours(process, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(cv2.contourArea(contours[0])/cv2.contourArea(contours[1]))

    img_outline = (np.zeros((img_body.shape[0], img_body.shape[1])) + 255).astype(np.uint8)
    cv2.drawContours(img_outline, contours[1], -1, 20, 3)
    if len(contours) == 1:
        ellipse = cv2.fitEllipse(contours[0])
    else:
        pass

    cv2.ellipse(img_outline,ellipse,20,3)
    cv2.namedWindow('1', 0)  # 可调大小
    cv2.imshow('1', img_outline)
    print(np.max(ellipse[1]))

def contour_test(img, angle):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    new_paint = (np.zeros((img.shape[0], img.shape[1])) + 255).astype(np.uint8)
    old_paint = (np.zeros((img.shape[0], img.shape[1])) + 255).astype(np.uint8)

    print(type(contours[0]))
    print(contours[0].shape)

    new_contour = contours[0].copy()

    MM = cv2.moments(contours[0])
    center_x = int(MM["m10"] / MM["m00"])
    center_y = int(MM["m01"] / MM["m00"])

    for i, point in enumerate(contours[0]):
        # center_x = img.shape[0]/2
        # center_y = img.shape[1]/2
        x = point[0][0] - center_x
        y = point[0][1] - (center_y)

        new_x = x * np.cos(angle) + y * np.sin(angle) + center_x
        new_y = -x * np.sin(angle) + y * np.cos(angle) + center_y
        new_contour[i][0][0] = new_x
        new_contour[i][0][1] = new_y

    cv2.drawContours(old_paint, contours[0], -1, 20, 3)
    cv2.drawContours(new_paint, new_contour, -1, 20, 3)

    cv2.namedWindow('new', 0)
    cv2.namedWindow('old', 0)

    cv2.imshow('old', old_paint)
    cv2.imshow('new', new_paint)

    # cv2.drawContours(old_paint, contours[0], -1, 20, 3)
    #
    # cv2.namedWindow('old', 0)
    # cv2.imshow('old', old_paint)
    #
    # M = cv2.getRotationMatrix2D((int(img.shape[0]/2), int(img.shape[1]/2)), 30, 1)
    # picture = cv2.warpAffine(old_paint, M, (img.shape[0], img.shape[1]))
    # cv2.imshow('ooo',picture)
    # new_contours, hierarchy = cv2.findContours(picture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    # print(new_contours[0].shape)
    # new_paint = (np.zeros((img.shape[0], img.shape[1])) + 255).astype(np.uint8)
    # cv2.drawContours(new_paint, new_contours[0], -1, 20, 3)
    #
    # cv2.namedWindow('new', 0)
    # cv2.imshow('new', new_paint)

def paint():
    painting = (np.zeros((512, 512)) + 255).astype(np.uint8)
    for x in range(40):
        for y in range(10,500):
            painting[x][y] = 100

    cv2.namedWindow('xmz', 0)
    cv2.imshow('xmz', painting)

src = cv2.imread('/Users/rain/Desktop/yyy.png')
# contour_test(src, 30/180*np.pi)
# test(src)
# boundingTest(src)

paint()

cv2.waitKey(0)
cv2.destroyAllWindows()
