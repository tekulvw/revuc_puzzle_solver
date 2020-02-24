import cv2 as cv
import numpy as np
from pathlib import Path


def resize(img):
    return cv.resize(img, (810, 1080))


def get_pieces(img):
    # cv.imshow("orig", resize(img))

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(hsv)
    cv.equalizeHist(v, v)
    # cv.GaussianBlur(h, (7, 7), 3, dst=h)
    cv.medianBlur(h, 13, dst=h)
    cv.merge((h, s, v), hsv)

    lower_gray = np.array([5, 5, 10])
    upper_gray = np.array([20, 70, 255])

    mask = cv.inRange(hsv, lower_gray, upper_gray)
    mask = cv.medianBlur(mask, 7)

    contours, he = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Display the resulting frame
    # cv.imshow('frame', resize(cv.cvtColor(hsv, cv.COLOR_HSV2BGR)))
    # cv.imshow('hsv', resize(hsv))
    # cv.imshow('mask', resize(mask))
    # cv.imshow('res', resize(res))

    pieces = sorted(contours, key=lambda c: cv.contourArea(c), reverse=True)[:4]

    return pieces


def get_pegs(img):
    # cv.imshow("orig", resize(img))

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(hsv)
    cv.equalizeHist(v, v)
    # cv.GaussianBlur(h, (7, 7), 3, dst=h)
    cv.medianBlur(h, 13, dst=h)
    cv.merge((h, s, v), hsv)

    lower_gray = np.array([5, 5, 10])
    upper_gray = np.array([20, 70, 255])

    mask = cv.inRange(hsv, lower_gray, upper_gray)
    mask = cv.medianBlur(mask, 13)

    contours, he = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Display the resulting frame
    # cv.imshow('frame', resize(cv.cvtColor(hsv, cv.COLOR_HSV2BGR)))
    # cv.imshow('hsv', resize(hsv))
    # cv.imshow('mask', resize(mask))
    # cv.imshow('res', resize(res))

    circles = cv.HoughCircles(h,cv.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=0)

    return circles


def get_corners(piece_contour, length=5, extrema_limit=0.5):
    ret = []

    slopes = []
    for i, point in enumerate(piece_contour):
        next_pt = piece_contour[(i + length) % (len(piece_contour) - 1)]
        diff = (next_pt - point)[0]
        angle = np.arctan2(diff[1], diff[0])
        if angle < 0:
            angle += 2*np.pi
        slopes.append(angle)

    normalized_angles = []
    for i, slope in enumerate(slopes):
        if i == 0:
            normalized_angles.append(slope)
        else:
            last_slope = normalized_angles[i - 1]
            next_slope = slopes[(i + 1) % (len(slopes) - 1)]
            normalized_angles.append(last_slope + np.arctan2(np.sin(next_slope - slope), np.cos(next_slope - slope)))

    derivative = np.gradient(normalized_angles)

    extrema_points = [piece_contour[i] for i, dtheta in enumerate(derivative) if np.abs(dtheta) > extrema_limit]

    # plt.plot(slopes)
    # plt.plot(derivative)
    # plt.plot(normalized_angles)
    # plt.show()

    #return find_parallelagram(extrema_points)

    #BENS CURRENT LINE:

    return find_rectangular_polygon(extrema_points)


    # print(len(extrema_points))
    # return extrema_points

def polygon_area(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def get_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def check_right_angle(corners, right_angle_limit):
    angle = 0
    for c11, corner_11 in enumerate(corners):
        for c22, corner_22 in enumerate(corners):
            if c11 == c22 or angle != 0:
                continue
            for c33, corner_33 in enumerate(corners):
                if c33 == c11 or c33 == c22 or angle != 0:
                    continue
                a = get_angle(corner_11, corner_22, corner_33)
                print(a)
                if np.abs(90 - a) <= right_angle_limit:
                    angle = a
        if angle == 0:
            return False
        if c11 != (len(corners) - 1):
            angle = 0
    return angle != 0

def find_rectangular_polygon(points, right_angle_limit = 5):
    max_area = 0
    max_corners = []
    for c1 in range(len(points)):
        for c2 in range(c1 + 1, len(points) - 1, 3):
            for c3 in range(c2 + 1, len(points) - 1, 3):
                for c4 in range(c3 + 1, len(points) - 1, 3):
                    corner_1 = points[c1]
                    corner_2 = points[c2]
                    corner_3 = points[c3]
                    corner_4 = points[c4]
                    corners = [corner_1[0], corner_2[0], corner_3[0], corner_4[0]]
                    area = polygon_area(corners)
                    if area > max_area and check_right_angle(corners, right_angle_limit):
                        max_corners = [corner_1, corner_2, corner_3, corner_4]
                        max_area = area
    return max_corners

def find_parallelagram(points, right_angle_limit = 2):

    max_area = 0
    max_corners = []
    allIndices=[]
    allPairs = []

    for c1 in range(len(points)):
        corner_1 = points[c1]
        for c2 in range(c1,len(points)):
            corner_2 = points[c2]
            dx = corner_2[0, 0] - corner_1[0, 0]
            dy = corner_2[0, 1] - corner_1[0, 1]

            allPairs.append([dx, dy])
            allIndices.append([c1, c2])


    minPairDist = 1000
    minPairIndices = [-1, -1]
    for p1, pair1 in enumerate(allPairs):
        print(p1)
        for p2, pair2 in enumerate(allPairs):
            if p1 == p2:
                continue
            #dist = np.sqrt((pair2[0]-pair1[0])*(pair2[0]-pair1[0]) + (pair2[1]-pair1[1])*(pair2[1]-pair1[1]))

            dist = np.abs(pair1[0]/pair1[1]-pair2[0]/pair2[1])
            #dist = (pair2[0]-pair1[0]) + (pair2[1]-pair1[1])

            p0s = allIndices[p1]
            p1s = allIndices[p2]
            i1 = p0s[0]
            i2 = p0s[1]
            i3 = p1s[0]
            i4 = p1s[1]

            if i1 == i2 or i1 == i3 or i1==i4 or i2==i3 or i2 == i4 or i3==i4:
                continue

            if dist < minPairDist:
                minPairDist = dist
                minPairIndices = [allIndices[p1], allIndices[p2]]

    #max_corners =

    minPair0 = minPairIndices[0]
    minPair1 = minPairIndices[1]
    i1 = minPair0[0]
    i2 = minPair0[1]
    i3 = minPair1[0]
    i4 = minPair1[1]

    p1 = points[i1]
    p2 = points[i2]
    p3 = points[i3]
    p4 = points[i4]

    parallelogram = (p1, p2, p3, p4)
    return parallelogram



def main():
    for img_path in (Path.cwd() / 'imgs' / 'flipped').glob("*.jpg"):
        img = cv.imread(str(img_path))
        pieces = get_pieces(img)
        corners = [c for p in pieces for c in get_corners(p, length=8, extrema_limit=0.0999)]

        # pegs = get_pegs(img)
        # cv.imshow('pegs?', resize(pegs))

        cv.drawContours(img, [cnt for p in pieces for cnt in p], -1, (255, 0, 0), thickness=2)
        cv.drawContours(img, corners, -1, (0, 255, 0), thickness=20)
        cv.imshow('corners?', resize(img))
        while True:
            k = cv.waitKey(5) & 0xFF
            if k == 27:
                break

        # cv.imshow(f"no_bg '{img_path.stem}'", cv.resize(no_bg_img, (1920, 1080)))

    while True:
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":
    main()
