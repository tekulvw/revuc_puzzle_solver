import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon


def resize(img):
    return cv.resize(img, (1920, 1080))


def get_img(fname):
    return cv.imread(f"imgs/{fname}")


def remove_bg(img):
    cv.imshow("orig", resize(img))

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # cv.imshow("hue", resize(hsv[:, :, 0]))
    # cv.imshow("sat", resize(hsv[:, :, 1]))
    # cv.imshow("val", resize(hsv[:, :, 2]))

    blurred_h = cv.GaussianBlur(hsv[:, :, 0], (51, 51), 21)

    dilate_h = cv.morphologyEx(blurred_h, cv.MORPH_DILATE, (13, 13), iterations=4)
    # cv.imshow('dilated_h', resize(dilate_h))

    _, dilate_threshed = cv.threshold(dilate_h, 18, 255, cv.THRESH_BINARY)
    # dilate_threshed = cv.adaptiveThreshold(dilate_h, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 111, 2)
    # cv.imshow('dilate_threshed', resize(dilate_threshed))

    thresh = cv.adaptiveThreshold(hsv[:, :, 1], 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 5)
    # cv.imshow("thresh", resize(thresh))

    median = cv.medianBlur(thresh, 5)
    # cv.imshow("median", resize(median))

    threshl = cv.adaptiveThreshold(hsv[:, :, 2], 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 5)
    # cv.imshow("threshl", resize(threshl))

    medianl = cv.medianBlur(threshl, 5)
    # cv.imshow("medianl", resize(medianl))

    combined = cv.bitwise_or(median, medianl)
    # combined = cv.morphologyEx(combined, cv.MORPH_CLOSE, (51, 51), iterations=10)
    # cv.imshow('combined', resize(combined))

    anded = cv.bitwise_and(combined, dilate_threshed)
    cv.imshow('anded', resize(anded))

    contours, hierarchy = cv.findContours(anded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv.imshow('contours', resize(img))

    boxes = []
    for cnt in contours:
        center, shape, rotation = cv.minAreaRect(cnt)
        if shape[0] * shape[1] < 10000:
            continue
        boxes.append(np.int0(cv.boxPoints((center, shape, rotation))))

    cv.drawContours(img, boxes, -1, (0, 0, 255), 2)
    cv.imshow('boxed', resize(img))

    return hsv


def main():
    img = get_img("mickey_mouse_24.jpg")
    no_bg_img = remove_bg(img)

    cv.imshow("no_bg", cv.resize(no_bg_img, (1920, 1080)))

    while True:
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":
    main()
