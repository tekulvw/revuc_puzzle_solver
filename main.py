import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def resize(img):
    return cv.resize(img, (1920, 1080))


def get_img(fname):
    return cv.imread(f"imgs/{fname}")


def remove_bg(img):
    cv.imshow("orig", resize(img))

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow("hue", resize(hsv[:, :, 0]))
    cv.imshow("sat", resize(hsv[:, :, 1]))
    cv.imshow("val", resize(hsv[:, :, 2]))

    thresh = cv.adaptiveThreshold(hsv[:, :, 1], 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 5)
    cv.imshow("thresh", resize(thresh))

    median = cv.medianBlur(thresh, 5)
    cv.imshow("median", resize(median))

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
