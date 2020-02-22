import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def resize(img):
    return cv.resize(img, (1920, 1080))


def get_img(fname):
    return cv.imread(f"imgs/{fname}")


def remove_bg(img):
    cv.imshow("orig", resize(img))

    gray = cv.equalizeHist(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    cv.imshow("gray", resize(gray))
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    cv.imshow("thresh", resize(thresh))

    blurred = cv.GaussianBlur(thresh, (5, 5), 7)
    cv.imshow("blurred", resize(blurred))

    denoise = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv.imshow("denoise", resize(denoise))

    return gray


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
