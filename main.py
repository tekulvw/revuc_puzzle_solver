import cv2 as cv
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from matplotlib import pyplot as plt


def resize(img):
    return cv.resize(img, (1920, 1080))


def remove_bg(img):
    # cv.imshow("orig", resize(img))

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(hsv)
    cv.equalizeHist(v, v)
    # cv.GaussianBlur(h, (7, 7), 3, dst=h)
    cv.medianBlur(h, 7, dst=h)
    cv.merge((h, s, v), hsv)

    lower_gray = np.array([5, 5, 10])
    upper_gray = np.array([20, 70, 255])

    mask = cv.inRange(hsv, lower_gray, upper_gray)
    mask = cv.medianBlur(mask, 7)

    res = cv.bitwise_and(img, img, mask=mask)

    contours, he = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    res = cv.drawContours(res, contours, -1, (0, 255, 0), 3)

    # Display the resulting frame
    # cv.imshow('frame', resize(cv.cvtColor(hsv, cv.COLOR_HSV2BGR)))
    # cv.imshow('hsv', resize(hsv))
    # cv.imshow('mask', resize(mask))
    # cv.imshow('res', resize(res))

    polygons = []
    for cnt in contours:
        try:
            p = Polygon([p[0] for p in cnt])
        except ValueError:
            pass
        else:
            polygons.append(p)

    polygons = sorted(polygons, key=lambda p: p.area, reverse=True)[:4]
    for p in polygons:
        plt.plot(*p.exterior.xy)

    plt.show()

    return hsv


def main():
    for img_path in (Path.cwd() / 'imgs' / 'flipped').glob("*.jpg"):
        img = cv.imread(str(img_path))
        no_bg_img = remove_bg(img)

        break

        # cv.imshow(f"no_bg '{img_path.stem}'", cv.resize(no_bg_img, (1920, 1080)))

    while True:
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":
    main()
