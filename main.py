import cv2 as cv
import numpy as np
from pathlib import Path


def resize(img):
    return cv.resize(img, (1920, 1080))


def remove_bg(img):
    cv.imshow("orig", resize(img))

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    return hsv


def main():
    for img_path in (Path.cwd() / 'imgs' / 'flipped').glob("*.jpg"):
        img = cv.imread(str(img_path))
        no_bg_img = remove_bg(img)

        cv.imshow(f"no_bg '{img_path.stem}", cv.resize(no_bg_img, (1920, 1080)))

    while True:
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":
    main()
