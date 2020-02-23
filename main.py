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
    cv.medianBlur(h, 7, dst=h)
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

    return extrema_points


def main():
    for img_path in (Path.cwd() / 'imgs' / 'flipped').glob("*.jpg"):
        img = cv.imread(str(img_path))
        pieces = get_pieces(img)
        corners = [c for p in pieces for c in get_corners(p, length=10, extrema_limit=0.1)]

        cv.drawContours(img, [cnt for p in pieces for cnt in p], -1, (255, 0, 0), thickness=2)
        cv.drawContours(img, corners, -1, (0, 255, 0), thickness=20)
        cv.imshow('corners?', resize(img))

        break

        # cv.imshow(f"no_bg '{img_path.stem}'", cv.resize(no_bg_img, (1920, 1080)))

    while True:
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":
    main()
