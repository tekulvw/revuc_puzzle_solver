import cv2 as cv


def get_img(fname):
    return cv.imread(f"imgs/{fname}")


def remove_bg(img):
    # Convert to hsv and equalize
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    hsv[:, :, 0] = cv.equalizeHist(hsv[:, :, 0])
    hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])

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
