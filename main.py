import cv2 as cv
import numpy as np


def get_img(fname):
    return cv.imread(f"imgs/{fname}")


def remove_bg(img):
    # Convert to hsv and equalize
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    hsv[:, :, 0] = cv.equalizeHist(hsv[:, :, 0])
    hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])

    return hsv

    
def get_regions(img):
    sizeBig = (1920,1080)
    sizeSmall = (800,600)
    cv.imshow("original", cv.resize(img, (800, 600)))

    imgMedian = cv.medianBlur(img,31)

    imgGauss = cv.GaussianBlur(img,(51,51),cv.BORDER_DEFAULT)
    #cv.imshow("smooth", cv.resize(imgGauss,sizeSmall))

    


    hsv = cv.cvtColor(imgGauss, cv.COLOR_RGB2HSV)
    #hsv[:, :, 0] = cv.equalizeHist(hsv[:, :, 0])
    #hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])


    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    #cv.imshow("hue", cv.resize(hue, sizeSmall))
    #cv.imshow("sat", cv.resize(sat, sizeSmall))
    #cv.imshow("val", cv.resize(val, sizeSmall))

    


    hueedges = cv.Canny(hue,40,100)
    satedges = cv.Canny(sat,40,100)
    valedges = cv.Canny(val,40,100)

    sum1 = cv.add(hueedges,satedges)
    sum2 = cv.add(sum1,valedges)

    closing = cv.morphologyEx(sum2, cv.MORPH_CLOSE, (21,21))
    cv.imshow("closed", cv.resize(closing, sizeSmall))

    #lines = cv.HoughLinesP(satedges,1,np.pi/180,100,minLineLength=100,maxLineGap=1)

    ##for line in lines:
    ##    x1,y1,x2,y2 = line[0]
    ##    cv.line(satedges,(x1,y1),(x2,y2),(0,255,0),2)

    cv.imshow("hue edges", cv.resize(hueedges, sizeSmall))
    cv.imshow("sat edges", cv.resize(satedges, sizeSmall))
    cv.imshow("val edges", cv.resize(valedges, sizeSmall))

    



    #cv.imshow("hsv", cv.resize(hsv, (1920, 1080)))
    #bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    #cv.imshow("rgb", cv.resize(bgr, (1920, 1080)))



    #edges = cv.Canny(bgr,300,400)
    #cv.imshow("edges", cv.resize(edges, (1920, 1080)))

    #harris = cv.cornerHarris(img,10,10,10)
    #cv.imshow("harris", cv.resize(harris, (1920, 1080)))


    #contours = cv.findContours(img, 1, 2)



    #v.drawContours(img, contours, -1, (0, 255, 0), 3) 
    #cv.imshow("gauss line", cv.resize(harris, (800, 600)))

def get_regions(img):
    sizeBig = (1920,1080)
    sizeSmall = (800,600)
    cv.imshow("original", cv.resize(img, (800, 600)))

    imgMedian = cv.medianBlur(img,31)

    imgGauss = cv.GaussianBlur(img,(111,111),cv.BORDER_DEFAULT)
    #cv.imshow("smooth", cv.resize(imgGauss,sizeSmall))

    #imgGray = cv.cvtColor(imgGauss, cv.COLOR_RGB2GRAY)

    #cv.imshow("gray", cv.resize(imgGray, sizeSmall))
    
    #imgGrayEdges = cv.Canny(imgGray,40,100)
    #cv.imshow("grayEdges", cv.resize(imgGrayEdges, sizeSmall))


    hsv = cv.cvtColor(imgGauss, cv.COLOR_RGB2HSV)
    hsv[:, :, 0] = cv.equalizeHist(hsv[:, :, 0])
    hsv[:, :, 1] = cv.equalizeHist(hsv[:, :, 1])
    hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])


    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    cv.imshow("hue", cv.resize(hue, sizeSmall))
    cv.imshow("sat", cv.resize(sat, sizeSmall))
    cv.imshow("val", cv.resize(val, sizeSmall))

    


    hueedges = cv.Canny(hue,100,120)
    satedges = cv.Canny(sat,100,120)
    valedges = cv.Canny(val,100,120)

    sum1 = cv.bitwise_or(hueedges,satedges)
    sum2 = cv.bitwise_or(sum1,valedges)

    imgGauss2 = cv.GaussianBlur(sum2,(111,111),cv.BORDER_DEFAULT)
    cv.imshow("guass2", cv.resize(sum2, sizeSmall))


    #lines = cv.HoughLinesP(satedges,1,np.pi/180,100,minLineLength=100,maxLineGap=1)

    #for line in lines:
    #    x1,y1,x2,y2 = line[0]
    #    cv.line(satedges,(x1,y1),(x2,y2),(0,255,0),2)

    #cv.imshow("hue edges", cv.resize(hueedges, sizeSmall))
    #cv.imshow("sat edges", cv.resize(satedges, sizeSmall))
    #cv.imshow("val edges", cv.resize(valedges, sizeSmall))

    



def main():
    img = get_img("Lab.jpg")
    get_regions(img)

    #cv.imshow("no_bg", cv.resize(img_exp,  (800, 600)))

    while True:
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break



if __name__ == "__main__":
    main()
