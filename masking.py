import cv2
import numpy as np 
import matplotlib.pyplot as plt

###############################################################################

# scale factors for display purposes
scale_percent = 30 # percent of original size
width = int(1920 * scale_percent / 100)
height = int(1080 * scale_percent / 100)
dim = (width, height)

# which filter to test/test individual pixels to grab hsv values
ONLY_HSV = True
TEST_PIXELS = False

RED_VAL = 0
BLUE_VAL = 255
GREEN_VAL = 0

###############################################################################

# testing hsv ranges for color filtering from hsv conversion
def hsv_filter(img):
    # test ranges
    img = cv2.resize(img, dim)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_range = (110, 100, 100)
    upper_range = (130, 255, 255)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    res = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('hsv filter', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# util function for simply grabbing pixels converted into hsv for informational purposes
def convert_pixel_to_hsv(red, green, blue):
    try:
        pixel = np.uint8([[[blue, green, red]]])
        return cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    except Exception as e:
        return e


# testing sobel edge gradient in x dirction
def sobel_filter(img):
    # kernel = np.ones((5,5),np.uint8)
    img = cv2.resize(img, dim)
    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel = np.absolute(sobel)
    sobel = np.uint8(sobel)
    cv2.imshow('sobel', sobel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if TEST_PIXELS:
        print(convert_pixel_to_hsv(RED_VAL, GREEN_VAL, BLUE_VAL))
    elif ONLY_HSV:
        img = cv2.imread('sample_dark.jpg')
        hsv_filter(img)
    else:
        img = cv2.imread('sample_dark.jpg')
        sobel_filter(img)