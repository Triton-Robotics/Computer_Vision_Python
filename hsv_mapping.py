# file for looking at valid ranges at which to separate the image into
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

# create hsv figure to look for ranges to threshold video frames at 
def gen_hsv_fig(img, using_rgb=False):

    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    hsv_img = None
    if using_rgb:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    else:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('sample_dark.jpg')
    gen_hsv_fig(img)