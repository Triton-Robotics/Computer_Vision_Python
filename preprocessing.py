import cv2
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import create_kalman

###############################################################################

# global filtering parameters for lightbars 
MIN_AREA = 20000 # min contour area considered valid

# hard coded scaling factors
scale_percent = 30 # percent of original size
width = int(1920 * scale_percent / 100)
height = int(1080 * scale_percent / 100)
dim = (width, height)

# flag to test with combined contours or individual contours 
TEST_COMBINED = True

###############################################################################

def resize_frame(frame):
    return cv2.resize(frame, dim)

    
def preprocess_frame(frame):
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get initial regions via thresholding
    ret, thresh = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)

    # create contours from thresholded regions found
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter out contours and store them 
    filter_contours = []
    area = cv2.contourArea(contours[0])
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > MIN_AREA:
            filter_contours.append(contours[i])

    bounding_box_edges = [] # get edges for bounding box - is going to be hard coded for now 
    centers = []
    for i in range(len(filter_contours)):
        rect = cv2.minAreaRect(filter_contours[i])
        # print(rect)
        
        # get all corners 
        center, offset, angle = rect
        x = center[0]
        y = center[1]
        w = offset[0]//2
        h = offset[1]//2

        test_pnt = np.int0(rect[0])
        # print(test_pnt)
        box = cv2.boxPoints(rect)
        
        # centers.append((test_pnt[0], test_pnt[1]))
        centers.append(tuple(test_pnt))
        # print(box)
        box = np.int0(box)
        # print(box)

        Xs = [i[0] for i in list(box)]
        Ys = [i[1] for i in list(box)]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        
        top_left = (x1, y1)
        bot_left = (x1, y2)
        top_right = (x2, y1)
        bot_right = (x2, y2)
        bounding_box_edges.append([top_left, top_right, bot_left, bot_right])

        # rectangles = cv2.circle(rectangles, centers[-1], 40, (255, 255, 0), -1)
        # rectangles = cv2.drawContours(rectangles,[box],-1,(255,255,255),15)
        
    # bounding_box_edges = np.int0(bounding_box_edges)
    # plt.imshow(rectangles, 'gray')

    # get corners for overall bounding box 
    top_left = (0, 0)
    bot_left = (0, 0)
    top_right = (0, 0)
    bot_right = (0, 0)

    if len(bounding_box_edges) == 2:
        top_left = bounding_box_edges[0][0]
        bot_left = bounding_box_edges[0][2]
        top_right = bounding_box_edges[1][1]
        bot_right = bounding_box_edges[1][3]

    # get center point for bounding box
    center_x = 0
    center_y = 0

    if len(centers) == 2:
        center_x = (centers[1][0] + centers[0][0])//2
        center_y = (centers[1][1] + centers[0][1])//2

    armor_box = None
    armor_box = cv2.rectangle(frame, (top_left[0], top_left[1]), (bot_right[0], bot_right[1]), (0, 255, 255), 20)
    # from above cell
    armor_box = cv2.circle(frame, (center_x, center_y), 40, (0, 255, 255), -1)
    return armor_box


# display each frame after resizing it 
def display_frames(original, preprocessed):
    screen_res = 1280, 720
    scale_width = screen_res[0] / original.shape[1]
    scale_height = screen_res[1] / original.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(original.shape[1] * scale)
    window_height = int(original.shape[0] * scale)

    # cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('original', window_width, window_height)

    cv2.namedWindow('preprocessed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('preprocessed', window_width, window_height)

    # original = cv2.resize(original, (window_width, window_height))
    preprocessed = cv2.resize(preprocessed, (window_width, window_height))

    # cv2.imshow('original', original)
    # cv2.imshow('preprocessed', preprocessed)


# function for testing drawing contours on each frame only 
def draw_frame_contours(frame, get_rectangles=True):
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not get_rectangles:
        res = cv2.drawContours(frame, contours, -1, (0, 255, 255), 20)
        return res
    
    return draw_frame_rectangles(frame, contours)

# convert contours into minimum angled bounding rectangles
def draw_frame_rectangles(frame, contours):
    rectangles = np.zeros((len(frame), len(frame[0])), dtype='uint8')
    centers = []
    for i in range(len(contours)):

        # might need to adjust this, some frames being ignored
        if cv2.contourArea(contours[i]) < 2000:
            continue

        rect = cv2.minAreaRect(contours[i])
        
        # get center of contour rectangle
        test_pnt = np.int0(rect[0])
        centers.append(tuple(test_pnt))

        box = cv2.boxPoints(rect)
        box = np.int0(box)

        rectangles = cv2.circle(frame, centers[-1], 20, (0, 255, 255), -1)
        rectangles = cv2.drawContours(rectangles,[box],-1,(0, 255, 255),6)
    
    return rectangles


# create contours from given frame
def get_frame_contours(frame):
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# additional parameters to try and filter contours
# can we append onto the contours array to bridge to regions together and 
# get the min rectangle area with a new center?
def filter_contours(contours):
    return [contour for contour in contours if cv2.contourArea(contour) > 1000]


# return (x, y) center coor and (w, h) from provided contour
def get_contour_center_and_wh(contour):
    if len(contour) == 0:
        return None

    rect = cv2.minAreaRect(contour)
    # center = tuple(np.int0(rect[0]))
    center_x = int(rect[0][0])
    center_y = int(rect[0][1])

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    Xs = [i[0] for i in list(box)]
    Ys = [i[1] for i in list(box)]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    w = x2 - x1
    h = y2 - y1

    return center_x, center_y, w, h


# attempting to combine contours into a single large contour
def combine_valid_contours(frame):
    contours = get_frame_contours(frame)
    contours = filter_contours(contours)
    combined = []

    for contour in contours:
        for point in contour:
            combined.append(point)

    return combined

# method of approach:
# maintain internal list of some past list of measurement vectors
# use the past list to push into kalman filter
# maybe we could assume center of rectangle starts at middle of frame?
# at each timestep dt, we get the center of the rectangle and also the 
# width and height of the rectangle to estimate size 
# what the measurement vector looks like: (cx, cy, w, h)
# currently works for drawing estimated center point
def test_kalman():
    cap = cv2.VideoCapture('dark_test.mp4')
    kalman = create_kalman() # create 8D kalman filter

    while(True):
        ok, frame = cap.read()
        frame = cv2.resize(frame, dim)

        valid_contours = combine_valid_contours(frame)
        measurement = get_contour_center_and_wh(np.array(valid_contours))

        if measurement == None:
            continue

        kalman.predict()
        kalman.update(measurement) # new measure calculated from detector
        curr_state = np.array(kalman.x).flatten()

        # need to think about how to draw rectangle on frame
        # could potentially be angled
        center_x = int(curr_state[0])
        center_y = int(curr_state[1])
        result = cv2.circle(frame, (center_x, center_y), 20, (0, 255, 255), -1)
        result = cv2.putText(result, '({x}, {y})'.format(x=center_x, y=center_y), 
                                (center_x - 40, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        # rectangles = cv2.drawContours(rectangles,[box],-1,(0, 255, 255),6)
        cv2.imshow('preprocessed', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
        

# run for provided video
def main():
    cap = cv2.VideoCapture('../dark_test.mp4')

    # if not cap.isOpened():
    #     raise Exception('could not start video capture object')
    counter = 0

    while (True):
        _, frame = cap.read()
        # res = preprocess_frame(frame)
        # screen_res = 1280, 720
        # scale_width = screen_res[0] / frame.shape[1]
        # scale_height = screen_res[1] / frame.shape[0]
        # scale = min(scale_width, scale_height)
        # window_width = int(frame.shape[1] * scale)
        # window_height = int(frame.shape[0] * scale)

        # cv2.namedWindow('original', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('original', window_width, window_height)

        # scale_percent = 40 # percent of original size
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100)
        # dim = (width, height)

        # cv2.namedWindow('preprocessed', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('preprocessed', width, height)

        # resize image
        resized = cv2.resize(frame, dim)
        print(resized.shape)
        # result = preprocess_frame(resized)
        result = draw_frame_contours(frame)
        print('done')
        print(counter)

        # save frames 
        # cv2.imwrite('dark' + str(counter) + '.jpg', result)

        counter += 1

        # test video stream
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # original = cv2.resize(original, (window_width, window_height))
        # preprocessed = cv2.resize(gray, (window_width, window_height))

        # test video stream preprocessing 
        cv2.imshow('preprocessed', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # cv2.destroyAllWindows()

def test_combined():

    cap = cv2.VideoCapture('../dark_test.mp4')

    while True:
        ok, frame = cap.read()
        # print(frame.shape)

        # scale_percent = 30 # percent of original size
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100)
        # dim = (width, height)
        frame = cv2.resize(frame, dim)

        # checking contour output:
        # img = cv2.imread('sample_dark.jpg')
        # print(filter_contours(get_frame_contours(img)))
        rectangle = combine_valid_contours(frame)
        if len(rectangle) == 0:
            continue

        rect = cv2.minAreaRect(np.array(rectangle))
        box = cv2.boxPoints(rect)

        center = tuple(np.int0(rect[0]))
        box = np.int0(box)

        # update frame with results from combining valid contours after filtering
        rectangles = cv2.circle(frame, center, 20, (0, 255, 255), -1)
        rectangles = cv2.putText(rectangles, '({x}, {y})'.format(x=center[0], y=center[1]), 
                                (center[0] - 40, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        rectangles = cv2.drawContours(rectangles,[box],-1,(0, 255, 255),6)
        # result = draw_frame_rectangles(img, rectangle)

        # result = cv2.drawContours(img, rectangle,-1,(0, 255, 255),40)
        # print(result.shape)
        cv2.imshow('combined_res', rectangles)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # cv2.imwrite('combined_sample.jpg',rectangles)
    
# it is obvious that this program is not very versatile, and the drawing 
# portion needs to be handled more effectively
# currently getting list index out of bounds for line 172
if __name__ == '__main__':
    # if TEST_COMBINED:
    #     test_combined()
    # else:
    #     main()
    test_kalman()