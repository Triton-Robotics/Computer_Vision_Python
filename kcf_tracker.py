import cv2
import numpy as np 
from preprocessing import combine_valid_contours, resize_frame

box_found = False # is false until first detection of armor
tracker = cv2.TrackerKCF_create() # create tracker object
tracker_init = False # false until a detection is found and the

cap = cv2.VideoCapture('dark_test.mp4')

if not cap.isOpened():
    print ("Could not open video")
    exit()

while True:
    ok, frame = cap.read()
    frame = resize_frame(frame)

    if not ok:
        break

    timer = cv2.getTickCount()

    if not box_found:
        contours = combine_valid_contours(frame)
        # box_found = True

        # found box
        if len(contours) > 0:
            box_found = True
            rect = cv2.minAreaRect(np.array(contours))
            box = cv2.boxPoints(rect)

            center = tuple(np.int0(rect[0]))
            box = np.int0(box)

            Xs = [i[0] for i in list(box)]
            Ys = [i[1] for i in list(box)]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)

            w = x2 - x1
            h = y2 - y1

            if not tracker_init:
                tracker_init = True
                ok = tracker.init(frame, (x1, y1, w, h))

    else:
        ok, bbox = tracker.update(frame)
    
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,255), 2, 1)
    
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    cv2.imshow("KCF result", frame)

    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
