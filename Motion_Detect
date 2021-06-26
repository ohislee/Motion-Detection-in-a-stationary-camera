import cv2 as cv
import numpy as np
from motion_tracker import *

#create tracker object
tracker = EuclideanDistTracker()

capture = cv.VideoCapture('Videos/vehicle_moving.mp4')

fps = capture.get(cv.CAP_PROP_FPS)  #FPS How many frames are transmitted per second
w = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (w, h)

video_output = cv.VideoWriter('Videos/motions.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, size)



#object detection from stable camera
object_detector = cv.createBackgroundSubtractorKNN(history=50, detectShadows=False)

i = 0  # nFrames: (0 -> nframes-1)
while 1:

    ret, frame = capture.read()

    mask = object_detector.apply(frame)
    _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    detections = []
    for cnt in contours:
        
        area = cv.contourArea(cnt)
        if area > 300:
            #cv.drawContours(roi, cnt, -1, (0,255,0), 2)
            x,y,w,h = cv.boundingRect(cnt)
            #cv.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 1)
            detections.append([x,y,w,h])


    #2. object_tracker
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        cv.putText(frame, str(id), (x, y-15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)


        #saving output
    if ret:
        video_output.write(frame)
    else:
        break
    i += 1

    cv.imshow('Mask', mask)
    cv.imshow('Contours', frame)
    if cv.waitKey(2) & 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
