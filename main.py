import cv2
import numpy as np
from LaneDetection import LaneDetection


video = cv2.VideoCapture("lane_detection_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("output.avi", fourcc, 20, (640, 360))

while video.isOpened():

    isopened, frame = video.read()

    lanedetect = LaneDetection()

    if not isopened:
        break

    height = frame.shape[0]
    width = frame.shape[1]

    edges = cv2.Canny(frame, 220, 120)

    pts = np.array([[0,height],[width/2, height*0.6],[width,height]],dtype=np.int32)

    regionofinterest = lanedetect.get_the_mask(edges, pts)

    lines = cv2.HoughLinesP(regionofinterest, rho = 2, theta= np.pi/180, threshold=50, minLineLength=40, maxLineGap=150)

    image_with_lines = lanedetect.draw_lines(frame, lines)

    writer.write(image_with_lines)

    cv2.imshow("Original Video", frame)
    cv2.imshow("Lane Video", image_with_lines)
    cv2.waitKey(20)


video.release()
cv2.destroyAllWindows()

