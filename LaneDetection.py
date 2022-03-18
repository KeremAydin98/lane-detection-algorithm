import numpy as np
import cv2


class LaneDetection():

    def draw_lines(self, image, lines):

        lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype="uint8")

        for line in lines:

            for x1,y1,x2,y2 in line:
                cv2.line(lines_image, (x1,y1), (x2,y2),(0,0,255),2)

        draw_image = cv2.addWeighted(image, 0.8, lines_image, 0.2, 0)

        return draw_image

    def get_the_mask(self, image, pts):

        mask = np.zeros(image.shape[:2], dtype="uint8")

        cv2.fillPoly(mask, [pts], color = 255)

        regionofinterest = cv2.bitwise_and(image, image, mask = mask)

        return regionofinterest

