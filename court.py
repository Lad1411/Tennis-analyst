import numpy as np
import cv2

class Court:
    def __init__(self):
        self.baseline_width = 8.23
        self.length = 23.77


    def view_transformer(self):
        source_point = np.array(
            [[238, 152], [402, 150], [485, 494], [155, 494]],
            dtype=np.float32
        )

        des_point = np.array(
            [[0, 0], [self.baseline_width, 0], [self.baseline_width, self.length], [0, self.length]],
            dtype= np.float32
        )

        matrix,_ = cv2.findHomography(source_point, des_point, cv2.RANSAC)

        return matrix

    def perspective_transform(self, x,y):
        point = np.array([[[x,y]]], dtype=np.float32)

        matrix = self.view_transformer()
        real_point = cv2.perspectiveTransform(point, matrix)
        return real_point

