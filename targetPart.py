import cv2
import numpy
import math
from enum import Enum

class TargetPart:
    """
    A part of a complete FRC 2019 Visual Target for Cargo Ship, Rocket and Loading Station
    """
    
    def __init__(self):
        """initializes all values to presets or None if need to be set
        """
        self.__rectPts = rectInput
        self.__type = None
        self.__distanceFromCenter=999

        __centerX = 100

    def process(self, contour):
        # get the minimum rectange of the target; rotated rectangle rather than bounding rect
        minRect = cv2.minAreaRect(contour)
		box = cv2.boxPoints(minRect)
		rectPts = perspective.order_points(box)

        #get slope to determine type (left or right)
        topRightX=rectPts.tr(0)
		topRightY=rectPts.tr(1)
		bottomRightX=rectPts.br(0)
		bottomRightY=rectPts.br(1)
        topLeftX=rectPts.tl(0)
        slope = calcSlope (bottomRightX, topRightX, , bottomRightY, topRightY)
        setType(slope)

    def setType(self, slope):
        if slope > 4:
            self.type = "L"
        if slope < -4:
            self.type = "R"

    def calcSlope(self, x1, x2, y1, y2) :
        return (y2-y1) / (x2-x1)

    def getType(self):
        return self.__type

    def getDistanceFromCenter(self):
        return self.__distanceFromCenter
    
    def order_points(pts):
	    # sort the points based on their x-coordinates
	    xSorted = pts[np.argsort(pts[:, 0]), :]
 
	    # grab the left-most and right-most points from the sorted
	    # x-roodinate points
	    leftMost = xSorted[:2, :]
	    rightMost = xSorted[2:, :]
 
	    # now, sort the left-most coordinates according to their
	    # y-coordinates so we can grab the top-left and bottom-left
	    # points, respectively
	    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	    (tl, bl) = leftMost
 
	    # now that we have the top-left coordinate, use it as an
	    # anchor to calculate the Euclidean distance between the
	    # top-left and right-most points; by the Pythagorean
	    # theorem, the point with the largest distance will be
	    # our bottom-right point
	    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	    # return the coordinates in top-left, top-right,
	    # bottom-right, and bottom-left order
	    return np.array([tl, tr, br, bl], dtype="float32")