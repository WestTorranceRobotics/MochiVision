#Import opencv
import cv2
#Import numpy, mainly for arrays
import numpy as np
#import cscore
import cscore
#import netowrktables
from networktables import NetworkTables
#import logging for network tables messages
import logging
import math

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


#set this thing
logging.basicConfig(level=logging.DEBUG)

#create the usb cam
cam = cscore.UsbCamera("usbcam", 0)

#The pixel format is YUYV becuase that is all the PS Eye supports
#other cameras can use others, probably mjpeg preferably
#the resolution is 320x240 at 30 FPS
cam.setVideoMode(cscore.VideoMode.PixelFormat.kYUYV, 320, 240, 30)

#set exposure mode to manual (0-auto 1-manual)
cam.getProperty("auto_exposure").set(1)
#set the exposure value to 0 (0-255)
cam.getProperty("exposure").set(0)
#Set the White balance to manual mode (0-manual 1-auto)
cam.getProperty("white_balance_automatic").set(0)
#set the white balance to a preset for indoor
#I don't think this camera supports this
#set the gain mode to manual (0-manual 1-auto)
cam.getProperty("gain_automatic").set(0)

#create a cv sink, which will grab images from the camera
cvsink = cscore.CvSink("cvsink")
cvsink.setSource(cam)

#preallocate memory for images so that we dont allocate it every loop
img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)
hsv = np.zeros(shape=(240, 320, 3), dtype=np.uint8)
thresh = np.zeros(shape=(240, 320, 3), dtype=np.uint8)
opening = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

#object and camera constants
objectInches=2.25
cameraXFieldPixels=320
cameraYFieldPixels=240
cameraXFieldAngle=67.4 
cameraYFieldAngle=48.4
cameraTanXAngle=math.tan(math.radians(cameraXFieldAngle/2))
cameraTanYAngle=math.tan(math.radians(cameraYFieldAngle/2))
min_ratio=0.6
max_ratio=0.75

#set up mjpeg server, the ip for this is 0.0.0.0:8081
mjpegServer = cscore.MjpegServer("httpserver", 8081)
mjpegServer.setSource(cam)


#initialize the netowrktable
NetworkTables.initialize(server='10.51.24.2')

#loop forever 
while True:

	#grab the frame from the sink, call it img
	#this resets img, so it is not drawn on anymore
	time, img = cvsink.grabFrame(img)

	#If there's an error or no frame, lets skip this loop fam
	if time == 0:
		#skip the rest of this iteration (no point in processing an image that doesnt exist)
		continue

	#convert the img from RGB colors to HSV colors
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	#threshhold for green retroreflection
	thresh = cv2.inRange(hsv, np.array([33, 127, 26]), np.array([96, 255, 230]))

	#the kernel for morphological operations, (maybe not the best for this application
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
	#open operation, which is just a erode followed by a dilate
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

	#find contours, outputs a tuple, we only care about the second value
	#which is a python list of all the contours, which is an array I think?
	_, contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	#loop through each contour in contours
	for contour in contours:
                
		#get the contour bounding rectangle

        x,y,w,h = cv2.boundingRect(contour)
		ratio = (float)(w) / h
		if (ratio < min_ratio or ratio > max_ratio):
                continue

		#draw a rectangle around the contour to calculate its center
                
		#calculate centerX and centerY
		#these are geometric, so a concave shape will be seen as a 
		#rectangle with no concave features
		geometricX = int(x+(w/2))
		geometricY = int(y+(h/2))
		distanceToTarget=(objectInches*cameraXFieldPixels)/(2*cameraTanXAngle*(w+h)/2)

		#min rectangle will show slope of contour
		minRect = cv2.minAreaRect(contour)
    	box = cv2.boxPoints(minRect)
		rectPts = order_points(box)
    		
		#draw the contours on the original image
		cv2.drawContours(img, contour, -1, (255, 0, 255), 3)
	
		#this draws the geometric center	
		#cv2.circle(img, (geometricX, geometricY), 3, (0, 255, 0), -1)
		posX=(geometricX-100) if geometricX > 100 else (geometricX+20)
		posY=(geometricY-20) if geometricY > 30 else (geometricY+20)
		cv2.putText(img, "C({0},{1})".format(geometricX,geometricY),
			(posX, posY), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 255, 0), 2)
		cv2.putText(img, "L {0} R {4} T {1} B {5} W {2} H {3}".format(x,y,w,h,(x+w),(y+h)),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
		cv2.putText(img, "Center({0}, {1})".format(int(geometricX-(cameraXFieldPixels/2)),int((cameraYFieldPixels/2)-geometricY)),
                        (100, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
		cv2.putText(img, "Distance: {0:.2f}{1}".format(distanceToTarget,'"'),
                        (10, 230), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)

	#display the img, which has contours drawn on it now
	cv2.imshow('Meme', img)
        
	#record keypresses
	c = cv2.waitKey(50)& 0xFF
    
	#if the escape key was pressed, stop the program
	if(c == 27):
		break

#destroy all the windows created by opencv
cv2.destroyAllWindows()

