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

#set up mjpeg server, the ip for this is 0.0.0.0:8081
mjpegServer = cscore.MjpegServer("httpserver", 8081)
mjpegServer.setSource(cam)

#h, s, and v values for sliders. h means high, l means low
#hl = 0
#hh = 0
#sl = 0
#sh = 0
#vl = 0
#vh = 0

#function for calling when each h, s, and v slider moves
#they essentially do nothing
#only here becuase unsure how to put no option, "None" may work

#def setH():
#       hh = hh

#def setS():
#       sh = sh

#def setV():
#       vh = vh

#create the window for the sliders, and add the sliders
#cv2.namedWindow('Sliders')
#cv2.createTrackbar('hl', 'Sliders', 0, 255, setH)
#cv2.createTrackbar('hh', 'Sliders', 0, 255, setH)      
#cv2.createTrackbar('sl', 'Sliders', 0, 255, setS)
#cv2.createTrackbar('sh', 'Sliders', 0, 255, setS)
#cv2.createTrackbar('vl', 'Sliders', 0, 255, setV)
#cv2.createTrackbar('vh', 'Sliders', 0, 255, setV)

#initialize the netowrktable
#NetworkTables.initialize(server='10.51.24.2')

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
	
	#grab the slider values
	#hl = cv2.getTrackbarPos('hl', 'Sliders')
	#hh = cv2.getTrackbarPos('hh', 'Sliders')
	#sl = cv2.getTrackbarPos('sl', 'Sliders')
	#sh = cv2.getTrackbarPos('sh', 'Sliders')
	#vl = cv2.getTrackbarPos('vl', 'Sliders')
	#vh = cv2.getTrackbarPos('vh', 'Sliders')

	#threshhold for adjustable sliders
	#thresh = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh]))

	#threshhold for green retroreflection
	#thresh = cv2.inRange(hsv, np.array([33, 127, 26]), np.array([96, 255, 230]))

	#threshhold for note 5 white retroreflection   
	#thresh = cv2.inRange(hsv, np.array([0, 52, 128]), np.array([56, 216, 255]))

	#threshhold for iphone white retroreflection
	thresh = cv2.inRange(hsv, np.array([0, 15, 9]), np.array([83, 179, 255]))

	#the kernel for morphological operations, (maybe not the best for this application
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
	#open operation, which is just a erode followed by a dilate
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

	#find contours, outputs a tuple, we only care about the second value
	#which is a python list of all the contours, which is an array I think?
	_, contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	#loop through each contour in contours
	for contour in contours:
                
		#get the contour area of the contour
		contourArea = cv2.contourArea(contour)
                
		#if the contour is not big enough, skip this iteration
		if(contourArea < 1000):
			continue		
                
		#draw a rectangle around the contour to calculate its center
		x, y, w, h = cv2.boundingRect(contour)
                
		#calculate centerX and centerY
		#these are geometric, so a concave shape will be seen as a 
		#rectangle with no concave features
		geometricX = int(x+(w/2))
		geometricY = int(y+(h/2))
	
		#this method of calculating the center uses moments, which will
		#calculate it based on center of mass, so concave and convex
		#features will be taken into account
		#I favor the geometric method, since the Stronghold target was
		#a "U" around the target. This moment method may be better in
		#some circumstances
		#moment = cv2.moments(contour)
		#contourX = int(moment['m10'] / moment['m00'])
		#contourY = int(moment['m01'] / moment['m00'])

		#draw the contours on the original image
		cv2.drawContours(img, contour, -1, (255, 0, 255), 3)

		#this draws the moment center
		#cv2.circle(img, (contourX, contourY), 3, (255, 255, 255), -1)
		#cv2.putText(img, "Mcenter", (contourX - 20, contourY - 20),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	
		#this draws the geometric center	
		cv2.circle(img, (geometricX, geometricY), 3, (0, 255, 0), -1)
		cv2.putText(img, "Center (" + str(geometricX) + ", " + str(geometricY) + ")",
			(geometricX - 20, geometricY - 20), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 255, 0), 2)

	#display the img, which has contours drawn on it now
	cv2.imshow('Meme', img)
        
	#record keypresses
	c = cv2.waitKey(50)& 0xFF
    
	#if the escape key was pressed, stop the program
	if(c == 27):
		break

#destroy all the windows created by opencv
cv2.destroyAllWindows()
