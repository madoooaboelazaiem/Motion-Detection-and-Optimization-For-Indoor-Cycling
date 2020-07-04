import cv2
import vg
import math
import numpy as np
import matplotlib.pyplot as plt
import imutils
from collections import deque
from imutils.video import VideoStream
from argparse import ArgumentParser
import time
import pandas as pd
from csv import DictWriter
import threading


def SquareBlobDetectorT1(inputSource):
    cap = cv2.VideoCapture(inputSource)
    hasFrame, image = cap.read()
    # image = cv2.imread(frame)
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()
    # loop over the contours
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
        # show the output image
        cv2.imshow("Image", image)
        cv2.imshow("2", thresh)
    cv2.waitKey(0)


def SquareBlobDetectorT2(inputSource):
    cap = cv2.VideoCapture(inputSource)
    hasFrame, image = cap.read()
    image = cv2.resize(image, (1024, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen, 127, 255, cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow('s',thresh)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # cv2.imshow('c',close)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 500
    max_area = 2000
    image_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            ROI = image[y:y+h, x:x+h]
            # cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 12), 2)
            image_number += 1

    cv2.imshow('sharpen', sharpen)
    # cv2.imshow('close', close)
    cv2.imshow('thresh', thresh)
    cv2.imshow('image', image)
    cv2.waitKey()

#Best Accuracy for test2Crop
def SimpleBlobWithCameraVtest2Crop(inputSource):
    cap = cv2.VideoCapture(inputSource)
    hasFrame, frame = cap.read()
    vid_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (frame.shape[1], frame.shape[0]))
    count = 0

    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        # print('fpsssssssssssssssss', cv2.CAP_PROP_POS_FRAMES)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        # count = count  # For Skipping Frames
        if not hasFrame:
            cv2.waitKey()
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

        edgeDetectedImage = cv2.threshold(sharpen,110, 256, cv2.THRESH_BINARY_INV)[1]
        
        cv2.imshow('Edge Detected Image', edgeDetectedImage)
        params = cv2.SimpleBlobDetector_Params()
        # im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('s',edgeDetectedImage)
        # Change thresholds
        params.minThreshold = 30
        params.maxThreshold = 256

        # Filter by Color

        params.filterByColor = True
        params.blobColor = 0  # for black

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 700        
        # params.minArea = 230
        params.maxArea = 1770

        # Filter by Circularity
        params.filterByCircularity = True
        # params.minCircularity = 0.4
        params.minCircularity = 0.2

        # Filter by Convexity
        params.filterByConvexity = True
        # params.minConvexity = 0.3
        params.minConvexity = 0.2

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.22
        # params.minInertiaRatio = 0.5


        # # Create a detector with the parameters
        # ver = (cv2.__version__).split('.')
        # if int(ver[0]) < 3 :
        # 	detector = cv2.SimpleBlobDetector(params)
        # else :
        # 	detector = cv2.SimpleBlobDetector_create(params)

        # Auto Scale Detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(edgeDetectedImage)
        blobPosition = []
        keypointCoordinates = []
        detected_keypoints_toString = []
        detected_keypoints = []
        keypoints_with_id = []
        keypoint_id = 1
        for keypoint in keypoints:
            blobPosition = (
                keypoint.pt[0],
                keypoint.pt[1],
                keypoint.size,
                keypoint.angle)
            keypointCoordinates.append(blobPosition)
        keypointCoordinates = sorted(
                keypointCoordinates, key=lambda k: (k[1],k[0]), reverse=True)
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypointCoordinates[i] + (keypoint_id,))
            detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(
                str(keypointCoordinates[i][2]))+' angle: '+(str(keypointCoordinates[i][3])))  # Converting the x and y positions to strings
            keypoint_id += 1

        nblobs = len(keypoints)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        # for curKey in keypointCoordinates:
        #     frame = cv2.circle(frame,(int(curKey[0]),int(curKey[1])),int(curKey[2]/2),(0, 0, 0), 10)
        # im_with_keypoints = frame 
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array(
            []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(nblobs):
            print(keypoints_with_id[i])
            cv2.putText(im_with_keypoints, str(keypoints_with_id[i][4]), (int(keypoints_with_id[i][0]), int(
                keypoints_with_id[i][1])), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        print(nblobs, 'From Blobs')
        # print(detected_keypoints_toString)
        # cv2.imwrite('Output.jpg', frameClone)
        # cimg = cv2.imread('Output.jpg',0)
        # cv2.imshow('gray',cimg)
        # Show keypoints
        # im_with_keypoints = blobConnection(im_with_keypoints)
        cv2.imshow("new Keypoints", im_with_keypoints)
        vid_writer.write(im_with_keypoints)

def SimpleBlobWithCameraV2(inputSource):
    cap = cv2.VideoCapture(inputSource)
    hasFrame, frame = cap.read()
    vid_writer = cv2.VideoWriter('test1.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (frame.shape[1], frame.shape[0]))
    count = 0

    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        # print('fpsssssssssssssssss', cv2.CAP_PROP_POS_FRAMES)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        # count = count  # For Skipping Frames
        if not hasFrame:
            cv2.waitKey()
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

        edgeDetectedImage = cv2.threshold(sharpen,190, 256, cv2.THRESH_BINARY_INV)[1]
        # cv2.imshow('Edge Detected Image', edgeDetectedImage)
        params = cv2.SimpleBlobDetector_Params()
        # im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('s',edgeDetectedImage)
        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 256

        # Filter by Color

        params.filterByColor = False
        params.blobColor = 0  # for black

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 800        
        # params.minArea = 230
        # params.maxArea = 1000

        # Filter by Circularity
        params.filterByCircularity = True
        # params.minCircularity = 0.4
        params.minCircularity = 0.2

        # Filter by Convexity
        params.filterByConvexity = True
        # params.minConvexity = 0.3
        params.minConvexity = 0.2

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.25
        # params.minInertiaRatio = 0.5


        # # Create a detector with the parameters
        # ver = (cv2.__version__).split('.')
        # if int(ver[0]) < 3 :
        # 	detector = cv2.SimpleBlobDetector(params)
        # else :
        # 	detector = cv2.SimpleBlobDetector_create(params)

        # Auto Scale Detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(edgeDetectedImage)
        blobPosition = []
        keypointCoordinates = []
        detected_keypoints_toString = []
        detected_keypoints = []
        keypoints_with_id = []
        keypoint_id = 1
        for keypoint in keypoints:
            blobPosition = (
                keypoint.pt[0],
                keypoint.pt[1],
                keypoint.size,
                keypoint.angle)
            keypointCoordinates.append(blobPosition)
        keypointCoordinates = sorted(
                keypointCoordinates, key=lambda k: (k[1],k[0]), reverse=True)
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypointCoordinates[i] + (keypoint_id,))
            detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(
                str(keypointCoordinates[i][2]))+' angle: '+(str(keypointCoordinates[i][3])))  # Converting the x and y positions to strings
            keypoint_id += 1

        nblobs = len(keypoints)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        # for curKey in keypointCoordinates:
        #     frame = cv2.circle(frame,(int(curKey[0]),int(curKey[1])),int(curKey[2]/2),(0, 0, 0), 10)
        # im_with_keypoints = frame 
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array(
            []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(nblobs):
            print(keypoints_with_id[i])
            cv2.putText(im_with_keypoints, str(keypoints_with_id[i][4]), (int(keypoints_with_id[i][0]), int(
                keypoints_with_id[i][1])), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        print(nblobs, 'From Blobs')
        # print(detected_keypoints_toString)
        # cv2.imwrite('Output.jpg', frameClone)
        # cimg = cv2.imread('Output.jpg',0)
        # cv2.imshow('gray',cimg)
        # Show keypoints
        # im_with_keypoints = blobConnection(im_with_keypoints)
        cv2.imshow("new Keypoints", im_with_keypoints)
        vid_writer.write(im_with_keypoints)

def SimpleBlobWithCameraV1(inputSource):
    cap = cv2.VideoCapture(inputSource)
    hasFrame, frame = cap.read()
    vid_writer = cv2.VideoWriter('test1.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (frame.shape[1], frame.shape[0]))
    count = 0

    while cv2.waitKey(50) < 0:
        hasFrame, frame = cap.read()
        # print('fpsssssssssssssssss', cv2.CAP_PROP_POS_FRAMES)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        # count = count + 0.5  # For Skipping Frames
        if not hasFrame:
            cv2.waitKey()
            break
        blurredImage = cv2.GaussianBlur(frame, (3, 3), 0)
        # cv2.imshow('Gaussian Blurred Image',blurredImage)

        # Detecting edges in Image using Canny edge Detector
        edgeDetectedImage = cv2.Canny(blurredImage, 60, 180)
        # cv2.imshow('Edge Detected Image', edgeDetectedImage)
        params = cv2.SimpleBlobDetector_Params()
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Change thresholds
        params.minThreshold = 35
        params.maxThreshold = 256

        # Filter by Color

        params.filterByColor = False
        # params.blobColor = 0

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 400        
        # params.minArea = 130
        params.maxArea = 1000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.2

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.2

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.3

        # # Create a detector with the parameters
        # ver = (cv2.__version__).split('.')
        # if int(ver[0]) < 3 :
        # 	detector = cv2.SimpleBlobDetector(params)
        # else :
        # 	detector = cv2.SimpleBlobDetector_create(params)

        # Auto Scale Detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(im)
        blobPosition = []
        keypointCoordinates = []
        detected_keypoints_toString = []
        detected_keypoints = []
        keypoints_with_id = []
        keypoint_id = 1
        for keypoint in keypoints:
            blobPosition = (
                keypoint.pt[0],
                keypoint.pt[1],
                keypoint.size,
                keypoint.angle)
            keypointCoordinates.append(blobPosition)
        keypointCoordinates = sorted(
                keypointCoordinates, key=lambda k: (k[1],k[0]), reverse=True)
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypointCoordinates[i] + (keypoint_id,))
            detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(
                str(keypointCoordinates[i][2]))+' angle: '+(str(keypointCoordinates[i][3])))  # Converting the x and y positions to strings
            keypoint_id += 1

        nblobs = len(keypoints)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(edgeDetectedImage, keypoints, np.array(
            []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(nblobs):
            print(keypoints_with_id[i])
            cv2.putText(im_with_keypoints, str(keypoints_with_id[i][4]), (int(keypoints_with_id[i][0]), int(
                keypoints_with_id[i][1])), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        print(nblobs, 'From Blobs')
        # print(detected_keypoints_toString)
        # cv2.imwrite('Output.jpg', frameClone)
        # cimg = cv2.imread('Output.jpg',0)
        # cv2.imshow('gray',cimg)
        # Show keypoints
        # im_with_keypoints = blobConnection(im_with_keypoints)
        cv2.imshow("new Keypoints", im_with_keypoints)
        vid_writer.write(im_with_keypoints)

def SimpleBlobWithCameraV3(inputSource): ## Big white blob = 23.x so area 530 make it 540 Small one 18 equals 324
    cap = cv2.VideoCapture(inputSource)
    hasFrame, frame = cap.read()
    vid_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (frame.shape[1], frame.shape[0]))
    count = 0
    # print(frame.shape[1], frame.shape[0])
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        # print('fpsssssssssssssssss', cv2.CAP_PROP_POS_FRAMES)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        # count = count  # For Skipping Frames
        if not hasFrame:
            cv2.waitKey()
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        # edgeDetectedImage = cv2.Canny(blur, 60, 100)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
        # edgeDetectedImage = cv2.filter2D(gray, -1, sharpen)
        # edgeDetectedImage = cv2.threshold(sharpen,250, 256, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] ## Make high light intensity on blobs and increase the threshold as you want
        #Default  
        edgeDetectedImage = cv2.threshold(sharpen,140, 256, cv2.THRESH_BINARY_INV)[1] ## Make high light intensity on blobs and increase the threshold as you want

        # edgeDetectedImage = cv2.threshold(sharpen,140, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] ## Make high light intensity on blobs and increase the threshold as you want
        cv2.imshow('Edge Detected Image', edgeDetectedImage)
        params = cv2.SimpleBlobDetector_Params()
        # im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('s',gray)
        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 256

        # Filter by Color

        params.filterByColor = False
        params.blobColor = 0  # for black

        # Filter by Area.
        params.filterByArea = True
        # params.minArea = 800        
        params.minArea = 300
        params.maxArea = 1200

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.2
        # params.minCircularity = 0.2

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.1
        # params.minConvexity = 0.2

        # Filter by Inertia
        params.filterByInertia = True
        # params.minInertiaRatio = 0.25
        params.minInertiaRatio = 0.15


        # # Create a detector with the parameters
        # ver = (cv2.__version__).split('.')
        # if int(ver[0]) < 3 :
        # 	detector = cv2.SimpleBlobDetector(params)
        # else :
        # 	detector = cv2.SimpleBlobDetector_create(params)

        # Auto Scale Detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(edgeDetectedImage)
        blobPosition = []
        keypointCoordinates = []
        detected_keypoints_toString = []
        detected_keypoints = []
        keypoints_with_id = []
        keypoint_id = 1
        for keypoint in keypoints:
            blobPosition = (
                keypoint.pt[0],
                keypoint.pt[1],
                keypoint.size,
                keypoint.angle)
            keypointCoordinates.append(blobPosition)
        keypointCoordinates = sorted(
                keypointCoordinates, key=lambda k: (k[1],k[0]), reverse=True)
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypointCoordinates[i] + (keypoint_id,))
            detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(
                str(keypointCoordinates[i][2]))+' angle: '+(str(keypointCoordinates[i][3])))  # Converting the x and y positions to strings
            keypoint_id += 1

        nblobs = len(keypoints)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        # for curKey in keypointCoordinates:
        #     frame = cv2.circle(frame,(int(curKey[0]),int(curKey[1])),int(curKey[2]/2),(0, 0, 0), 10)
        # im_with_keypoints = frame 
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array(
            []), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(nblobs):
            print(keypoints_with_id[i])
            cv2.putText(im_with_keypoints, str(keypoints_with_id[i][4]), (int(keypoints_with_id[i][0]), int(
                keypoints_with_id[i][1])), font, 0.5, (0, 255, 124), 4, cv2.LINE_AA)

        print(nblobs, 'From Blobs')
        # print(detected_keypoints_toString)
        # cv2.imwrite('Output.jpg', frameClone)
        # cimg = cv2.imread('Output.jpg',0)
        # cv2.imshow('gray',cimg)
        # Show keypoints
        # im_with_keypoints = blobConnection(im_with_keypoints)
        cv2.imshow("new Keypoints", im_with_keypoints)
        vid_writer.write(im_with_keypoints)
def SimpleBlobWithCameraV4(inputSource): # Best Performance ## Big white blob = 23.x so area 530 make it 540 Small one 18 equals 324
    cap = cv2.VideoCapture(2)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps',fps)
    hasFrame, frame = cap.read()
    vid_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (frame.shape[1], frame.shape[0]))
    count = 0
    df = pd.DataFrame(columns=['1','2','3','4','5','6'])
    # print(frame.shape[1], frame.shape[0])
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        # print('fpsssssssssssssssss', cv2.CAP_PROP_POS_FRAMES)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        # count = count  # For Skipping Frames
        if not hasFrame:
            cv2.waitKey()
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        # edgeDetectedImage = cv2.Canny(blur, 60, 100)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
        ret, x = cv2.threshold(blur, 155, 255, cv2.THRESH_BINARY)
        cv2.imshow('sh',x)
        # edgeDetectedImage = cv2.threshold(sharpen,140, 256, cv2.THRESH_BINARY_INV + cv2.THRESH_BINARY)[1] ## Make high light intensity on blobs and increase the threshold as you want
        edgeDetectedImage = np.invert(x)
        #Default  
        # edgeDetectedImage = cv2.threshold(sharpen,140, 256, cv2.THRESH_BINARY_INV)[1] ## Make high light intensity on blobs and increase the threshold as you want

        # edgeDetectedImage = cv2.threshold(sharpen,140, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] ## Make high light intensity on blobs and increase the threshold as you want
        cv2.imshow('Edge Detected Image', edgeDetectedImage)
        params = cv2.SimpleBlobDetector_Params()
        # im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('s',gray)
        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 256

        # Filter by Color

        params.filterByColor = True
        params.blobColor = 0  # for black

        # Filter by Area.
        params.filterByArea = True
        # params.minArea = 800        
        params.minArea = 200
        # params.maxArea = 800

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.25
        # params.minCircularity = 0.2

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.1
        # params.minConvexity = 0.2

        # Filter by Inertia
        params.filterByInertia = True
        # params.minInertiaRatio = 0.25
        params.minInertiaRatio = 0.2


        # # Create a detector with the parameters
        # ver = (cv2.__version__).split('.')
        # if int(ver[0]) < 3 :
        # 	detector = cv2.SimpleBlobDetector(params)
        # else :
        # 	detector = cv2.SimpleBlobDetector_create(params)

        # Auto Scale Detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(edgeDetectedImage)
        blobPosition = []
        keypointCoordinates = []
        detected_keypoints_toString = []
        detected_keypoints = []
        keypoints_with_id = []
        keypoint_id = 1
        for keypoint in keypoints:
            blobPosition = (
                keypoint.pt[0],
                keypoint.pt[1],
                keypoint.size)
            keypointCoordinates.append(blobPosition)
        keypointCoordinates = sorted(
                keypointCoordinates, key=lambda k: (k[1],k[0]), reverse=True)
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypointCoordinates[i] + (keypoint_id,))
            detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(
                str(keypointCoordinates[i][2])))  # Converting the x and y positions to strings
            keypoint_id += 1

        nblobs = len(keypoints)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        # for curKey in keypointCoordinates:
        #     frame = cv2.circle(frame,(int(curKey[0]),int(curKey[1])),int(curKey[2]/2),(0, 0, 0), 10)
        # im_with_keypoints = frame 
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array(
            []), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(nblobs):
            print(keypoints_with_id[i])
            cv2.putText(im_with_keypoints, str(keypoints_with_id[i][3]), (int(keypoints_with_id[i][0]), int(
                keypoints_with_id[i][1])), font, 0.5, (0, 255, 124), 4, cv2.LINE_AA)
        # print('keu',keypoints_with_id)
        angle = angleCalculationSimple(keypoints_with_id)
        pos = 30
        for i in range(len(angle)):
            cv2.putText(frame, 'Angle of point '+str(angle[i][2])+' is ' + str(
                round((angle[i][3]), 2)), (100, pos), font, 1, (255, 255, 255), 3, cv2.LINE_AA)
            pos = pos + 30
        print(nblobs, 'From Blobs')
        # print(detected_keypoints_toString)
        # cv2.imwrite('Output.jpg', frameClone)
        # cimg = cv2.imread('Output.jpg',0)
        # cv2.imshow('gray',cimg)
        # Show keypoints
        # im_with_keypoints = blobConnection(im_with_keypoints)
        if count%30 == 0 :
            print('count',count)
            field_names = []
            row_dict = {}
            for i in range(len(angle)):
                # print('anglle',angle)
                col = str(angle[i][3])
                # print('col',col)
                field_names.append(col)
                row_dict[col] = angle[i][4]
                # print('row,',row_dict[col])
    
                # Append a dict as a row in csv file
            df = addRow(df,row_dict)
            print(df)  
        count+=1
        
        cv2.imshow("new Keypoints", im_with_keypoints)
        vid_writer.write(im_with_keypoints)



def SimpleBlobDetection(frame):
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 256

    # Filter by Color
    # default false
    params.filterByColor = True
    params.blobColor = 255

    # Filter by Area.
    # Working minArea = 70
    params.filterByArea = True
    params.minArea = 305
    # params.maxArea = 80

    # Filter by Circularity
    params.filterByCircularity = True
    # default was 0.5 minCir
    params.minCircularity = 0.7

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.6

    # # Create a detector with the parameters
    # ver = (cv2.__version__).split('.')
    # if int(ver[0]) < 3 :
    # 	detector = cv2.SimpleBlobDetector(params)
    # else :
    # 	detector = cv2.SimpleBlobDetector_create(params)

    # Auto Scale Detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(frame)
    blobPosition = []
    keypointCoordinates = []
    detected_keypoints_toString = []
    detected_keypoints = []
    keypoints_with_id = []
    keypoint_id = 1
    for keypoint in keypoints:
        # print(keypoint)
        blobPosition = (
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            keypoint.angle)
        keypointCoordinates.append(blobPosition)
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypointCoordinates[i] + (keypoint_id,))
        detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(
            str(keypointCoordinates[i][2]))+' angle: '+(str(keypointCoordinates[i][3])))  # Converting the x and y positions to strings
        keypoint_id += 1

    nblobs = len(keypoints)
    # print(nblobs)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array(
        []), (0, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    font = cv2.FONT_HERSHEY_SIMPLEX

    print(nblobs, 'From Blobs')
    # print(detected_keypoints_toString)
    # cv2.imwrite('Output.jpg', frameClone)
    # cimg = cv2.imread('Output.jpg',0)
    # cv2.imshow('gray',cimg)
    # Show keypoints
    # im_with_keypoints = blobConnection(im_with_keypoints)
    # cv2.imshow("new Keypoints", im_with_keypoints)
    plt.imshow(cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('Output.jpg', im_with_keypoints)


def houghCirclesDetection(img):
    countBlobs = 0
    # Read image as gray-scale
    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    img_blur = cv2.medianBlur(gray, 5)
    # Apply hough transform on the image
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1,
                               img.shape[0]/32, param1=200, param2=17, minRadius=20, maxRadius=20)
    # Draw detected circles
    if circles is not None:
        blobPositions = []
        blobID = 1
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            blobPositions.append((i[0], i[1], i[2]))
            countBlobs = countBlobs + 1
            # Draw outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw inner circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        blobPositions = sorted(blobPositions, key=lambda k: k[1], reverse=True)
        nBlobs = 1
        newCoord = []
        for i in range(len(blobPositions)):
            newCoord.append((blobPositions[i][0], blobPositions[i][1], nBlobs))
            nBlobs = nBlobs + 1
        blobPositions = newCoord
        print('mado', blobPositions)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(countBlobs):
            # print(blobPositions[i])
            cv2.putText(img, str(blobPositions[i][2]), (int(blobPositions[i][0]), int(
                blobPositions[i][1])), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
    # angle = angleCalculation(blobPositions)
    angle = angleCalculationV2(blobPositions)
    pos = 30
    for i in range(len(angle)):
        angle1 = angle[i][3][0]
        angle2 = angle[i][3][1]
        if(angle2 == 0):
            cv2.putText(img, 'Angle of point '+str(angle[i][2])+' is ' + str(
                round((angle1), 2)), (100, pos), font, 1, (255, 255, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(img, 'Angle of point '+str(angle[i][2])+' is ' + str(
                round((angle1), 2)), (100, pos), font, 1, (255, 255, 255), 3, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(img, 'Angle of point '+str(angle[i][2])+' dash is ' + str(
                round((angle2), 2)), (100, pos), font, 1, (255, 255, 255), 3, cv2.LINE_AA)
        pos = pos + 30

    print('Blobs Detected ', countBlobs)
    print(angle)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('OutputHoughCirlces2.jpg', img)


def houghCircleDetectionVideoSorted(inputsource): #big circle radius 12
    cap = cv2.VideoCapture(inputSource)
    hasFrame, frame = cap.read()
    #(frame.shape[1], frame.shape[0])
    vid_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (frame.shape[1], frame.shape[0]))
    count = 0

    while cv2.waitKey(50) < 0:
        hasFrame, frame = cap.read()
        fps = cap.get(cv2.cv2.CAP_PROP_POS_FRAMES)
        # frame = cv2.resize(frame, (800,700))
        # print('fpsssssssssssssssss', fps)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        # count = count  # For Skipping Frames
        if not hasFrame:
            cv2.waitKey()
            break
        countBlobs = 0
        # Read image as gray-scale
        # Convert to gray-scale

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur the image to reduce noise
        img_blur = cv2.medianBlur(gray, 5)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_blur = cv2.filter2D(img_blur, -1, sharpen_kernel)
        cv2.imshow('s,',img_blur)
        # Apply hough transform on the image
        # Default circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/32, param1=200, param2=17, minRadius=20, maxRadius=20)
        # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=18, minRadius=7, maxRadius=7)
        circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1,
                                   img.shape[0]/32, param1=200, param2=17, minRadius=9, maxRadius=13)
        # Draw detected circles
        # print(circles)
        # circles = sorted(circles, key=lambda tup: tup)
        # print(circles[0])
        if circles is not None:
            blobPositions = []
            blobCoord = []
            blobID = 1
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # print('size',i[3])
                blobPositions.append((i[0], i[1], i[2]))
                countBlobs = countBlobs + 1
                # Draw outer circle
                cv2.circle(frame, (i[0], i[1]), i[2], (255, 255, 0), 2)
                # print('colored')
                # Draw inner circle
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            blobPositions = sorted(
                blobPositions, key=lambda k: k[1], reverse=True)
            nBlobs = 1
            newCoord = []
            for i in range(len(blobPositions)):
                newCoord.append(
                    (blobPositions[i][0], blobPositions[i][1], nBlobs))
                nBlobs = nBlobs + 1
            print('mado', blobPositions)
            blobPositions = newCoord
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(countBlobs):
                # print(blobPositions[i])
                cv2.putText(frame, str(blobPositions[i][2]), (int(blobPositions[i][0]), int(
                    blobPositions[i][1])), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
            angle = angleCalculation(blobPositions)
            pos = 30
            for i in range(len(angle)):
                cv2.putText(frame, 'Angle of point '+str(angle[i][2])+' is ' + str(
                    round((angle[i][3]), 2)), (100, pos), font, 1, (255, 255, 255), 3, cv2.LINE_AA)
                pos = pos + 30
            # blobCoord = sorted(blobCoord, key=lambda tup: (tup[0],tup[1]))
            # frame = houghCirclesConnection(frame,blobCoord)
            print('Blobs Detected ', countBlobs)
            # print(blobPositions)
        cv2.imshow("new Keypoints", frame)
        vid_writer.write(frame)


def houghCirclesConnection(img, pts):
    # pts = [k.pt for k in keypoints]#Opencv can't draw an arrow between a single point center and a list of points. So we'll have to go over it in a for loop as such
    # max(pts,key=lambda item:item[1])
    # print('points',pts)
    # max(lis,key=lambda item:item[1])
    # nearest = min(cooList, key=lambda x: distance(x, coordinate))
    # centre = (246, 234) # This should be changed to the center of your image
    for i in range(len(pts)):
        for j in range(len(pts)):
            if(j+1 <= i):
                FirstPt = tuple(map(int, pts[j]))
                pt = tuple(map(int, pts[j+1]))
                # print('xxxxxxxxxxxx',(pt,FirstPt))
                # print(soFar)
                img = cv2.line(img=img, pt1=(FirstPt), pt2=(pt),
                               color=(0, 255, 255), thickness=2)
    return img


def blobDetLive(inputSource):
    ap = ArgumentParser()
    args = vars(ap.parse_args())
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=32,
                    help="max buffer size")
    # define the lower and upper boundaries of the "green"
    # ball in the HSV color space
    greenLower = (176, 12, 86)
    greenUpper = (255, 0, 0)

    # initialize the list of tracked points, the frame counter,
    # and the coordinate deltas
    pts = deque(maxlen=64)
    counter = 0
    (dX, dY) = (0, 0)
    direction = ""

    # if a video path was not supplied, grab the reference
    # to the webcam
    cap = cv2.VideoCapture(inputSource)
    hasFrame, frame = cap.read()
    vid_writer = cv2.VideoWriter('outputHough2.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (frame.shape[1], frame.shape[0]))
    count = 0
    time.sleep(2.0)

    while cv2.waitKey(1) < 0:

        hasFrame, frame = cap.read()
        fps = cap.get(cv2.cv2.CAP_PROP_POS_FRAMES)
        print('fpsssssssssssssssss', fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        count = count + 0.5  # For Skipping Frames
        if not hasFrame:
            cv2.waitKey()
            break
    # allow the camera or video file to warm up

    # keep looping

        # handle the frame from VideoCapture or VideoStream

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video

        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                pts.appendleft(center)

        # loop over the set of tracked points
        for i in np.arange(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # check to see if enough points have been accumulated in
            # the buffer
            if counter >= 10 and i == 1 and pts[-10] is not None:
                # compute the difference between the x and y
                # coordinates and re-initialize the direction
                # text variables
                dX = pts[-10][0] - pts[i][0]
                dY = pts[-10][1] - pts[i][1]
                (dirX, dirY) = ("", "")

                # ensure there is significant movement in the
                # x-direction
                if np.abs(dX) > 20:
                    dirX = "East" if np.sign(dX) == 1 else "West"

                # ensure there is significant movement in the
                # y-direction
                if np.abs(dY) > 20:
                    dirY = "North" if np.sign(dY) == 1 else "South"

                # handle when both directions are non-empty
                if dirX != "" and dirY != "":
                    direction = "{}-{}".format(dirY, dirX)

                # otherwise, only one direction is non-empty
                else:
                    direction = dirX if dirX != "" else dirY

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # show the movement deltas and the direction of movement on
        # the frame
        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 3)
        cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1)

        # show the frame to our screen and increment the frame counter
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        counter += 1


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def angle3(a, b, c):
    a = np.array([a[0], a[1], 1])
    b = np.array([b[0], b[1], 1])
    c = np.array([c[0], c[1], 1])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)*180.0 / math.pi
    return angle


def length(v):
    return math.sqrt(dotproduct(v, v))


def angleCalculation2(p1, p2):
    # vector1 = [1,0,0]
    # vector2 = [0,1,0]

    # unit_vector1 = vector1 / np.linalg.norm(vector1)
    # unit_vector2 = vector2 / np.linalg.norm(vector2)

    # dot_product = np.dot(unit_vector1, unit_vector2)

    # angle = np.arccos(dot_product) #angle in radian
    # vec1 = np.array([p1[0], p1[1], 1])
    # vec2 = np.array([p2[0], p2[1], 1])

    # r = vg.angle(vec1, vec2)
    p21x = p1[0] - p2[0] if p1[0] > p2[0] else p2[0] - p1[0]
    p21y = p1[1]-p2[1] if p1[1] > p2[1] else p2[1] - p1[1]
    angle = float(math.atan2(p21x, p21y))
    angle = angle * 180 / math.pi
    return angle


def angleCalculation(data):
    newData = []
    # threading.Timer(1.0, angleCalculation).start()
    for i in range(len(data)):
        newCord = []
        if(i+2 < len(data)):
            print(i+2, len(data))
            p1 = (data[i][0], data[i][1])
            p2 = (data[i+1][0], data[i+1][1])
            p3 = (data[i+2][0], data[i+2][1])
            p21x = p1[0] - p2[0] if p1[0] > p2[0] else p2[0] - p1[0]
            p21y = p1[1]-p2[1] if p1[1] > p2[1] else p2[1] - p1[1]
            p21 = (p21x, p21y)
            p23x = p3[0] - p2[0] if p3[0] > p2[0] else p2[0] - p3[0]
            p23y = p3[1]-p2[1] if p3[1] > p2[1] else p2[1] - p3[1]
            p23 = (p23x, p23y)
            # angle = angleCalculation2(p21,p23)
            angle = angle3(p1, p2, p3)
            newData.append((data[i+1][0], data[i+1][1], data[i+1][2], angle))
            # newData.append((data[i][0],data[i][1]))
            # data[i+1] = newCord
            print('angleeee', angle)
    return newData
def angleCalculationSimple(data):
    newData = []
    # threading.Timer(1.0, angleCalculation).start()
    for i in range(len(data)):
        newCord = []
        if(i+2 < len(data)):
            print(i+2, len(data))
            p1 = (data[i][0], data[i][1])
            p2 = (data[i+1][0], data[i+1][1])
            p3 = (data[i+2][0], data[i+2][1])
            p21x = p1[0] - p2[0] if p1[0] > p2[0] else p2[0] - p1[0]
            p21y = p1[1]-p2[1] if p1[1] > p2[1] else p2[1] - p1[1]
            p21 = (p21x, p21y)
            p23x = p3[0] - p2[0] if p3[0] > p2[0] else p2[0] - p3[0]
            p23y = p3[1]-p2[1] if p3[1] > p2[1] else p2[1] - p3[1]
            p23 = (p23x, p23y)
            # angle = angleCalculation2(p21,p23)
            angle = angle3(p1, p2, p3)
            newData.append((data[i+1][0], data[i+1][1], data[i+1][2],data[i+1][3], angle))
            # newData.append((data[i][0],data[i][1]))
            # data[i+1] = newCord
            print('angleeee', angle)
    return newData
def angleCalculationV2(data):
    newData = []
    p1 = (data[0][0], data[0][1])
    p2 = (480, 985)
    angle = angleCalculation2(p1,p2)
    newData.append((data[0][0], data[0][1], data[0][2], (angle,0)))
    # threading.Timer(1.0, angleCalculation).start()
    for i in range(len(data)):
        newCord = []
        angle2 = 0
        # if(i == len(data)-2): # knee outer angle calc Not Working
        #     p1 = (data[i+1][0],data[i+1][1])
        #     p2 = (data[i][0],data[i][1])
        #     angle2 = angleCalculation2(p1,p2)
        #     print('da5al',angle2, p1,p2)
        if(i == len(data)-1): # Last Point with horizontal calc
            p1 = (data[i-1][0], data[i-1][1])
            p2 = (data[i][0], data[i][1])
            p3 = (712, 284)
            # angle = angleCalculation2(p21,p23)
            # print('da5al' ,p1, p2, p3)
            angle2 = angle3(p1, p2, p3)
            newData.append((data[i][0], data[i][1], data[i][2], (angle2,0)))
        if(i+2 < len(data)): 
            print(i+2, len(data))
            p1 = (data[i][0], data[i][1])
            p2 = (data[i+1][0], data[i+1][1])
            p3 = (data[i+2][0], data[i+2][1])
            # p21x = p1[0] - p2[0] if p1[0] > p2[0] else p2[0] - p1[0]
            # p21y = p1[1]-p2[1] if p1[1] > p2[1] else p2[1] - p1[1]
            # p21 = (p21x, p21y)
            # p23x = p3[0] - p2[0] if p3[0] > p2[0] else p2[0] - p3[0]
            # p23y = p3[1]-p2[1] if p3[1] > p2[1] else p2[1] - p3[1]
            # p23 = (p23x, p23y)
            # angle = angleCalculation2(p21,p23)
            angle = angle3(p1, p2, p3)
            if(i+1 == 2):
                p1 = (data[i-1][0], data[i][1])
                p2 = (data[i+1][0], data[i+1][1])
                p3 = (data[i+2][0], data[i+2][1])
                angle2 = angle3(p1, p2, p3)
                # print(angle2)
                # print(angle2 , 'sssssssssssssssssssssssssssssssssssssss', p1,(data[i-2][0],data[i-2][1]))
            if(i+1 == len(data)-2):
                angle2 = 180 - angle
            # print('an2',angle2)
            newData.append((data[i+1][0], data[i+1][1], data[i+1][2], (angle,angle2)))
            # newData.append((data[i][0],data[i][1]))
            # data[i+1] = newCord
            # print('angleeee', angle)
    return newData


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def find_squares(inputSource):
    cap = cv2.VideoCapture(inputSource)
    hasFrame, img = cap.read()
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = cv2.resize(img, (656, 368))
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(
                    gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(
                bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max(
                        [angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    for c in squares:
        cv2.drawContours(img, [c], -1, (0, 255, 0), 3)
    cv2.imshow('squares', img)
    return squares


def findBlobVid(inputSource):
    cap = cv2.VideoCapture(inputSource)
    hasFrame, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    vid_writer = cv2.VideoWriter('temp.avi',fourcc, 30, (900, 600))
    count = 0
    df = pd.DataFrame(columns=['1','2','3','4','5','6','7','8','9'])
    # df.to_csv('data.csv') 
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        frame = cv2.resize(frame, (900, 600))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

        thresh = cv2.threshold(sharpen, 127, 255, cv2.THRESH_BINARY_INV)[1]
        # print('fpsssssssssssssssss', cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        count = count + 0.5  # For Skipping Frames
        if not hasFrame:
            cv2.waitKey()
            break
        # blurredImage = cv2.GaussianBlur(frame, (3, 3), 0)
        # cv2.imshow('Gaussian Blurred Image',blurredImage)

        # Detecting edges in Image using Canny edge Detector
        # edgeDetectedImage = cv2.Canny(blurredImage, 60, 100)
        # cv2.imshow('Edge Detected Image', edgeDetectedImage)
        params = cv2.SimpleBlobDetector_Params()
        # im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 256

        # Filter by Color
        # default false
        params.filterByColor = True
        params.blobColor = 0  # for black

        # Filter by Area.
        # Working minArea = 70
        params.filterByArea = True
        params.minArea = 170
        params.maxArea = 410

        # Filter by Circularity
        params.filterByCircularity = True
        # default was 0.5 minCir
        params.minCircularity = 0.2

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.9

        # # Create a detector with the parameters
        # ver = (cv2.__version__).split('.')
        # if int(ver[0]) < 3 :
        # 	detector = cv2.SimpleBlobDetector(params)
        # else :
        # 	detector = cv2.SimpleBlobDetector_create(params)

        # Auto Scale Detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(thresh)
        blobPosition = []
        blobPositions = []
        keypointCoordinates = []
        detected_keypoints_toString = []
        detected_keypoints = []
        keypoints_with_id = []
        angle_arr = []
        keypoint_id = 1
        for keypoint in keypoints:
            blobPosition = (
                keypoint.pt[0],
                keypoint.pt[1],
                keypoint.size,
                keypoint.angle)
            keypointCoordinates.append(blobPosition)
        for i in range(len(keypoints)):
            blobPositions.append((keypointCoordinates[i][0], keypointCoordinates[i][1],keypoint_id))
            keypoints_with_id.append(keypointCoordinates[i] + (keypoint_id,))
            detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(
                str(keypointCoordinates[i][2]))+' angle: '+(str(keypointCoordinates[i][3])))  # Converting the x and y positions to strings
            keypoint_id += 1

        nblobs = len(keypoints)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array(
            []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(nblobs):
            # print(keypoints_with_id[i])
            cv2.putText(im_with_keypoints, str(keypoints_with_id[i][4]), (int(keypoints_with_id[i][0]), int(
                keypoints_with_id[i][1])), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        angle = angleCalculation(blobPositions)
        # print(angle)
        pos = 30
        for i in range(len(angle)):
            print(len(angle))
            cv2.putText(im_with_keypoints, 'Angle of point '+str(angle[i][2])+' is ' + str(round((angle[i][3]), 2)), (40, pos), font, 0.75, (0, 255, 255), 3, cv2.LINE_AA)
            pos = pos + 30
        print(nblobs, 'From Blobs')
        # print(detected_keypoints_toString)
        # cv2.imwrite('Output.jpg', frameClone)
        # cimg = cv2.imread('Output.jpg',0)
        # cv2.imshow('gray',cimg)
        # Show keypoints
        # im_with_keypoints = blobConnection(im_with_keypoints)
        field_names = []
        row_dict = {}
        for i in range(len(angle)):
            col = str(angle[i][2])
            field_names.append(col)
            row_dict[col] = angle[i][3]
 
            # Append a dict as a row in csv file
        df = addRow(df,row_dict)
        cv2.imshow("thresh", thresh)
        cv2.imshow("new Keypoints", im_with_keypoints)
        vid_writer.write(im_with_keypoints)

def addRow(df,row_dict):
    # threading.Timer(1.0, addRow).start()
    df = df.append(row_dict, ignore_index=True)
    # print(row_dict)
    # append_dict_as_row('data.csv', row_dict, field_names)
    df.to_csv('data.csv') 
    return df
def findBlobs(inputSource):
    cap = cv2.VideoCapture(inputSource)
    hasFrame, image = cap.read()
    image = cv2.resize(image, (1024, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen, 127, 255, cv2.THRESH_BINARY_INV)[1]
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 256

    # Filter by Color
    # default false
    params.filterByColor = True
    params.blobColor = 0  # for black

    # Filter by Area.
    # Working minArea = 70
    params.filterByArea = True
    params.minArea = 240
    # params.maxArea = 80

    # Filter by Circularity
    params.filterByCircularity = True
    # default was 0.5 minCir
    params.minCircularity = 0.2

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.9

    # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.6

    # # Create a detector with the parameters
    # ver = (cv2.__version__).split('.')
    # if int(ver[0]) < 3 :
    # 	detector = cv2.SimpleBlobDetector(params)
    # else :
    # 	detector = cv2.SimpleBlobDetector_create(params)

    # Auto Scale Detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(thresh)
    blobPosition = []
    keypointCoordinates = []
    detected_keypoints_toString = []
    detected_keypoints = []
    keypoints_with_id = []
    keypoint_id = 1
    for keypoint in keypoints:
        # print(keypoint)
        blobPosition = (
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            keypoint.angle)
        keypointCoordinates.append(blobPosition)
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypointCoordinates[i] + (keypoint_id,))
        detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(
            str(keypointCoordinates[i][2]))+' angle: '+(str(keypointCoordinates[i][3])))  # Converting the x and y positions to strings
        keypoint_id += 1

    nblobs = len(keypoints)
    print(nblobs)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array(
        []), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.imshow('thresh', thresh)
    cv2.imshow('image', im_with_keypoints)
    cv2.waitKey()




 
def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)

img = cv2.imread("./images/cyclingP.png", 1)
inputSource = 'FirstTrial.mp4'
# img = cv2.resize(img,(656,368))
# SimpleBlobDetection(img)
# SimpleBlobDetection(img)
SimpleBlobWithCameraV3(inputSource)
# blobDetLive(inputSource)
# findBlobVid(inputSource)
# SimpleBlobWithCameraV1(inputSource)
cv2.waitKey(0)
