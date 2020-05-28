import cv2
import vg
import math
import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial.distance import euclidean
from math import sqrt
def SimpleBlobWithCamera():
   inputsource = 'output.avi'
   cap = cv2.VideoCapture(inputsource)
   hasFrame, frame = cap.read()
   vid_writer = cv2.VideoWriter('blobLines.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1],frame.shape[0]))
   count = 0
   width  = frame.shape[1] # float
   height = frame.shape[0] # float
   print(width,height)
   while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    count = count + 2 # For Skipping Frames
    if not hasFrame:
        cv2.waitKey()
        break

    params = cv2.SimpleBlobDetector_Params()
    
    
    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 256
    
    # Filter by Color
    
    params.filterByColor = False
    # params.blobColor = 0
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 70
    # params.maxArea = 80
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    
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
    keypointCoordinates = []
    keypointSizes = []
    detected_keypoints_toString = []
    keypoints_with_id = []
    keypoint_id = 1
    # keypoints = sorted(keypoints, key=lambda tup: (tup.pt[0],tup.pt[1]))
    keypoints = sorted(keypoints, key=lambda tup: tup.pt[0])
    for keypoint in keypoints:
        keypointCoordinates.append((keypoint.pt[0],
                keypoint.pt[1]))
        keypointSizes.append(keypoint.size)
    # pts = [k.pt for k in keypoints]
    # print('keypoints',keypoints)
    # print('sortedKeypoints',testy)
    # print('keypoints',keypointCoordinates)
    # keypointCoordinates = sortTuples(keypointCoordinates)
    # print('Sorted Keypoints',keypointCoordinates)
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypointCoordinates[i] + (keypoint_id,))
        detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(str(keypointSizes[i]))) #Converting the x and y positions to strings
        keypoint_id += 1

    nblobs = len(keypoints)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.circle(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)),3,  (0, 255, 0), 3)
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(nblobs):
        # print(keypoints_with_id[i])
        cv2.putText(im_with_keypoints, str(keypoints_with_id[i][2]) ,(int(keypoints_with_id[i][0]),int(keypoints_with_id[i][1])), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    print(nblobs, 'From Blobs')
    # print(detected_keypoints_toString)
    # cv2.imwrite('Output.jpg', frameClone)
    # cimg = cv2.imread('Output.jpg',0)
    # cv2.imshow('gray',cimg)
    # Show keypoints
    # im_with_keypoints=connectingBlobs(im_with_keypoints,keypoints)
    keypoints_with_id = getAngleWithRespectToCenter(keypoints_with_id,(frame.shape[1],frame.shape[0]))
    keypoints_with_id = getAngleTwoPoints(keypoints_with_id)   
    # print(keypoints_with_id)
    im_with_keypoints = connectingBlobs4(im_with_keypoints,keypointCoordinates)
    # im_with_keypoints = blobConnection(im_with_keypoints)
    # cv2.namedWindow('new Keypoints', flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
    cv2.imshow("new Keypoints", im_with_keypoints)
    # plt.show()
    # plt.pause(0.0001) #Note this correction
    vid_writer.write(im_with_keypoints)

def SimpleBlobDetection(frame):
    params = cv2.SimpleBlobDetector_Params()
    
    
    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 256
    
    # Filter by Color
    #default false
    params.filterByColor = True
    params.blobColor = 255
    
    # Filter by Area.
    #Working minArea = 70
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
    keypointCoordinates = []
    keypointSizes = []
    detected_keypoints_toString = []
    keypoints_with_id = []
    keypoint_id = 1
    keypoints = sorted(keypoints, key=lambda tup: (tup.pt[0],tup.pt[1]))
    # keypoints = sorted(keypoints, key=lambda tup: tup.pt[1]) sorted correctly 
    for keypoint in keypoints:
        # print(keypoint)
        keypointCoordinates.append((keypoint.pt[0],
                keypoint.pt[1]))
        keypointSizes.append(keypoint.size)
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypointCoordinates[i] + (keypoint_id,))
        detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(str(keypointSizes[i]))) #Converting the x and y positions to strings
        keypoint_id += 1
    keypoints_with_id = getAngleTwoPoints(keypoints_with_id)   
    # print(keypoints_with_id)
    nblobs = len(keypoints)
    # print(nblobs)
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # x = (0,0)
    # y = (0,0)
    # z = (0,0)
    # a , b = (0,0)
    for i in range(nblobs):
        # print(keypoints_with_id[i])
        cv2.putText(im_with_keypoints, str(keypoints_with_id[i][2]) ,(int(keypoints_with_id[i][0]),int(keypoints_with_id[i][1])), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    #     if(keypoints_with_id[i][4] == 2):
    #         x = (keypoints_with_id[i][0],keypoints_with_id[i][1])
    #     if(keypoints_with_id[i][4] == 5):
    #         y = (keypoints_with_id[i][0],keypoints_with_id[i][1])   
    #     if(keypoints_with_id[i][4] == 4):
    #         z = (keypoints_with_id[i][0],keypoints_with_id[i][1])   
        
    # a = (y[0]-x[0],y[1]-x[1])
    # b = (y[0]-z[0],y[1]-z[1])    
    # angleCalculation(a,b)
    print(nblobs, 'From Blobs')
    # print(detected_keypoints_toString)
    # cv2.imwrite('Output.jpg', frameClone)
    # cimg = cv2.imread('Output.jpg',0)
    # cv2.imshow('gray',cimg)
    # Show keypoints
    # im_with_keypoints = blobConnection(im_with_keypoints)
    # cv2.imshow("new Keypoints", im_with_keypoints)
    im_with_keypoints = houghCirclesConnection(im_with_keypoints,keypointCoordinates)
    # im_with_keypoints = blobConnection(im_with_keypoints)
    plt.imshow(cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('Output3.jpg', im_with_keypoints)
def blobConnection(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    cv2.bitwise_not(threshold, threshold)
    contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    listx = []
    listy=[]

    for i in range(0, len(contours)):
        c = contours[i]
        size = cv2.contourArea(c)
        if size < 1000:
            # M = cv2.moments(c)
            M = cv2.moments(threshold)
            print(M)
            # print(M)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            listx.append(cX)
            listy.append(cY)

    listxy = list(zip(listx,listy))
    listxy = np.array(listxy)

    for i in range(0, len(listxy)):    
        x1 = listxy[i,0]
        y1 = listxy[i,1]
        distance = 0
        secondx = []
        secondy = []
        dist_listappend = []
        sort = []   
        for j in range(0, len(listxy)):      
            if i == j:
                pass     
            else:
                x2 = listxy[j,0]
                y2 = listxy[j,1]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                secondx.append(x2)
                secondy.append(y2)
                dist_listappend.append(distance)               
        secondxy = list(zip(dist_listappend,secondx,secondy))
        sort = sorted(secondxy, key=lambda second: second[0])
        sort = np.array(sort)
        cv2.line(img, (x1,y1), (int(sort[0,1]), int(sort[0,2])), (0,0,255), 2)
    cv2.imshow("new Keypoints", img)
    return img
def angleCalculation(p1,p2):
    # vector1 = [1,0,0]
    # vector2 = [0,1,0]

    # unit_vector1 = vector1 / np.linalg.norm(vector1)
    # unit_vector2 = vector2 / np.linalg.norm(vector2)

    # dot_product = np.dot(unit_vector1, unit_vector2)

    # angle = np.arccos(dot_product) #angle in radian
    vec1 = np.array([p1[0], p1[1], 1])
    vec2 = np.array([p2[0], p2[1], 1])

    r = vg.angle(vec1, vec2)
    print(r)
    #You can also specify a viewing angle to compute the angle via projection: vg.angle(vec1, vec2, look=vg.basis.z)
    # Signed Angle via projection: vg.signed_angle(vec1, vec2, look=vg.basis.z)
def houghCirclesDetection(img):
    countBlobs = 0
    # Read image as gray-scale
    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    img_blur = cv2.medianBlur(gray, 5)
    # Apply hough transform on the image
    # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/32, param1=200, param2=17, minRadius=4, maxRadius=15)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/32, param1=200, param2=17, minRadius=20, maxRadius=20)
    # Draw detected circles
    blobPositions = []
    blobCoord = []
    if circles is not None:
        blobID = 1
        # print(circles)
        circles = np.uint16(np.around(circles))
        # circles = sorted(circles, key=lambda tup: tup)

        # print(circles)
        for i in circles[0, :]:
            blobPositions.append((i[0] ,i[1] , i[2],blobID))
            blobCoord.append((i[0] ,i[1]))
            countBlobs = countBlobs + 1
            # Draw outer circle
            img = cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 3)
            # Draw inner circle
            img = cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
            img = cv2.putText(img, str(blobID) ,(int(i[0]),int(i[1])),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 13), 3
            , cv2.LINE_AA)
            blobID = blobID + 1   
        blobPositions = sorted(blobPositions,key=lambda tup: tup[1],reverse=True)   
        blobCoord = sorted(blobCoord,key=lambda tup: tup[1],reverse=True)   
        # print(blobPositions)    
        blobPositions = getAngleTwoPoints(blobPositions)   
        # print(blobPositions)
        img = houghCirclesConnection(img,blobCoord)
    # print('Blobs Detected ',blobPositions)
    # print(blobPositions)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('OutputHoughCirlces.jpg', img)
def houghCircleDetectionVideo():
   inputsource = 'output.avi'
   cap = cv2.VideoCapture(inputsource)
   hasFrame, frame = cap.read()
   vid_writer = cv2.VideoWriter('outputHough.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1],frame.shape[0]))
   count = 0

   while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        count = count + 2 # For Skipping Frames
        if not hasFrame:
            cv2.waitKey()
            break
        countBlobs = 0
        # Read image as gray-scale
        # Convert to gray-scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur the image to reduce noise
        img_blur = cv2.medianBlur(gray, 5)
        # Apply hough transform on the image
        # Default circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/32, param1=200, param2=17, minRadius=20, maxRadius=20)
        circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/32, param1=200, param2=17, minRadius=4, maxRadius=15)
        # Draw detected circles
        blobPositions = []
        blobCoord = []
        if circles is not None:
            blobID = 1
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                blobPositions.append((i[0] ,i[1] , i[2],blobID))
                blobCoord.append((i[0] ,i[1]))
                countBlobs = countBlobs + 1
                # Draw outer circle
                frame = cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw inner circle
                frame = cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
                frame = cv2.putText(frame, str(blobID) ,(int(i[0]),int(i[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                blobID = blobID + 1
            frame = houghCirclesConnection(frame,blobCoord)
        print('Blobs Detected ',countBlobs)
        # print(blobPositions)
        cv2.imshow("new Keypoints", frame)
        vid_writer.write(frame)
def connectingBlobs(img,pts):
    # pts = [k.pt for k in keypoints]#Opencv can't draw an arrow between a single point center and a list of points. So we'll have to go over it in a for loop as such
    # max(pts,key=lambda item:item[1])
    # print('points',pts)
    #max(lis,key=lambda item:item[1])
    # nearest = min(cooList, key=lambda x: distance(x, coordinate))
    # centre = (246, 234) # This should be changed to the center of your image
    FirstPt = tuple(map(int, pts[0]))
    soFar = []
    for pt in pts:
        temp = pts
        temp.remove(pt)

        # pt = tuple(map(int, pt))
        # pt = min(soFar,key=lambda item:euclidean((item[0],item[1]), FirstPt))
        # nearest = min(soFar, key=lambda x: distance(x, FirstPt))
        # nearest = min(soFar, key=lambda c: (c[0]- FirstPt[0])**2 + (c[1]-FirstPt[1])**2)
        try:
            nearest = min(temp, key=lambda x: ((abs(x[0]-FirstPt[0]),abs(x[1]-FirstPt[1]))))
            pt = tuple(map(int, nearest))
            # print('xxxxxxxxxxxx',(pt,FirstPt),temp)
            # print(soFar)
            img = cv2.line(img=img, pt1=(FirstPt), pt2=(pt), color=(0, 0, 255), thickness = 2)
            FirstPt = pt
        except:
            continue
    return img
def connectingBlobs3(img,pts):
    # pts = [k.pt for k in keypoints]#Opencv can't draw an arrow between a single point center and a list of points. So we'll have to go over it in a for loop as such
    # max(pts,key=lambda item:item[1])
    # print('points',pts)
    #max(lis,key=lambda item:item[1])
    # nearest = min(cooList, key=lambda x: distance(x, coordinate))
    # centre = (246, 234) # This should be changed to the center of your image
    FirstPt = tuple(map(int, pts[0]))
    soFar = []
    for i in range(len(pts)):
        for pt in pts:
            temp = pts[:]
            temp.remove(pt)

            # pt = tuple(map(int, pt))
            # pt = min(soFar,key=lambda item:euclidean((item[0],item[1]), FirstPt))
            # nearest = min(soFar, key=lambda x: distance(x, FirstPt))
            # nearest = min(soFar, key=lambda c: (c[0]- FirstPt[0])**2 + (c[1]-FirstPt[1])**2)
            try:
                nearest = min(temp, key=lambda x: ((abs(x[0]-FirstPt[0]),abs(x[1]-FirstPt[1]))))
                pt = tuple(map(int, nearest))
                # print('xxxxxxxxxxxx',(pt,FirstPt),temp)
                # print(soFar)
                img = cv2.line(img=img, pt1=(FirstPt), pt2=(pt), color=(0, 0, 255), thickness = 2)
                FirstPt = pt
            except:
                continue
    return img
def connectingBlobs4(img,pts):
    # pts = [k.pt for k in keypoints]#Opencv can't draw an arrow between a single point center and a list of points. So we'll have to go over it in a for loop as such
    # max(pts,key=lambda item:item[1])
    # print('points',pts)
    #max(lis,key=lambda item:item[1])
    # nearest = min(cooList, key=lambda x: distance(x, coordinate))
    # centre = (246, 234) # This should be changed to the center of your image
    for i in range(len(pts)):
        for j in range(len(pts)):
            # if(j == int(len(pts)/2)-1):
            #     j = j + 1
            if(j+1 <= i):
                FirstPt = tuple(map(int, pts[j]))
                pt = tuple(map(int, pts[j+1]))
                # print('xxxxxxxxxxxx',(pt,FirstPt))
                # print(soFar)
                img = cv2.line(img=img, pt1=(FirstPt), pt2=(pt), color=(0, 0, 255), thickness = 2)
    return img
def connectingBlobs2(img,pts):
   #Opencv can't draw an arrow between a single point center and a list of points. So we'll have to go over it in a for loop as such
   # pts = [k.pt for k in keypoints]
    # max(pts,key=lambda item:item[1])
    # print('points',pts)
    #max(lis,key=lambda item:item[1])
    # nearest = min(cooList, key=lambda x: distance(x, coordinate))
    # centre = (246, 234) # This should be changed to the center of your image
    # 1. Generating array with the distances to the current blob that contains [(distance,coordinates)] get min tuple based on distance that is our coordinates
    
    n = len(pts)
    for i in range(n):
        FirstPt = pts[i]
        temp =  pts[:]  # right way of cloning a list .. python is pass by reference so manipulation to the current reference would occur if t = pts
        for j in range(n):
            # print('j',j)
            # print('length',len(pts))
            # print('n',n)
            # print('currentpoints',pts)
            # print(pts[j])
            # print(len(temp), 'tempp',temp)
            temp.remove(pts[j]) # removing the current blob and the last blob not to draw a line to the same point again
            # if FirstPt in temp:
            #     temp.remove(FirstPt)
            generateDistanceArray = getDistances(pts[j],temp)
            temp =  pts[:]
            # pt = tuple(map(int, pt))
            # pt = min(soFar,key=lambda item:euclidean((item[0],item[1]), FirstPt))
            # nearest = min(soFar, key=lambda x: distance(x, FirstPt))
            # nearest = min(soFar, key=lambda c: (c[0]- FirstPt[0])**2 + (c[1]-FirstPt[1])**2)
  
            # nearest = min(temp, key=lambda x: ((abs(x[0]-FirstPt[0]),abs(x[1]-FirstPt[1]))))
            try:
                nearest = min(generateDistanceArray, key=lambda x: x[0])
                # print('Nearest',nearest)
                pt = tuple(map(int, nearest[1]))
                FirstPt = tuple(map(int, FirstPt))
                # print('xxxxxxxxxxxx',(pt,FirstPt),temp)
                # print(soFar)
                img = cv2.line(img=img, pt1=(FirstPt), pt2=(pt), color=(0, 0, 255), thickness = 2)
                FirstPt = pts[j]
            except:
                continue
         
    return img
def houghCirclesConnection(img,pts):
    # pts = [k.pt for k in keypoints]#Opencv can't draw an arrow between a single point center and a list of points. So we'll have to go over it in a for loop as such
    # max(pts,key=lambda item:item[1])
    # print('points',pts)
    #max(lis,key=lambda item:item[1])
    # nearest = min(cooList, key=lambda x: distance(x, coordinate))
    # centre = (246, 234) # This should be changed to the center of your image
    for i in range(len(pts)):
        for j in range(len(pts)):
            if(j+1 <= i):
                FirstPt = tuple(map(int, pts[j]))
                pt = tuple(map(int, pts[j+1]))
                # print('xxxxxxxxxxxx',(pt,FirstPt))
                # print(soFar)
                img = cv2.line(img=img, pt1=(FirstPt), pt2=(pt), color=(0, 255, 255), thickness = 2)
    return img
def houghCircleDetectionVideoSorted():
   inputsource = 'output.avi'
   cap = cv2.VideoCapture(inputsource)
   hasFrame, frame = cap.read()
   vid_writer = cv2.VideoWriter('outputHough2.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1],frame.shape[0]))
   count = 0

   while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        count = count + 2 # For Skipping Frames
        if not hasFrame:
            cv2.waitKey()
            break
        countBlobs = 0
        # Read image as gray-scale
        # Convert to gray-scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur the image to reduce noise
        img_blur = cv2.medianBlur(gray, 5)
        # Apply hough transform on the image
        # Default circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/32, param1=200, param2=17, minRadius=20, maxRadius=20)
        circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/32, param1=200, param2=17, minRadius=4, maxRadius=15)
        # Draw detected circles
        # print(circles)
        # circles = sorted(circles, key=lambda tup: tup)
        # print(circles[0])  
        blobPositions = []
        blobCoord = []
        if circles is not None:
            blobID = 1
            circles = np.uint16(np.around(circles))
            for c in circles[0,:]:
                blobCoord.append((c[0] ,c[1]))
                blobPositions.append((c[0] ,c[1] , c[2]))
                countBlobs = countBlobs + 1
            blobCoord = sorted(blobCoord, key=lambda tup: (tup[0],tup[1]))
            blobPositions = sorted(blobPositions, key=lambda tup: (tup[0],tup[1]))
            for i in range(len(circles[0, :])):
                # Draw outer circle
                blobPositions[i] = blobPositions[i]+(blobID,)
                frame = cv2.circle(frame, (blobPositions[i][0], blobPositions[i][1]), blobPositions[i][2], (0, 255, 0), 2)
                # Draw inner circle
                frame = cv2.circle(frame, (blobPositions[i][0], blobPositions[i][1]), 2, (0, 0, 255), 3)
                frame = cv2.putText(frame, str(blobPositions[i][3]) ,(int(blobPositions[i][0]),int(blobPositions[i][1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                blobID = blobID + 1
            frame = houghCirclesConnection(frame,blobCoord)
        print('Blobs Detected ',countBlobs)
        # print(blobPositions)
        cv2.imshow("new Keypoints", frame)
        vid_writer.write(frame)

def angle3(a,b,c):
    a = np.array([a[0],a[1] ,1])
    b = np.array([b[0], b[1],1])
    c = np.array([c[0], c[1],1])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)*180.0/ math.pi
    return angle


def getAngleWithRespectToCenter(data,cP):
    x = cP[0]/2
    y = cP[1]/2
    p = (x,y)
    for i in range(len(data)):
        # print(len(data))
        p2 = (data[i][0], data[i][1])
        x2 = data[i][0]
        print(x2)
        y2 = data[i][1]
        ydiff = y2-y if y2>y else y-y2
        # print(y1,y2,ydiff)
        # print(ydiff)
        xdiff = x2-x if x2>x else x-x2
        # print(x1,x2,xdiff)
        slope = ydiff/xdiff if xdiff > 0 else 0
        # slope = abs(y2-y1)/abs(x2-x1)
        # slope = abs(p1[1]-p2[1])/abs(p1[0]-p2[0])
        angle = 180.0 * np.arctan(slope) / np.pi
        print('all ',p,'    ',p2,'   ',slope,'   ',angle)
        data[i] = (data[i]) + (angle,)

    return data
def getAngleTwoPoints(data):
    x = data[0][0]
    y = data[0][1]
    p = (x,y)
    for i in range(len(data)):
        # print(len(data))
        if(i+1 < len(data)):
            p1 = (data[i][0], data[i][1])
            p2 = (data[i+1][0], data[i+1][1])
            x1 = data[i][0]
            y1 = data[i][1]
            x2 = data[i+1][0]
            y2 = data[i+1][1]
            ydiff = y2-y1 if y2>y1 else y1-y2
            # print(y1,y2,ydiff)
            # print(ydiff)
            xdiff = x2-x1 if x2>x1 else x1-x2
            # print(x1,x2,xdiff)
            slope = ydiff/xdiff if xdiff > 0 else 0
            # slope = abs(y2-y1)/abs(x2-x1)
            # slope = abs(p1[1]-p2[1])/abs(p1[0]-p2[0])
            angle = 180.0 * np.arctan(slope) / np.pi
            # print('all ',p1,'    ',p2,'   ',slope,'   ',angle)
            data[i] = (data[i]) + (angle,)
        else:
            p1 = (data[i][0], data[i][1])
            x1 = data[i][0]
            y1 = data[i][1]
            # slope = abs(y1-y)/abs(x1-x)
            ydiff = y-y1 if y>y1 else y1-y
            # print(ydiff)
            xdiff = x-x1 if x>x1 else x1-x
            # print(xdiff)
            slope = ydiff/xdiff if xdiff > 0 else 0
            # slope = abs(p1[1]-p[1])/abs(p1[0]-p[0])
            angle = 180.0 * np.arctan(slope) / np.pi
            # print('last'  ,p1,'    ',p,'   ',slope,'   ',angle)
            data[i] = (data[i]) + (angle,)

    return data
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang
def angleCalculationVector(data):
    # vector1 = [1,0,0]
    # vector2 = [0,1,0]

    # unit_vector1 = vector1 / np.linalg.norm(vector1)
    # unit_vector2 = vector2 / np.linalg.norm(vector2)

    # dot_product = np.dot(unit_vector1, unit_vector2)

    # angle = np.arccos(dot_product) #angle in radian
    firstPoint = np.array([data[0][0],data[0][1],1])
    for i in range(len(data)):
        print(len(data))
        if(i+1 < len(data)):
            vec1 = np.array([data[i][0], data[i][1], 1])
            vec2 = np.array([data[i+1][0], data[i+1][1], 1])
            r = vg.angle(vec1, vec2)
            data[i] = data[i] + (r,)
        else:
            vec1 = np.array([data[i][0], data[i][1], 1])
            r = vg.angle(vec1, firstPoint)
            data[i] = data[i] + (r,)

    return data
def distance(co1, co2):
    co1 = tuple(map(int, co1))
    # print('wwwwwwwwwwwwwwwwwwwwwwwwwww',co1 , co2)
    print('sssssssssssss',sqrt(pow(abs(co1[0] - co2[0]), 2) + pow(abs(co1[1] - co2[1]), 2)))
    return sqrt(pow(abs(co1[0] - co2[0]), 2) + pow(abs(co1[1] - co2[1]), 2))
def sortTuples(data):
    sort = sorted(data, key=lambda tup: (tup[0],tup[1]))
    return sort
def findingMinmumTask():
    cords =  [(455, 12), (188, 90), (74, 366), (10,10)]
    point = (18, 448)
    x = []
    for c in cords:
        dst = euclidean(c, point)
        x.append((dst,c))
    # closest_dst = min(euclidean(c, point) for c in cords)
    closest = min(x,key=lambda item:item[0])
    print(x)
    print(closest)
def getDistances(point,data):
    distanceArray = []
    for i in range(len(data)):
        distanceArray.append((euclidean((data[i][0],data[i][1]), point),(data[i][0],data[i][1])))
        print(distanceArray)
    return distanceArray
img = cv2.imread("./images/cyclingP.png",1)
# img = cv2.resize(img,(656,368))
    # SimpleBlobDetection(img)
# blobConnection(img)
# houghCirclesDetection(img)
# houghCircleDetectionVideoSorted()
# SimpleBlobDetection(img)
SimpleBlobWithCamera()
cv2.waitKey(0)
