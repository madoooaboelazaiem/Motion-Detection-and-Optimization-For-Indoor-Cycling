import cv2
import vg
import numpy as np
import matplotlib.pyplot as plt 
def SimpleBlobWithCamera():
   inputsource = 'output.avi'
   cap = cv2.VideoCapture(inputsource)
   hasFrame, frame = cap.read()
   vid_writer = cv2.VideoWriter('output2.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1],frame.shape[0]))
   count = 0

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
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypointCoordinates[i] + (keypoint_id,))
        detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(str(keypointCoordinates[i][2]))+' angle: '+(str(keypointCoordinates[i][3]))) #Converting the x and y positions to strings
        keypoint_id += 1

    nblobs = len(keypoints)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(nblobs):
        print(keypoints_with_id[i])
        cv2.putText(im_with_keypoints, str(keypoints_with_id[i][4]) ,(int(keypoints_with_id[i][0]),int(keypoints_with_id[i][1])), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    print(nblobs, 'From Blobs')
    # print(detected_keypoints_toString)
    # cv2.imwrite('Output.jpg', frameClone)
    # cimg = cv2.imread('Output.jpg',0)
    # cv2.imshow('gray',cimg)
    # Show keypoints
    # im_with_keypoints = blobConnection(im_with_keypoints)
    cv2.imshow("new Keypoints", im_with_keypoints)
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
        detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(str(keypointCoordinates[i][2]))+' angle: '+(str(keypointCoordinates[i][3]))) #Converting the x and y positions to strings
        keypoint_id += 1

    nblobs = len(keypoints)
    # print(nblobs)
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = (0,0)
    y = (0,0)
    z = (0,0)
    a , b = (0,0)
    for i in range(nblobs):
        # print(keypoints_with_id[i])
        cv2.putText(im_with_keypoints, str(keypoints_with_id[i][4]) ,(int(keypoints_with_id[i][0]),int(keypoints_with_id[i][1])), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        if(keypoints_with_id[i][4] == 2):
            x = (keypoints_with_id[i][0],keypoints_with_id[i][1])
        if(keypoints_with_id[i][4] == 5):
            y = (keypoints_with_id[i][0],keypoints_with_id[i][1])   
        if(keypoints_with_id[i][4] == 4):
            z = (keypoints_with_id[i][0],keypoints_with_id[i][1])   
        
    a = (y[0]-x[0],y[1]-x[1])
    b = (y[0]-z[0],y[1]-z[1])    
    angleCalculation(a,b)
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
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/32, param1=200, param2=17, minRadius=20, maxRadius=20)
    # Draw detected circles
    if circles is not None:
        blobPositions = []
        blobID = 1
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            blobPositions.append((i[0] ,i[1] , i[2],blobID))
            blobID = blobID + 1
            countBlobs = countBlobs + 1
            # Draw outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw inner circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    print('Blobs Detected ',countBlobs)
    print(blobPositions)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('OutputHoughCirlces.jpg', img)


img = cv2.imread("./images/cyclingP.png",1)
# img = cv2.resize(img,(656,368))
    # SimpleBlobDetection(img)
# SimpleBlobWithCamera()
houghCirclesDetection(img)
cv2.waitKey(0)
