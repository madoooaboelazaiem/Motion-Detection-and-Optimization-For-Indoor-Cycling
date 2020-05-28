import cv2
import vg
import math
import numpy as np
import matplotlib.pyplot as plt 
def SimpleBlobWithCamera(inputSource):
   cap = cv2.VideoCapture(inputSource)
   hasFrame, frame = cap.read()
   vid_writer = cv2.VideoWriter('output2.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1],frame.shape[0]))
   count = 0

   while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    count = count + 4 # For Skipping Frames
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
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/32, param1=200, param2=17, minRadius=20, maxRadius=20)
    # Draw detected circles
    if circles is not None:
        blobPositions = []
        blobID = 1
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            blobPositions.append((i[0] ,i[1] , i[2]))
            countBlobs = countBlobs + 1
            # Draw outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw inner circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        blobPositions = sorted(blobPositions , key=lambda k: k[1],reverse = True)
        nBlobs = 1
        newCoord = []
        for i in range(len(blobPositions)):
            newCoord.append((blobPositions[i][0],blobPositions[i][1],nBlobs))
            nBlobs = nBlobs + 1
        blobPositions = newCoord
        print('mado',blobPositions)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(countBlobs):
            # print(blobPositions[i])
            cv2.putText(img, str(blobPositions[i][2]) ,(int(blobPositions[i][0]),int(blobPositions[i][1])), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
    angle = angleCalculation(blobPositions)
    pos = 30
    for i in range(len(angle)):
        cv2.putText(img, 'Angle of point '+str(angle[i][2])+' is ' +str(round((angle[i][3]),2)) ,(100,pos), font, 1, (255, 255, 255), 3, cv2.LINE_AA)
        pos= pos +30

    print('Blobs Detected ',countBlobs)
    print(angle)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('OutputHoughCirlces.jpg', img)


def houghCircleDetectionVideoSorted(inputsource):
   cap = cv2.VideoCapture(inputsource)
   hasFrame, frame = cap.read()
   vid_writer = cv2.VideoWriter('outputHough2.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1],frame.shape[0]))
   count = 0

   while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        count = count + 4 # For Skipping Frames
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
        circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=20, minRadius=6, maxRadius=5)
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
                blobPositions.append((i[0] ,i[1] , i[2]))
                countBlobs = countBlobs + 1
                # Draw outer circle
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # print('colored')
                # Draw inner circle
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            blobPositions = sorted(blobPositions , key=lambda k: k[1],reverse = True)
            nBlobs = 1
            newCoord = []
            for i in range(len(blobPositions)):
                newCoord.append((blobPositions[i][0],blobPositions[i][1],nBlobs))
                nBlobs = nBlobs + 1
            blobPositions = newCoord
            print('mado',blobPositions)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(countBlobs):
                # print(blobPositions[i])
                cv2.putText(frame, str(blobPositions[i][2]) ,(int(blobPositions[i][0]),int(blobPositions[i][1])), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
            angle = angleCalculation(blobPositions)
            pos = 30
            for i in range(len(angle)):
                cv2.putText(frame, 'Angle of point '+str(angle[i][2])+' is ' +str(round((angle[i][3]),2)) ,(100,pos), font, 1, (255, 255, 255), 3, cv2.LINE_AA)
                pos= pos +30
            #blobCoord = sorted(blobCoord, key=lambda tup: (tup[0],tup[1]))
            #frame = houghCirclesConnection(frame,blobCoord)
            print('Blobs Detected ',countBlobs)
            # print(blobPositions)
        cv2.imshow("new Keypoints", frame)
        vid_writer.write(frame)

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

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))
def angle3(a,b,c):
    a = np.array([a[0],a[1] ,1])
    b = np.array([b[0], b[1],1])
    c = np.array([c[0], c[1],1])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)*180.0/ math.pi
    return angle
def length(v):
  return math.sqrt(dotproduct(v, v))
def angleCalculation2(p1,p2):
    # vector1 = [1,0,0]
    # vector2 = [0,1,0]

    # unit_vector1 = vector1 / np.linalg.norm(vector1)
    # unit_vector2 = vector2 / np.linalg.norm(vector2)

    # dot_product = np.dot(unit_vector1, unit_vector2)

    # angle = np.arccos(dot_product) #angle in radian
    vec1 = np.array([p1[0], p1[1], 1])
    vec2 = np.array([p2[0], p2[1], 1])

    r = vg.angle(vec1, vec2)
    return r
def angleCalculation(data):
    newData = []
    for i in range(len(data)):
        newCord = []
        if(i+2 < len(data)):
            print(i+2 , len(data))
            p1 = (data[i][0],data[i][1])
            p2 = (data[i+1][0],data[i+1][1])
            p3 = (data[i+2][0],data[i+2][1])
            p21x = p1[0] - p2[0]  if p1[0] > p2[0] else p2[0] - p1[0]
            p21y = p1[1]-p2[1] if p1[1] > p2[1] else p2[1] - p1[1]
            p21  = (p21x,p21y)
            p23x = p3[0] - p2[0] if p3[0] > p2[0] else p2[0] - p3[0]
            p23y = p3[1]-p2[1] if p3[1] > p2[1] else p2[1] - p3[1]
            p23 = (p23x,p23y)
            # angle = angleCalculation2(p21,p23)
            angle = angle3(p1,p2,p3)
            newData.append((data[i+1][0],data[i+1][1],data[i+1][2],angle))
            # newData.append((data[i][0],data[i][1]))
            # data[i+1] = newCord
            print('angleeee',angle)
    return newData

img = cv2.imread("./images/cyclingP.png",1)
inputSource = 'Video_2_EDIT.mp4'
# img = cv2.resize(img,(656,368))
    # SimpleBlobDetection(img)
# SimpleBlobDetection(img)
# houghCirclesDetection(img)
# SimpleBlobWithCamera(inputSource)
houghCircleDetectionVideoSorted(inputSource)
cv2.waitKey(0)
