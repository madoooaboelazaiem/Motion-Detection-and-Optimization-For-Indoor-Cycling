import cv2
import numpy as np
import matplotlib.pyplot as plt 
def SimpleBlobWithCamera():
   inputsource = 'output.avi'
   cap = cv2.VideoCapture(inputsource)
   hasFrame, frame = cap.read()
   vid_writer = cv2.VideoWriter('output2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))
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
        detected_keypoints_toString.append((str(keypointCoordinates[i][0]))+(',')+(str(keypointCoordinates[i][1]))) #Converting the x and y positions to strings
        keypoint_id += 1

    nblobs = len(keypoints)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print(nblobs, 'From Blobs')
    print(keypoints_with_id)
    # cv2.imwrite('Output.jpg', frameClone)
    # cimg = cv2.imread('Output.jpg',0)
    # cv2.imshow('gray',cimg)
    # Show keypoints
    cv2.imshow("new Keypoints", im_with_keypoints)
    vid_writer.write(im_with_keypoints)

def SimpleBlobDetection(frameClone):
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
    keypoints = detector.detect(frameClone)
    nblobs = len(keypoints)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(frameClone, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print(nblobs, 'From Blobs')
    # cv2.imwrite('Output.jpg', frameClone)
    # cimg = cv2.imread('Output.jpg',0)
    # cv2.imshow('gray',cimg)
    # Show keypoints
    cv2.imshow("new Keypoints", im_with_keypoints)
    cv2.imwrite('Output.jpg', im_with_keypoints)

img = cv2.imread("person2.png",1)
# SimpleBlobDetection(img)
SimpleBlobWithCamera()
cv2.waitKey(0)
