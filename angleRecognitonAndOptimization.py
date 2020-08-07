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
import joblib


def blobDetectionAndOptimization(inputSource,mlInputSource,modelUsed):
    cap = cv2.VideoCapture(inputSource)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print('fps', fps)
    hasFrame, frame = cap.read()
    vid_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (800, 600))
    count = 0
    rowNames = ['PedalBlob', 'HeelBlob', 'AnkleBlob',
        'KneeBlob', 'KneeBlob2', 'HipBlob']
    model2 = joblib.load(modelUsed)
    df = pd.DataFrame(columns=['PedalBlob', 'HeelBlob',
                      'AnkleBlob', 'KneeBlob', 'KneeBlob2', 'HipBlob'])
    dfInput = pd.read_csv(mlInputSource)  # , sep=';'
    dfInput = dfInput.apply(np.ceil)
    dfInput = dfInput.astype(int, errors='ignore')
    pedalArray = []
    dfInput = dfInput.drop(
        columns=['PedalBlob', 'HeelBlob', 'AnkleBlob', 'KneeBlob', 'KneeBlob2', 'HipBlob'])
    Powercolumn = dfInput["Power"]
    Powermax_value = Powercolumn.max()
    RPMcolumn = dfInput["Rpm"]
    RPMmax_value = RPMcolumn.max()
    BPMcolumn = dfInput["Bpm"]
    BPMmax_value = BPMcolumn.max()
    dfInput["Power"] = dfInput["Power"]/366
    dfInput["Rpm"] = dfInput["Rpm"]/106
    dfInput["Bpm"] = dfInput["Bpm"]/173
    inputLayer = dfInput.values
    rowIncrementer = 0
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        frame = cv2.resize(frame, (800, 600))
        if not hasFrame:
            cv2.waitKey()
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
        ret, x = cv2.threshold(blur, 155, 255, cv2.THRESH_BINARY)
        cv2.imshow('sh',x)
        edgeDetectedImage = np.invert(x)
        cv2.imshow('Edge Detected Image', edgeDetectedImage)
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 1
        params.maxThreshold = 256
        params.filterByColor = True
        params.blobColor = 0  # for black

        # Filter by Area.
        params.filterByArea = True
        # params.minArea = 150
        params.minArea = 350
        params.maxArea = 1500

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.3
        # params.minCircularity = 0.2

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.1
        # params.minConvexity = 0.2

        # Filter by Inertia
        params.filterByInertia = True
        # params.minInertiaRatio = 0.25
        params.minInertiaRatio = 0.18

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
                keypointCoordinates, key=lambda k: (k[1], k[0]), reverse=True)
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypointCoordinates[i] + (keypoint_id,))
            detected_keypoints_toString.append((('X: ')+str(keypointCoordinates[i][0]))+('  Y: ')+(str(keypointCoordinates[i][1]))+' size: '+(
                str(keypointCoordinates[i][2])))  # Converting the x and y positions to strings
            keypoint_id += 1
        keypoints_with_id = flipCoord(keypoints_with_id)
        nblobs = len(keypoints)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array(
            []), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(nblobs):
            cv2.putText(im_with_keypoints, str(keypoints_with_id[i][3]), (int(keypoints_with_id[i][0]), int(
                keypoints_with_id[i][1])), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        if count % 30 == 0:
            angle = angleCalculationV3(keypoints_with_id)
            currentPedalAngle = angle[0][4][0]/360
            pedalArray.append(currentPedalAngle)
            currentPower = inputLayer[rowIncrementer][0]
            currentRPM = inputLayer[rowIncrementer][1]
            currentBPM = inputLayer[rowIncrementer][2]
            currentInputRow = np.array(
                [[currentPower, currentRPM, currentBPM, currentPedalAngle]])
            # print(currentInputRow)
            # print(currentInputRow.shape)
            prediction = model2.predict(currentInputRow)
            prediction = np.ceil(prediction*360)
            prediction = prediction.astype(int)
            rowIncrementer = rowIncrementer + 1
        pos = 90
        if len(angle) > 4:
            angleHeel = angle[1][4][0]
            angleAnkle = angle[2][4][0]
            angleKnee = angle[3][4][0]
            angleKnee2 = angle[3][4][1]
            angleHip = angle[4][4][0]
            cv2.putText(im_with_keypoints, 'Angle of point '+str(angle[1][3])+'(Heel) is ' + str(
                    (angleHeel)), (20, pos), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of (Heel) should be '+ str(
                    (prediction[0][0])), (20, pos), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of point '+str(angle[2][3])+'(Ankle) is ' + str(
                    (angleAnkle)), (20, pos), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of (Ankle) should be '+ str(
                    (prediction[0][1])), (20, pos), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of point '+str(angle[3][3])+'(Knee) is ' + str(
                    (angleKnee))+' Knee2 '+str(angleKnee2) , (20, pos), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of (Knee) should be '+ str(
                    (prediction[0][2]))+' Knee2 '+str(180 - prediction[0][2]), (20, pos), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of point '+str(angle[4][3])+'(Hip) is ' + str(
                    (angleHip)) , (20, pos), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of (Hip) should be '+ str(
                    (prediction[0][3])), (20, pos), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        elif len(angle) > 3:
            angleHeel = angle[1][4][0]
            angleAnkle = angle[2][4][0]
            angleKnee = angle[3][4][0]
            angleKnee2 = angle[3][4][1]
            cv2.putText(im_with_keypoints, 'Angle of point '+str(angle[1][3])+'(Heel) is ' + str(
                    (angleHeel)), (20, pos), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of (Heel) should be '+ str(
                    (prediction[0][0])), (20, pos), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of point '+str(angle[2][3])+'(Ankle) is ' + str(
                    (angleAnkle)), (20, pos), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of (Ankle) should be '+ str(
                    (prediction[0][1])), (20, pos), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of point '+str(angle[3][3])+'(Knee) is ' + str(
                    (angleKnee))+' Knee2 '+str(angleKnee2) , (20, pos), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of (Knee) should be '+ str(
                    (prediction[0][2]))+' Knee2 '+str(180 - prediction[0][2]), (20, pos), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        elif len(angle) > 2:
            angleHeel = angle[1][4][0]
            angleAnkle = angle[2][4][0]
            cv2.putText(im_with_keypoints, 'Angle of point '+str(angle[1][3])+'(Heel) is ' + str(
                    (angleHeel)), (20, pos), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of (Heel) should be '+ str(
                    (prediction[0][0])), (20, pos), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of point '+str(angle[2][3])+'(Ankle) is ' + str(
                    (angleAnkle)), (20, pos), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of (Ankle) should be '+ str(
                    (prediction[0][1])), (20, pos), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)         
              
        elif len(angle) > 1:
            angleHeel = angle[1][4][0]
            cv2.putText(im_with_keypoints, 'Angle of point '+str(angle[1][3])+'(Heel) is ' + str(
                    (angleHeel)), (20, pos), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            pos = pos + 30
            cv2.putText(im_with_keypoints, 'Angle of (Heel) should be ' + str(
                    (prediction[0][0])), (20, pos), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        print(nblobs, 'From Blobs')
        count+=1
        
        cv2.imshow("new Keypoints", im_with_keypoints)
        vid_writer.write(im_with_keypoints)

def anglesOptimizer(data):
    # print(data , 'dataaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    # Covering both normal cycling and aero cycling techniques   
    # +ve means Extend -ve means flex
    # Average Knee 65 total flexation per cycle 
    minKnee = 70 
    maxKnee = 148
    maxKnee2 = 110
    minKnee2 = 32
    # Average 20 to 40 total flex per cycle 
    minHip = 18
    maxHip = 50
    
    # Average Knee 15 total flexation per cycle
    minAnkle = 75 ## Used as heel in our implementation
    maxAnkle = 105

    optimizingAngle = [(False)]*len(data)  
    ankleAngle = 0
    kneeAngle = 0
    hipAngle = 0
    optimizedKnee2 = 0
    extend = 'Extended'
    flex = 'Flexed'
    if len(data) > 1:
        ankleAngle = data[1][4][0]
        if ankleAngle < minAnkle :
            optimizedAnkle = minAnkle - ankleAngle # Extend
            optimizingAngle[1] = (True,extend,optimizedAnkle)
        elif ankleAngle > maxAnkle:
            optimizedAnkle = ankleAngle - maxAnkle # Flex
            optimizingAngle[1] = (True,flex,optimizedAnkle)
        else:
            optimizingAngle[1] = (False)

    if len(data) > 3:
        kneeAngle = data[3][4][0]
        kneeAngle2 = data[3][4][1]
        if kneeAngle < minKnee :
            optimizedKnee = minKnee - kneeAngle # Extend
            if kneeAngle2 != 0:
                optimizedKnee2 = maxKnee2 - kneeAngle2
            optimizingAngle[3] = (True,extend,optimizedKnee,optimizedKnee2)
        elif kneeAngle > maxKnee:
            optimizedKnee = kneeAngle - maxKnee # Flex
            if kneeAngle2 != 0:
                optimizedKnee2 = kneeAngle2 - minKnee2
            optimizingAngle[3] = (True,flex,optimizedKnee,optimizedKnee2)
        else:
            optimizingAngle[3] = (False)
    if len(data) > 4:
        hipAngle = data[4][4][0]
        if hipAngle < minHip :
            optimizedHip = minHip - hipAngle # Extend
            optimizingAngle[4] = (True,extend,optimizedHip)
        elif hipAngle > maxHip:
            optimizedHip = hipAngle - maxHip # Flex
            optimizingAngle[4] = (True,flex,optimizedHip)
        else:
            optimizingAngle[4] = (False)
    return optimizingAngle
def angle3(a, b, c):
    a = np.array([a[0], a[1], 1])
    b = np.array([b[0], b[1], 1])
    c = np.array([c[0], c[1], 1])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)*180.0 / math.pi
    return angle
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
    angle = float(math.atan2(p21y, p21x))
    angle = angle * 180 / math.pi
    return angle
def flipCoord(data):
    if len(data) > 3 :
        first = data[0]
        second = data[1]
        third = data[2]
        result = sorted([first,second,third], key=lambda x: x[0],reverse = True)
        out1 = list(result[0])
        out1[3] = 1
        out2 = list(result[1])
        out2[3] = 2
        out3 = list(result[2])
        out3[3] = 3
        res1 = tuple(out1)
        data[0] = res1
        res2 = tuple(out2)
        data[1] = res2
        res3 = tuple(out3)
        data[2] = res3
    return data
        

    newData = []
    p1 = (data[0][0], data[0][1])
    p2 = (408, 390)
    angle = angleCalculation2(p1,p2)
    newData.append((data[0][0], data[0][1], data[0][2], (angle,0)))
    # threading.Timer(1.0, angleCalculation).start()
    for i in range(len(data)):
        newCord = []
        angle2 = 0
        if(i == len(data)-1): # Last Point with horizontal calc
            p1 = (data[i-1][0], data[i-1][1])
            p2 = (data[i][0], data[i][1])
            p3 = (490, 21) #Horizontal Coordinate with the hip
            angle2 = angle3(p1, p2, p3)
            newData.append((data[i][0], data[i][1], data[i][2], (angle2,0)))
        if(i+2 < len(data)): 
            print(i+2, len(data))
            p1 = (data[i][0], data[i][1])
            p2 = (data[i+1][0], data[i+1][1])
            p3 = (data[i+2][0], data[i+2][1])
            angle = angle3(p1, p2, p3)
            if(i+1 == 2):
                p1 = (data[i-1][0], data[i][1])
                p2 = (data[i+1][0], data[i+1][1])
                p3 = (data[i+2][0], data[i+2][1])
                angle2 = angle3(p1, p2, p3)
            if(i+1 == len(data)-2): # Second way of calculating knee angle
                angle2 = 180 - angle
            newData.append((data[i+1][0], data[i+1][1], data[i+1][2], (angle,angle2)))
    return newData
def angleCalculationV3(data):
    newData = []
    p1 = (data[0][0], data[0][1])
    p2 = (372,448) # The center of axis of the bike
    angle = angleCalculation2(p1,p2)
    newData.append((data[0][0], data[0][1], data[0][2],data[0][3], (math.ceil(angle),0)))
    for i in range(len(data)):
        newCord = []
        angle2 = 0
        if(i == len(data)-1): # Last Point with horizontal calc
            p1 = (data[i-1][0], data[i-1][1])
            p2 = (data[i][0], data[i][1])
            p3 = (data[i][0]+300,  data[i][1]) #Horizontal Coordinate with the hip
            angle2 = angle3(p1, p2, p3)
            newData.append((data[i][0], data[i][1], data[i][2],data[i][3], (math.ceil(angle2),0)))
        if(i+2 < len(data)): 
            if(i+1 != 1):
                # print(i+2, len(data))
                p1 = (data[i][0], data[i][1])
                p2 = (data[i+1][0], data[i+1][1])
                p3 = (data[i+2][0], data[i+2][1])
                angle = angle3(p1, p2, p3)
                if(i+1 == 3): # The fourth blob ankle right way to calculate the angle
                    p1 = (data[i-1][0], data[i-1][1])
                    p2 = (data[i+1][0], data[i+1][1])
                    p3 = (data[i+2][0], data[i+2][1])
                    angle = angle3(p1, p2, p3)
                if(i+1 == len(data)-2): # Second way of calculating knee angle
                    angle2 = 180 - angle
                newData.append((data[i+1][0], data[i+1][1], data[i+1][2],data[i+1][3], (math.ceil(angle),math.ceil(angle2))))
    return newData
def addRow(df,row_dict):
    df = df.append(row_dict, ignore_index=True)
    df.to_csv('data.csv') 
    return df




# Main Method
inputSource = './Video_Trials/SecondTrial.mp4'
xgboostModel = './Models/modelXGB.pkl'
machineLearningInputSource = './CSV_Trials/SecondTrialOutputMerged.csv'
blobDetectionAndOptimization(inputSource,machineLearningInputSource,xgboostModel)
cv2.waitKey(0)
