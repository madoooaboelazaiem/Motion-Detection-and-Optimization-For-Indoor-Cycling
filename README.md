# Bachelor Thesis Project
## The thesis discusses a way to enhance the cyclist's performance by optimizing his posture by correcting the lower limb joint angles of the cyclist during his training.
## Tools Used:
- OpenCV (cv2)
- Numpy and Pandas (np,pd)
- Matplotlib (plt)
- Joblib
- Machine Learning Imports (pip,xgboost(1.1),keras)

### Steps to be able to run the program:
1. Install all the required tools and libraries
2. (If you want live detection ** Skip ** to step 3) ... Include the video of the training session of the cyclist in the ** input source parameter ** .. Use the Video Trials folder in order to make a similar video with similar sized blobs.
3. Inorder to have live detection replace the input source with an integer where the integer is the established connection port with your camera (Know which drive(usb port) is being used in your laptop either 0 , 1 , 2 , 3) where 0 most of the time stands for the built in laptop camera. 
4. Add the csv file of the input parameters used by ML model (Power,Rpm,Bpm,PedalBlob) in the CSV_Trials folder and replace it with the one used in the parameter ** machineLearningInputSource **
5. Run the program by (python (name of the file.py))
6. autopep8 -i nameoffile.py (to remove unwanted indentation in python)
7. The output will be saved as output.mp4 after the session ends
### Note 
** In angle angleCalculationV3 replace the p2 which is the center of axis of the bike with the exact coordinates of the new bike ... the center of axis is the green blob found in the image (blobsDetected) **