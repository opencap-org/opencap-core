import sys
sys.path.append("./mmpose") # utilities in child directory
import cv2 
import numpy as np 
import pandas as pd
import os 
import glob 
import pickle
import json
import subprocess
import urllib.request
import shutil
import utilsDataman
import requests
import ffmpeg
import logging
import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt, butter, find_peaks
from scipy.interpolate import pchip_interpolate
from scipy.spatial.transform import Rotation 
from itertools import combinations
import copy
from utilsCameraPy3 import Camera, nview_linear_triangulations
from utils import getOpenPoseMarkerNames, getOpenPoseFaceMarkers
from utils import numpy2TRC, rewriteVideos, delete_multiple_element,loadCameraParameters
from utils import makeRequestWithRetry
from utilsAPI import getAPIURL

from utilsAuth import getToken

API_TOKEN = getToken()
API_URL = getAPIURL()

# %%
def download_file(url, file_name):
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


# %% 
def getVideoLength(filename):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filename],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return float(result.stdout)

# %%
def video2Images(videoPath, nImages=12, tSingleImage=None, filePrefix='output', skipIfRun=True, outputFolder='default'):
    # Pops images out of a video.
    # If tSingleImage is defined (time, not frame number), only one image will be popped
    if outputFolder == 'default':
        outputFolder = os.path.dirname(videoPath)
    
    # already written out?
    if not os.path.exists(os.path.join(outputFolder, filePrefix + '_0.jpg')) or not skipIfRun: 
        if tSingleImage is not None: # pop single image at time value
            CMD = ('ffmpeg -loglevel error -skip_frame nokey -y -ss ' + str(tSingleImage) + ' -i ' + videoPath + 
                   " -qmin 1 -q:v 1 -frames:v 1 -vf select='-eq(pict_type\,I)' " + 
                   os.path.join(outputFolder,filePrefix + '0.png'))
            os.system(CMD)
            outImagePath = os.path.join(outputFolder,filePrefix + '0.png')
           
        else: # pop multiple images from video
            lengthVideo = getVideoLength(videoPath)
            timeImageSamples = np.linspace(1,lengthVideo-1,nImages) # disregard first and last second
            for iFrame,t_image in enumerate(timeImageSamples):
                CMD = ('ffmpeg -loglevel error -skip_frame nokey -ss ' + str(t_image) + ' -i ' + videoPath + 
                       " -qmin 1 -q:v 1 -frames:v 1 -vf select='-eq(pict_type\,I)' " + 
                       os.path.join(outputFolder,filePrefix) + '_' + str(iFrame) + '.jpg')
                os.system(CMD)
                outImagePath = os.path.join(outputFolder,filePrefix) + '0.jpg'
                
    return outImagePath
        

# %%                
def calcIntrinsics(folderName, CheckerBoardParams=None, filenames=['*.jpg'], 
                   imageScaleFactor=1, visualize=False, saveFileName=None):
    if CheckerBoardParams is None:
        # number of black to black corners and side length (cm)
        CheckerBoardParams = {'dimensions': (6,9), 'squareSize': 2.71}
    
    if '*' in filenames[0]:
        imageFiles = glob.glob(folderName + '/' + filenames[0])
        
    else:
        imageFiles = [] ;
        for fName in filenames:
            imageFiles.append(folderName + '/' + fName)    
           
    # stop the iteration when specified 
    # accuracy, epsilon, is reached or 
    # specified number of iterations are completed. 
    criteria = (cv2.TERM_CRITERIA_EPS + 
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)       
      
    # Vector for 3D points 
    threedpoints = [] 
      
    # Vector for 2D points 
    twodpoints = []       
      
    #  3D points real world coordinates 
    # objectp3d = generate3Dgrid(CheckerBoardParams) 
            
    # Load images in for calibration
    for iImage, pathName in enumerate(imageFiles):
        image = cv2.imread(pathName) 
        if imageScaleFactor != 1:
            dim = (int(imageScaleFactor*image.shape[1]),int(imageScaleFactor*image.shape[0]))
            image = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
        imageSize = np.reshape(np.asarray(np.shape(image)[0:2]).astype(np.float64),(2,1)) # This all to be able to copy camera param dictionary
        
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        print(pathName + ' used for intrinsics calibration.')
      
        # Find the chess board corners 
        # If desired number of corners are 
        # found in the image then ret = true 
        # ret, corners = cv2.findChessboardCorners( 
        #                 grayColor, CheckerBoardParams['dimensions'],  
        #                 cv2.CALIB_CB_ADAPTIVE_THRESH
        #                 + cv2.CALIB_CB_FAST_CHECK + 
        #                 cv2.CALIB_CB_NORMALIZE_IMAGE) 
        
        ret,corners,meta = cv2.findChessboardCornersSBWithMeta(	grayColor, CheckerBoardParams['dimensions'],
                                                        cv2.CALIB_CB_EXHAUSTIVE + 
                                                        cv2.CALIB_CB_ACCURACY + 
                                                        cv2.CALIB_CB_LARGER)
      
        # If desired number of corners can be detected then, 
        # refine the pixel coordinates and display 
        # them on the images of checker board 
        if ret == True: 
            # 3D points real world coordinates 
            checkerCopy = copy.copy(CheckerBoardParams)
            checkerCopy['dimensions'] = meta.shape[::-1] # reverses order so width is first
            objectp3d = generate3Dgrid(checkerCopy)
            
            threedpoints.append(objectp3d) 
      
            # Refining pixel coordinates 
            # for given 2d points. 
            # corners2 = cv2.cornerSubPix( 
            #     grayColor, corners, (11, 11), (-1, -1), criteria) 
            
            corners2 = corners/imageScaleFactor # Don't need subpixel refinement with findChessboardCornersSBWithMeta
            twodpoints.append(corners2) 
            
            # Draw and display the corners 
            image = cv2.drawChessboardCorners(image,  
                                                meta.shape[::-1],  
                                                corners2, ret) 
                
            #findAspectRatio
            ar = imageSize[1]/imageSize[0]            
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL) 
            cv2.resize(image,(int(600*ar),600))
            
            # Save intrinsic images
            imageSaveDir = os.path.join(folderName,'IntrinsicCheckerboards')
            if not os.path.exists(imageSaveDir):
                os.mkdir(imageSaveDir)
            cv2.imwrite(os.path.join(imageSaveDir,'intrinsicCheckerboard' + str(iImage) + '.jpg'), image)
                
            if visualize:
                print('Press enter or close image to continue')
                cv2.imshow('img', image) 
                cv2.waitKey(0)  
                cv2.destroyAllWindows() 
                
        if ret == False:
            print("Couldn't find checkerboard in " + pathName)
  
    if len(twodpoints) < .5*len(imageFiles):
       print('Checkerboard not detected in at least half of intrinsic images. Re-record video.')
       return None
       
     
    # Perform camera calibration by 
    # passing the value of above found out 3D points (threedpoints) 
    # and its corresponding pixel coordinates of the 
    # detected corners (twodpoints) 
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera( 
        threedpoints, twodpoints, grayColor.shape[::-1], None, None)     
    
    CamParams = {'distortion':distortion,'intrinsicMat':matrix,'imageSize':imageSize}
    
    if saveFileName is not None:
        saveCameraParameters(saveFileName,CamParams)
  
    return CamParams

# %%
def computeAverageIntrinsics(session_path,trialIDs,CheckerBoardParams,nImages=25):
    CamParamList = []
    camModels = []
    
    for trial_id in trialIDs:
        resp = makeRequestWithRetry('GET',
                                    API_URL + "trials/{}/".format(trial_id),
                                    headers = {"Authorization": "Token {}".format(API_TOKEN)})
        trial = resp.json()
        camModels.append(trial['videos'][0]['parameters']['model'])
        trial_name = trial['name']
        if trial_name == 'null':
            trial_name = trial_id
        
        # Make directory (folder for trialname, intrinsics also saved there)
        video_dir = os.path.join(session_path,trial_name)
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir,trial_name + ".mov")
        
        # Download video if not done
        if not os.path.exists(video_path):
            download_file(trial["videos"][0]["video"], video_path)
            
        if not os.path.exists(os.path.join(video_dir,'cameraIntrinsics.pickle')):
            
            # Compute intrinsics from images popped out of intrinsic video.
            video2Images(video_path,filePrefix=trial_name,nImages=nImages)
            CamParams = calcIntrinsics(os.path.join(session_path,trial_name), CheckerBoardParams=CheckerBoardParams,
                                       filenames=['*.jpg'], 
                                       saveFileName=os.path.join(video_dir,'cameraIntrinsics.pickle'),
                                       visualize = False)
            if CamParams is None:
                saveCameraParameters(os.path.join(video_dir,'cameraIntrinsics.pickle'),CamParams)
            
        else:
            CamParams = loadCameraParameters(os.path.join(video_dir,'cameraIntrinsics.pickle'))
        
        if CamParams is not None:
            CamParamList.append(CamParams)
    
    # Compute results
    intComp = {}
    intComp['fx'] = [c['intrinsicMat'][0,0] for c in CamParamList]
    intComp['fy'] = [c['intrinsicMat'][1,1] for c in CamParamList]
    intComp['cx'] = [c['intrinsicMat'][0,2] for c in CamParamList]
    intComp['cy'] = [c['intrinsicMat'][1,2] for c in CamParamList]
    intComp['d'] = np.asarray([c['distortion'] for c in CamParamList])
   
    intCompNames = list(intComp.keys())
    for v in intCompNames:
        intComp[v + '_u'] = np.mean(intComp[v],axis=0)
        intComp[v + '_std'] = np.std(intComp[v],axis=0)
        intComp[v + '_stdPerc'] = np.divide(intComp[v + '_std'],intComp[v + '_u']) * 100
        
    phoneModel = camModels[0]
    if any([camModel != phoneModel for camModel in camModels]):
        raise Exception('You are averaging intrinsics across different phone models.')
    
    # output averaged parameters
    CamParamsAverage = {}
    params = list(CamParamList[0].keys())
    for param in params:
        CamParamsAverage[param] = np.mean(np.asarray([c[param] for c in CamParamList]),axis=0)

    return CamParamsAverage, CamParamList, intComp, phoneModel


# %%
def generate3Dgrid(CheckerBoardParams):
    #  3D points real world coordinates. Assuming z=0
    objectp3d = np.zeros((1, CheckerBoardParams['dimensions'][0]  
                          * CheckerBoardParams['dimensions'][1],  
                          3), np.float32) 
    objectp3d[0, :, :2] = np.mgrid[0:CheckerBoardParams['dimensions'][0], 
                                    0:CheckerBoardParams['dimensions'][1]].T.reshape(-1, 2) 
    
    objectp3d = objectp3d * CheckerBoardParams['squareSize'] 
    
    return objectp3d

# %%
def saveCameraParameters(filename,CameraParams):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename),exist_ok=True)
    
    open_file = open(filename, "wb")
    pickle.dump(CameraParams, open_file)
    open_file.close()
    
    return True

#%% 
def getVideoRotation(videoPath):
    
    
    meta = ffmpeg.probe(videoPath)
    try:
        rotation = meta['format']['tags']['com.apple.quicktime.video-orientation']
    except:
        # For AVI (after we rewrite video), no rotation paramter, so just using h and w. 
        # For now this is ok, we don't need leaning right/left for this, just need to know
        # how to orient the pose estimation resolution parameters.
        try: 
            if meta['format']['format_name'] == 'avi':
                if meta['streams'][0]['height']>meta['streams'][0]['width']:
                    rotation = 90
                else:
                    rotation = 0
            else:
                raise Exception('no rotation info')
        except:
            rotation = 90 # upright is 90, and intrinsics were captured in that orientation
        
    return int(rotation)

#%% 
def rotateIntrinsics(CamParams,videoPath):
    rotation = getVideoRotation(videoPath)
    
    # upright is 90, which is how intrinsics recorded so no rotation needed
    if rotation !=90:
        originalCamParams = copy.deepcopy(CamParams)
        fx = originalCamParams['intrinsicMat'][0,0]
        fy = originalCamParams['intrinsicMat'][1,1]
        px = originalCamParams['intrinsicMat'][0,2]
        py = originalCamParams['intrinsicMat'][1,2]
        lx = originalCamParams['imageSize'][1]
        ly = originalCamParams['imageSize'][0]
           
        if rotation == 0: # leaning left
            # Flip image size
            CamParams['imageSize'] = np.flipud(CamParams['imageSize'])
            # Flip focal lengths
            CamParams['intrinsicMat'][0,0] = fy
            CamParams['intrinsicMat'][1,1] = fx
            # Change principle point - from upper left
            CamParams['intrinsicMat'][0,2] = py
            CamParams['intrinsicMat'][1,2] = lx-px   
        elif rotation == 180: # leaning right
            # Flip image size
            CamParams['imageSize'] = np.flipud(CamParams['imageSize'])
            # Flip focal lengths
            CamParams['intrinsicMat'][0,0] = fy
            CamParams['intrinsicMat'][1,1] = fx
            # Change principle point - from upper left
            CamParams['intrinsicMat'][0,2] = ly-py
            CamParams['intrinsicMat'][1,2] = px
        elif rotation == 270: # upside down
            # Change principle point - from upper left
            CamParams['intrinsicMat'][0,2] = lx-px
            CamParams['intrinsicMat'][1,2] = ly-py
            
    return CamParams

# %% 
def calcExtrinsics(imageFileName, CameraParams, CheckerBoardParams,
                   imageScaleFactor=1,visualize=False,
                   imageUpsampleFactor=1,useSecondExtrinsicsSolution=False):
    # Camera parameters is a dictionary with intrinsics
    
    # stop the iteration when specified 
    # accuracy, epsilon, is reached or 
    # specified number of iterations are completed. 
    criteria = (cv2.TERM_CRITERIA_EPS + 
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
      
    # Vector for 3D points 
    threedpoints = [] 
      
    # Vector for 2D points 
    twodpoints = [] 
    
    #  3D points real world coordinates. Assuming z=0
    objectp3d = generate3Dgrid(CheckerBoardParams)
    
    # Load and resize image - remember calibration image res needs to be same as all processing
    image = cv2.imread(imageFileName)
    if imageScaleFactor != 1:
        dim = (int(imageScaleFactor*image.shape[1]),int(imageScaleFactor*image.shape[0]))
        image = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
        
    if imageUpsampleFactor != 1:
        dim = (int(imageUpsampleFactor*image.shape[1]),int(imageUpsampleFactor*image.shape[0]))
        imageUpsampled = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    else:
        imageUpsampled = image

    
    # Find the chess board corners 
    # If desired number of corners are 
    # found in the image then ret = true 
    
    #TODO need to add a timeout to the findChessboardCorners function
    grayColor = cv2.cvtColor(imageUpsampled, cv2.COLOR_BGR2GRAY)
    
    ## Contrast TESTING - openCV does thresholding already, but this may be a bit helpful for bumping contrast
    # grayColor = grayColor.astype('float64')
    # cv2.imshow('Grayscale', grayColor.astype('uint8'))
    # savePath = os.path.join(os.path.dirname(imageFileName),'extrinsicGray.jpg')
    # cv2.imwrite(savePath,grayColor)
    
    # grayContrast = np.power(grayColor,2)
    # grayContrast = grayContrast/(np.max(grayContrast)/255)    
    # # plt.figure()
    # # plt.imshow(grayContrast, cmap='gray')
    
    # # cv2.imshow('ContrastEnhanced', grayContrast.astype('uint8'))
    # savePath = os.path.join(os.path.dirname(imageFileName),'extrinsicGrayContrastEnhanced.jpg')
    # cv2.imwrite(savePath,grayContrast)
      
    # grayContrast = grayContrast.astype('uint8')
    # grayColor = grayColor.astype('uint8')

    ## End contrast Testing
    
    ## Testing settings - slow and don't help 
    # ret, corners = cv2.findChessboardCorners( 
    #                 grayContrast, CheckerBoardParams['dimensions'],  
    #                 cv2.CALIB_CB_ADAPTIVE_THRESH  
    #                 + cv2.CALIB_CB_FAST_CHECK + 
    #                 cv2.CALIB_CB_NORMALIZE_IMAGE) 
    
    # Note I tried findChessboardCornersSB here, but it didn't find chessboard as reliably
    ret, corners = cv2.findChessboardCorners( 
                grayColor, CheckerBoardParams['dimensions'],  
                cv2.CALIB_CB_ADAPTIVE_THRESH) 

    # Fallback: if the standard detector fails, try the SB variant (more robust to
    # certain lighting/contrast conditions where adaptive thresholding struggles).
    # Using ACCURACY|LARGER without EXHAUSTIVE to keep the fallback reasonably fast.
    corners2_from_sb = False
    if not ret:
        ret_sb, corners_sb, _ = cv2.findChessboardCornersSBWithMeta(
            grayColor, CheckerBoardParams['dimensions'],
            cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_LARGER)
        if ret_sb:
            ret = True
            corners = corners_sb
            corners2_from_sb = True

    # If desired number of corners can be detected then, 
    # refine the pixel coordinates and display 
    # them on the images of checker board 
    if ret == True: 
        # 3D points real world coordinates       
        threedpoints.append(objectp3d) 
  
        # Refining pixel coordinates 
        # for given 2d points. 
        # SBWithMeta corners are already subpixel-accurate; skip cornerSubPix in that case.
        if corners2_from_sb:
            corners2 = corners / imageUpsampleFactor
        else:
            corners2 = cv2.cornerSubPix( 
                grayColor, corners, (11, 11), (-1, -1), criteria) / imageUpsampleFactor
  
        twodpoints.append(corners2) 
  
        # For testing: Draw and display the corners 
        # image = cv2.drawChessboardCorners(image,  
        #                                  CheckerBoardParams['dimensions'],  
        #                                   corners2, ret) 
        # Draw small dots instead
        # Choose dot size based on size of squares in pixels
        circleSize = 1
        squareSize = np.linalg.norm((corners2[1,0,:] - corners2[0,0,:]).squeeze())
        if squareSize >12:
            circleSize = 2

        for iPoint in range(corners2.shape[0]):
            thisPt = corners2[iPoint,:,:].squeeze()
            cv2.circle(image, tuple(thisPt.astype(int)), circleSize, (255,255,0), 2) 
        
        #cv2.imshow('img', image) 
        #cv2.waitKey(0) 
  
        #cv2.destroyAllWindows()
    if ret == False:
        print('No checkerboard detected. Will skip cam in triangulation.')
        return None
        
        
    # Find position and rotation of camera in board frame.
    # ret, rvec, tvec = cv2.solvePnP(objectp3d, corners2,
    #                                CameraParams['intrinsicMat'], 
    #                                CameraParams['distortion'])
    
    # This function gives two possible solutions.
    # It helps with the ambiguous cases with small checkerboards (appears like
    # left handed coord system). Unfortunately, there isn't a clear way to 
    # choose the correct solution. It is the nature of the solvePnP problem 
    # with a bit of 2D point noise.
    rets, rvecs, tvecs, reprojError = cv2.solvePnPGeneric(
        objectp3d, corners2, CameraParams['intrinsicMat'], 
        CameraParams['distortion'], flags=cv2.SOLVEPNP_IPPE)
    rvec = rvecs[1]
    tvec = tvecs[1]
   
    if rets < 1 or np.max(rvec) == 0 or np.max(tvec) == 0:
        print('solvePnPGeneric failed. Use SolvePnPRansac')
        # Note: can input extrinsics guess if we generally know where they are.
        # Add to lists to look like solvePnPRansac results
        rvecs = []
        tvecs = []
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectp3d, corners2, CameraParams['intrinsicMat'],
            CameraParams['distortion'])
        if ret is True:
            rets = 1
            rvecs.append(rvec)
            tvecs.append(tvec)
        else:
            print('Extrinsic calculation failed. Will skip cam in triangulation.')
            return None
    
    # Select which extrinsics solution to use
    extrinsicsSolutionToUse = 0
    if useSecondExtrinsicsSolution:
        extrinsicsSolutionToUse = 1
        
    topLevelExtrinsicImageFolder = os.path.abspath(
        os.path.join(os.path.dirname(imageFileName),
                     '../../../../CalibrationImages'))
    if not os.path.exists(topLevelExtrinsicImageFolder):
        os.makedirs(topLevelExtrinsicImageFolder,exist_ok=True)
        
    for iRet,rvec,tvec in zip(range(rets),rvecs,tvecs):
        theseCameraParams = copy.deepcopy(CameraParams)
        # Show reprojections
        img_points, _ = cv2.projectPoints(objectp3d, rvec, tvec, 
                                          CameraParams['intrinsicMat'],  
                                          CameraParams['distortion'])
    
        # Plot reprojected points
        # for c in img_points.squeeze():
        #     cv2.circle(image, tuple(c.astype(int)), 2, (0, 255, 0), 2)
        
        # Show object coordinate system
        imageCopy = copy.deepcopy(image)
        imageWithFrame = cv2.drawFrameAxes(
            imageCopy, CameraParams['intrinsicMat'], 
            CameraParams['distortion'], rvec, tvec, 200, 4)
        
        # Create zoomed version.
        ht = image.shape[0]
        wd = image.shape[1]
        bufferVal = 0.05 * np.mean([ht,wd])
        topEdge = int(np.max([np.squeeze(np.min(img_points,axis=0))[1]-bufferVal,0]))
        leftEdge = int(np.max([np.squeeze(np.min(img_points,axis=0))[0]-bufferVal,0]))
        bottomEdge = int(np.min([np.squeeze(np.max(img_points,axis=0))[1]+bufferVal,ht]))
        rightEdge = int(np.min([np.squeeze(np.max(img_points,axis=0))[0]+bufferVal,wd]))
        
        # imageCopy2 = copy.deepcopy(imageWithFrame)
        imageCropped = imageCopy[topEdge:bottomEdge,leftEdge:rightEdge,:]
                
        
        # Save extrinsics picture with axis
        imageSize = np.shape(image)[0:2]
        #findAspectRatio
        ar = imageSize[1]/imageSize[0]
        # cv2.namedWindow("axis", cv2.WINDOW_NORMAL) 
        cv2.resize(imageWithFrame,(600,int(np.round(600*ar))))
     
        # save crop image to local camera file
        savePath2 = os.path.join(os.path.dirname(imageFileName), 
                                'extrinsicCalib_soln{}.jpg'.format(iRet))
        cv2.imwrite(savePath2,imageCropped)
          
        if visualize:   
            print('Close image window to continue')
            cv2.imshow('axis', image)
            cv2.waitKey()
            
            cv2.destroyAllWindows()
        
        R_worldFromCamera = cv2.Rodrigues(rvec)[0]
        
        theseCameraParams['rotation'] = R_worldFromCamera
        theseCameraParams['translation'] = tvec
        theseCameraParams['rotation_EulerAngles'] = rvec
        
        # save extrinsics parameters to video folder
        # will save the selected parameters in Camera folder in main
        saveExtPath = os.path.join(
            os.path.dirname(imageFileName),
            'cameraIntrinsicsExtrinsics_soln{}.pickle'.format(iRet))
        saveCameraParameters(saveExtPath,theseCameraParams)
        
        # save images to top level folder and return correct extrinsics
        camName = os.path.split(os.path.abspath(
                  os.path.join(os.path.dirname(imageFileName), '../../')))[1] 
            
        if iRet == extrinsicsSolutionToUse:
            fullCamName = camName 
            CameraParamsToUse = copy.deepcopy(theseCameraParams)
        else:
            fullCamName = 'altSoln_{}'.format(camName)
        savePath = os.path.join(topLevelExtrinsicImageFolder, 'extrinsicCalib_' 
                                + fullCamName + '.jpg')
        cv2.imwrite(savePath,imageCropped)   
            
    return CameraParamsToUse

# %% 
def calcExtrinsicsFromVideo(videoPath, CamParams, CheckerBoardParams,
                            visualize=False, imageUpsampleFactor=2,
                            useSecondExtrinsicsSolution=False):    
    # Get video parameters.
    vidLength = getVideoLength(videoPath)
    videoDir, videoName = os.path.split(videoPath)    
    # Pick end of video as only sample point. For some reason, won't output
    # video with t close to vidLength, so we count down til it does.
    tSampPts = [np.round(vidLength-0.3, decimals=1)]    
    upsampleIters = 0
    for iTime,t in enumerate(tSampPts):
        # Pop an image.
        imagePath = os.path.join(videoDir, 'extrinsicImage0.png')
        if os.path.exists(imagePath):
            os.remove(imagePath)
        while not os.path.exists(imagePath) and t>=0:
            video2Images(videoPath, nImages=1, tSingleImage=t, filePrefix='extrinsicImage', skipIfRun=False)
            t -= 0.2
        # Default to beginning if can't find a keyframe.
        if not os.path.exists(imagePath):
            video2Images(videoPath, nImages=1, tSingleImage=0.01, filePrefix='extrinsicImage', skipIfRun=False)
        # Throw error if it can't find a keyframe.
        if not os.path.exists(imagePath):
            exception = 'No calibration image could be extracted for at least one camera. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about camera calibration and https://www.opencap.ai/troubleshooting for potential causes for a failed calibration.'
            raise Exception(exception, exception)
        # Try to find the checkerboard; return None if you can't find it.           
        CamParamsTemp = calcExtrinsics(
            os.path.join(videoDir, 'extrinsicImage0.png'),
            CamParams, CheckerBoardParams, visualize=visualize, 
            imageUpsampleFactor=imageUpsampleFactor,
            useSecondExtrinsicsSolution=useSecondExtrinsicsSolution)
        while iTime == 0 and CamParamsTemp is None and upsampleIters < 3:
            if imageUpsampleFactor > 1: 
                imageUpsampleFactor = 1
            elif imageUpsampleFactor == 1:
                imageUpsampleFactor = .5
            elif imageUpsampleFactor < 1:
                imageUpsampleFactor = 1
            CamParamsTemp = calcExtrinsics(
                os.path.join(videoDir, 'extrinsicImage0.png'),
                CamParams, CheckerBoardParams, visualize=visualize, 
                imageUpsampleFactor=imageUpsampleFactor,
                useSecondExtrinsicsSolution=useSecondExtrinsicsSolution)
            upsampleIters += 1
        if CamParamsTemp is not None:
            # If checkerboard was found, exit.
            CamParams = CamParamsTemp.copy()
            return CamParams

    # If made it through but didn't return camera params, throw an error.
    exception = 'The checkerboard was not detected by at least one camera. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about camera calibration and https://www.opencap.ai/troubleshooting for potential causes for a failed calibration.'
    raise Exception(exception, exception)
    
    return None

# %%
def isCheckerboardUpsideDown(CameraParams):
    # With backwall orientation, R[1,1] will always be positive in correct orientation
    # and negative if upside down
    for cam in list(CameraParams.keys()):
        if CameraParams[cam] is not None:
            upsideDown = CameraParams[cam]['rotation'][1,1] < 0
            break
        #Default if no camera params (which is a garbage case anyway)
        upsideDown = False

    return upsideDown

# %% 
def autoSelectExtrinsicSolution(sessionDir,keypoints2D,confidence,extrinsicsOptions):
    keypoints2D = copy.copy(keypoints2D)
    confidence = copy.copy(confidence)
       
    camNames = list(extrinsicsOptions.keys()) 
    
    optimalCalibrationDict = {}
    
    # Order the cameras based on biggest difference between solutions. Want to start
    # with these
    if len(camNames)>2:
        camNames = orderCamerasForAutoCalDetection(extrinsicsOptions)
    
    # Find first pair of cameras
    optimalCalibrationDict[camNames[0]], optimalCalibrationDict[camNames[1]] = computeOptimalCalibrationCombination(
        keypoints2D,confidence,extrinsicsOptions,[camNames[0],camNames[1]])
    
    # Continue for third and additional cameras
    additionalCameras = []
    if len(camNames)>2:
        additionalCameras = camNames[2:]
    
    for camName in additionalCameras:
        _, optimalCalibrationDict[camName] = computeOptimalCalibrationCombination(
        keypoints2D,confidence,extrinsicsOptions,[camNames[0],camName],
        firstCamSoln=optimalCalibrationDict[camNames[0]])
    
    # save calibrationJson to Videos
    calibrationOptionsFile = os.path.join(sessionDir,'Videos','calibOptionSelections.json')
    with open(calibrationOptionsFile, 'w') as f:
        json.dump(optimalCalibrationDict, f)
    f.close()
    
    # Make camera params dict
    CamParamDict = {}
    for camName in camNames:
        CamParamDict[camName] = extrinsicsOptions[camName][optimalCalibrationDict[camName]]
        
    # Switch any cameras in local file system. 
    for cam,val in optimalCalibrationDict.items():
        if val == 1:
            saveFileName = os.path.join(sessionDir,'Videos',cam,'cameraIntrinsicsExtrinsics.pickle')
            saveCameraParameters(saveFileName,extrinsicsOptions[cam][1])
    
    return CamParamDict
    
# %%
def computeOptimalCalibrationCombination(keypoints2D,confidence,extrinsicsOptions,
                                         CamNames,firstCamSoln=None):
    if firstCamSoln is None:
        firstCamOptions = range(len(extrinsicsOptions[CamNames[0]]))
    else:
        firstCamOptions = [firstCamSoln]
        
    # Remove face markers - they are intermittent.
    _, idxFaceMarkers = getOpenPoseFaceMarkers()
    
    #find most confident frame
    confidenceMat = np.minimum(confidence[CamNames[0]],confidence[CamNames[1]])
    confidenceMat = np.delete(confidenceMat, idxFaceMarkers,axis=0)
    iFrame = np.argmax(confidenceMat.mean(axis=1))
    
    # put keypoints in list and delete face markers
    keypointList = [np.expand_dims(np.delete(keypoints2D[CamNames[0]],idxFaceMarkers,axis=0)[:,iFrame],axis=1),
                    np.expand_dims(np.delete(keypoints2D[CamNames[1]],idxFaceMarkers,axis=0)[:,iFrame],axis=1)]
    
    meanReprojectionErrors = []
    combinations = []  
    
    for iCam0 in firstCamOptions:
        for iCam1 in range(len(extrinsicsOptions[CamNames[1]])):
            combinations.append([iCam0,iCam1])
            
            CameraParamList = [extrinsicsOptions[CamNames[0]][iCam0],
                               extrinsicsOptions[CamNames[1]][iCam1]]
                                           
            # triangulate           
            points3D,_ = triangulateMultiview(CameraParamList,keypointList)
            
            # reproject
            # Make list of camera objects
            cameraObjList = []       
            for camParams in CameraParamList:
                c = Camera()
                c.set_K(camParams['intrinsicMat'])
                c.set_R(camParams['rotation'])
                c.set_t(np.reshape(camParams['translation'],(3,1)))
                cameraObjList.append(c)
               
            # Organize points for reprojectionError function
            stackedPoints = np.stack([k[:,None,0,:] for k in keypointList])
            pointsInput = []
            for i in range(stackedPoints.shape[1]):
                pointsInput.append(stackedPoints[:,i,0,:].T)
           
            # Calculate combined reprojection error
            reprojError = calcReprojectionError(cameraObjList,pointsInput,points3D,
                                                normalizeError=True)
            meanReprojectionErrors.append(np.mean(reprojError))
      
    # Select solution with minimum error
    idx = np.argmin(meanReprojectionErrors)
    
    if (sorted(meanReprojectionErrors)[1]-sorted(meanReprojectionErrors)[0])/sorted(meanReprojectionErrors)[0] < .5:
        # This only happens when the vector from checker board origin to camera is < a few degrees. If you offset the board
        # vertically by a few feet, the solution becomes very clear again (ratio of ~7). We could throw an error.
        print("Warning: not much separability between auto checker board selection options. Try moving checkerboard closer to cameras, and move it so it is not straight-on with any of the cameras.")
        # Pick default to first solution when they are super close like this.
        idx = 0
        
    return combinations[idx][0], combinations[idx][1]

# %%
def orderCamerasForAutoCalDetection(extrinsicsOptions):
    # dict of rotations between first and second solutions
    rotDifs = []
    testVec = np.array((1,0,0))
    for cals in extrinsicsOptions.values():
        rot = np.matmul(cals[0]['rotation'],cals[1]['rotation'])
        # This will be close to 0 if the rotation was small, 2 if it is 180
        rotDifs.append(1-np.dot(np.matmul(testVec,rot),testVec))
    
    sortedCams = [cam for _, cam in sorted(zip(rotDifs, extrinsicsOptions.keys()),reverse=True)]
    
    return sortedCams
    

# %% 
def getLargestBoundingBox(data, bbox, confThresh=0.6):
    # Select the person/timepoint with the greatest bounding box area, with
    # reasonable mean confidence (i.e., closest to the camera).

    # Parameters (may require some tuning).
    # Don't consider frame if > this many keypoints with 0s or low confidence.
    nGoodKeypoints = 10 
    # Threshold for low confidence keypoints. 
    confThreshRemoveRow = 0.4 
    # Don't consider frame if feet don't average at this confidence.
    footConfThresh = 0.5

    # Copy data
    c_data = np.copy(data)
    c_data[c_data==0] = np.nan
    conf = c_data[:,2::3]
    c_bbox = np.copy(bbox)
    
    # Detect rows where < nGoodKeypoints markers have non-zeros.
    rows_nonzeros = np.count_nonzero(c_data, axis=1)        
    rows_nonzeros_10m = np.argwhere(rows_nonzeros < nGoodKeypoints*3)
    
    # Detect rows where < nGoodKeypoints markers have high confidence.
    nHighConfKeypoints = np.count_nonzero((conf > confThreshRemoveRow), 
                                          axis=1) 
    rows_lowConf = np.argwhere(nHighConfKeypoints < nGoodKeypoints)
       
    # Detect rows where the feet have low confidence.
    markerNames = getOpenPoseMarkerNames()
    feetMarkers = ['RAnkle', 'RHeel', 'RBigToe', 'LAnkle', 'LHeel', 'LBigToe']
    idxFeet = [markerNames.index(i) for i in feetMarkers]
    confFeet = conf[:,idxFeet]
    confFeet[np.isnan(confFeet)] = 0
    rows_badFeet = np.argwhere(np.mean(confFeet, axis=1) < footConfThresh)
    
    # Set bounding box to 0 for bad rows.
    badRows = np.unique(np.concatenate((rows_nonzeros_10m, rows_lowConf,
                                        rows_badFeet)))
    
    # Only remove rows if it isn't removing all rows
    if len(badRows) < c_data.shape[0]: 
        c_bbox[badRows, :] = 0
    
    # Find bbox size.
    bbArea = np.multiply(c_bbox[:,2], c_bbox[:,3])
    
    # Find rows with high enough average confidence.
    confMask = np.zeros((conf.shape[0]), dtype=bool)
    nonNanRows = np.argwhere(np.any(~np.isnan(c_data), axis=1))
    confMask[nonNanRows] = np.nanmean(conf[nonNanRows,:],axis=2) > confThresh
    maskedArea = np.multiply(confMask, bbArea)
    maxArea = np.nanmax(maskedArea)
    try:
        idxMax = np.nanargmax(maskedArea)
    except:
        idxMax = np.nan
   
    return maxArea, idxMax

#%%
def keypointsToBoundingBox(data,confidenceThreshold=0.3):
    # input: nFrames x 75.
    # output: nFrames x 4 (xTopLeft, yTopLeft, width, height).
    
    c_data = np.copy(data)
    
    # Remove face markers - they are intermittent.
    _, idxFaceMarkers = getOpenPoseFaceMarkers()
    idxToRemove = np.hstack([np.arange(i*3,i*3+3) for i in idxFaceMarkers])
    c_data = np.delete(c_data, idxToRemove, axis=1)    
    
    # nan the data if below a threshold
    confData = c_data[:,2::3]<confidenceThreshold
    confMask = np.repeat(confData, 3, axis=1)
    c_data[confMask] = np.nan  
    nonNanRows = np.argwhere(np.any(~np.isnan(c_data), axis=1))
    
    bbox = np.zeros((c_data.shape[0], 4))
    bbox[nonNanRows,0] = np.nanmin(c_data[nonNanRows,0::3], axis=2)
    bbox[nonNanRows,1] = np.nanmin(c_data[nonNanRows,1::3], axis=2)
    bbox[nonNanRows,2] = (np.nanmax(c_data[nonNanRows,0::3], axis=2) - 
                          np.nanmin(c_data[nonNanRows,0::3], axis=2))
    bbox[nonNanRows,3] = (np.nanmax(c_data[nonNanRows,1::3], axis=2) - 
                          np.nanmin(c_data[nonNanRows,1::3], axis=2))
    
    # Go a bit above head (this is for image-based tracker).
    bbox[:,1] = np.maximum(0, bbox[:,1] - .05 * bbox[:,3])
    bbox[:,3] = bbox[:,3] * 1.05
    
    return bbox

#%%    
def findClosestBox(bbox,keyBoxes,imageSize,iPerson=None):
    # bbox: the bbox selected from the previous frame.
    # keyBoxes: bboxes detected in the current frame.
    # imageSize: size of the image
    # iPerson: index of the person to track..   
    
    # Parameters.
    # Proportion of mean image dimensions that corners must change to be
    # considered different person
    cornerChangeThreshold = 0.2 
    
    keyBoxCorners = []
    for keyBox in keyBoxes:
        keyBoxCorners.append(np.array([keyBox[0], keyBox[1], 
                                       keyBox[0] + keyBox[2],
                                       keyBox[1] + keyBox[3]]))
    bboxCorners = np.array(
        [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    
    boxErrors = [
        np.linalg.norm(keyBox - bboxCorners) for keyBox in keyBoxCorners]
    try:
        if iPerson is None:
            iPerson = np.nanargmin(boxErrors)
        bbox = keyBoxes[iPerson]
    except:
        return None, None, False
    
    # If large jump in bounding box, break.
    samePerson = True
    if (boxErrors[iPerson] > cornerChangeThreshold*np.mean(imageSize)):
        samePerson = False
        
    return iPerson,bbox,samePerson

#%%
def trackKeypointBox(videoPath,bbStart,allPeople,allBoxes,dataOut,frameStart = 0 ,
                     frameIncrement = 1, visualize = False, poseDetector='OpenPose',
                     badFramesBeforeStop = 0):
    
    # Extract camera name
    if videoPath.split('InputMedia')[0][-5:-2] == 'Cam': # <= 10 cams
        camName = videoPath.split('InputMedia')[0][-5:-1]
    else:
        camName = videoPath.split('InputMedia')[0][-6:-1]

    # Tracks closest keypoint bounding boxes until the box changes too much.
    bboxKey = bbStart # starting bounding box
    frameNum = frameStart

    # initiate video capture
    # Read video
    video = cv2.VideoCapture(videoPath.replace('.mov', '_rotated.avi'))
    nFrames = allBoxes[0].shape[0]
    
    # Read desiredFrames.
    video.set(1, frameNum)
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        raise Exception('Cannot read video file')
        
    imageSize = (frame.shape[0],frame.shape[1])
    justStarted = True
    count = 0   
    badFrames = []
    while frameNum > -1 and frameNum < nFrames:
        # Read a new frame
        
        if visualize:
            video.set(1, frameNum)
            ok, frame = video.read()
            if not ok:
                break
        
        # Find person closest to tracked bounding box, and fill their keypoint data
        keyBoxes = [box[frameNum] for box in allBoxes]        
        
        iPerson, bboxKey_new, samePerson = findClosestBox(bboxKey, keyBoxes, 
                                                imageSize)
        
        # We allow badFramesBeforeStop of samePerson = False to account for an
        # errant frame(s) in the pose detector. Once we reach badFramesBeforeStop,
        # we break and output to the last good frame.
        if len(badFrames) > 0 and samePerson:
            badFrames = []
            
        if not samePerson and not justStarted:
            if len(badFrames) >= badFramesBeforeStop:
                print('{}: not same person at {}'.format(camName, frameNum - frameIncrement*badFramesBeforeStop))
                # Replace the data from the badFrames with zeros
                if len(badFrames) > 1:
                    dataOut[badFrames,:] = np.zeros(len(badFrames),dataOut.shape[0])
                break     
            else:
                badFrames.append(frameNum)
                
       # Don't update the bboxKey for the badFrames
        if len(badFrames) == 0:
            bboxKey = bboxKey_new

        
        dataOut[frameNum,:] = allPeople[iPerson][frameNum,:]
        
        # Next frame 
        frameNum += frameIncrement
        justStarted = False
        
        if visualize: 
            p3 = (int(bboxKey[0]), int(bboxKey[1]))
            p4 = (int(bboxKey[0] + bboxKey[2]), int(bboxKey[1] + bboxKey[3]))
            cv2.rectangle(frame, p3, p4, (0,255,0), 2, 1)
            
            # Display result
            cv2.imshow("Tracking", frame)
            
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break
        
        count += 1
  
    return dataOut
 
#%%
def trackBoundingBox(videoPath,bbStart,allPeople,allBoxes,dataOut,frameStart = 0 ,frameIncrement = 1, visualize = False):
    # Uses image-based tracking to track person thru video
    # returns dataOut with single person nFrames x 75 
        
    # Initialize tracker. KCF is accurate and semi-fast
    tracker = cv2.TrackerKCF_create()
    
    # Initialize box
    bbox = bbStart
    bboxKey = bbStart
    frameNum = frameStart
    
    # Read video
    video = cv2.VideoCapture(videoPath.replace('.mov', '_rotated.avi'))
    nFrames = allBoxes[0].shape[0]

    # Read desiredFrames.
    video.set(1, frameNum)
    ok, frame = video.read()
    if not ok:
        raise Exception('Cannot read video file')
         
    # Initialize tracker with first frame and bounding box
    try:
        ok = tracker.init(frame, bbox.astype(int))
    except:
        ok = tracker.init(frame,tuple(bbox.astype(int)))  # bbox has to be tuple for legacy trackers, like MOSSE
        
    justStarted = True
    updateCounter = 0
    imageSize = (frame.shape[0],frame.shape[1])
    
    while frameNum > -1 and frameNum < nFrames:
        # Read a new frame
        
        video.set(1, frameNum)
        ok, frame = video.read()
        if not ok:
            break
                     
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        if visualize: 
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    
            # Draw bounding box
            if ok:
                # Tracking success - draw box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                
            else :
                # Tracking failure - assume subject left scene. this means no re-entry
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                print('tracking failed at ' + str(frameNum))
                break
                
            # Display tracker type on frame
            cv2.putText(frame, "KCF Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
      
        # Find person closest to tracked bounding box, and fill their keypoint data
        keyBoxes = [box[frameNum] for box in allBoxes]
        iPerson, bboxKey, samePerson = findClosestBox(bbox,keyBoxes,imageSize) # it may be better to find the one closest to bboxKey, but then it wouldn't be leveraging
                                                                              # image based tracker except when person leaves scene

        if not samePerson and not justStarted:
            print('not same person at ' + str(frameNum))
            break
        
        dataOut[frameNum,:] = allPeople[iPerson][frameNum,:]
        
        # Next frame 
        frameNum += frameIncrement
        updateCounter += 1
        justStarted = False

        # Update Bounding Box with keypoints to keep the size right, every 20 frames
        if updateCounter > 20:
            updateCounter = 0
            tracker = cv2.TrackerKCF_create()
            try:
                ok = tracker.init(frame, bboxKey.astype(int)) 
            except:
                ok = tracker.init(frame, tuple(bboxKey.astype(int))) # bbox has to be tuple for legacy trackers, like MOSSE
        
        if visualize: 

            p3 = (int(bboxKey[0]), int(bboxKey[1]))
            p4 = (int(bboxKey[0] + bboxKey[2]), int(bboxKey[1] + bboxKey[3]))
            cv2.rectangle(frame, p3, p4, (0,255,0), 2, 1)

            # Display result
            cv2.imshow("Tracking", frame)
            
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break
        
    return dataOut


# %%
def unpackKeypointList(keypointList):
    nFrames = keypointList[0].shape[1]
    unpackedKeypoints = []
    for iFrame in range(nFrames):
        tempList = []
        for keyArray in keypointList:
            tempList.append(keyArray[:,iFrame,None,:])
        unpackedKeypoints.append(tempList.copy())
        
    return unpackedKeypoints

# %% 
def filter3DPointsButterworth(points3D,filtFreq,sampleFreq,order=4):
    points3D_out = np.copy(points3D)
    wn = filtFreq/(sampleFreq/2)
    if wn>1:
        print('You tried to filter ' + str(sampleFreq) + ' signal with cutoff freq of ' + str(filtFreq) + ', which is above the Nyquist Frequency. Will filter at ' + str(sampleFreq/2) + 'instead.')
        wn=0.99
    elif wn==1:
        wn=0.99
        
    sos = butter(order/2,wn,btype='low',output='sos')
    
    for i in range(2):
        points3D_out = sosfiltfilt(sos,points3D_out,axis=0)             
        
    return points3D_out


# %% Triangulation
# If you set ignoreMissingMarkers to True, and pass the DISTORTED keypoints
# as keypoints2D, the triangulation will ignore data from cameras that
# returned (0,0) as marker coordinates.
def triangulateMultiview(CameraParamList, points2dUndistorted, 
                          imageScaleFactor=1, useRotationEuler=False,
                          ignoreMissingMarkers=False,selectCamerasMinReprojError = False,
                          ransac = False, keypoints2D=[],confidence=None):
    # create a list of cameras (says sequence in documentation) from CameraParamList
    cameraList = []    
    nCams = len(CameraParamList) 
    nMkrs = np.shape(points2dUndistorted[0])[0]
    
    for camParams in CameraParamList:
        # get rotation matrix
        if useRotationEuler:
            rotMat = cv2.Rodrigues(camParams['rotation_EulerAngles'])[0]
        else:
            rotMat = camParams['rotation']
        
        c = Camera()
        c.set_K(camParams['intrinsicMat'])
        c.set_R(rotMat)
        c.set_t(np.reshape(camParams['translation'],(3,1)))
        cameraList.append(c)
           
   
    # triangulate
    stackedPoints = np.stack(points2dUndistorted)
    pointsInput = []
    for i in range(stackedPoints.shape[1]):
        pointsInput.append(stackedPoints[:,i,0,:].T)
    
   
    points3d,confidence3d = nview_linear_triangulations(cameraList,pointsInput,weights=confidence)

    
    # Below are some outlier rejection methods
    
    # A slow, hacky way of rejecting outliers, like RANSAC. 
    # Select the combination of cameras that minimize mean reprojection error for all cameras
    if selectCamerasMinReprojError and nCams>2:
        
        # Function definitions
        def generateCameraCombos(nCams):
            comb = [] ;
            k=nCams-1 ;           
            comb.append(tuple(range(nCams)))
            while k>=2:
                comb = comb + list(combinations(np.arange(nCams),k))
                k=k-1
            return comb
        

        # generate a list of all possible camera combinations down to 2
        camCombos = generateCameraCombos(len(cameraList))
        
        # triangulate for all camera combiations
        points3DList = []
        nMkrs = confidence[0].shape[0]
        reprojError = np.empty((nMkrs,len(camCombos)))
                
        for iCombo,camCombo in enumerate(camCombos):
            camList = [cameraList[i] for i in camCombo]
            points2DList = [pts[:,camCombo] for pts in pointsInput]
            conf = [confidence[i] for i in camCombo]
            
            points3DList.append(nview_linear_triangulations(camList,points2DList,weights=conf))
        
            # compute per-marker, confidence-weighted reprojection errors for all camera combinations
            # reprojError[:,iCombo] = calcReprojectionError(camList,points2DList,points3DList[-1],weights=conf)
            reprojError[:,iCombo] = calcReprojectionError(cameraList,pointsInput,points3DList[-1],weights=confidence)

        # select the triangulated point from camera set that minimized reprojection error
        new3Dpoints = np.empty((3,nMkrs))
        for iMkr in range(nMkrs):
            new3Dpoints[:,iMkr] = points3DList[np.argmin(reprojError[iMkr,:])][:,iMkr]
        
        points3d = new3Dpoints
        
    #RANSAC for outlier rejection - on a per-marker basis, not a per-camera basis. Could be part of the problem
    #Not clear that this is helpful 4/23/21
    if ransac and nCams>2:
        nIter = np.round(np.log(.01)/np.log(1-np.power(.75,2))) # log(1 - prob of getting optimal set) / log(1-(n_inliers/n_points)^minPtsForModel)
        #TODO make this a function of image resolution
        errorUB = 20 # pixels, below this reprojection error, another camera gets added. 
        nGoodModel =3  # model must have this many cameras to be considered good
        
        #functions
        def triangulateLimitedCameras(cameraList,pointsInput,confidence,cameraNumList):
            camList = [cameraList[i] for i in cameraNumList]
            points2DList = [pts[:,cameraNumList] for pts in pointsInput]
            conf = [confidence[i] for i in cameraNumList]
            points3D = nview_linear_triangulations(camList,points2DList,weights=conf)
            return points3D
        
        def reprojErrorLimitedCameras(cameraList,pointsInput,points3D,confidence,cameraNumList):
            if type(cameraNumList) is not list: cameraNumList = [cameraNumList]
            camList = [cameraList[i] for i in cameraNumList]
            points2DList = [pts[:,cameraNumList] for pts in pointsInput]
            conf = [confidence[i] for i in cameraNumList]
            reprojError = calcReprojectionError(camList,points2DList,points3D,weights=conf)
            return reprojError

        
        #initialize
        bestReprojError = 1000 * np.ones(nMkrs) # initial, large value
        best3Dpoint = np.empty((points3d.shape))
        camComboList = [[] for _ in range(nMkrs)]
        
        for iIter in range(int(nIter)):
            np.random.seed(iIter)
            camCombo = np.arange(nCams)
            np.random.shuffle(camCombo) # Seed setting should give same combos every run
            maybeInliers = list(camCombo[:2])
            alsoInliers = [[] for _ in range(nMkrs)]
            
            
            # triangulate maybe inliers
            points3D = triangulateLimitedCameras(cameraList,pointsInput,confidence,maybeInliers)
            
            #error on next camera
            for iMkr in range(nMkrs):
                for j in range(nCams-2):
                    er = reprojErrorLimitedCameras(cameraList,pointsInput,points3D,confidence,camCombo[2+j])
                    # print(er[iMkr])
                    if er[iMkr] < errorUB:
                        alsoInliers[iMkr].append(camCombo[2+j])
                # see if error is bigger than previous, if not, use this combo to triangulate
                # just 1 marker
                if (len(maybeInliers) + len(alsoInliers[iMkr])) >= nGoodModel:
                    thisConf = [np.atleast_1d(c[iMkr]) for c in confidence]
                    point3D = triangulateLimitedCameras(cameraList,[pointsInput[iMkr]],thisConf,maybeInliers + alsoInliers[iMkr])
                    er3D = reprojErrorLimitedCameras(cameraList,[pointsInput[iMkr]],point3D,thisConf,maybeInliers + alsoInliers[iMkr])    
                    # if er3D<bestReprojError[iMkr]:
                    if (len(maybeInliers) + len(alsoInliers[iMkr])) > len(camComboList[iMkr]):
                        best3Dpoint[:,iMkr] = point3D.T
                        bestReprojError[iMkr] = er3D
                        camComboList[iMkr] = maybeInliers.copy() + alsoInliers[iMkr].copy()

        points3d = best3Dpoint
                                    
    
    if ignoreMissingMarkers and nCams>2:        
        # For markers that were not identified by certain cameras,
        # we re-compute their 3D positions but only using cameras that could
        # identify them (ie cameras that did not return (0,0) as coordinates).
        missingCams, missingMarkers = getMissingMarkersCameras(keypoints2D)     
    
        for missingMarker in np.unique(missingMarkers):
            idx_missingMarker = np.where(missingMarkers == missingMarker)[0]
            idx_missingCam = missingCams[idx_missingMarker]
            
            idx_viewedCams = list(range(0, len(cameraList)))
            for i in idx_missingCam:
                idx_viewedCams.remove(i)
                
            CamParamList_viewed = [cameraList[i] for i in idx_viewedCams]
            c_pointsInput = copy.deepcopy(pointsInput)
            for count, pointInput in enumerate(c_pointsInput):
                c_pointsInput[count] = pointInput[:,idx_viewedCams]
            
            points3d_missingMarker = nview_linear_triangulations(
                CamParamList_viewed, c_pointsInput,weights=confidence)
            
            # overwritte marker
            points3d[:, missingMarker] = points3d_missingMarker[:, missingMarker]
    
    return points3d, confidence3d


# %% Get 3D keypoints by triangulation.
# If you set ignoreMissingMarkers to True, and pass the DISTORTED keypoints
# as keypoints2D, the triangulation will ignore data from cameras that
# returned (0,0) as marker coordinates.
# TODO: imageScaleFactor isn't used for now.
def triangulateMultiviewVideo(CameraParamDict,keypointDict,imageScaleFactor=1,
                              ignoreMissingMarkers=False, keypoints2D=[],
                              cams2Use = ['all'],confidenceDict={},trimTrial=True,
                              spline3dZeros = False, splineMaxFrames=5, nansInOut=[],
                              CameraDirectories = None, trialName = None,
                              startEndFrames=None, trialID='',
                              outputMediaFolder=None):
    # cams2Use is a list of cameras that you want to use in triangulation. 
    # if first entry of list is ['all'], will use all
    # otherwise, ['Cam0','Cam2']
    CameraParamList = [CameraParamDict[i] for i in CameraParamDict]
    if cams2Use[0] == 'all' and not None in CameraParamList:
        keypointDict_selectedCams = keypointDict
        CameraParamDict_selectedCams = CameraParamDict
        confidenceDict_selectedCams = confidenceDict
    else:
        if cams2Use[0] == 'all': #must have been a none (uncalibrated camera)
            cams2Use = []
            for camName in CameraParamDict:
                if CameraParamDict[camName] is not None:
                    cams2Use.append(camName)
            
        keypointDict_selectedCams = {}
        CameraParamDict_selectedCams = {}
        if confidenceDict:
            confidenceDict_selectedCams = {}
        for camName in cams2Use:
            if CameraParamDict[camName] is not None:
                keypointDict_selectedCams[camName] = keypointDict[camName]
                CameraParamDict_selectedCams[camName] = CameraParamDict[camName]
                if confidenceDict:
                    confidenceDict_selectedCams[camName] = confidenceDict[camName]
    
    keypointList_selectedCams = [keypointDict_selectedCams[i] for i in keypointDict_selectedCams]
    confidenceList_selectedCams = [confidenceDict_selectedCams[i] for i in confidenceDict_selectedCams]
    CameraParamList_selectedCams = [CameraParamDict_selectedCams[i] for i in CameraParamDict_selectedCams]
    unpackedKeypoints = unpackKeypointList(keypointList_selectedCams)
    points3D = np.zeros((3,keypointList_selectedCams[0].shape[0],keypointList_selectedCams[0].shape[1]))
    confidence3D = np.zeros((1,keypointList_selectedCams[0].shape[0],keypointList_selectedCams[0].shape[1]))
    
    
    for iFrame,points2d in enumerate(unpackedKeypoints):
        # If confidence weighting
        if confidenceDict:
            thisConfidence = [c[:,iFrame] for c in confidenceList_selectedCams]
        else:
            thisConfidence = None
        
        points3D[:,:,iFrame], confidence3D[:,:,iFrame] = triangulateMultiview(CameraParamList_selectedCams, points2d, 
                          imageScaleFactor=1, useRotationEuler=False,
                          ignoreMissingMarkers=ignoreMissingMarkers, keypoints2D=keypoints2D,confidence=thisConfidence)
        
    if trimTrial:
        # Delete confidence and 3D keypoints if markers, except for face 
        # markers, have 0 confidence (they're garbage b/c <2 cameras saw them).
        markerNames = getOpenPoseMarkerNames()        
        allMkrInds = np.arange(len(markerNames))
        _, idxFaceMarkers = getOpenPoseFaceMarkers()        
        includedMkrs = np.delete(allMkrInds,idxFaceMarkers)
        
        nZeroConf = (len(includedMkrs) - 
                     np.count_nonzero(confidence3D[:,includedMkrs,:],axis=1))        
        if not True in (nZeroConf<1).flatten():
            points3D = np.zeros((3,25,10))
            confidence3D = np.zeros((1,25,10))
            startInd = 0
            endInd = confidence3D.shape[2]
            
        else:            
            startInd = np.argwhere(nZeroConf<1)[0,1]
            endInd = confidence3D.shape[2] - np.argwhere(np.flip(nZeroConf<1))[0,1]
            
            # If there were less than 3 cameras, then we also take into account the
            # inleading and exiting nans, which result in garbage interpolated
            # keypoints.        
            if nansInOut:
                nans_in = [nansInOut[cam][0] for cam in nansInOut]
                nans_out = [nansInOut[cam][1] for cam in nansInOut]
                # When >2 cameras, all list entries will be nan. If >2 cameras, 
                # but only 2 see person, will have 2 non-nan entries.
                if not any(np.isnan(nans_in)) or np.sum(~np.isnan(nans_in)) == 2:  
                    startInd = int(np.nanmax(
                        np.array([startInd, np.nanmax(np.asarray(nans_in))])))
                    endInd = int(np.nanmin(
                        np.array([endInd, np.nanmin(np.asarray(nans_out))])))
            
            # nPointsOriginal = copy.deepcopy(points3D.shape[2])
            points3D = points3D[:,:,startInd:endInd]
            confidence3D = confidence3D[:,:,startInd:endInd]   
    else:
        startInd = 0
        endInd = confidence3D.shape[2]
                
    # Rewrite videos based on sync time and trimmed trc.
    if CameraDirectories != None and trialName !=None:       
        print('Writing synchronized videos')
        outputVideoDir = os.path.abspath(os.path.join(
                        list(CameraDirectories.values())[0],'../../','VisualizerVideos',trialName))
        # Check if the directory already exists
        if os.path.exists(outputVideoDir):
            # If it exists, delete it and its contents
            shutil.rmtree(outputVideoDir)
        os.makedirs(outputVideoDir,exist_ok=True)
        for iCam,camName in enumerate(keypointDict):
                        
            nFramesToWrite = endInd-startInd
            
            if outputMediaFolder is None:
                outputMediaFolder = 'OutputMedia*'
                
            inputPaths = glob.glob(os.path.join(CameraDirectories[camName],outputMediaFolder,trialName,trialID + '*'))
            if len(inputPaths) > 0:
                inputPath = inputPaths[0]
            else:
                inputPaths = glob.glob(os.path.join(CameraDirectories[camName],'InputMedia*',trialName,trialID + '*'))
                inputPath = inputPaths[0]
            
            # get frame rate and assume all the same for sync'd videos
            if iCam==0: 
                thisVideo = cv2.VideoCapture(inputPath.replace('.mov', '_rotated.avi'))
                frameRate = np.round(thisVideo.get(cv2.CAP_PROP_FPS))
                thisVideo.release()
            
            # Only rewrite if camera in cams2use and wasn't kicked out earlier
            if (camName in cams2Use or cams2Use[0] == 'all') and startEndFrames[camName] != None:
                _, inputName = os.path.split(inputPath)
                inputRoot,inputExt = os.path.splitext(inputName)
                
                # Let's use mp4 since we write for the internet
                outputFileName = inputRoot + '_syncd_' + camName + ".mp4 "# inputExt
                
                thisStartFrame = startInd + startEndFrames[camName][0]
                
                rewriteVideos(inputPath, thisStartFrame, nFramesToWrite, frameRate,
                              outputDir=outputVideoDir, imageScaleFactor = .5,
                              outputFileName = outputFileName)
        
    if spline3dZeros:
    # Spline across positions with 0 3D confidence (i.e., there weren't 2 cameras
    # to use for triangulation).
        points3D = spline3dPoints(points3D, confidence3D, splineMaxFrames)
    
    return points3D, confidence3D

# %% 
def spline3dPoints(points3D, confidence3D, splineMaxFrames=5):
    c_p3d = copy.deepcopy(points3D)
    c_conf = copy.deepcopy(confidence3D)
    
    # Find internal stretches of 0 that are shorter than splineMaxFrames
    for iPt in np.arange(c_p3d.shape[1]):
        thisConf = c_conf[0,iPt,:]        
        zeroInds, nonZeroInds = findInternalZeroInds(thisConf,splineMaxFrames)
        
        # spline these internal zero stretches
        if zeroInds is not None and zeroInds.shape[0] > 0:
            c_p3d[:,iPt,zeroInds] = pchip_interpolate(nonZeroInds,
                                                  c_p3d[:,iPt,nonZeroInds],
                                                  zeroInds,axis=1)    

    return c_p3d

# %%
def findInternalZeroInds(x,maxLength):
       
    # skip splining if x is all 0
    if all(x==0):
        return None, None
    
    zeroInds = np.argwhere(x==0)
    dZeroInds = np.diff(zeroInds,axis=0,prepend=-1)
    
    # check if first/last values are 0, don't spline the beginning and end
    start0 = x[0] == 0
    end0 = x[-1] == 0
    
    if start0:
        # delete string of 0s at beginning
        while(dZeroInds.shape[0] > 0 and dZeroInds[0] == 1):
            zeroInds = np.delete(zeroInds,0)
            dZeroInds = np.delete(dZeroInds,0)
    if end0: 
        while(dZeroInds.shape[0] > 0 and dZeroInds[-1] == 1):
            # keep deleting end inds if there is a string of 0s before end
            zeroInds = np.delete(zeroInds,-1)
            dZeroInds = np.delete(dZeroInds,-1)
        # delete last index before jump - value will be greater than 1
        zeroInds = np.delete(zeroInds,-1)
        dZeroInds = np.delete(dZeroInds,-1)
    
    # check if any stretches are longer than maxLength
    thisStretch = np.array([0]) # initialize with first value b/c dZeroInds[0] ~=1
    indsToDelete = np.array([])
    for iIdx,d in enumerate(dZeroInds):
        if d == 1:
            thisStretch = np.append(thisStretch,iIdx)
        else:
            if len(thisStretch) >= maxLength:
                indsToDelete = np.append(indsToDelete,np.copy(thisStretch))
            thisStretch = np.array([iIdx])
    # if final stretch is too long, add it too
    if len(thisStretch) >= maxLength:
        indsToDelete = np.append(indsToDelete,np.copy(thisStretch))
        
    if len(indsToDelete) > 0:
        zeroInds = np.delete(zeroInds,indsToDelete.astype(int))
        dZeroInds = np.delete(dZeroInds,indsToDelete.astype(int))
            
    nonZeroInds = np.delete(np.arange(len(x)),zeroInds)

        
    return zeroInds, nonZeroInds

# %%
def getMissingMarkersCameras(keypoints2D):
    # Identify cameras that returned (0,0) as marker coordinates, ie that could
    # not identify the keypoints.
    # missingCams contains the indices of the cameras
    # missingMarkers contains the indices of the markers
    # Eg, missingCams[0] = 5 and missingMarkers[0] = 17 means that the camera
    # with index 5 returned (0,0) as coordinates of the marker with index 17.
    keypoints2D_res = np.reshape(np.stack(keypoints2D), 
                                  (np.stack(keypoints2D).shape[0], 
                                  np.stack(keypoints2D).shape[1], 
                                  np.stack(keypoints2D).shape[3]))
    missingCams, missingMarkers = np.where(np.sum(keypoints2D_res, axis=2) == 0)
    
    return missingCams, missingMarkers

#%%
def calcReprojectionError(cameraList,points2D,points3D,weights=None,normalizeError=False):
    reprojError = np.empty((points3D.shape[1],len(cameraList)))
    
    if weights==None:
        weights = [1 for i in range(len(cameraList))]
    for iCam,cam in enumerate(cameraList):
        reproj = cam.world_to_image(points3D)[:2,:]
        this2D = np.array([pt2D[:,iCam] for pt2D in points2D]).T
        reprojError[:,iCam] = np.linalg.norm(np.multiply((reproj-this2D),weights[iCam]),axis=0)
        
        if normalizeError: # Normalize by height of bounding box 
            nonZeroYVals = this2D[1,:][this2D[1,:]>0]
            boxHeight = np.nanmax(nonZeroYVals) - np.nanmin(nonZeroYVals)
            reprojError[:,iCam] /= boxHeight
    weightedReprojError_u = np.mean(reprojError,axis=1)
    return weightedReprojError_u

# %% Write TRC file for use with OpenSim.
def writeTRCfrom3DKeypoints(keypoints3D, pathOutputFile, keypointNames, 
                            frameRate=60, rotationAngles={}):
    
    keypoints3D_res = np.empty((keypoints3D.shape[2],
                                keypoints3D.shape[0]*keypoints3D.shape[1]))
    for iFrame in range(keypoints3D.shape[2]):
        keypoints3D_res[iFrame,:] = np.reshape(
            keypoints3D[:,:,iFrame], 
            (1,keypoints3D.shape[0]*keypoints3D.shape[1]),"F")
    
    # Change units to save data in m.
    keypoints3D_res /= 1000
    
    # Do not write face markers, they are unreliable and useless.
    faceMarkers = getOpenPoseFaceMarkers()[0]
    idxFaceMarkers = [keypointNames.index(i) for i in faceMarkers]    
    idxToRemove = np.hstack([np.arange(i*3,i*3+3) for i in idxFaceMarkers])
    keypoints3D_res_sel = np.delete(keypoints3D_res, idxToRemove, axis=1)  
    keypointNames_sel = [i for i in keypointNames if i not in faceMarkers]

    with open(pathOutputFile,"w") as f:
        numpy2TRC(f, keypoints3D_res_sel, keypointNames_sel, fc=frameRate, 
                  units="m")
    
    # Rotate data to match OpenSim conventions; this assumes the chessboard
    # is behind the subject and the chessboard axes are parallel to those of
    # OpenSim.
    trc_file = utilsDataman.TRCFile(pathOutputFile)    
    for axis,angle in rotationAngles.items():
        trc_file.rotate(axis,angle)

    trc_file.write(pathOutputFile)   
    
    return None

#%%
def loadPklVideo(pklPath, videoFullPath, imageBasedTracker=False, poseDetector='OpenPose',
                 confidenceThresholdForBB=0.3, visualizeKeypointAnimation=False):
    
    open_file = open(pklPath, "rb")
    frames = pickle.load(open_file)
    open_file.close()

    nFrames = len(frames)

    # Read in JSON files.
    allPeople = []
    iPerson = 0
    anotherPerson = True
    
    while anotherPerson is True:             
        anotherPerson = False
        res = np.zeros((nFrames, 75))
        res[:] = np.nan
        
        for c_frame, frame in enumerate(frames):    
            # Get this persons keypoints if they exist.
            if len(frame) > iPerson:
                person = frame[iPerson]
                keypoints = person['pose_keypoints_2d']
                res[c_frame,:] = keypoints
            
            # See if there are more people.
            if len(frame) > iPerson+1:
                # If there was someone else in any frame, loop again.
                anotherPerson = True 
        
        allPeople.append(res.copy())
        iPerson +=1     
        
    # Creates a browser animation of the data in each person detected. This
    # may not be continuous yet. That happens later with person tracking.
    if visualizeKeypointAnimation:
        import plotly.express as px
        import plotly.io as pio
        pio.renderers.default = 'browser'
    
        for i,data in enumerate(allPeople):
            
            # Remove the selected columns from 'data'
            data = np.delete(data, np.s_[2::3], axis=1)        
            data = data.reshape(nFrames, 25, 2).transpose(1, 0, 2)
        
            nPoints,nFrames,_ = data.shape
            # Reshape the 3D numpy array to 2D, preserving point and frame indices
            data_reshaped = np.copy(data).reshape(-1, 2)
    
            # Create DataFrame
            df = pd.DataFrame(data_reshaped, columns=['x', 'y'])
    
            # Add columns for point number and frame number
            df['Point'] = np.repeat(np.arange(nPoints), nFrames)
            df['Frame'] = np.tile(np.arange(nFrames), nPoints)
    
            # Reorder columns if needed
            df = df[['Point', 'Frame', 'x', 'y']]
    
            # Create a figure and add an animated scatter plot
            fig = px.scatter(df,x='x', y='y', title=videoFullPath + ' Box ' + str(i),
                              animation_frame='Frame',
                              range_x=[0, 1200], range_y=[1200,0])
    
            # Show the animation
            fig.show()   
 
    # Track People, or if only one person, skip tracking
    if len(allPeople) >1: 
        # Select the largest keypoint-based bounding box as the subject of interest
        bbFromKeypoints = [keypointsToBoundingBox(data,confidenceThreshold=confidenceThresholdForBB) for data in allPeople]
        maxArea, maxIdx = zip(*[getLargestBoundingBox(data,bbox) for data,bbox in zip(allPeople,bbFromKeypoints)]) # may want to find largest bounding box size in future instead of height
        
        # Check if a person has been detected, ie maxArea >= 0.0. If not, set
        # keypoints and confidence scores to 0, such that the camera is later
        # kicked out of the synchronization and triangulation.
        maxArea_np = np.array(maxArea)
        if np.max(maxArea_np) == 0.0:
            key2D = np.zeros((25,nFrames,2))
            confidence = np.zeros((25,nFrames))
            return key2D, confidence
        
        startPerson = np.nanargmax(maxArea)
        startFrame = maxIdx[startPerson]
        startBb = bbFromKeypoints[startPerson][startFrame]
        
        # initialize output data
        res = np.zeros((nFrames, 75))
        # res[:] = np.nan
          
        if imageBasedTracker:
            # This uses an image-based tracker and picks keypoint-based bounding box that is close to image-based tracker
            # It is slow
            # track this bounding box backwards until it can't be tracked
            res = trackBoundingBox(videoFullPath , startBb , allPeople , 
                                   bbFromKeypoints , res , frameStart = startFrame, 
                                   frameIncrement = -1 , visualize=True)
            
            # track this bounding box forward until it can't be tracked            
            res = trackBoundingBox(videoFullPath , startBb , allPeople , 
                                   bbFromKeypoints , res , frameStart = startFrame, 
                                   frameIncrement = 1 , visualize=True)
        else:
            # This just tracks the keypoint bounding box until there is a frame where the norm of the bbox corner change is above some
            # threshold (currently 20% of average image size). This percentage may need tuning
            
            # track this bounding box backwards until it can't be tracked
            res = trackKeypointBox(videoFullPath , startBb , allPeople ,
                                   bbFromKeypoints , res , frameStart = startFrame, 
                                   frameIncrement = -1 , visualize=False, 
                                   poseDetector=poseDetector)
            
            # track this bounding box forward until it can't be tracked            
            res = trackKeypointBox(videoFullPath , startBb , allPeople , 
                                   bbFromKeypoints , res , frameStart = startFrame, 
                                   frameIncrement = 1 , visualize=False, 
                                   poseDetector=poseDetector)
    else:
        res = allPeople[0]

    key2D = np.zeros((25,nFrames,2))
    confidence = np.zeros((25,nFrames))
    for i in range(0,25):
        key2D[i,:,0:2] = res[:,i*3:i*3+2]
        confidence[i,:] = res[:,i*3+2]
        
    # replace confidence nans with 0. 0 isn't used at all, nan is splined and used
    confidence = np.nan_to_num(confidence,nan=0)
        
    return key2D, confidence

# %%
def popNeutralPoseImages(cameraDirectories, camerasToUse, tSingleImage,
                         staticImagesFolderDir, trial_id, writeVideo = False):    
    
    if camerasToUse[0] == 'all':
        cameras2Use = list(cameraDirectories.keys())
    else:
        cameras2Use = camerasToUse
    
    cameraDirectories_selectedCams = {}
    for iCam,cam in enumerate(cameras2Use):
        cameraDirectories_selectedCams[cam] = cameraDirectories[cam]                
        videoPath = os.path.join(cameraDirectories_selectedCams[cam], 
                                 'InputMedia', 'neutral', 
                                 '{}_rotated.avi'.format(trial_id))
        
        imagePath = video2Images(videoPath, tSingleImage=tSingleImage, 
                     filePrefix=(str(cam)+'_'), 
                     outputFolder=staticImagesFolderDir)        
        
        if writeVideo:
            if iCam ==0 :
                print('writing Neutral video')                
            videoFolder,videoFile = os.path.split(imagePath)
            videoFolder = os.path.join(videoFolder,'Videos')
            os.makedirs(videoFolder, exist_ok=True)            
            fileRoot,_ = os.path.splitext(videoFile)
            # ensure this ends in Cam#
            camStart = fileRoot.rfind('Cam')+3
            for i in range(len(fileRoot) - camStart):
                if not fileRoot[camStart+i].isdigit():
                    break
            
            camName = 'Cam' + fileRoot[camStart:camStart+i]            
            
            videoPath = os.path.join(videoFolder,'neutralVid_' + camName + '.mp4')   
            # Write a video from a single image
            ffmpegCmd = ('ffmpeg -loglevel error -r 0.01 -loop 1 -i ' + imagePath + 
                         ' -c:v libx264 -tune stillimage -preset  ultrafast -ss 00:00:00 -t 00:00:1   -c:a aac  -b:a 96k -pix_fmt yuv420p  -shortest ' 
                         + videoPath + ' -y')
            os.system(ffmpegCmd)
        
    return
