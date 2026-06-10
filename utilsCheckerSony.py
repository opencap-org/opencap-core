import sys
sys.path.append("./mmpose") # utilities in child directory
import cv2 
import numpy as np 
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
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import gaussian, sosfiltfilt, butter, find_peaks
from scipy.interpolate import pchip_interpolate
from scipy.spatial.transform import Rotation 
import scipy.linalg
from itertools import combinations
import copy
from utilsCameraPy3 import Camera, nview_linear_triangulations
from utils import getOpenPoseMarkerNames, getOpenPoseFaceMarkers
from utils import numpy2TRC, rewriteVideos, delete_multiple_element,loadCameraParameters


API_TOKEN = None
API_URL = None

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
def video2Images(videoPath, nImages=12, tSingleImage=None, filePrefix='output', skipIfRun=False, outputFolder='default'):
    if outputFolder == 'default':
        outputFolder = os.path.dirname(videoPath)
    
    # Check if first image exists to skip if requested
    first_image_check = os.path.join(outputFolder, f"{filePrefix}_0.jpg")
    if not os.path.exists(first_image_check) or not skipIfRun: 
        
        if tSingleImage is not None: # pop single image at time value
            outImagePath = os.path.join(outputFolder, f"{filePrefix}0.png")
            
            # Construct command as a list
            CMD = [
                'ffmpeg', '-loglevel', 'error', '-skip_frame', 'nokey', '-y',
                '-ss', str(tSingleImage),
                '-i', videoPath,
                '-qmin', '1', '-q:v', '1', '-frames:v', '1',
                '-vf', "select='eq(pict_type,I)'",
                outImagePath
            ]
            
            subprocess.run(CMD)
           
        else: # pop multiple images from video
            lengthVideo = getVideoLength(videoPath)
            timeImageSamples = np.linspace(1, lengthVideo - 1, nImages) 
            
            for iFrame, t_image in enumerate(timeImageSamples):
                outImagePath = os.path.join(outputFolder, f"{filePrefix}_{iFrame}.jpg")
                
                CMD = [
                    'ffmpeg', '-loglevel', 'error', '-skip_frame', 'nokey', '-y',
                    '-ss', str(t_image),
                    '-i', videoPath,
                    '-qmin', '1', '-q:v', '1', '-frames:v', '1',
                    '-vf', "select='eq(pict_type,I)'",
                    outImagePath
                ]
                
                subprocess.run(CMD)
                # Note: outImagePath updates in loop; returning final path below
                outImagePath = os.path.join(outputFolder, f"{filePrefix}0.jpg")
                
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
def computeAverageIntrinsics(session_path,trialIDs,CheckerBoardParams,nImages=25,cameraModel= None,videoType=".mov"):
    CamParamList = []
    camModels = []
    trial_name = ''
    
    for trial_id in trialIDs:
        if cameraModel is None:
            resp = requests.get(API_URL + "trials/{}/".format(trial_id),
                              headers = {"Authorization": "Token {}".format(API_TOKEN)})
            trial = resp.json()
            camModels.append(trial['videos'][0]['parameters']['model'])
            trial_name = trial['name']
        else:
            camModels.append(cameraModel)
        if trial_name == '':
            trial_name = trial_id
        
        # Make directory (folder for trialname, intrinsics also saved there)
        video_dir = os.path.join(session_path,trial_name)
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir,trial_name + videoType)
        
        # Download video if not done
        if not os.path.exists(video_path):
            download_file(trial["videos"][0]["video"], video_path)
            
        if not os.path.exists(os.path.join(video_dir,'cameraIntrinsics.pickle')):
            
            # Compute intrinsics from images popped out of intrinsic video. Step in
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

    # If desired number of corners can be detected then, 
    # refine the pixel coordinates and display 
    # them on the images of checker board 
    if ret == True: 
        # 3D points real world coordinates       
        threedpoints.append(objectp3d) 
  
        # Refining pixel coordinates 
        # for given 2d points. 
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
def synchronizeVideos(CameraDirectories, trialRelativePath, pathPoseDetector,
                      undistortPoints=False, CamParamDict=None, 
                      confidenceThreshold=0.3, 
                      filtFreqs={'gait':12,'default':30},
                      imageBasedTracker=False, cams2Use=['all'],
                      poseDetector='OpenPose', trialName=None, bbox_thr=0.8,
                      resolutionPoseDetection='default'):
    
    markerNames = getOpenPoseMarkerNames()
    
    # Create list of cameras.
    if cams2Use[0] == 'all':
        cameras2Use = list(CameraDirectories.keys())
    else:
        cameras2Use = cams2Use
    cameras2Use_in = copy.deepcopy(cameras2Use)

    # Initialize output lists
    pointList = []
    confList = []
    
    CameraDirectories_selectedCams = {}
    CamParamList_selectedCams = []
    for cam in cameras2Use:
        CameraDirectories_selectedCams[cam] = CameraDirectories[cam]
        CamParamList_selectedCams.append(CamParamDict[cam])
        
    # Load data.
    camsToExclude = []
    for camName in CameraDirectories_selectedCams:
        cameraDirectory = CameraDirectories_selectedCams[camName]
        videoFullPath = os.path.normpath(os.path.join(cameraDirectory,
                                                      trialRelativePath))
        trialPrefix, _ = os.path.splitext(
            os.path.basename(trialRelativePath))
        if poseDetector == 'OpenPose':
            outputPklFolder = "OutputPkl_" + resolutionPoseDetection
        elif poseDetector == 'mmpose':
            outputPklFolder = "OutputPkl_mmpose_" + str(bbox_thr)
        openposePklDir = os.path.join(outputPklFolder, trialName)
        pathOutputPkl = os.path.join(cameraDirectory, openposePklDir)
        ppPklPath = os.path.join(pathOutputPkl, trialPrefix+'_rotated_pp.pkl')
        key2D, confidence = loadPklVideo(
            ppPklPath, videoFullPath, imageBasedTracker=imageBasedTracker,
            poseDetector=poseDetector)
        thisVideo = cv2.VideoCapture(videoFullPath[:-4] + '_rotated.avi')
        frameRate = np.round(thisVideo.get(cv2.CAP_PROP_FPS))        
        if key2D.shape[1] == 0 and confidence.shape[1] == 0:
            camsToExclude.append(camName)
        else:
            pointList.append(key2D)
            confList.append(confidence)
        
    # If video is not existing, the corresponding camera should be removed.
    idx_camToExclude = []
    for camToExclude in camsToExclude:
        cameras2Use.remove(camToExclude)
        CameraDirectories_selectedCams.pop(camToExclude)
        idx_camToExclude.append(cameras2Use_in.index(camToExclude))
        # By removing the cameras in CamParamDict and CameraDirectories, we
        # modify those dicts in main, which is needed for the next stages.
        CamParamDict.pop(camToExclude)
        CameraDirectories.pop(camToExclude)        
    delete_multiple_element(CamParamList_selectedCams, idx_camToExclude)    

    # Synchronize keypoints.
    pointList, confList, nansInOutList,startEndFrameList = synchronizeVideoKeypoints(
        pointList, confList, confidenceThreshold=confidenceThreshold,
        filtFreqs=filtFreqs, sampleFreq=frameRate, visualize=False,
        maxShiftSteps=2*frameRate, CameraParams=CamParamList_selectedCams,
        cameras2Use=cameras2Use, 
        CameraDirectories=CameraDirectories_selectedCams, trialName=trialName)
    
    if undistortPoints:
        if CamParamList_selectedCams is None:
            raise Exception('Need to have CamParamList to undistort Images')
        # nFrames = pointList[0].shape[1]
        unpackedPoints = unpackKeypointList(pointList) ;
        undistortedPoints = []
        for points in unpackedPoints:
            undistortedPoints.append(undistort2Dkeypoints(
                points, CamParamList_selectedCams, useIntrinsicMatAsP=True))
        pointList = repackKeypointList(undistortedPoints)
        
    pointDir = {}
    confDir = {}
    nansInOutDir = {}
    startEndFrames = {}
    for iCam, camName in enumerate(CameraDirectories_selectedCams):
        pointDir[camName] =  pointList[iCam]
        confDir[camName] =  confList[iCam] 
        nansInOutDir[camName] = nansInOutList[iCam] 
        startEndFrames[camName] = startEndFrameList[iCam]
        
    return pointDir, confDir, markerNames, frameRate, nansInOutDir, startEndFrames, cameras2Use

# %%
def synchronizeVideoKeypoints(keypointList, confidenceList,
                              confidenceThreshold=0.3, 
                              filtFreqs = {'gait':12,'default':30},
                              sampleFreq=30, visualize=False, maxShiftSteps=30,
                              isGait=False, CameraParams = None,
                              cameras2Use=['none'],CameraDirectories = None,
                              trialName=None, trialID=''):
    visualize2Dkeypoint = False # this is a visualization just for testing what filtered input data looks like
    
    # keypointList is a mCamera length list of (nmkrs,nTimesteps,2) arrays of camera 2D keypoints
    print('Synchronizing Keypoints')
    
    # Deep copies such that the inputs do not get modified.
    c_CameraParams = copy.deepcopy(CameraParams)
    c_cameras2Use = copy.deepcopy(cameras2Use)
    c_CameraDirectoryDict = copy.deepcopy(CameraDirectories)
    
    # Turn Camera Dict into List
    c_CameraDirectories = list(c_CameraDirectoryDict.values())
    # Check if one camera has only 0s as confidence scores, which would mean
    # no one has been properly identified. We want to kick out this camera
    # from the synchronization and triangulation. We do that by popping out
    # the corresponding data before syncing and add back 0s later.
    badCameras = []
    for icam, conf in enumerate(confidenceList):
        if np.max(conf) == 0.0:
            badCameras.append(icam)
    idxBadCameras = [badCameras[i]-i for i in range(len(badCameras))]
    for idxBadCamera in idxBadCameras:
        print('{} kicked out of synchronization'.format(
            c_cameras2Use[idxBadCamera]))
        keypointList.pop(idxBadCamera)
        confidenceList.pop(idxBadCamera)
        c_CameraParams.pop(idxBadCamera)
        c_cameras2Use.pop(idxBadCamera)
        c_CameraDirectories.pop(idxBadCamera)
        
        
    markerNames = getOpenPoseMarkerNames()
    mkrDict = {mkr:iMkr for iMkr,mkr in enumerate(markerNames)}
    
    # First, remove occluded markers
    footMkrs = {'right':[mkrDict['RBigToe'], mkrDict['RSmallToe'], mkrDict['RHeel'],mkrDict['RAnkle']],
                'left':[mkrDict['LBigToe'], mkrDict['LSmallToe'], mkrDict['LHeel'],mkrDict['LAnkle']]}
    armMkrs = {'right':[mkrDict['RElbow'], mkrDict['RWrist']],
                'left':[mkrDict['LElbow'], mkrDict['LWrist']]}
    
    plt.close('all')
    
    # Copy for visualization
    keypointListUnfilt = copy.deepcopy(keypointList)
    
    # remove occluded foot markers (uses large differences in confidence)
    keypointList,confidenceList = zip(*[removeOccludedSide(keys,conf,footMkrs,confidenceThreshold,visualize=False) for keys,conf in zip(keypointList,confidenceList)])
    # remove occluded arm markers
    keypointList,confidenceList = zip(*[removeOccludedSide(keys,conf,armMkrs,confidenceThreshold,visualize=False) for keys,conf in zip(keypointList,confidenceList)])
       
    # Copy for visualization 
    keypointListOcclusionRemoved = copy.deepcopy(keypointList)
    
    # Don't change these. The ankle markers are used for gait detector
    markers4VertVel = [mkrDict['RAnkle'], mkrDict['LAnkle']] # R&L Ankles and Heels did best. There are some issues though - like when foot marker velocity is aligned with camera ray
    markers4HandPunch = [mkrDict['RWrist'], mkrDict['LWrist'],mkrDict['RShoulder'],mkrDict['LShoulder']]
    markers4Ankles = [mkrDict['RAnkle'],mkrDict['LAnkle']]
    
    # find velocity signals for synchronization
    nCams = len(keypointList)
    vertVelList = []
    mkrSpeedList = []
    handPunchVertPositionList = []
    allMarkerList = []
    for (keyRaw,conf) in zip(keypointList,confidenceList):
        keyRaw_clean, _, _, _ = clean2Dkeypoints(keyRaw,conf,confidenceThreshold=0.3,nCams=nCams,linearInterp=True)        
        keyRaw_clean_smooth = smoothKeypoints(keyRaw_clean, sdKernel=3) 
        handPunchVertPositionList.append(getPositions(keyRaw_clean_smooth,markers4HandPunch,direction=1)) 
        vertVelList.append(getVertVelocity(keyRaw_clean_smooth)) # doing it again b/c these settings work well for synchronization
        mkrSpeedList.append(getMarkerSpeed(keyRaw_clean_smooth,markers4VertVel,confidence=conf,averageVels=False)) # doing it again b/c these settings work well for synchronization
        allMarkerList.append(keyRaw_clean_smooth)
        
    # Find indices with high confidence that overlap between cameras.    
    # Note: Could get creative and do camera pair syncing in the future, based
    # on cameras with greatest amount of overlapping confidence.
    overlapInds_clean, minConfLength_all = findOverlap(confidenceList,
                                                   markers4VertVel)
    
    # If no overlap found, try with fewer cameras.
    c_nCams = len(confidenceList)
    while not np.any(overlapInds_clean) and c_nCams>2:
        print("Could not find overlap with {} cameras - trying with {} cameras".format(c_nCams, c_nCams-1))
        cam_list = [i for i in range(nCams)]
        # All possible combination with c_nCams-1 cameras.
        from itertools import combinations
        combs = set(combinations(cam_list, c_nCams-1))
        overlapInds_clean_combs = []
        for comb in combs:
            confidenceList_sel = [confidenceList[i] for i in list(comb)]
            overlapInds_clean_c, minConfLength_c = findOverlap(
                confidenceList_sel, markers4VertVel)
            overlapInds_clean_combs.append(overlapInds_clean_c.flatten())
        longest_stretch = 0
        for overlapInds_clean_comb in overlapInds_clean_combs:
            stretch_size = overlapInds_clean_comb.shape[0]
            if stretch_size > longest_stretch:
                longest_stretch = stretch_size
                overlapInds_clean = overlapInds_clean_comb
        c_nCams -= 1
        
    # If no overlap found, return 0s.
    if not np.any(overlapInds_clean):
        keypointsSync = []
        confidenceSync = []
        nansInOutSync = []
        for i in range(len(cameras2Use)):
            keypointsSync.insert(i, np.zeros((keypointList[0].shape[0], 10,
                                              keypointList[0].shape[2])))
            confidenceSync.insert(i, np.zeros((keypointList[0].shape[0], 10)))
            nansInOutSync.insert(i, np.array([np.nan, np.nan]))     
        return keypointsSync, confidenceSync, nansInOutSync
                
    [idxStart, idxEnd] = [np.min(overlapInds_clean), np.max(overlapInds_clean)]
    idxEnd += 1 # Python indexing system.
    # Take max shift between cameras into account.
    idxStart = int(np.max([0,idxStart - maxShiftSteps]))
    idxEnd = int(np.min([idxEnd+maxShiftSteps,minConfLength_all]))
    # Re-sample the lists    
    vertVelList = [v[idxStart:idxEnd] for v in vertVelList]
    mkrSpeedList = [v[:,idxStart:idxEnd] for v in mkrSpeedList]
    handPunchVertPositionList = [p[:,idxStart:idxEnd] for p in handPunchVertPositionList]
    allMarkerList = [p[:,idxStart:idxEnd] for p in allMarkerList]
    confSyncList= [c[:,idxStart:idxEnd] for c in confidenceList]
    
    # Detect whether it is a gait trial, which determines what sync algorithm
    # to use. Input right and left ankle marker speeds. Gait should be
    # detected for all cameras (all but one camera is > 2 cameras) for the
    # trial to be considered a gait trial.
    
    # Detect activity, which determines sync function that gets used
    
    isHandPunch,handForPunch = detectHandPunchAllVideos(handPunchVertPositionList,sampleFreq)
    if isHandPunch:
        syncActivity = 'handPunch'
    elif detectGaitAllVideos(mkrSpeedList,allMarkerList,confSyncList,markers4Ankles,sampleFreq):
        syncActivity = 'gait'
    else:
        syncActivity = 'general'
        
    print('Using ' + syncActivity + ' sync function.')
    
    
    # Select filtering frequency based on if it is gait
    if syncActivity == 'gait': 
        filtFreq = filtFreqs['gait']
    else: 
        filtFreq = filtFreqs['default']
    
    # Filter keypoint data
    # sdKernel = sampleFreq/(2*np.pi*filtFreq) # not currently used, but in case using gaussian smoother (smoothKeypoints function) instead of butterworth to filter keypoints
    keyFiltList = []
    confFiltList = []
    confSyncFiltList = []
    nansInOutList = []
    for (keyRaw,conf) in zip(keypointList,confidenceList):
        keyRaw_clean, conf_clean, nans_in_out, conf_sync_clean = clean2Dkeypoints(keyRaw,conf,confidenceThreshold,nCams=nCams)
        keyRaw_clean_filt = filterKeypointsButterworth(keyRaw_clean,filtFreq,sampleFreq,order=4)
        keyFiltList.append(keyRaw_clean_filt)
        confFiltList.append(conf_clean)
        confSyncFiltList.append(conf_sync_clean)
        nansInOutList.append(nans_in_out)

    # Copy for visualization
    keypointListFilt = copy.deepcopy(keyFiltList)
    confidenceListFilt = copy.deepcopy(confFiltList)
    confidenceSyncListFilt = copy.deepcopy(confSyncFiltList)

    # find nSample shift relative to the first camera
    # nSamps = keypointList[0].shape[1]
    shiftVals = []
    shiftVals.append(0)
    timeVecs = []
    tStartEndVec = np.zeros((len(keypointList),2))
    for iCam,vertVel in enumerate(vertVelList):
        timeVecs.append(np.arange(keypointList[iCam].shape[1]))
        if iCam>0:
            # if no keypoints in Cam0 or the camera of interest, do not use cross_corr to sync.
            if np.max(np.abs(vertVelList[iCam])) == 0 or np.max(np.abs(vertVelList[0])) == 0:
                lag = 0
            elif syncActivity == 'general':
                dataForReproj = {'CamParamList':c_CameraParams,
                                 'keypointList':keypointListFilt,
                                 'cams2UseReproj': [0, c_cameras2Use.index(c_cameras2Use[iCam])],
                                 'confidence': confidenceSyncListFilt,
                                 'cameras2Use': c_cameras2Use
                                 }
                corVal,lag = cross_corr(vertVel,vertVelList[0],multCorrGaussianStd=20,visualize=False,dataForReproj=dataForReproj) # gaussian curve gets multipled by correlation plot - helping choose the smallest shift value for periodic motions
            elif syncActivity == 'gait':
                
                dataForReproj = {'CamParamList':c_CameraParams,
                                 'keypointList':keypointListFilt,
                                 'cams2UseReproj': [0, c_cameras2Use.index(c_cameras2Use[iCam])],
                                 'confidence': confidenceSyncListFilt,
                                 'cameras2Use': c_cameras2Use
                                 }
                corVal,lag = cross_corr_multiple_timeseries(mkrSpeedList[iCam],
                                            mkrSpeedList[0],
                                            multCorrGaussianStd=maxShiftSteps/2,
                                            dataForReproj=dataForReproj,
                                            visualize=False)     
            elif syncActivity == 'handPunch':
                corVal,lag = syncHandPunch([handPunchVertPositionList[i] for i in [0,iCam]],
                                           handForPunch,maxShiftSteps=maxShiftSteps)
            if np.abs(lag) > maxShiftSteps: # if this fails and we get a lag greater than maxShiftSteps (units=timesteps)
                lag = 0 
                print('Did not use cross correlation to sync {} - computed shift was greater than specified {} frames. Shift set to 0.'.format(c_cameras2Use[iCam], maxShiftSteps))
            shiftVals.append(lag)
            timeVecs[iCam] = timeVecs[iCam] - shiftVals[iCam]
        tStartEndVec[iCam,:] = [timeVecs[iCam][0], timeVecs[iCam][-1]]
        
    # align signals - will start at the latest-starting frame (most negative shift) and end at
    # nFrames - the end of the earliest starting frame (nFrames - max shift)
    tStart = np.max(tStartEndVec[:,0])
    tEnd = np.min(tStartEndVec[:,1])
    
    keypointsSync = []
    confidenceSync = []
    startEndFrames = []
    nansInOutSync = []
    for iCam,key in enumerate(keyFiltList):
        # Trim the keypoints and confidence lists
        confidence = confFiltList[iCam]
        iStart = int(np.argwhere(timeVecs[iCam]==tStart))
        iEnd = int(np.argwhere(timeVecs[iCam]==tEnd))
        keypointsSync.append(key[:,iStart:iEnd+1,:])
        confidenceSync.append(confidence[:,iStart:iEnd+1])
        if shiftVals[iCam] > 0:
            shiftednNansInOut = nansInOutList[iCam] - shiftVals[iCam]
        else:
            shiftednNansInOut = nansInOutList[iCam]
        nansInOutSync.append(shiftednNansInOut)        
        # Save start and end frames to list, so can rewrite videos in
        # triangulateMultiviewVideo
        startEndFrames.append([iStart,iEnd])
        
    # Plot synchronized velocity curves
    if visualize:
        # Vert Velocity
        f, (ax0,ax1) = plt.subplots(1,2)
        for (timeVec,vertVel) in zip(timeVecs,vertVelList):
            ax0.plot(timeVec[range(len(vertVel))],vertVel)
        legNames = [c_cameras2Use[iCam] for iCam in range(len(vertVelList))]
        ax0.legend(legNames)
        ax0.set_title('summed vertical velocities')
        
        # Marker speed
        for (timeVec,mkrSpeed) in zip(timeVecs,mkrSpeedList):
            ax1.plot(timeVec[range(len(vertVel))],mkrSpeed[2])
        legNames = [c_cameras2Use[iCam] for iCam in range(len(vertVelList))]
        ax1.legend(legNames)
        ax1.set_title('Right Ankle Speed')
    
    # Plot a single marker trajectory to see effect of filtering and occlusion removal
    if visualize2Dkeypoint:
        nCams = len(keypointListFilt)
        nCols = int(np.ceil(nCams/2))
        mkr = mkrDict['RBigToe']
        mkrXY = 0 # x=0, y=1
        
        fig = plt.figure()
        fig.set_size_inches(12,7,forward=True)
        for camNum in range(nCams):
            ax = plt.subplot(2,nCols,camNum+1)
            ax.set_title(c_cameras2Use[iCam])
            ax.set_ylabel('yPos (pixel)')
            ax.set_xlabel('frame')
            ax.plot(keypointListUnfilt[camNum][mkr,:,mkrXY],linewidth = 2)
            ax.plot(keypointListOcclusionRemoved[camNum][mkr,:,mkrXY],linewidth=1.6)
            ax.plot(keypointListFilt[camNum][mkr,:,mkrXY],linewidth=1.3)
            
            # find indices where conf> thresh or nan (the marker is used in triangulation)
            usedInds = np.argwhere(np.logical_or(confidenceListFilt[camNum][mkr,:] > confidenceThreshold ,
                                                 np.isnan(confidenceListFilt[camNum][mkr,:])))
            if len(usedInds) > 0:
                ax.plot(usedInds,keypointListFilt[camNum][mkr,usedInds,mkrXY],linewidth=1)
                ax.set_ylim((.9*np.min(keypointListFilt[camNum][mkr,usedInds,mkrXY]),1.1*np.max(keypointListFilt[camNum][mkr,usedInds,mkrXY])))
            else:
                ax.text(0.5,0.5,'no data used',horizontalalignment='center',transform=ax.transAxes)
            
        plt.tight_layout()
        ax.legend(['unfilt','occlusionRemoved','filt','used for triang'],bbox_to_anchor=(1.05,1))
        
    # We need to add back the cameras that have been kicked out.
    # We just add back zeros, they will be kicked out of the triangulation.
    for badCamera in badCameras:
        keypointsSync.insert(badCamera, np.zeros(keypointsSync[0].shape))
        confidenceSync.insert(badCamera, np.zeros(confidenceSync[0].shape))
        nansInOutSync.insert(badCamera, np.array([np.nan, np.nan]))
        startEndFrames.insert(badCamera, None)
 
    return keypointsSync, confidenceSync, nansInOutSync, startEndFrames

# %%
def detectHandPunchAllVideos(handPunchPositionList,sampleFreq,punchDuration=3):
    
    isPunch = []
    for pos in handPunchPositionList:
        isPunchThisVid = []
        relPosList = []
        maxIndList = []
        for iSide in range(2):
            relPos = -np.diff(pos[(iSide, iSide+2),:],axis=0) # vertical position of wrist over shoulder
            relPosList.append(relPos)

            maxInd = np.argmax(relPos)
            maxIndList.append(maxInd)
            maxPos = relPos[:,maxInd]
                        
            isPunchThisVid.append(False)
            if maxPos>0:
                zeroCrossings = np.argwhere(np.diff(np.sign(np.squeeze(relPos))) != 0)
                if any(zeroCrossings>maxInd) and any(zeroCrossings<maxInd):
                    startLength = np.abs(np.min([zeroCrossings-maxInd]))/sampleFreq
                    endLength = np.abs(np.min([-zeroCrossings+maxInd]))
                    if (startLength+endLength)/sampleFreq<punchDuration:
                        isPunchThisVid[-1] = True
                        
        # Ensure only one arm is punching
        hand = None
        isPunch.append(False)
        if isPunchThisVid[0]:
            if relPosList[1][:,maxIndList[0]] < 0:
                isPunch[-1] = True
                hand = 'r'
        elif isPunchThisVid[1]:
            if relPosList[0][:,maxIndList[1]] < 0:
                isPunch[-1] = True
                hand = 'l' 
        
    isTrialPunch = all(isPunch)
    
    return isTrialPunch, hand

# %% 
def syncHandPunch(positions,hand,maxShiftSteps=600):
    if hand == 'r':
        startInd = 0
    else:
        startInd = 1
        
    relVel = []
    for pos in positions:
        relPos = -np.diff(pos[(startInd, startInd+2),:],axis=0) # vertical position of wrist over shoulder
        relVel.append(np.squeeze(np.diff(relPos)))
        
    corr_val,lag = cross_corr(relVel[1],relVel[0],multCorrGaussianStd=maxShiftSteps,visualize=False)
    
    return corr_val, lag

# %% 
def getPositions(keypoints,indsMarkers,direction=1):
                     
    positions = np.max(keypoints.max(axis=1)[:,1])-keypoints[indsMarkers,:,:]
                    
    return positions[:,:,direction]

# %% 
def detectGait(rSpeed,lSpeed,frameRate):
    
    # cross correlate to see if they are in or out of phase
    corr,lag = cross_corr(rSpeed,lSpeed,multCorrGaussianStd=frameRate*3)
    
    # default false in case feet are static
    isGait = False
    if corr > 0.55:
        if np.abs(lag) > 0.1 * frameRate and np.abs(lag) < frameRate:
            isGait = True
    
    return isGait

# %%
def detectFeetMoving(allMarkers,confidence,ankleInds,motionThreshold=.5):
    # motion threshold is a percent of bounding box height/width
    
    # Get bounding box height or width
    # nFrames x(nMkrsx3)
    nFrames = confidence.shape[1]
    nMkrs = confidence.shape[0]
    cMkrs = np.copy(allMarkers)
    reshapedMkrs = np.ndarray((nFrames,0))
    for i in range(nMkrs):
        reshapedMkrs = np.append(reshapedMkrs,np.squeeze(cMkrs[i,:,:]),axis=1)
    
    inData = np.insert(reshapedMkrs.T,np.arange(2,nMkrs*2+1,2),confidence,axis=0)
    
    bbox=keypointsToBoundingBox(inData.T)
    # normalize by the average width of the bounding box
    normValue = np.mean(bbox[:,2])
    
    # compute max distance
    ankleMkrs = np.divide(allMarkers[ankleInds,:,:],normValue)
    ankleConf = confidence[ankleInds,:]
    confThresh = 0.4
    maxMvt = []
    for i in range(2):
        confidentInds = ankleConf[i]>confThresh
        if len(confidentInds)>0:
            confidentMarkers = ankleMkrs[i,ankleConf[i]>confThresh,:]
            # need to find the two points that are furthest from each other. A naive
            # search is O(n^2). Let's assume we are looking for motion in horizontal 
            # direction.
            idxMax = np.argmax(confidentMarkers[:,0])
            idxMin = np.argmin(confidentMarkers[:,0])
            maxMvt.append(scipy.linalg.norm(
                confidentMarkers[idxMax,:]-confidentMarkers[idxMin,:]))
        else:
            # if we did not see the foot, assume it did not move
            maxMvt.append=0

    
    # did both feet move greater than the motion threshold
    anyFootMoving = all([m>motionThreshold for m in maxMvt])
    
    return anyFootMoving
    

# %% 
def detectGaitAllVideos(mkrSpeedList,allMarkers,confidence,ankleInds,sampleFreq):
    isGaits = []
    feetMoving = []
    for c_mkrSpeed,allMkrs,conf in zip(mkrSpeedList,allMarkers,confidence):
        isGaits.append(detectGait(c_mkrSpeed[0], c_mkrSpeed[1], sampleFreq))
        feetMoving.append(detectFeetMoving(allMkrs,conf,ankleInds))
    if len(isGaits) > 2:
        true_count = sum(isGaits)
        isGait = true_count>=len(isGaits)-1
    else:
        isGait = all(isGaits)
        
    isGait = isGait and any(feetMoving)
        
    return isGait

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
def keypointsToBoundingBox(data):
    # input: nFrames x 75.
    # output: nFrames x 4 (xTopLeft, yTopLeft, width, height).
    
    c_data = np.copy(data)
    
    # Remove face markers - they are intermittent.
    _, idxFaceMarkers = getOpenPoseFaceMarkers()
    idxToRemove = np.hstack([np.arange(i*3,i*3+3) for i in idxFaceMarkers])
    c_data = np.delete(c_data, idxToRemove, axis=1)    
    
    c_data[c_data==0] = np.nan
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
    # Tracks closest keypoint bounding boxes until the box changes too much.
    bboxKey = bbStart # starting bounding box
    frameNum = frameStart

    # initiate video capture
    # Read video
    video = cv2.VideoCapture(videoPath)
    nFrames = allBoxes[0].shape[0]
    
    # Read desiredFrames.
    video.set(1, frameNum)
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
        
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
        
        # For OpenPose: a person is not always referred to by the same index,
        # so we identify the box that 'contains' the subject of interest. For
        # mmpose, a person is always referred to by the same index, since
        # persons are detected first and joints second. Here, we therefore
        # distinguish between both detection algorithms, and detect the person
        # for mmpose only for the first frame and then use the same person 
        # index for the rest of the frames. For OpenPose, we detect the person 
        # at each frame.
        if count == 0 or poseDetector == "OpenPose":
            iPerson, bboxKey_new, samePerson = findClosestBox(bboxKey, keyBoxes,
                                                          imageSize)
        else:
            _, bboxKey_new, samePerson = findClosestBox(bboxKey, keyBoxes, 
                                                    imageSize, iPerson)
        
        # We allow badFramesBeforeStop of samePerson = False to account for an
        # errant frame(s) in the pose detector. Once we reach badFramesBeforeStop,
        # we break and output to the last good frame.
        if len(badFrames) > 0 and samePerson:
            badFrames = []
            
        if not samePerson and not justStarted:
            if len(badFrames) >= badFramesBeforeStop:
                print('not same person at ' + str(frameNum - frameIncrement*badFramesBeforeStop))
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
    video = cv2.VideoCapture(videoPath)
    nFrames = allBoxes[0].shape[0]

    # Read desiredFrames.
    video.set(1, frameNum)
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
         
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
def removeOccludedSide(key2D,confidence,mkrInds,confThresh,visualize=False):
    
    key2D_out = np.copy(key2D)
    confidence_out = np.copy(confidence)
    
    #Parameters
    confDif = .2 # the difference in mean confidence between R and L sided markers. If dif > confDif, must be occluded
        
    rMkrs = mkrInds['right']
    lMkrs = mkrInds['left']
    
    # nan confidences should be turned to 0 at this point...not sure why I used a nanmean
    rConf = np.nanmean(confidence_out[rMkrs,:],axis=0)
    lConf = np.nanmean(confidence_out[lMkrs,:],axis=0)
    dConf = rConf-lConf

    
    if visualize:
        plt.figure()
        plt.plot(rConf,color='r')
        plt.plot(lConf,color='k')
        plt.plot(dConf,color=(.5,.5,.5))
        plt.legend(['rightConfidence','leftConfidence','dConf'])
        plt.xlabel('frame')
        plt.ylabel('mean confidence')
    
    
    rOccluded = np.where(np.logical_and(np.less(dConf,confDif) , np.less(rConf,confThresh)))[0]
    lOccluded = np.where(np.logical_and(np.greater(dConf,confDif) , np.less(lConf,confThresh)))[0]
    
    # if inleading and exiting values are in occluded list, don't change them
    def zeroPad(occInds,nSamps):
    
        # Default: not setting confidence of any indices to 0
        zeroInds = np.empty([0],dtype=np.int64)
        
        if len(occInds) == 0:
            return occInds,zeroInds

        # find gaps
        dInds = np.diff(occInds,prepend=1)

        #all first vals        
        if occInds[0] == 0:
            #special case - occlusion starts at beginning but there are no gaps afterwards
            if not any(dInds>1):
                zeroInds = np.concatenate((zeroInds, np.copy(occInds)))
                occInds = []
                return occInds, zeroInds

            else:
                # otherwise, remove in-leading indices
                firstGap = np.argwhere(dInds>1)[0]
                occInds = np.delete(occInds,np.arange(firstGap))
                dInds = np.delete(dInds,np.arange(firstGap))
                zeroInds = np.concatenate((zeroInds, np.arange(firstGap)))

        #all end vals
        if occInds[-1] == nSamps-1:
            #special case - occlusion goes to end but there are no gaps beforehand
            if not any(dInds>1):
                zeroInds = np.concatenate((zeroInds, np.copy(occInds)))
                occInds = []
            else:
                # otherwise, remove end zeros
                lastGap = np.argwhere(dInds>1)[-1]
                occInds_init = np.copy(occInds)
                occInds = np.delete(occInds,np.arange(lastGap,len(occInds)))
                zeroInds = np.concatenate((zeroInds, np.arange(occInds_init[lastGap[0]],occInds_init[-1]+1)))

        return occInds, zeroInds

    nSamps = len(rConf)
    # Assumes that occlusion can't happen starting in first frame or ending in last frame. Looks for first "gap"
    rOccluded,rZeros = zeroPad(rOccluded,nSamps)
    lOccluded,lZeros = zeroPad(lOccluded,nSamps)    
    
    # Couldn't figure out how to index multidimensions...
    # Set occlusion confidences to nan. Later, the keypoints with associated nan confidence will be cubic splines, 
    # and the confidence will get set to something like 0.5 for weighted triangulation in utilsCameraPy3 
    for mkr in lMkrs:
        if np.count_nonzero(confidence_out[mkr,:]>0)>3:
            lNan = np.intersect1d(lOccluded,np.arange(np.argwhere(confidence_out[mkr,:]>0)[1],np.argwhere(confidence_out[mkr,:]>0)[-1])) # don't add nans to 0 pad
        else:
            lNan = np.array([],dtype='int64')
        key2D_out[mkr,lNan.astype(int),:] = np.nan
        confidence_out[mkr,lNan.astype(int)] = np.nan
        confidence_out[mkr,lZeros] = 0
        
    for mkr in rMkrs:
        if np.count_nonzero(confidence_out[mkr,:]>0)>3:  
            rNan = np.intersect1d(rOccluded,np.arange(np.argwhere(confidence_out[mkr,:]>0)[1],np.argwhere(confidence_out[mkr,:]>0)[-1])) # don't add nans to 0 pad
        else:
            rNan = np.array([],dtype='int64')
        key2D_out[mkr,rNan.astype(int),:] = np.nan
        confidence_out[mkr,rNan.astype(int)] = np.nan
        confidence_out[mkr,rZeros] = 0

    
    if visualize:
        # TESTING
        rConf = np.nanmean(confidence_out[rMkrs,:],axis=0)
        lConf = np.nanmean(confidence_out[lMkrs,:],axis=0)
        dConf = rConf-lConf
        plt.figure()
        plt.plot(rConf,color='r')
        plt.plot(lConf,color='k')
        plt.plot(dConf,color=(.5,.5,.5))
        plt.legend(['rightFootConfidence','leftFootConfidence','dConf'])
        plt.xlabel('frame')
        plt.ylabel('mean confidence')
       
    return key2D_out, confidence_out


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
def repackKeypointList(unpackedKeypointList):
    nFrames = len(unpackedKeypointList)
    nCams = len(unpackedKeypointList[0])
    nMkrs = unpackedKeypointList[0][0].shape[0]
    
    repackedKeypoints = []
    
    for iCam in range(nCams):
        tempArray = np.empty((nMkrs,nFrames,2))
        for iFrame in range(nFrames):
            tempArray[:,iFrame,:] = np.squeeze(unpackedKeypointList[iFrame][iCam])
        repackedKeypoints.append(np.copy(tempArray))
    
    return repackedKeypoints

# %%
def undistort2Dkeypoints(pointList2D, CameraParamList, useIntrinsicMatAsP=True):
    # list of 2D points per image
    pointList2Dundistorted = []
    
    for i,points2D in enumerate(pointList2D):
        if useIntrinsicMatAsP:
            res = cv2.undistortPoints(points2D,CameraParamList[i]['intrinsicMat'],CameraParamList[i]['distortion'],P=CameraParamList[i]['intrinsicMat'])
        else:
            res = cv2.undistortPoints(points2D,CameraParamList[i]['intrinsicMat'],CameraParamList[i]['distortion'])
        pointList2Dundistorted.append(res.copy())
    
    return pointList2Dundistorted

# %%
def getVertVelocity(key2D):
    vertVel = np.diff(key2D[:,:,1],axis=1)
    vertVelTotal = np.mean(vertVel,axis=0)
    vertVelTotal = np.append(vertVelTotal,0) # keep length
    if not np.max(np.abs(vertVelTotal)) == 0: # only if markers were found, otherwise div by 0
        vertVelTotal = vertVelTotal / np.max(np.abs(vertVelTotal))
    return vertVelTotal

# %%
# get 2D speed of specified markers
def getMarkerSpeed(key2D,idxMkrs = [0],confidence = None, confThresh = 0.2, averageVels=False):
    c_conf = copy.deepcopy(confidence)
    
    diffOrder = 1 ;
    # 2d marker speed
    vertVel = np.abs(np.linalg.norm(np.diff(key2D,axis=1,n=diffOrder,append=np.tile(key2D[:,[-1],:],(1,diffOrder,1))),axis=2))
    
    # Set velocity for low confidence to 0
    lowConf = c_conf < confThresh
    vertVel[lowConf] = 0
    
    if len(idxMkrs) >1:
        if averageVels:
            vertVelTotal = np.mean(vertVel[idxMkrs,:],axis=0)
            # np.append(vertVelTotal,0)
        else:
            vertVelTotal = vertVel[idxMkrs,:]
            # vertVelTotal = np.hstack((vertVelTotal,np.zeros((vertVelTotal.shape[0],1)))) # keep length
    else:
        vertVelTotal = vertVel[idxMkrs,:]
        # np.append(vertVelTotal,0)
    if not np.max(np.abs(vertVelTotal)) == 0: # only if markers were found, otherwise div by 0
        vertVelTotal = vertVelTotal / np.max(np.abs(vertVelTotal))
    return vertVelTotal

# %%
def smoothKeypoints(key2D,sdKernel=1):
    key2D_out = np.copy(key2D)
    for i in range(25):
        for j in range(2):
            key2D_out[i,:,j] = np.apply_along_axis(
                lambda x: gaussian_filter1d(x, sdKernel),
                arr = key2D_out[i,:,j],
                axis = 0)
    return key2D_out

# %% 
def filterKeypointsButterworth(key2D,filtFreq,sampleFreq,order=4):
    key2D_out = np.copy(key2D)
    wn = filtFreq/(sampleFreq/2)
    if wn>1:
        print('You tried to filter ' + str(sampleFreq) + ' signal with cutoff freq of ' + str(filtFreq) + ', which is above the Nyquist Frequency. Will filter at ' + str(sampleFreq/2) + 'instead.')
        wn=0.99
    elif wn==1:
        wn=0.99
        
    sos = butter(order/2,wn,btype='low',output='sos')
    
    for i in range(2):
        key2D_out[:,:,i] = sosfiltfilt(sos,key2D_out[:,:,i],axis=1)             
        
    return key2D_out

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

# %%
def clean2Dkeypoints(key2D, confidence, confidenceThreshold=0.5, nCams=2, 
                     linearInterp=False):
    
    key2D_out = np.copy(key2D)
    confidence_out = np.copy(confidence)
    confidence_sync_out = np.copy(confidence)
    
    nMkrs = key2D_out.shape[0]    
    markerNames = getOpenPoseMarkerNames()
    
    # Turn all 0s into nans.
    key2D_out[key2D_out==0] = np.nan    
    
    # Turn low confidence values to 0s if >2 cameras and nans otherwise.
    for i in range(nMkrs):        
        # If a marker has at least two frames with positive confidence,
        # then identify frames where confidence is lower than threshold.
        if len(np.argwhere(confidence_out[i,:]>0))>2:
            nanInds = np.where(confidence_out[i,:] < confidenceThreshold)
        # If a marker doesn't have two frames with positive confidence,
        # then return empty list.    
        else:
            nanInds = []
            faceMarkers = getOpenPoseFaceMarkers()[0]
            # no warning if face marker
            if not markerNames[i] in faceMarkers:
                print('There were <2 frames with >0 confidence for {}'.format(
                    markerNames[i]))
        # Turn low confidence values to 0s if >2 cameras.
        # Frames with confidence values of 0s are ignored during triangulation.
        if nCams>2:
            confidence_out[i,nanInds] = 0
        # Turn low confidence values to nans if 2 cameras.
        # Frames with nan confidence values are splined, and nan confidences
        # are replaced by 0.5 during triangulation.
        else:
            confidence_out[i,nanInds] = np.nan            
        # Turn inleading and exiting nans into 0s
        idx_nonnans = ~np.isnan(confidence_out[i,:])
        idx_nonzeros  = confidence_out[i,:] != 0            
        idx_nonnanszeros = idx_nonnans & idx_nonzeros
        if True in idx_nonnanszeros:
            idx_nonnanszeros_first = np.where(idx_nonnanszeros)[0][0]
            idx_nonnanszeros_last = np.where(idx_nonnanszeros)[0][-1]
            confidence_out[i,:idx_nonnanszeros_first] = 0
            confidence_out[i,idx_nonnanszeros_last:] = 0 
        else:
            confidence_out[i,:] = 0                    
            
        # Turn low confidence values to 0s for confidence_sync_out whatever
        # the number of cameras; confidence_sync_out is used for 
        # calculating the reprojection error, and using nans rather than 
        # 0s might affect the outcome.
        confidence_sync_out[i,nanInds] = 0
        # Turn inleading and exiting nans into 0s
        idx_nonnans = ~np.isnan(confidence_sync_out[i,:])
        idx_nonzeros  = confidence_sync_out[i,:] != 0            
        idx_nonnanszeros = idx_nonnans & idx_nonzeros
        if True in idx_nonnanszeros:
            idx_nonnanszeros_first = np.where(idx_nonnanszeros)[0][0]
            idx_nonnanszeros_last = np.where(idx_nonnanszeros)[0][-1]
            confidence_sync_out[i,:idx_nonnanszeros_first] = 0
            confidence_sync_out[i,idx_nonnanszeros_last:] = 0 
        else:
            confidence_sync_out[i,:] = 0
        
        # Turn keypoint values to nan if confidence is low. Keypoints with nan
        # will be interpolated. In cases with more than 2 cameras, this will
        # have no impact on triangulation, since corresponding confidence is 0.
        # But with 2 cameras, we need the keypoint 2D coordinates to
        # be interpolated. In both cases, not relying on garbage keypoints for
        # interpolation matters when computing keypoint speeds, which are used
        # for synchronization.            
        key2D_out[i,nanInds,:] = np.nan
    
    # Helper function.
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    
    # Interpolate keypoints with nans.
    for i in range(nMkrs):
        for j in range(2):
            if np.isnan(key2D_out[i,:,j]).all(): # only nans
                key2D_out[i,:,j] = 0
            elif np.isnan(key2D_out[i,:,j]).any():  # partially nans
                nans, x = nan_helper(key2D_out[i,:,j])
                if linearInterp:
                    # Note: with linear interpolation, values are carried over
                    # backward and forward for inleading and exiting nans.
                    key2D_out[i,nans,j] = np.interp(x(nans), x(~nans), 
                                                    key2D_out[i,~nans,j]) 
                else:
                    # Note: with cubic interpolate, values are garbage for
                    # inleading and exiting nans.
                    try:
                        key2D_out[i,:,j] = pchip_interpolate(
                            x(~nans), key2D_out[i,~nans,j], 
                            np.arange(key2D_out.shape[1]))    
                    except:
                        key2D_out[i,nans,j] = np.interp(x(nans), x(~nans), 
                                                        key2D_out[i,~nans,j])
                        
    # Keep track of inleading and exiting nans when less than 3 cameras.
    if nCams>2:
        nans_in_out = np.array([np.nan, np.nan])        
    else:
        _, idxFaceMarkers = getOpenPoseFaceMarkers()       
        nans_in = []
        nans_out = []
        for i in range(nMkrs):
            if i not in idxFaceMarkers:
                idx_nans = np.isnan(confidence_out[i,:])
                if False in idx_nans:
                    nans_in.append(np.where(idx_nans==False)[0][0])
                    nans_out.append(np.where(idx_nans==False)[0][-1])
                else:
                    nans_in.append(np.nan)
                    nans_out.append(np.nan)                    
        in_max = np.max(np.asarray(nans_in))
        out_min = np.min(np.asarray(nans_out))        
        nans_in_out = np.array([in_max, out_min])
    

    return key2D_out, confidence_out, nans_in_out, confidence_sync_out

# %%
def cross_corr(y1, y2,multCorrGaussianStd=None,visualize=False, dataForReproj=None):
    """Calculates the cross correlation and lags without normalization.
    
    The definition of the discrete cross-correlation is in:
    https://www.mathworks.com/help/matlab/ref/xcorr.html
    
    Args:
    y1, y2: Should have the same length.
    
    Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
    """
    # Pad shorter signal with 0s
    if len(y1) > len(y2):
        temp = np.zeros(len(y1))
        temp[0:len(y2)] = y2
        y2 = np.copy(temp)
    elif len(y2)>len(y1):
        temp = np.zeros(len(y2))
        temp[0:len(y1)] = y1
        y1 = np.copy(temp)
        
    y1_auto_corr = np.dot(y1, y1) / len(y1)
    y2_auto_corr = np.dot(y2, y2) / len(y1)
    corr = np.correlate(y1, y2, mode='same')
    # The unbiased sample size is N - lag.
    unbiased_sample_size = np.correlate(np.ones(len(y1)), np.ones(len(y1)), mode='same')
    corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
    shift = len(y1) // 2
    max_corr = np.max(corr)
    argmax_corr = np.argmax(corr)    

    # find correlation peak with minimum reprojection error
    if dataForReproj is not None:
        _,peaks = find_peaks(corr, height=.1)
        idxPeaks = np.squeeze(np.asarray([np.argwhere(peaks['peak_heights'][i]==corr) for i in range(len(peaks['peak_heights']))]))
        lags = idxPeaks-shift
        # look at 3 lags closest to 0
        if np.isscalar(lags):
            max_corr = corr[lags+shift]
            return max_corr, lags
        if len(lags)>3:
            lags = lags[np.argsort(np.abs(lags))[:3]]
        
        reprojError = np.empty((len(lags),1))
        reprojSuccess = []
        for iPeak,lag in enumerate(lags):
            # calculate reprojection error for each potential lag
            reprojError[iPeak,0], reprojSuccessi = calcReprojectionErrorForSync(
                dataForReproj['CamParamList'], dataForReproj['keypointList'],
                lag, dataForReproj['cams2UseReproj'], 
                dataForReproj['confidence'], dataForReproj['cameras2Use'])
            reprojSuccess.append(reprojSuccessi)
        
        # find if min reproj error is clearly smaller than other peaks. If it is not,
        # don't use reproj error min for sync. E.g. with treadmill walking, reproj error may not work as
        # robustly as for overground walking
        reprojSort = np.sort(reprojError,axis=0)
        reprojErrorRatio = reprojSort[0]/reprojSort[1]
        if reprojErrorRatio < 0.6 and not False in reprojSuccess: # tunable parameter. Usually around 0.25 for overground walking
            # find idx with minimum reprojection error 
            lag = lags[np.argmin(reprojError)]
            max_corr = corr[lag+shift]
            
            if multCorrGaussianStd is not None:
                print('For {}, used reprojection error minimization to sync.'.format(dataForReproj['cameras2Use'][dataForReproj['cams2UseReproj'][1]]))
            return max_corr, lag
        
    if visualize:
        plt.figure()
        plt.plot(corr)
        plt.title('vertical velocity correlation')
        
    # Multiply correlation curve by gaussian (prioritizing lag solution closest to 0)
    if multCorrGaussianStd is not None:
        corr = np.multiply(corr,gaussian(len(corr),multCorrGaussianStd))
        if visualize: 
            plt.plot(corr,color=[.4,.4,.4])
            plt.legend(['corr','corr*gaussian'])  
    
    argmax_corr = np.argmax(corr)
    max_corr = np.nanmax(corr)
    
    lag = argmax_corr-shift
    
    return max_corr, lag


# %%
def cross_corr_multiple_timeseries(Y1, Y2,multCorrGaussianStd=None,dataForReproj=None,visualize=False):
    
    # SHAPE OF Y1,Y2 is nMkrs by nSamples
    """Calculates the cross correlation and lags without normalization.
    
    The definition of the discrete cross-correlation is in:
    https://www.mathworks.com/help/matlab/ref/xcorr.html
    
    Args:
    y1, y2: Should have the same length.
    
    Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
    """
    nMkrs = Y1.shape[0]
    corrMat = np.empty(Y1.shape)
    for iMkr in range(nMkrs):
        y1=Y1[iMkr,:]
        y2=Y2[iMkr,:]
        # Pad shorter signal with 0s
        if len(y1) > len(y2):
            temp = np.zeros(len(y1))
            temp[0:len(y2)] = y2
            y2 = np.copy(temp)
        elif len(y2)>len(y1):
            temp = np.zeros(len(y2))
            temp[0:len(y1)] = y1
            y1 = np.copy(temp)
            
        y1_auto_corr = np.dot(y1, y1) / len(y1)
        y2_auto_corr = np.dot(y2, y2) / len(y1)
        corr = np.correlate(y1, y2, mode='same')
        # The unbiased sample size is N - lag
        unbiased_sample_size = np.correlate(np.ones(len(y1)), np.ones(len(y1)), mode='same')
        corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
        shift = len(y1) // 2
        corrMat[iMkr,:] = corr  
    
    if visualize:
        plt.figure()
        plt.plot(corrMat.T)
        plt.plot(np.nansum(corrMat,axis=0),color='k')
        plt.title('multi-marker correlation')
        legNames = ['mkr' + str(iMkr) for iMkr in range(nMkrs)]
        legNames.append('summedCorrelation')
        plt.legend(legNames)
        
        plt.figure()
        plt.plot(Y1.T)
        plt.plot(Y2.T)
    
    summedCorr = np.nansum(corrMat,axis=0)
    
    # find correlation peak with minimum reprojection error
    if dataForReproj is not None:
        _,peaks = find_peaks(summedCorr, height=.75)
        idxPeaks = np.squeeze(np.asarray([np.argwhere(peaks['peak_heights'][i]==summedCorr) for i in range(len(peaks['peak_heights']))]))
        lags = idxPeaks-shift
        # look at 3 lags closest to 0
        if np.isscalar(lags):
            max_corr = summedCorr[lags+shift]
            return max_corr, lags
        if len(lags)>3:
            lags = lags[np.argsort(np.abs(lags))[:3]]
        
        reprojError = np.empty((len(lags),1))
        reprojSuccess = []
        for iPeak,lag in enumerate(lags):
            # calculate reprojection error for each potential lag
            reprojError[iPeak,0], reprojSuccessi = calcReprojectionErrorForSync(
                dataForReproj['CamParamList'], dataForReproj['keypointList'],
                lag, dataForReproj['cams2UseReproj'], 
                dataForReproj['confidence'], dataForReproj['cameras2Use'])
            reprojSuccess.append(reprojSuccessi)
        
        # find if min reproj error is clearly smaller than other peaks. If it is not,
        # don't use reproj error min for sync. E.g. with treadmill walking, reproj error may not work as
        # robustly as for overground walking
        reprojSort = np.sort(reprojError,axis=0)
        reprojErrorRatio = reprojSort[0]/reprojSort[1]
        if reprojErrorRatio < 0.6 and not False in reprojSuccess: # tunable parameter. Usually around 0.25 for overground walking
            # find idx with minimum reprojection error 
            lag = lags[np.argmin(reprojError)]
            max_corr = summedCorr[lag+shift]
            
            if multCorrGaussianStd is not None:
                print('For {}, used reprojection error minimization to sync.'.format(dataForReproj['cameras2Use'][dataForReproj['cams2UseReproj'][1]]))
            return max_corr, lag
        
    # Multiply correlation curve by gaussian (prioritizing lag solution closest to 0)
    if multCorrGaussianStd is not None:
        summedCorr = np.multiply(summedCorr,gaussian(len(summedCorr),multCorrGaussianStd))
        if visualize: 
            plt.plot(summedCorr,color=(.4,.4,.4))
    
    argmax_corr = np.argmax(summedCorr)
    max_corr = np.nanmax(summedCorr)/corrMat.shape[0] 
    
    lag = argmax_corr-shift
    return max_corr, lag

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
                              startEndFrames=None, trialID=''):
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
        os.makedirs(outputVideoDir,exist_ok=True)
        for iCam,camName in enumerate(keypointDict):
                        
            nFramesToWrite = endInd-startInd
    
            inputPaths = glob.glob(os.path.join(CameraDirectories[camName],'OutputMedia*',trialName,trialID + '*'))
            if len(inputPaths) > 0:
                inputPath = inputPaths[0]
            else:
                inputPaths = glob.glob(os.path.join(CameraDirectories[camName],'InputMedia*',trialName,trialID + '*'))
                inputPath = inputPaths[0]
            
            # get frame rate and assume all the same for sync'd videos
            if iCam==0: 
                thisVideo = cv2.VideoCapture(inputPath)
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
def calcReprojectionErrorForSync(CamParamList, keypointList, lagVal,
                                 cams2UseReproj, confidence, cameras2Use):
    
    # Number of timesteps to triangulate. Will average reprojection error over all nTimesteps.
    nTimesteps = 5 
    
    keypoints2D = copy.deepcopy(keypointList)
    conf = copy.deepcopy(confidence)
    CamParamListCopy = copy.deepcopy(CamParamList)
    
    # Find the range of overlapping confidence for this lag value
    confSel = []
    for cam in cams2UseReproj:
        confSel.append(conf[cam])
        
    # find confidence ranges in original indices
    confThresh = [.5*np.nanmax(c) for c in confSel] # Threshold for saying this camera confidently sees the person
    avgConf = [np.nanmean(c,axis=0) for c in confSel]
    
    # Breakdown steps to catch potential error.
    confRanges = []
    for i,c in enumerate(avgConf):
        temp = c > confThresh[i]
        if True in temp:
            confRanges.append(np.array([np.argwhere(temp)[0], np.argwhere(temp)[-1]+1]))
        else:
            reprojErrorAcrossFrames = 0.1
            reprojSuccess = False
            return reprojErrorAcrossFrames, reprojSuccess
        
    # shift second camera based on lag, so indices are "aligned," then find overlapping range
    shiftedConfRanges = copy.deepcopy(confRanges)
    shiftedConfRanges[1] = shiftedConfRanges[1] - lagVal # shift the indices for second camera
    # Ignore the first and last few timesteps here as confidence drops
    shiftedOverlapInds = [np.max((shiftedConfRanges[0][0],shiftedConfRanges[1][0]))+3,
                          np.min((shiftedConfRanges[0][1],shiftedConfRanges[1][1]))-3]
    
    # Sample nTimesteps between the shifted Overlap Inds
    shiftedSampleInds = np.linspace(shiftedOverlapInds[0],shiftedOverlapInds[1],nTimesteps).astype(int)
    
    sampleInds = []
    sampleInds.append(shiftedSampleInds) # no shift for first camera
    sampleInds.append(shiftedSampleInds + lagVal) # unshifts the indices for second camera
    
    # Select keypoints and confidence at appropriate timesteps
    keypoints2DSelected = []
    cameraListSelected = []
    confListSelected = []
    for iCam, cam in enumerate(cams2UseReproj):
        keypoints2D[cam] = keypoints2D[cam][:,sampleInds[iCam],:]
        conf[cam] = conf[cam][:,sampleInds[iCam]]
        keypoints2DSelected.append(keypoints2D[cam])
        cameraListSelected.append(CamParamListCopy[cam])
        confListSelected.append(conf[cam])
        
    # Triangulate at each of the nTimesteps
    # We here need to turn the lists back into dicts.
    CamParamListCopy_dict = {}
    keypoints2D_dict = {}
    conf_dict = {}
    for iCam, cam in enumerate(cameras2Use):
        CamParamListCopy_dict[cam] = CamParamListCopy[iCam]
        keypoints2D_dict[cam] = keypoints2D[iCam]
        conf_dict[cam] = conf[iCam]
    cameras2UseReproj = [cameras2Use[i] for i in cams2UseReproj]
    keypoints3D, _ = triangulateMultiviewVideo(
        CamParamListCopy_dict, keypoints2D_dict, ignoreMissingMarkers=False, 
        cams2Use=cameras2UseReproj, confidenceDict=conf_dict,trimTrial=False)
    
    # Make list of camera objects
    cameraObjList = []       
    for camParams in cameraListSelected:
        c = Camera()
        c.set_K(camParams['intrinsicMat'])
        c.set_R(camParams['rotation'])
        c.set_t(np.reshape(camParams['translation'],(3,1)))
        cameraObjList.append(c)
        
    # Compute confidence-weighted reprojection error for each of nTimesteps
    reprojErrorVec = []
    for tStep in range(nTimesteps):
       
        # Organize points for reprojectionError function
        stackedPoints = np.stack([k[:,None,tStep,:] for k in keypoints2DSelected])
        pointsInput = []
        for i in range(stackedPoints.shape[1]):
            pointsInput.append(stackedPoints[:,i,0,:].T)
        
        confForWeights = [c[:,None,tStep].T for c in confListSelected] # needs transpose for reprojection error function
        confForWeights = [np.nan_to_num(c,nan=0) for c in confForWeights] # sometimes confidence has nans, don't want to use as weights in this case
        
        # Calculate combined reprojection error
        key3D = np.squeeze(keypoints3D[:,:,tStep])
        reprojErrors = calcReprojectionError(cameraObjList,pointsInput,key3D,
                                             weights=confForWeights,normalizeError=True)
        
        # multiply minimum confidence between cameras times marker-wise reproj errors
        # so we don't include errors for markers that had low confidence in one of the cameras
        minConfVec = np.min(np.asarray(confForWeights), axis=0)
        minConfVec[np.where(minConfVec<0.5)] = 0 # Set low conf markers to 0
        weightedReprojErrors = np.multiply(reprojErrors,minConfVec)
        if np.any(weightedReprojErrors):
            reprojErrorVec.append(np.mean(weightedReprojErrors[minConfVec>0]))
        else:
            reprojErrorVec.append(1000) # in cases where no position is confident set to large reproj error. typical values are on the order of  0.1
        
    reprojErrorAcrossFrames = np.mean(reprojErrorVec)
    reprojSuccess = True    
    
    return reprojErrorAcrossFrames, reprojSuccess

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

# %% Find indices with high confidence that overlap between cameras.
def findOverlap(confidenceList, markers4VertVel):    
    c_confidenceList = copy.deepcopy(confidenceList)
    
    # Find overlapping indices.
    confMean = [
        np.mean(cMat[markers4VertVel,:],axis=0) for cMat in c_confidenceList]
    minConfLength = np.min(np.array([len(x) for x in confMean]))
    confArray = np.array([x[0:minConfLength] for x in confMean])
    minConfidence = np.nanmin(confArray,axis=0)
    confThresh = .5*np.nanmax(minConfidence)
    overlapInds = np.asarray(np.where(minConfidence>confThresh))
    
    # Find longest stretch of high confidence.
    difOverlap = np.diff(overlapInds)
    
    # Create array with last elements of continuous stretches
    # (more than 2 timesteps between high confidence).
    unConfidentInds = np.argwhere(np.squeeze(difOverlap)>4)
    
    # Identify and select the longest continuous stretch.
    if unConfidentInds.shape[0] > 0:        
        # First stretch (ie, between 0 and first unconfident index).
        stretches = []
        stretches.append(np.arange(0, unConfidentInds[0]+1))
        # Middle stretches.
        if unConfidentInds.shape[0] >1:
            for otherStretch in range(unConfidentInds.shape[0]-1):
                stretches.append(np.arange(unConfidentInds[otherStretch]+1,
                                           unConfidentInds[otherStretch+1]+1))
        # Last stretch (ie, between last unconfident index and last element).
        stretches.append(np.arange(unConfidentInds[-1]+1,
                                   overlapInds.shape[1]))
        # Identify longest stretch.
        longestStretch = stretches[
            np.argmax([stretch.shape[0] for stretch in stretches])]
        # Select longest stretch
        overlapInds_clean = overlapInds[0,longestStretch]
    else:
        overlapInds_clean = overlapInds
        
    return overlapInds_clean, minConfLength

#%%
def loadPklVideo(pklPath, videoFullPath, imageBasedTracker=False, poseDetector='OpenPose'):
    
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
 
    # Track People, or if only one person, skip tracking
    if len(allPeople) >1: 
        # Select the largest keypoint-based bounding box as the subject of interest
        bbFromKeypoints = [keypointsToBoundingBox(data) for data in allPeople]
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
