"""
    @authors: Scott Uhlrich, Antoine Falisse, Łukasz Kidziński

    This script takes as inputs a video, sessionMetadata.yaml, the camera model name, and
    computes camera intrinsics. The intrinsic parameters are then saved to file in a general
    location for future use of this phone model.
"""

import os 
import yaml
import pickle

from utils import importMetadata
from utils import getDataDirectory
from utilsChecker import computeAverageIntrinsics
from utilsChecker import saveCameraParameters

# %% Required user inputs.
sessionName = 'iPhone15_720_60FPS_20231107'
CheckerBoardParams = {'dimensions':(11,8),'squareSize':60} # gets replaced if metadata present in sessionName folder

#List of trials - intrinsics from each video are averaged
trials = ['3dc7b63a-71d8-48d6-b906-28372a527129']

loadTrialInfo = False # Load previous trial names and CheckerBoardParams from file
saveIntrinsicsForDeployment = True

deployedFolderNames = ['Deployed_720_60fps','Deployed'] # both folder names if want to keep the detailed folder

cameraModel = "iPhoneTest"  
videoType = ".mov" #can be .avi or other file formats
    
# %% Paths to data folder for local testing.
dataDir = os.path.join(getDataDirectory(),'Data')
sessionDir = os.path.join(dataDir,'IntrinsicCaptures',sessionName)
trialFile = os.path.join(sessionDir,'trialInfo.yaml')
intrinsicComparisonFile = os.path.join(sessionDir,'intrinsicComparison.pkl')

# %% Get checker parameters and filenames if they exist
# TODO this should come from the server API
# Get checkerboard parameters from metadata.
metadataPath = os.path.join(sessionDir,'sessionMetadata.yaml')

if os.path.exists(metadataPath):
    sessionMetadata = importMetadata()
    CheckerBoardParams = {
        'dimensions': (sessionMetadata['checkerBoard']['black2BlackCornersWidth_n'],
                        sessionMetadata['checkerBoard']['black2BlackCornersHeight_n']),
        'squareSize': sessionMetadata['checkerBoard']['squareSideLength_mm']}
    print('CheckerBoardParams replaced from metadata file.')

if loadTrialInfo:
    if os.path.exists(trialFile):
        with open(trialFile, 'r') as f:
            trialInfo = yaml.safe_load(f)
            trials = trialInfo['trials']
            CheckerBoardParams = {
                        'dimensions': (trialInfo['nSquaresWidth'],
                                       trialInfo['nSquaresHeight']),
                        'squareSize': trialInfo['squareSize']}
    else:
        raise Exception('trialFile doesn''t exist. Enter trials and CheckerBoardParams manually.')
        
     
# Compute average intrinsic values from multiple trials of same camera
CamParamsAverage, CamParamList, intrinsicComparisons, cameraModel = computeAverageIntrinsics(sessionDir,trials,CheckerBoardParams,nImages=50,cameraModel=cameraModel,videoType=videoType)


# Save intrinsics from first camera for deployement 
if saveIntrinsicsForDeployment:
    for deployedFolderName in deployedFolderNames:
        permIntrinsicDir = os.path.join(os.getcwd(), 'CameraIntrinsics',
                                    cameraModel,deployedFolderName)
        saveCameraParameters(os.path.join(
                permIntrinsicDir, 'cameraIntrinsics.pickle'), CamParamsAverage)
        
# Save trial info
trialInfo = {'trials':trials,
             'nSquaresWidth':CheckerBoardParams['dimensions'][0],
             'nSquaresHeight':CheckerBoardParams['dimensions'][1],
             'squareSize':CheckerBoardParams['squareSize'],
             'cameraModel':cameraModel,}
with open(trialFile, 'w') as f:
    yaml.dump(trialInfo,f)
    
with open(intrinsicComparisonFile, 'wb') as f:
    pickle.dump(intrinsicComparisons,f)