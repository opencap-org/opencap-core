import yaml
import json
import os
import socket
import requests
import urllib.request
import shutil
import utilsDataman
import pickle
import glob
import mimetypes
import subprocess
import zipfile
import time
import datetime

import numpy as np
import pandas as pd
from scipy import signal
from urllib3.util.retry import Retry

import platform
import opensim

import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian

# Local-only: cloud API disabled (no OpenCap login required).
API_URL = None
API_TOKEN = None

#%% Rest of utils

def getDataDirectory(isDocker=False):
    computername = socket.gethostname()
    # Paths to OpenPose folder for local testing.
    if computername == 'SUHLRICHHPLDESK':
        dataDir = 'C:/Users/scott.uhlrich/MyDrive/mobilecap/'
    elif computername == "LAPTOP-7EDI4Q8Q":
        dataDir = 'C:\MyDriveSym/mobilecap/'
    elif computername == 'DESKTOP-0UPR1OH':
        dataDir = 'C:/Users/antoi/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'HPL1':
        dataDir = 'C:/Users/opencap/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'DESKTOP-GUEOBL2':
        dataDir = 'C:/Users/opencap/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'DESKTOP-L9OQ0MS':
        dataDir = 'C:/Users/antoi/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'clarkadmin-MS-7996':
        dataDir = '/home/clarkadmin/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'DESKTOP-NJMGEBG':
        dataDir = 'C:/Users/opencap/Documents/MyRepositories/mobilecap_data/'
    elif computername == 'WIN-FKQTPLS40OV':
        dataDir = 'C:/Users/steudelkri/Documents/'
    elif isDocker:
        dataDir = os.getcwd()
    else:
        dataDir = os.getcwd()
    return dataDir

def getOpenPoseDirectory(isDocker=False):
    computername = os.environ.get('COMPUTERNAME', None)
    
    # Paths to OpenPose folder for local testing.
    if computername == "DESKTOP-0UPR1OH":
        openPoseDirectory = "C:/Software/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose"
    elif computername == "HPL1":
        openPoseDirectory = "C:/Users/opencap/Documents/MySoftware/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose"
    elif computername == "DESKTOP-GUEOBL2":
        openPoseDirectory = "C:/Software/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose"
    elif computername == "DESKTOP-L9OQ0MS":
        openPoseDirectory = "C:/Software/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose"
    elif isDocker:
        openPoseDirectory = "docker"
    elif computername == 'SUHLRICHHPLDESK':
        openPoseDirectory = "C:/openpose/"
    elif computername == "LAPTOP-7EDI4Q8Q":
        openPoseDirectory = "C:/openpose/"
    elif computername == "DESKTOP-NJMGEBG":
        openPoseDirectory = "C:/openpose/"
    else:
        openPoseDirectory = "C:/openpose/"
    return openPoseDirectory

def getMMposeDirectory(isDocker=False):
    computername = socket.gethostname()
    
    # Paths to OpenPose folder for local testing.
    if computername == "clarkadmin-MS-7996":
        mmposeDirectory = "/home/clarkadmin/Documents/MyRepositories/MoVi_analysis/model_ckpts"
    else:
        mmposeDirectory = ''
    return mmposeDirectory

def loadCameraParameters(filename):
    open_file = open(filename, "rb")
    cameraParams = pickle.load(open_file)
    
    open_file.close()
    return cameraParams

def importMetadata(filePath):
    myYamlFile = open(filePath)
    parsedYamlFile = yaml.load(myYamlFile, Loader=yaml.FullLoader)
    
    return parsedYamlFile

def download_file(url, file_name):
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        
def getTrialJson(trial_id):
    response = makeRequestWithRetry('GET',
                                    API_URL + "trials/{}/".format(trial_id),
                                    headers = {"Authorization": "Token {}".format(API_TOKEN)})
    trialJson = response.json()
    return trialJson

def getSessionJson(session_id):
    response = makeRequestWithRetry('GET',
                                    API_URL + "sessions/{}/".format(session_id),
                                    headers = {"Authorization": "Token {}".format(API_TOKEN)})
    sessionJson = response.json()
    
    # sort trials by time recorded
    def getCreatedAt(trial):
        return trial['created_at']
    sessionJson['trials'].sort(key=getCreatedAt)
    
    return sessionJson

def getSubjectJson(subject_id):
    response = makeRequestWithRetry('GET',
                                    API_URL + "subjects/{}/".format(subject_id),
                                    headers = {"Authorization": "Token {}".format(API_TOKEN)})
    subjectJson = response.json()
    return subjectJson
    
def getTrialName(trial_id):
    trial = getTrialJson(trial_id)
    trial_name = trial['name']
    trial_name = trial_name.replace(' ', '')
    
    return trial_name

def writeMediaToAPI(API_URL,media_path,trial_id,tag=None,deleteOldMedia=False):
    
    if deleteOldMedia:
        deleteResult(trial_id, tag=tag)
    
    for filename in os.listdir(media_path):
        thisMimeType = mimetypes.guess_type(filename)
        if thisMimeType[0] is not None and not os.path.isdir(filename):
            print(filename)
            fileType = thisMimeType[0][0:thisMimeType[0].find('/')]
            if fileType == 'image' or fileType == 'video' or fileType == 'application':
                fullpath = "{}/{}".format(media_path, filename)
                
                if fileType == 'image' and tag == "calibration-img":
                    cam = filename[filename.find('Cam'):filename.find('.')]
                    if "altSoln" in filename:
                        altSoln = '_altSoln'
                    else:
                        altSoln = ''
                    device_id = cam + altSoln
                
                else:
                    device_id = None
                               
                postFileToTrial(fullpath,trial_id,tag,device_id)

    return


def getTrialNameIdMapping(session_id):
    trials = getSessionJson(session_id)['trials']
    
    # dict of session name->id and date
    trialDict = {}
    for t in trials:
        trialDict[t['name']] = {'id':t['id'],'date':t['created_at']}
        
    return trialDict


def postCalibrationOptions(session_path,session_id,overwrite=False):
    calibration_id = getCalibrationTrialID(session_id)
    trial = getTrialJson(calibration_id)
   
    if trial['meta'] is None or overwrite == True:
        calibOptionsJsonPath = os.path.join(session_path,'Videos','calibOptionSelections.json')
        f = open(calibOptionsJsonPath)
        calibOptionsJson = json.load(f)
        f.close()
        data = {
                "meta":json.dumps({'calibration':calibOptionsJson})
            }
        trial_url = "{}{}{}/".format(API_URL, "trials/", calibration_id)
        r = makeRequestWithRetry('PATCH',
                                 trial_url,
                                 data=data,
                                 headers = {"Authorization": "Token {}".format(API_TOKEN)})
        
        if r.status_code == 200:
            print('Wrote calibration selections to metadata.')

def downloadVideosFromServer(session_id,trial_id, isDocker=True,
                             isCalibration=False, isStaticPose=False,
                             trial_name=None, session_name=None, 
                             session_path=None, benchmark=False):
    
    if session_name is None:
        session_name = session_id
    data_dir = getDataDirectory(isDocker)   
    if session_path is None:
        session_path = os.path.join(data_dir,'Data', session_name)  
    if not os.path.exists(session_path): 
        os.makedirs(session_path, exist_ok=True)
    
    trial = getTrialJson(trial_id)

    if trial_name is None:
        trial_name = trial['name']
    trial_name = trial_name.replace(' ', '')

    
    print("\nProcessing {}".format(trial_name))

    # The videos are not always organized in the same order. Here, we save
    # the order during the first trial processed in the session such that we
    # can use the same order for the other trials.
    if not benchmark:
        if not os.path.exists(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle')):
            mappingCamDevice = {}
            for k, video in enumerate(trial["videos"]):
                os.makedirs(os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name), exist_ok=True)
                video_path = os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name, trial_id + ".mov")
                download_file(video["video"], video_path)                
                mappingCamDevice[video["device_id"].replace('-', '').upper()] = k
            with open(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle'), 'wb') as handle:
                pickle.dump(mappingCamDevice, handle)
        else:
            with open(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle'), 'rb') as handle:
                mappingCamDevice = pickle.load(handle)            
            for video in trial["videos"]:            
                k = mappingCamDevice[video["device_id"].replace('-', '').upper()] 
                videoDir = os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name)
                os.makedirs(videoDir, exist_ok=True)
                video_path = os.path.join(videoDir, trial_id + ".mov")
                if not os.path.exists(video_path):
                    if video['video'] :
                        download_file(video["video"], video_path)
    
        # Import and save metadata
        sessionYamlPath = os.path.join(session_path, "sessionMetadata.yaml")
        if not os.path.exists(sessionYamlPath) or isStaticPose or isCalibration:
            if isCalibration: # subject parameters won't be entered yet
                session_desc = getMetadataFromServer(session_id,justCheckerParams = isCalibration)
            else: # subject parameters will be entered when capturing static pose
                session_desc = getMetadataFromServer(session_id)      
                
            # Load iPhone models.
            phoneModel= []
            for i,video in enumerate(trial["videos"]):    
                phoneModel.append(video['parameters']['model'])
            session_desc['iphoneModel'] = {'Cam' + str(i) : phoneModel[i] for i in range(len(phoneModel))}
        
            # Save metadata.
            with open(sessionYamlPath, 'w') as file:
                yaml.dump(session_desc, file)
                
    return trial_name


def deleteCalibrationFiles(session_path):
    calImagePath = os.path.join(session_path,'CalibrationImages')
    if os.path.exists(calImagePath):
        shutil.rmtree(calImagePath)
    
    # Delete camera directories
    camDirs = glob.glob(os.path.join(session_path,'Videos','Cam*'))
    
    # Find extrinsic Filename
    extrinsicFileFound = False
    if len(camDirs)>1 and os.path.exists(os.path.join(camDirs[0], 'InputMedia')):
        inputDir = os.path.join(camDirs[0], 'InputMedia')
        dirContents = os.listdir(inputDir)
        trialNames = [tName for tName in dirContents if os.path.isdir(os.path.join(inputDir,tName))]
        for tName in trialNames:
            if os.path.exists(os.path.join(inputDir,tName,'extrinsicImage0.png')):
                extrinsicTrialName = tName
                extrinsicFileFound = True
    
    for camDir in camDirs:
        extPath = os.path.join(camDir,'cameraIntrinsicsExtrinsics.pickle')
        if os.path.exists(extPath):
            os.remove(extPath)
        #Find extrinsic Filename
        if extrinsicFileFound:
            extFolder = os.path.join(camDir,'InputMedia',extrinsicTrialName)
            if os.path.isdir(extFolder):
                shutil.rmtree(extFolder)
                
def deleteStaticFiles(session_path,staticTrialName='neutral'):
        
    vidDir = os.path.join(session_path,'Videos')
    camDirs = glob.glob(os.path.join(vidDir,'Cam*'))
    markerDirs = glob.glob(os.path.join(session_path,'MarkerData'))
    openSimDir = os.path.join(session_path,'OpenSimData')
    
    # This is a hack, but os.walk doesn't work on attached server drives
    for camDir in camDirs:
        mediaDirs = glob.glob(os.path.join(camDir,'*'))
        for medDir in mediaDirs:
            try:
                shutil.rmtree(os.path.join(camDir,medDir,staticTrialName))
                _,camName = os.path.split()
                print('deleting ' + camName + '/' + medDir + '/' + staticTrialName)
            except:
                pass
            
    for mkrDir in markerDirs:
        mkrFiles = glob.glob(os.path.join(mkrDir,'*'))
        for mkrFile in mkrFiles:
            if staticTrialName in mkrFile:
                os.remove(mkrFile)
                _,fName = os.split(mkrFile)
                print('deleting '+ fName)
           
    if os.path.exists(openSimDir):
        shutil.rmtree(openSimDir) # Static will be the first opensim data saved, so this is safe
        print('deleting openSimDir')
    
    # # this works locally, but not on server drives. Saving in case we change storage
    # for root, dirList, fileList in os.walk(session_path + './'):
    #     for thisFile in fileList:
    #         print(thisFile)
    #         if (bool(regex.match(staticTrialName + '.trc', thisFile)) or 
    #             bool(regex.match(staticTrialName + '.mot', thisFile)) or 
    #             bool(regex.match(staticTrialName + '.sto', thisFile))):
    #             filePath = os.path.join(root,thisFile)
    #             os.remove(filePath)
    #             print('removing ' + thisFile)
        
    #     for thisDir in dirList:
    #         print(thisDir)
    #         if thisDir == staticTrialName:
    #             dirPath = os.path.join(root,thisDir)
    #             shutil.rmtree(dirPath)
    #             print('removing ' + thisDir)

def switchCalibrationForCamera(cam,trial_id,session_path):
    trialName = getTrialName(trial_id)
    camPath = os.path.join(session_path,'Videos',cam)
    trialPath = os.path.join(camPath,'InputMedia',trialName)
    
    # change Picture 
    src = os.path.join(trialPath,'extrinsicCalib_soln1.jpg')
    dest = os.path.join(session_path,'CalibrationImages','extrinsicCalib' + cam + '.jpg')
    if os.path.exists(dest):
        os.remove(dest)
    shutil.copyfile(src,dest)
    
    # change calibration parameters
    src = os.path.join(trialPath,'cameraIntrinsicsExtrinsics_soln1.pickle')
    dest = os.path.join(camPath,'cameraIntrinsicsExtrinsics.pickle')
    if os.path.exists(dest):
        os.remove(dest)
    shutil.copyfile(src,dest)    
    
                 
def getMetadataFromServer(session_id,justCheckerParams=False):
    
    defaultMetadataPath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'defaultSessionMetadata.yaml')
    session_desc = importMetadata(defaultMetadataPath)
    
    # Get session-specific metadata from api.
    session = getSessionJson(session_id) 
    if session['meta'] is not None:
        if not justCheckerParams:
            # Backward compatibility
            if 'subject' in session['meta']:
                session_desc["subjectID"] = session['meta']['subject']['id']
                session_desc["mass_kg"] = float(session['meta']['subject']['mass'])
                session_desc["height_m"] = float(session['meta']['subject']['height'])
                if 'gender' in session['meta']['subject']:
                    session_desc["gender_mf"] = getGendersDict().get(session['meta']['subject']['gender'])
                # Before implementing the subject feature, the posemodel was stored
                # in session['meta']['subject']. After implementing the subject
                # feature, the posemodel is stored in session['meta']['settings']
                # and there is no session['meta']['subject'].
                try:
                    session_desc["posemodel"] = session['meta']['subject']['posemodel']
                except:
                    session_desc["posemodel"] = 'openpose'
                # This might happen if openSimModel/augmentermodel/filterfrequency/scalingsetup was changed post data collection.
                if 'settings' in session['meta']:
                    try:
                        session_desc["openSimModel"] = session['meta']['settings']['openSimModel']
                    except:
                        session_desc["openSimModel"] = 'LaiUhlrich2022'
                    try:
                        session_desc["augmentermodel"] = session['meta']['settings']['augmentermodel']
                    except:
                        session_desc["augmentermodel"] = 'v0.2'
                    try:
                        session_desc["filterfrequency"] = session['meta']['settings']['filterfrequency']
                        if session_desc["filterfrequency"] != 'default':
                            session_desc["filterfrequency"] = float(session_desc["filterfrequency"])
                    except:
                        session_desc["filterfrequency"] = 'default'
                    try:
                        session_desc["scalingsetup"] = session['meta']['settings']['scalingsetup']
                    except:
                        session_desc["scalingsetup"] = 'upright_standing_pose'
            else:                
                subject_info = getSubjectJson(session['subject'])                
                session_desc["subjectID"] = subject_info['name']
                session_desc["mass_kg"] = subject_info['weight']
                session_desc["height_m"] = subject_info['height']
                session_desc["gender_mf"] = getGendersDict().get(subject_info['gender'])
                try:
                    session_desc["posemodel"] = session['meta']['settings']['posemodel']
                except:
                    session_desc["posemodel"] = 'openpose'
                try:
                    session_desc["openSimModel"] = session['meta']['settings']['openSimModel']
                except:
                    session_desc["openSimModel"] = 'LaiUhlrich2022'
                try:
                    session_desc["augmentermodel"] = session['meta']['settings']['augmentermodel']
                except:
                    session_desc["augmentermodel"] = 'v0.2'
                try:
                    session_desc["filterfrequency"] = session['meta']['settings']['filterfrequency']
                    if session_desc["filterfrequency"] != 'default':
                        session_desc["filterfrequency"] = float(session_desc["filterfrequency"])
                except:
                    session_desc["filterfrequency"] = 'default'
                try:
                    session_desc["scalingsetup"] = session['meta']['settings']['scalingsetup']
                except:
                    session_desc["scalingsetup"] = 'upright_standing_pose'

        if 'sessionWithCalibration' in session['meta'] and 'checkerboard' not in session['meta']:
            newSessionId = session['meta']['sessionWithCalibration']['id']
            session = getSessionJson(newSessionId)
                                   
        session_desc['checkerBoard']["squareSideLength_mm"] =  float(session['meta']['checkerboard']['square_size'])
        session_desc['checkerBoard']["black2BlackCornersWidth_n"] = int(session['meta']['checkerboard']['cols'])
        session_desc['checkerBoard']["black2BlackCornersHeight_n"] = int(session['meta']['checkerboard']['rows'])
        session_desc['checkerBoard']["placement"] = session['meta']['checkerboard']['placement']   
        

          
    else:
        print('Couldn''t find session metadata in API, using default metadata. May be issues.')
    
    return session_desc

def deleteResult(trial_id, tag=None,resultNum=None):
    # Delete specific result number, or all results with a specific tag, or all results if tag==None
    if resultNum != None:
        resultNums = [resultNum]
    elif tag != None:
        trial = getTrialJson(trial_id)
        resultNums = [r['id'] for r in trial['results'] if r['tag']==tag]
        
    elif tag == None: 
        trial = getTrialJson(trial_id)
        resultNums = [r['id'] for r in trial['results']]

    for rNum in resultNums:
        makeRequestWithRetry('DELETE',
                             API_URL + "results/{}/".format(rNum),
                             headers = {"Authorization": "Token {}".format(API_TOKEN)})
        
def deleteAllResults(session_id):

    session = getSessionJson(session_id)
    
    for trial in session['trials']:
        deleteResult(trial['id'])

def writeCalibrationOptionsToAPI(session_path,session_id,calibration_id=None,trialName = None):
    if calibration_id == None:
        calibration_id = getCalibrationTrialID(session_id)
    
    if trialName == None:
        trial = getTrialJson(calibration_id)
        trialName = trial['name']
    
    deleteResult(calibration_id, tag='camera_mapping')
    videoDir = os.path.join(session_path,'Videos')
    camDirs = glob.glob(os.path.join(videoDir,'Cam*'))
    mapPath = os.path.join(videoDir, 'mappingCamDevice.pickle')
    postFileToTrial(mapPath,calibration_id,'camera_mapping','all')
    
    tag = 'calibration_parameters_options'
    deleteResult(calibration_id, tag=tag)
    for camDir in camDirs:
        _,camName = os.path.split(camDir)
        calibDir = os.path.join(camDir,'InputMedia',trialName)        
        # Post both solutions
        for i in range(2):
            filePath = os.path.join(calibDir,'cameraIntrinsicsExtrinsics_soln{}.pickle'.format(i))
            device_id = camName+'_soln{}'.format(i)
            postFileToTrial(filePath,calibration_id,tag,device_id)    

def getCalibrationTrialID(session_id):
    session = getSessionJson(session_id)
    
    calib_ids = [t['id'] for t in session['trials'] if t['name'] == 'calibration']
                                                          
    if len(calib_ids)>0:
        calibID = calib_ids[-1]
    elif session['meta']['sessionWithCalibration']:
        calibID = getCalibrationTrialID(session['meta']['sessionWithCalibration']['id'])
    else:
        raise Exception('No calibration trial in session.')
    
    return calibID

def getNeutralTrialID(session_id):
    session = getSessionJson(session_id)
    
    neutral_ids = [t['id'] for t in session['trials'] if t['name'] == 'neutral']
    
    if len(neutral_ids)>0:
        neutralID = neutral_ids[-1]
    elif session['meta']['neutral_trial']:
        neutralID = session['meta']['neutral_trial']['id']
    else:
        raise Exception('No neutral trial in session.')
    
    return neutralID       

def postCalibration(session_id,session_path,calibTrialID=None):
    
    videoDir = os.path.join(session_path,'Videos')
    videoFolders = glob.glob(os.path.join(videoDir,'Cam*'))
        
    if calibTrialID == None:
        calibTrialID = getCalibrationTrialID(session_id)
    
    # remove 'calibration_parameters' in case they exist already.
    tag = 'calibration_parameters'
    deleteResult(calibTrialID, tag=tag)
    for vf in videoFolders:
        _, camName = os.path.split(vf)
        fPath = os.path.join(vf,'cameraIntrinsicsExtrinsics.pickle')
        deviceID = camName
        postFileToTrial(fPath,calibTrialID,'calibration_parameters',deviceID)
    
    return

def getCalibration(session_id,session_path,trial_type='dynamic',getCalibrationOptions=False):
    # look for calibration pickles on Django. If they are not there, then see if 
    # we need to do any switch calibration, then post the good calibration to django.
    calibration_id = getCalibrationTrialID(session_id)

    # Check if calibration has been posted to session
    trial = getTrialJson(calibration_id)
    calibResultTags = [res['tag'] for res in trial['results']]

    # download the mapping
    videoFolder = os.path.join(session_path,'Videos')
    os.makedirs(videoFolder, exist_ok=True)
    mapURL = trial['results'][calibResultTags.index('camera_mapping')]['media']
    mapLocalPath = os.path.join(videoFolder,'mappingCamDevice.pickle')
    download_file(mapURL,mapLocalPath)
    
    # download calibration parameters and switch if necessary.
    calibrationOptions = downloadAndSwitchCalibrationFromDjango(session_id,session_path,
                                                                calibTrialID=calibration_id,
                                                                getCalibrationOptions=getCalibrationOptions)
    
    # Post calibration if neutral trial. The posted parameters are no longer
    # used, but it is handy to know which ones were selected from both options.
    if trial_type == 'static':
        postCalibration(session_id,session_path,calibTrialID=calibration_id)   

    if getCalibrationOptions:
        return calibrationOptions                             

def downloadAndSwitchCalibrationFromDjango(session_id,session_path,calibTrialID = None,
                                           getCalibrationOptions=False):
    if calibTrialID == None:
        calibTrialID = getCalibrationTrialID(session_id)
    trial = getTrialJson(calibTrialID)
       
    calibURLs = {t['device_id']:t['media'] for t in trial['results'] if t['tag'] == 'calibration_parameters_options'}
    
    if 'meta' in trial.keys() and trial['meta'] is not None and 'calibration' in trial['meta'].keys() and trial['meta']['calibration']:
        calibDict = trial['meta']['calibration']
    else:
        print('No metadata for camera switching. Using first solution.')
        calibDict = {'Cam'+str(i):0 for i in range(len(trial['videos']))}
        
    for cam,calibNum in calibDict.items():
        camDir = os.path.join(session_path,'Videos',cam)
        os.makedirs(camDir,exist_ok=True)
        file_name = os.path.join(camDir,'cameraIntrinsicsExtrinsics.pickle')
        if calibNum == 0:
            download_file(calibURLs[cam+'_soln0'], file_name)
            print('Downloading calibration for ' + cam)
        elif calibNum == 1:
            download_file(calibURLs[cam+'_soln1'], file_name)                  
            print('Downloading alternate calibration camera for ' + cam)
    
    # If static trial and we are automatically selecting a calibration
    if getCalibrationOptions:
        tempPath = os.path.join(session_path,'tempCalib.pickle')
        calibrationOptions = {}
        for cam in calibDict.keys():
            calibrationOptions[cam] = []
            download_file(calibURLs[cam+'_soln0'], tempPath)
            calibrationOptions[cam].append(loadCameraParameters(tempPath))
            os.remove(tempPath)
            download_file(calibURLs[cam+'_soln1'], tempPath)
            calibrationOptions[cam].append(loadCameraParameters(tempPath))
            os.remove(tempPath)            
    
        return calibrationOptions
    else:
        return None
    
def changeSessionMetadata(session_ids,newMetaDict):

    if 'filterfrequency' in newMetaDict and newMetaDict['filterfrequency'] != 'default':
        if type(newMetaDict['filterfrequency']) is not str:
            newMetaDict['filterfrequency'] = str(newMetaDict['filterfrequency'])
        else:
            raise Exception('Filter frequency should be a number or default.')
        
    if 'datasharing' in newMetaDict:
        if newMetaDict['datasharing'] not in ['Share processed data and identified videos',
                                                'Share processed data and de-identified videos',
                                                'Share processed data',
                                                'Share no data']:
                raise Exception('datasharing is {} but should be one of the following options: "Share processed data and identified videos", "Share processed data and de-identified videos", "Share processed data", "Share no data".'.format(newMetaDict['datasharing']))
   
    meta_settings_fields = [
        'framerate', 'posemodel', 'openSimModel', 'augmentermodel',
        'filterfrequency', 'scalingsetup', 'camerastouse', 'sync_ver',
    ]

    for session_id in session_ids:
        session_url = "{}{}{}/".format(API_URL, "sessions/", session_id)
        
        # get metadata
        session = getSessionJson(session_id)
        existingMeta = session['meta']
        
        # Check if framerate is in metadata. If not, set to 60
        if 'framerate' not in existingMeta:
            framerate = 60
        else:
            framerate = existingMeta['framerate']
        if 'filterfrequency' in newMetaDict:
            if newMetaDict['filterfrequency'] != 'default':
                if float(newMetaDict['filterfrequency']) > framerate/2:
                    raise Exception('Filter frequency cannot exceed Nyquist frequency (here {}Hz).'.format(framerate/2))
                elif float(newMetaDict['filterfrequency']) < 0:
                    raise Exception('Filter frequency cannot be negative.')        
        
        # change metadata
        # Hack: wrong mapping between metadata and yaml
        # mass in metadata is mass_kg in yaml
        # height in metadata is height_m in yaml
        mapping_metadata = {'mass': 'mass_kg',
                            'height': 'height_m'}
        addedKey= {}
        for key in existingMeta.keys():
            if key in mapping_metadata:
                key_t = mapping_metadata[key]
            else:
                key_t = key
            if key_t in newMetaDict.keys():
                existingMeta[key] = newMetaDict[key_t]
                addedKey[key_t] = newMetaDict[key_t]
            if type(existingMeta[key]) is dict:
                for key2 in existingMeta[key].keys():                    
                    if key2 in mapping_metadata:
                        key_t = mapping_metadata[key2]
                    else:
                        key_t = key2                     
                    if key_t in newMetaDict.keys():
                        existingMeta[key][key2] = newMetaDict[key_t]
                        addedKey[key_t] = newMetaDict[key_t]
                        
        # add metadata if not existing (eg, specifying OpenSim model)
        # only entries in settings_fields below are supported.
        for newMeta in newMetaDict:
            if not newMeta in addedKey:
                print("Could not find {} in existing metadata, trying to add it.".format(newMeta))
                
                if newMeta in meta_settings_fields:
                    if 'settings' not in existingMeta:
                        existingMeta['settings'] = {}
                    existingMeta['settings'][newMeta] = newMetaDict[newMeta]
                    addedKey[newMeta] = newMetaDict[newMeta]
                    print("Added {}={} to settings in metadata".format(newMeta, newMetaDict[newMeta]))
                else:
                    print("Could not add {}={} to the metadata; not recognized".format(newMeta, newMetaDict[newMeta]))
        
        data = {"meta":json.dumps(existingMeta)}
        
        r = makeRequestWithRetry('PATCH',
                                 session_url,
                                 data=data,
                                 headers = {"Authorization": "Token {}".format(API_TOKEN)})
        
        if r.status_code !=200:
            print('Changing metadata failed.')
            
        # Also change this in the metadata yaml in neutral trial
        trial_id = getNeutralTrialID(session_id)
        trial = getTrialJson(trial_id)
        resultTags = [res['tag'] for res in trial['results']]
        
        metaPath = os.path.join(os.getcwd(),'sessionMetadata.yaml')
        if 'session_metadata' in resultTags:
            yamlURL = trial['results'][resultTags.index('session_metadata')]['media']
            download_file(yamlURL,metaPath)
            
            metaYaml = importMetadata(metaPath)
            
            addedKey= {}
            for key in metaYaml.keys():
                if key in newMetaDict.keys():
                    metaYaml[key] = newMetaDict[key]
                    addedKey[key] = newMetaDict[key]
                if type(metaYaml[key]) is dict:
                    for key2 in metaYaml[key].keys():
                        if key2 in newMetaDict.keys():
                            metaYaml[key][key2] = newMetaDict[key2] 
                            addedKey[key2] = newMetaDict[key2]
                            
            for newMeta in newMetaDict:
                if not newMeta in addedKey:
                    print("Could not find {} in existing yaml, adding it.".format(newMeta))               
                    metaYaml[newMeta] = newMetaDict[newMeta]
                            
            with open(metaPath, 'w') as file:
                yaml.dump(metaYaml, file)
                
            deleteResult(trial_id, tag='session_metadata')
            postFileToTrial(metaPath,trial_id,tag='session_metadata',device_id='all')
            os.remove(metaPath)
        
def makeSessionPublic(session_id,publicStatus=True):
    
    session_url = "{}{}{}/".format(API_URL, "sessions/", session_id)
    
    data = {
            "public":publicStatus
        }
    
    r = makeRequestWithRetry('PATCH',
                             session_url,
                             data=data,
                             headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
    if r.status_code == 200:
        print('Successfully made ' + session_id + ' public.')
    else:
        print('server resp was ' + str(r.status_code))
        
    return

        
def postMotionData(trial_id,session_path,trial_name=None,isNeutral=False,
                   poseDetector='OpenPose', resolutionPoseDetection='default',
                   bbox_thr=0.8):
    
    if trial_name == None:
        trial_name = getTrialJson(trial_id)['id']

    if poseDetector.lower() == 'openpose':
        pklDir = os.path.join("OutputPkl_" + resolutionPoseDetection, trial_name)
    elif poseDetector.lower() == 'hrnet':
        pklDir = os.path.join("OutputPkl_mmpose_" + str(bbox_thr), trial_name)
    else:
        raise Exception('Unknown pose detector: {}'.format(poseDetector))
        
    markerDir = os.path.join(session_path,'MarkerData','PostAugmentation')
        
    # post settings
    deleteResult(trial_id, tag='main_settings')
    mainSettingsPath = os.path.join(markerDir,'Settings_{}.yaml'.format(trial_id))
    postFileToTrial(mainSettingsPath,trial_id,tag='main_settings',device_id='all')
        
    # post pose pickles
    # If we parallelize this, this will be redundant, and we will want to delete this posting of pickles
    deleteResult(trial_id, tag='pose_pickle')
    camDirs = glob.glob(os.path.join(session_path,'Videos','Cam*'))
    for camDir in camDirs:
        outputPklFolder = os.path.join(camDir,pklDir)
        pickle_files = glob.glob(os.path.join(outputPklFolder,'*_pp.pkl'))
        if pickle_files:
            pklPath = pickle_files[0]
            _,camName = os.path.split(camDir)
            postFileToTrial(pklPath,trial_id,tag='pose_pickle',device_id=camName)
        
    # post marker data
    deleteResult(trial_id, tag='marker_data')
    markerPath = os.path.join(markerDir,trial_id + '.trc')
    postFileToTrial(markerPath,trial_id,tag='marker_data',device_id='all')
    
    if isNeutral:
        # post model
        deleteResult(trial_id, tag='opensim_model')
        modelFolder = os.path.join(session_path,'OpenSimData','Model')
        modelPath = glob.glob(modelFolder + '/*_scaled.osim')[0]
        postFileToTrial(modelPath,trial_id,tag='opensim_model',device_id='all')
        
        # post metadata
        deleteResult(trial_id, tag='session_metadata')
        metadataPath = os.path.join(session_path,'sessionMetadata.yaml')
        postFileToTrial(metadataPath,trial_id,tag='session_metadata',device_id='all')
    else:
        # post ik data
        deleteResult(trial_id, tag='ik_results')
        ikPath = os.path.join(session_path,'OpenSimData','Kinematics',trial_id + '.mot')
        postFileToTrial(ikPath,trial_id,tag='ik_results',device_id='all')
        
    return

def getMotionData(trial_id, session_path, 
                  simplePath=False, 
                  include_pose_pickles=False):
    trial = getTrialJson(trial_id)
    trial_name = trial['name']
    resultTags = [res['tag'] for res in trial['results']]

    # get marker data
    if 'marker_data' in resultTags:
        markerFolder = os.path.join(session_path,'MarkerData','PostAugmentation',trial_name)
        if simplePath:
            markerFolder = os.path.join(session_path,'MarkerData')
        markerPath = os.path.join(markerFolder,trial_name + '.trc')
        os.makedirs(markerFolder, exist_ok=True)
        markerURL = trial['results'][resultTags.index('marker_data')]['media']
        download_file(markerURL,markerPath)
    
    # get IK data
    if 'ik_results' in resultTags:
        ikFolder = os.path.join(session_path,'OpenSimData','Kinematics')
        ikPath = os.path.join(ikFolder,trial_name + '.mot')
        os.makedirs(ikFolder, exist_ok=True)
        ikURL = trial['results'][resultTags.index('ik_results')]['media']
        download_file(ikURL,ikPath)
    
    # get pose pickles
    if include_pose_pickles and 'pose_pickle' in resultTags:
        # metadata needed for pose pickle folder naming
        main_settings = getMainSettings(trial_id)
        poseDetector = main_settings['poseDetector']

        # sometimes mmpose is used instead of hrnet
        if poseDetector.lower() == 'mmpose':
            poseDetector = 'hrnet'

        # infer pose detection from main settings to get correct folders
        if poseDetector.lower() == 'openpose':
            getPosePickles(trial_id, session_path,
                           poseDetector=poseDetector,
                           resolutionPoseDetection=main_settings['resolutionPoseDetection'])

        elif poseDetector.lower() == 'hrnet':
            # shared check with `checkAndGetPosePickles()`
            if 'bbox_thr' in main_settings:
                bbox_thr = main_settings['bbox_thr']
            else:
                # There was a bug in main, where bbox_thr was not saved in main_settings.yaml.
                # Since there is in practice no option to change bbox_thr in the GUI, we can
                # assume that the default value was used.
                bbox_thr = 0.8

            getPosePickles(trial_id, session_path,
                           poseDetector=poseDetector,
                           bbox_thr=bbox_thr)
        else:
            print(f'pose pickles found, but specified pose detector  \
                    {poseDetector} does not exist. skipping pose pickle \
                    download')
        
    return
        
def getModelAndMetadata(session_id,session_path,simplePath=False):
    neutral_id = getNeutralTrialID(session_id)
    trial = getTrialJson(neutral_id)
    resultTags = [res['tag'] for res in trial['results']]
    
    # get metadata
    metadataPath = os.path.join(session_path,'sessionMetadata.yaml')
    if not os.path.exists(metadataPath) :
        metadataURL = trial['results'][resultTags.index('session_metadata')]['media']
        download_file(metadataURL, metadataPath)
    
    # get model if does not exist
    modelURL = trial['results'][resultTags.index('opensim_model')]['media']
    modelName = modelURL[modelURL.rfind('-')+1:modelURL.rfind('?')]
    modelFolder = os.path.join(session_path,'OpenSimData','Model')
    if simplePath:
       modelFolder = os.path.join(session_path,'OpenSimData','Model')
    modelPath = os.path.join(modelFolder,modelName)
    if not os.path.exists(modelPath):
        os.makedirs(modelFolder, exist_ok=True)
        download_file(modelURL, modelPath)
        
    return
    
def postFileToTrial(filePath,trial_id,tag,device_id):
        
    # get S3 link
    data = {'fileName':os.path.split(filePath)[1]}
    response = makeRequestWithRetry('GET',
                                    API_URL + "sessions/null/get_presigned_url/",
                                    data=data)
    r = response.json()
    
    # upload to S3
    files = {'file': open(filePath, 'rb')}
    makeRequestWithRetry('POST',
                         r['url'],
                         data=r['fields'],
                         files=files)
    files["file"].close()

    # post link to and data to results   
    data = {
        "trial": trial_id,
        "tag": tag,
        "device_id" : device_id,
        "media_url" : r['fields']['key']
    }
    
    rResult = makeRequestWithRetry('POST',
                                   API_URL + "results/", 
                                   data=data,
                                   headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
    if rResult.status_code != 201:
        print('server response was + ' + str(r.status_code))
    else:
        print('Result posted to S3.')
    
    return

def getSyncdVideos(trial_id,session_path):
    trial = getTrialJson(trial_id)
    trial_name = trial['name']
    
    if trial['results']:
        for result in trial['results']:
            if result['tag'] == 'video-sync':
                url = result['media']
                cam,suff = os.path.splitext(url[url.rfind('_')+1:])
                lastIdx = suff.find('?') 
                if lastIdx >0:
                    suff = suff[:lastIdx]
                
                syncVideoPath = os.path.join(session_path,'Videos',cam,'InputMedia',trial_name,trial_name + '_sync' + suff)
                download_file(url,syncVideoPath)

def getPosePickles(trial_id,session_path, poseDetector='OpenPose', 
                   resolutionPoseDetection='default', bbox_thr=0.8):
    trial = getTrialJson(trial_id)
    trial_name = trial['name']

    if poseDetector.lower() == 'openpose':
        pklDir = os.path.join("OutputPkl_" + resolutionPoseDetection, trial_name)
    elif poseDetector.lower() == 'hrnet':
        pklDir = os.path.join("OutputPkl_mmpose_" + str(bbox_thr), trial_name)
    else:
        raise Exception('Unknown pose detector: {}'.format(poseDetector))
    
    trialPrefix = trial_id + "_rotated_pp.pkl"
    
    if trial['results']:
        for result in trial['results']:
            if result['tag'] == 'pose_pickle':
                url = result['media']                
                cam = result['device_id']
                posePickleDir = os.path.join(session_path,'Videos',cam,pklDir)
                os.makedirs(posePickleDir,exist_ok=True)
                posePicklePath = os.path.join(posePickleDir,trialPrefix)
                download_file(url,posePicklePath)

def checkAndGetPosePickles(trial_id, session_path, poseDetector, resolutionPoseDetection, bbox_thr):
    # Check if the pose pickles for that set of settings exist.
    # Load main_settings yaml.
    main_settings = getMainSettings(trial_id)
    if 'poseDetector' in main_settings:
        usedPoseDetector = main_settings['poseDetector']
        if poseDetector.lower() == 'openpose':
            if 'resolutionPoseDetection' in main_settings:
                usedResolution = main_settings['resolutionPoseDetection']
                if usedPoseDetector.lower() == poseDetector.lower() and usedResolution == resolutionPoseDetection:
                    print('The pose pickles for {} {} already exist in the database. We will download them to avoid re-running pose estimation'.format(poseDetector, resolutionPoseDetection))
                    getPosePickles(trial_id,session_path, poseDetector=poseDetector, resolutionPoseDetection=resolutionPoseDetection)
                else:
                    print('The pose pickles in the database are for {} {}, but you are now using {} {}. We will re-run pose estimation'.format(usedPoseDetector, usedResolution, poseDetector, resolutionPoseDetection))
            else:
                print('It is unclear which settings were used for pose estimation. We will re-run pose estimation')
        elif poseDetector.lower() == 'hrnet':
            # Hack: hrnet is sometimes called mmpose
            if usedPoseDetector.lower() == 'mmpose':
                usedPoseDetector = 'hrnet'
            if 'bbox_thr' in main_settings:
                usedBbox_thr = main_settings['bbox_thr']
            else:
                # There was a bug in main, where bbox_thr was not saved in main_settings.yaml.
                # Since there is in practice no option to change bbox_thr in the GUI, we can
                # assume that the default value was used.
                usedBbox_thr = 0.8
            if usedPoseDetector.lower() == poseDetector.lower() and usedBbox_thr == bbox_thr:
                print('The pose pickles for {} {} already exist in the database. We will download them to avoid re-running pose estimation'.format(poseDetector, bbox_thr))
                getPosePickles(trial_id,session_path, poseDetector=poseDetector, bbox_thr=bbox_thr)
            else:
                print('The pose pickles in the database are for {} {}, but you are now using {} {}. We will re-run pose estimation'.format(usedPoseDetector, usedBbox_thr, poseDetector, bbox_thr))
        else:
            print('It is unclear which settings were used for pose estimation. We will re-run pose estimation')
    else:
        print('It is unclear which settings were used for pose estimation. We will re-run pose estimation')

def getMainSettings(trial_id):
    trial = getTrialJson(trial_id)
    if len(trial['results'])>1:
        for result in trial['results']:
            if result['tag'] == 'main_settings':
                url = result['media']
                # Load yaml file
                try:
                    with urllib.request.urlopen(url) as response:
                        yaml_content = response.read()
                        data = yaml.safe_load(yaml_content)
                        return data
                except Exception as e:
                    print("An error occurred:", e)
                    return {}  # Return an empty dictionary in case of an error
    else:
        return {}
        
def downloadAndZipSession(session_id,deleteFolderWhenZipped=True,isDocker=True,
                          writeToDjango=False,justDownload=False,data_dir=None,
                          useSubjectNameFolder=False,
                          include_pose_pickles=False):
    
    session = getSessionJson(session_id)
    
    if data_dir is None:
        data_dir = os.path.join(getDataDirectory(isDocker=isDocker),'Data')
    if useSubjectNameFolder:
        folderName = session['name']
    else:
        folderName = session_id
    session_path = os.path.join(data_dir,folderName)
    
    calib_id = getCalibrationTrialID(session_id)
    neutral_id = getNeutralTrialID(session_id)
    dynamic_ids = [t['id'] for t in session['trials'] if (t['name'] != 'calibration' and t['name'] !='neutral')]
       
    # Calibration
    downloadVideosFromServer(session_id,calib_id,isDocker=isDocker,
                         isCalibration=True,isStaticPose=False) 
    getCalibration(session_id,session_path)
    
    # Neutral
    getModelAndMetadata(session_id,session_path)
    getMotionData(neutral_id,session_path,include_pose_pickles=include_pose_pickles)
    downloadVideosFromServer(session_id,neutral_id,isDocker=isDocker,
                     isCalibration=False,isStaticPose=True,session_path=session_path)
    getSyncdVideos(neutral_id,session_path)

    # Dynamic
    for dynamic_id in dynamic_ids:
        getMotionData(dynamic_id,session_path,include_pose_pickles=include_pose_pickles)
        downloadVideosFromServer(session_id,dynamic_id,isDocker=isDocker,
                 isCalibration=False,isStaticPose=False,session_path=session_path)
        getSyncdVideos(dynamic_id,session_path)

   
    if not justDownload:
        # Zip   
        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file), 
                               os.path.relpath(os.path.join(root, file), 
                                               os.path.join(path, '..')))
        
        session_zip = '{}.zip'.format(session_path)
    
        if os.path.isfile(session_zip):
            os.remove(session_zip)
      
        zipf = zipfile.ZipFile(session_zip, 'w', zipfile.ZIP_DEFLATED)
        zipdir(session_path, zipf)
        zipf.close()
        
        # write zip as a result to last trial for now
        if writeToDjango:
            postFileToTrial(session_zip,dynamic_ids[-1],tag='session_zip',device_id='all')
        
        if deleteFolderWhenZipped:
            if os.path.exists(session_path):
                shutil.rmtree(session_path)
            if os.path.exists(session_zip):
                os.remove(session_zip)
    
    return
#test session
# downloadAndZipSession('a24a895a-aa62-4403-bd9e-cf637ac02eb6',deleteFolderWhenZipped=False,isDocker=False)


def numpy2TRC(f, data, headers, fc=50.0, t_start=0.0, units="m"):
    
    header_mapping = {}
    for count, header in enumerate(headers):
        header_mapping[count+1] = header 
        
    # Line 1.
    f.write('PathFileType  4\t(X/Y/Z) %s\n' % os.getcwd())
    
    # Line 2.
    f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\t'
                'Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
    
    num_frames=data.shape[0]
    num_markers=len(header_mapping.keys())
    
    # Line 3.
    f.write('%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n' % (
            fc, fc, num_frames,
            num_markers, units, fc,
            1, num_frames))
    
    # Line 4.
    f.write("Frame#\tTime\t")
    for key in sorted(header_mapping.keys()):
        f.write("%s\t\t\t" % format(header_mapping[key]))

    # Line 5.
    f.write("\n\t\t")
    for imark in np.arange(num_markers) + 1:
        f.write('X%i\tY%s\tZ%s\t' % (imark, imark, imark))
    f.write('\n')
    
    # Line 6.
    f.write('\n')

    for frame in range(data.shape[0]):
        f.write("{}\t{:.8f}\t".format(frame+1,(frame)/fc+t_start)) # opensim frame labeling is 1 indexed

        for key in sorted(header_mapping.keys()):
            f.write("{:.5f}\t{:.5f}\t{:.5f}\t".format(data[frame,0+(key-1)*3], data[frame,1+(key-1)*3], data[frame,2+(key-1)*3]))
        f.write("\n")
        
def numpy2storage(labels, data, storage_file):
    
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"
    
    f = open(storage_file, 'w')
    f.write('name %s\n' %storage_file)
    f.write('datacolumns %d\n' %data.shape[1])
    f.write('datarows %d\n' %data.shape[0])
    f.write('range %f %f\n' %(np.min(data[:, 0]), np.max(data[:, 0])))
    f.write('endheader \n')
    
    for i in range(len(labels)):
        f.write('%s\t' %labels[i])
    f.write('\n')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' %data[i, j])
        f.write('\n')
        
    f.close() 
      
    
def lowpassFilter(inputData, filtFreq, order=4):
    # Input is an array of nSteps x (nMeasures +1) because time is the first column
    time = inputData[:,0]
    fs=1/np.mean(np.diff(time))
    wn = filtFreq/(fs/2)
    sos = signal.butter(order/2,wn,btype='low',output='sos')
    inputDataFilt = signal.sosfiltfilt(sos,inputData[:,1:],axis=0)    
    data = np.concatenate((np.expand_dims(time,1), inputDataFilt), axis=1)

    return data

        
def TRC2numpy(pathFile, markers,rotation=None):
    # rotation is a dict, eg. {'y':90} with axis, angle for rotation
    
    trc_file = utilsDataman.TRCFile(pathFile)
    time = trc_file.time
    num_frames = time.shape[0]
    data = np.zeros((num_frames, len(markers)*3))
    
    if rotation != None:
        for axis,angle in rotation.items():
            trc_file.rotate(axis,angle)
    for count, marker in enumerate(markers):
        data[:,3*count:3*count+3] = trc_file.marker(marker)    
    this_dat = np.empty((num_frames, 1))
    this_dat[:, 0] = time
    data_out = np.concatenate((this_dat, data), axis=1)
    
    return data_out

def getOpenPoseMarkerNames():
    
    markerNames = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                   "LShoulder", "LElbow", "LWrist", "midHip", "RHip",
                   "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                   "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
                   "LHeel", "RBigToe", "RSmallToe", "RHeel"]
    
    return markerNames

def getOpenPoseFaceMarkers():
    
    faceMarkerNames = ['Nose', 'REye', 'LEye', 'REar', 'LEar']
    markerNames = getOpenPoseMarkerNames()
    idxFaceMarkers = [markerNames.index(i) for i in faceMarkerNames]
    
    return faceMarkerNames, idxFaceMarkers

def getMMposeMarkerNames():
    
    markerNames = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", 
                   "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
                   "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
                   "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe",
                   "RHeel"]        
    
    return markerNames


def rewriteVideos(inputPath,startFrame,nFrames,frameRate,outputDir=None,
                  imageScaleFactor = .5,outputFileName=None):
        
    inputDir, vidName = os.path.split(inputPath)
    vidName, vidExt = os.path.splitext(vidName)

    if outputFileName is None:
        outputFileName = vidName + '_sync' + vidExt
    if outputDir is not None:
        outputFullPath = os.path.join(outputDir, outputFileName)
    else:
        outputFullPath = os.path.join(inputDir, outputFileName)
      
    imageScaleArg = '' # None if want to keep image size the same
    maintainQualityArg = '-acodec copy -vcodec copy'
    if imageScaleFactor is not None:
        imageScaleArg = '-vf scale=iw/{:.0f}:-1'.format(1/imageScaleFactor)
        maintainQualityArg = ''

    startTime = startFrame/frameRate

    # We need to replace double space to single space for split to work
    # That's a bit hacky but works for now. (TODO)
    ffmpegCmd = 'ffmpeg -loglevel error -y -ss {:.3f} -i "{}" {} -vframes {:.0f} {} "{}"'.format(
                startTime, inputPath, maintainQualityArg, 
                nFrames, imageScaleArg, outputFullPath).rstrip().replace("  ", " ")

    import shlex
    subprocess.run(shlex.split(ffmpegCmd))
    
    return

# %%  Found here: https://github.com/chrisdembia/perimysium/ => thanks Chris
def storage2numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
            skip_header=skip_header)

    return data

def storage2df(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out
	
def getIK(storage_file, joints, degrees=False):
    # Extract data
    data = storage2numpy(storage_file)
    Qs = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, joint in enumerate(joints):  
        if ((joint == 'pelvis_tx') or (joint == 'pelvis_ty') or 
            (joint == 'pelvis_tz')):
            Qs.insert(count + 1, joint, data[joint])         
        else:
            if degrees == True:
                Qs.insert(count + 1, joint, data[joint])                  
            else:
                Qs.insert(count + 1, joint, data[joint] * np.pi / 180)              
            
    # Filter data    
    fs=1/np.mean(np.diff(Qs['time']))    
    fc = 6  # Cut-off frequency of the filter
    order = 4
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(order/2, w, 'low')  
    output = signal.filtfilt(b, a, Qs.loc[:, Qs.columns != 'time'], axis=0, 
                             padtype='odd', padlen=3*(max(len(b),len(a))-1))    
    output = pd.DataFrame(data=output, columns=joints)
    QsFilt = pd.concat([pd.DataFrame(data=data['time'], columns=['time']), 
                        output], axis=1)    
    
    return Qs, QsFilt

# %% Markers for augmenters.
def getOpenPoseMarkers_fullBody():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers

def getMMposeMarkers_fullBody():

    # Here we replace RSmallToe_mmpose and LSmallToe_mmpose by RSmallToe and
    # LSmallToe, since this is how they are named in the triangulation.
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe", 
        "RElbow", "LElbow", "RWrist", "LWrist"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers        

def getOpenPoseMarkers_lowerExtremity():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers

# Different order of markers compared to getOpenPoseMarkers_lowerExtremity 
def getOpenPoseMarkers_lowerExtremity2():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]

    response_markers = [
        'r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study',
        'L.PSIS_study', 'r_knee_study', 'r_mknee_study', 
        'r_ankle_study', 'r_mankle_study', 'r_toe_study', 
        'r_5meta_study', 'r_calc_study', 'L_knee_study', 
        'L_mknee_study', 'L_ankle_study', 'L_mankle_study',
        'L_toe_study', 'L_calc_study', 'L_5meta_study', 
        'r_shoulder_study', 'L_shoulder_study', 'C7_study', 
        'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
        'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
        'r_sh1_study', 'r_sh2_study', 'r_sh3_study', 'L_sh1_study',
        'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']

    return feature_markers, response_markers

def getMMposeMarkers_lowerExtremity():

    # Here we replace RSmallToe_mmpose and LSmallToe_mmpose by RSmallToe and
    # LSmallToe, since this is how they are named in the triangulation.
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]

    return feature_markers, response_markers

def getMarkers_upperExtremity_pelvis():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RElbow", "LElbow",
        "RWrist", "LWrist"]

    response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers

def getMarkers_upperExtremity_noPelvis():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist",
        "LWrist"]

    response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers

# Different order of markers compared to getMarkers_upperExtremity_noPelvis.
def getMarkers_upperExtremity_noPelvis2():

    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist",
        "LWrist"]

    response_markers = ["r_lelbow_study", "r_melbow_study", "r_lwrist_study",
                        "r_mwrist_study", "L_lelbow_study", "L_melbow_study",
                        "L_lwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def getVideoExtension(pathFileWithoutExtension):
    
    pathVideoDir = os.path.split(pathFileWithoutExtension)[0]
    videoName = os.path.split(pathFileWithoutExtension)[1]
    for file in os.listdir(pathVideoDir):
        if videoName == file.rsplit('.', 1)[0]:
            extension = '.' + file.rsplit('.', 1)[1]
            
    return extension

# check how much time has passed since last status check
def checkTime(t,minutesElapsed=30):
    t2 = time.localtime()
    return (t2.tm_hour - t.tm_hour) * 3600 + (t2.tm_min - t.tm_min)*60 + (t2.tm_sec - t.tm_sec) >= minutesElapsed*60

# check for trials with certain status
def checkForTrialsWithStatus(status,hours=9999999,relativeTime='newer'):
    
    # get trials with statusOld
    params = {'status':status,
              'hoursSinceUpdate':hours,
              'justNumber':1,
              'relativeTime':relativeTime}
    
    response = makeRequestWithRetry('GET',
                                    API_URL+"trials/get_trials_with_status/",
                                    params=params,
                                    headers = {"Authorization": "Token {}".format(API_TOKEN)})
    r = response.json()
    
    return r['nTrials']

# send status email
def sendStatusEmail(message=None,subject=None):
    import smtplib, ssl
    from utilsAPI import getStatusEmails
    from email.message import EmailMessage
    
    emailInfo = getStatusEmails()
    if emailInfo is None:
        return('No email info or wrong email info in env file.')
    
    if 'ip' in emailInfo:
        ip = emailInfo['ip']
        message = message + ' IP: ' + ip
       
    if message is None:
        message = "A backend server is down and has been stopped."
    if subject is None:
        subject = "OpenCap backend server down"
        
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"  
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(emailInfo['fromEmail'], emailInfo['password'])
        for toEmail in emailInfo['toEmails']:
            # server.(emailInfo['fromEmail'], toEmail, message)
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = emailInfo['fromEmail']
            msg['To'] = toEmail
            msg.set_content(message)
            server.send_message(msg)
        server.quit()

def checkResourceUsage(stop_machine_and_email=True):
    import psutil
    
    resourceUsage = {}
    
    memory_info = psutil.virtual_memory()
    resourceUsage['memory_gb'] = memory_info.used / (1024 ** 3)
    resourceUsage['memory_perc'] = memory_info.percent 

    # Get the disk usage information of the root directory
    disk_usage = psutil.disk_usage('/')

    # Get the percentage of disk usage
    resourceUsage['disk_gb'] = disk_usage.used / (1024 ** 3)
    resourceUsage['disk_perc'] = disk_usage.percent
    
    if stop_machine_and_email and resourceUsage['disk_perc'] > 95:
            
        message = "Disc is full on an OpenCap backend machine. It has been stopped. Data: " \
                            + json.dumps(resourceUsage)
        sendStatusEmail(message=message)
        
        raise Exception('Not enough available disc space. Stopped.')
    
    return resourceUsage

def checkCudaTF():
    import tensorflow as tf

    if tf.config.list_physical_devices('GPU'):
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Found {len(gpus)} GPU(s).")
        for gpu in gpus:
            print(f"GPU: {gpu.name}")
    else:
        message = "Cuda check failed on an OpenCap backend machine. It has been stopped."
        sendStatusEmail(message=message)
        raise Exception("No GPU detected. Exiting.")

def writeToJsonLog(path, new_dict, max_entries=1000, indent=2):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append(new_dict)

    while len(data) > max_entries:
        data.pop(0)

    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)

def writeToErrorLog(path, session_id, trial_id, error, stack, max_entries=1000):
    error_entry = {
        'session_id': session_id,
        'trial_id': trial_id,
        'datetime': datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        'error': str(error),
        'stack': stack
    }
    writeToJsonLog(path, error_entry, max_entries)

# %% Some functions for loading subject data

def getSubjectNumber(subjectName):
    response = makeRequestWithRetry('GET',
                                    API_URL + "subjects/",
                                    headers = {"Authorization": "Token {}".format(API_TOKEN)})
    subjects = response.json()
    sNum = [s['id'] for s in subjects if s['name'] == subjectName]
    if len(sNum)>1:
        print(len(sNum) + ' subjects with the name ' + subjectName + '. Will use the first one.')   
    elif len(sNum) == 0:
        raise Exception('no subject found with this name.')
        
    return sNum[0]

def getUserSessions():
    response = makeRequestWithRetry('GET',
                                    API_URL + "sessions/valid/",
                                    headers = {"Authorization": "Token {}".format(API_TOKEN)})
    sessionJson = response.json()
    return sessionJson

def getSubjectSessions(subjectName):
    sessions = getUserSessions()
    subNum = getSubjectNumber(subjectName)
    sessions2 = [s for s in sessions if (s['subject'] == subNum)]
    
    return sessions2

def getTrialNames(session):
    trialNames = [t['name'] for t in session['trials']]
    return trialNames

def findSessionWithTrials(subjectTrialNames,trialNames):
    hasTrials = []
    for trials in trialNames:
        hasTrials.append(None)
        for i,sTrials in enumerate(subjectTrialNames):
            if all(elem in sTrials for elem in trials):
                hasTrials[-1] = i
                break
            
    return hasTrials

def get_entry_with_largest_number(trialList):
    max_entry = None
    max_number = float('-inf')

    for entry in trialList:
        # Extract the number from the string
        try:
            number = int(entry.split('_')[-1])
            if number > max_number:
                max_number = number
                max_entry = entry
        except ValueError:
            continue

    return max_entry

def getGendersDict():
    genders_dict = {
          "woman": "Woman",
          "man": "Man",
          "transgender": "Transgender",
          "non-binary": "Non-Binary/Non-Conforming",
          "prefer-not-respond": "Prefer not to respond",
        }
    return genders_dict

# Get local client info and update

def getCommitHash():
    """Get the git commit hash stored in the environment variable
    GIT_COMMIT_HASH. This is assumed to be set in the Docker build
    step. If not set, returns Null (default value for os.getenv())
    """
    return os.getenv('GIT_COMMIT_HASH')

def getHostname():
    """Get the hostname. For a docker container, this is the container ID."""
    return socket.gethostname()

def postLocalClientInfo(trial_url):
    """Given a trial_url, updates the Trial fields for 
    'git_commit' and 'hostname'.
    """
    data = {
            "git_commit": getCommitHash(),
            "hostname": getHostname()
        }
    r = makeRequestWithRetry('PATCH',
                             trial_url,
                             data=data,
                             headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
    return r

def postProcessedDuration(trial_url, duration):
    """Given a trial_url and duration (formed from difference in datetime
    objects), updates the Trial field for 'processed_duration'.
    """
    data = {
        "processed_duration": duration
    }
    r = makeRequestWithRetry('PATCH',
                             trial_url,
                             data=data,
                             headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
    return r

# utils for common HTTP requests
def makeRequestWithRetry(method, url,
                         headers=None, data=None, params=None, files=None,
                         retries=5, backoff_factor=1):
    """
    Makes an HTTP request with retry logic and returns the Response object.

    Args:
        method (str): HTTP method (e.g., 'GET', 'POST', 'PUT', etc.) as used in 
            requests.Session().request()
        url (str): The endpoint URL.
        headers (dict): Headers to include in the request.
        data (dict): Data to send in the request body.
        params (dict): URL query parameters.
        retries (int): Number of retry attempts.
        backoff_factor (float): Backoff factor for exponential delays.

    Returns:
        requests.Response: The response object for further processing.
    """
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={'DELETE', 'GET', 'POST', 'PUT', 'PATCH'}
    )

    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    with requests.Session() as session:
        session.mount("https://", adapter)
        response = session.request(method,
                                    url,
                                    headers=headers,
                                    data=data,
                                    params=params,
                                    files=files)
    response.raise_for_status()
    return response

'''
    ---------------------------------------------------------------------------
    OpenCap processing: utils.py
    ---------------------------------------------------------------------------

    Copyright 2022 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''
def download_file(url, file_name):
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

def get_session_json(session_id):
    resp = requests.get(
        API_URL + "sessions/{}/".format(session_id),
        headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
    if resp.status_code == 500:
        raise Exception('No server response. Likely not a valid session id.')
        
    sessionJson = resp.json()
    if 'trials' not in sessionJson.keys():
        raise Exception('This session is not in your username, nor is it public. You do not have access.')
    
    # Sort trials by time recorded.
    def get_created_at(trial):
        return trial['created_at']
    sessionJson['trials'].sort(key=get_created_at)
    
    return sessionJson
    
# Returns a list of all sessions of the user.
def get_user_sessions():
    sessions = requests.get(
        API_URL + "sessions/valid/", 
        headers = {"Authorization": "Token {}".format(API_TOKEN)}).json()
    
    return sessions

# Returns a list of all sessions of the user.
# TODO: this also contains public sessions of other users.
def get_user_sessions_all(user_token=API_TOKEN):
    sessions = requests.get(
        API_URL + "sessions/", 
        headers = {"Authorization": "Token {}".format(user_token)}).json()
    
    return sessions

# Returns a list of all subjects of the user.
def get_user_subjects(user_token=API_TOKEN):
    subjects = requests.get(
            API_URL + "subjects/", 
            headers = {"Authorization": "Token {}".format(user_token)}).json()
    
    return subjects

# Returns a list of all sessions of a subject.
def get_subject_sessions(subject_id, user_token=API_TOKEN):
    sessions = requests.get(
        API_URL + "subjects/{}/".format(subject_id),
        headers = {"Authorization": "Token {}".format(user_token)}).json()['sessions']
    
    return sessions

def get_trial_json(trial_id):
    trialJson = requests.get(
        API_URL + "trials/{}/".format(trial_id),
        headers = {"Authorization": "Token {}".format(API_TOKEN)}).json()
    
    return trialJson

def get_neutral_trial_id(session_id):
    session = get_session_json(session_id)    
    neutral_ids = [t['id'] for t in session['trials'] if t['name']=='neutral']
    
    if len(neutral_ids)>0:
        neutralID = neutral_ids[-1]
    elif session['meta']['neutral_trial']:
        neutralID = session['meta']['neutral_trial']['id']
    else:
        raise Exception('No neutral trial in session.')
    
    return neutralID 
 

def get_calibration_trial_id(session_id):
    session = get_session_json(session_id)
    
    calib_ids = [t['id'] for t in session['trials'] if t['name'] == 'calibration']
                                                          
    if len(calib_ids)>0:
        calibID = calib_ids[-1]
    elif session['meta']['sessionWithCalibration']:
        calibID = get_calibration_trial_id(session['meta']['sessionWithCalibration']['id'])
    else:
        raise Exception('No calibration trial in session.')
    
    return calibID

def get_camera_mapping(session_id, session_path):
    calibration_id = get_calibration_trial_id(session_id)
    trial = get_trial_json(calibration_id)
    resultTags = [res['tag'] for res in trial['results']]

    mappingPath = os.path.join(session_path,'Videos','mappingCamDevice.pickle')
    os.makedirs(os.path.join(session_path,'Videos'), exist_ok=True)
    if not os.path.exists(mappingPath):
        mappingURL = trial['results'][resultTags.index('camera_mapping')]['media']
        download_file(mappingURL, mappingPath)
    

def get_model_and_metadata(session_id, session_path):
    neutral_id = get_neutral_trial_id(session_id)
    trial = get_trial_json(neutral_id)
    resultTags = [res['tag'] for res in trial['results']]
    
    # Metadata.
    metadataPath = os.path.join(session_path,'sessionMetadata.yaml')
    if not os.path.exists(metadataPath) :
        metadataURL = trial['results'][resultTags.index('session_metadata')]['media']
        download_file(metadataURL, metadataPath)
    
    # Model.
    modelURL = trial['results'][resultTags.index('opensim_model')]['media']
    modelName = modelURL[modelURL.rfind('-')+1:modelURL.rfind('?')]
    modelFolder = os.path.join(session_path, 'OpenSimData', 'Model')
    modelPath = os.path.join(modelFolder, modelName)
    if not os.path.exists(modelPath):
        os.makedirs(modelFolder, exist_ok=True)
        download_file(modelURL, modelPath)
        
    return modelName

def get_main_settings(session_folder,trial_name):
    settings_path = os.path.join(session_folder,'MarkerData',
                                 'Settings','settings_' + trial_name + '.yaml')
    main_settings = import_metadata(settings_path)
    
    return main_settings

        
def get_model_name_from_metadata(sessionFolder,appendText='_scaled'):
    metadataPath = os.path.join(sessionFolder,'sessionMetadata.yaml')
    
    if os.path.exists(metadataPath):
        metadata = import_metadata(os.path.join(sessionFolder,'sessionMetadata.yaml'))
        modelName = metadata['openSimModel'] + appendText + '.osim'
    else:
        raise Exception('Session metadata not found, could not identify OpenSim model.')
        
    return modelName

        
def get_motion_data(trial_id, session_path):
    trial = get_trial_json(trial_id)
    trial_name = trial['name']
    resultTags = [res['tag'] for res in trial['results']]

    # Marker data.
    if 'marker_data' in resultTags:
        markerFolder = os.path.join(session_path, 'MarkerData')
        markerPath = os.path.join(markerFolder, trial_name + '.trc')
        os.makedirs(markerFolder, exist_ok=True)
        if not os.path.exists(markerPath):
            markerURL = trial['results'][resultTags.index('marker_data')]['media']
            download_file(markerURL, markerPath)
    
    # IK data.
    if 'ik_results' in resultTags:
        ikFolder = os.path.join(session_path, 'OpenSimData', 'Kinematics')
        ikPath = os.path.join(ikFolder, trial_name + '.mot')
        os.makedirs(ikFolder, exist_ok=True)
        if not os.path.exists(ikPath):
            ikURL = trial['results'][resultTags.index('ik_results')]['media']
            download_file(ikURL, ikPath)
        
    # Main settings
    if 'main_settings' in resultTags:
        settingsFolder = os.path.join(session_path, 'MarkerData', 'Settings')
        settingsPath = os.path.join(settingsFolder, 'settings_' + trial_name + '.yaml')
        os.makedirs(settingsFolder, exist_ok=True)
        if not os.path.exists(settingsPath):
            settingsURL = trial['results'][resultTags.index('main_settings')]['media']
            download_file(settingsURL, settingsPath)
        
        
def get_geometries(session_path, modelName='LaiUhlrich2022_scaled'):
        
    geometryFolder = os.path.join(session_path, 'OpenSimData', 'Model', 'Geometry')
    try:
        # Download.
        os.makedirs(geometryFolder, exist_ok=True)
        if 'Lai' in modelName:
            modelType = 'LaiArnold'
            vtpNames = [
                'capitate_lvs','capitate_rvs','hamate_lvs','hamate_rvs',
                'hat_jaw','hat_ribs_scap','hat_skull','hat_spine','humerus_lv',
                'humerus_rv','index_distal_lvs','index_distal_rvs',
                'index_medial_lvs', 'index_medial_rvs','index_proximal_lvs',
                'index_proximal_rvs','little_distal_lvs','little_distal_rvs',
                'little_medial_lvs','little_medial_rvs','little_proximal_lvs',
                'little_proximal_rvs','lunate_lvs','lunate_rvs','l_bofoot',
                'l_femur','l_fibula','l_foot','l_patella','l_pelvis','l_talus',
                'l_tibia','metacarpal1_lvs','metacarpal1_rvs',
                'metacarpal2_lvs','metacarpal2_rvs','metacarpal3_lvs',
                'metacarpal3_rvs','metacarpal4_lvs','metacarpal4_rvs',
                'metacarpal5_lvs','metacarpal5_rvs','middle_distal_lvs',
                'middle_distal_rvs','middle_medial_lvs','middle_medial_rvs',
                'middle_proximal_lvs','middle_proximal_rvs','pisiform_lvs',
                'pisiform_rvs','radius_lv','radius_rv','ring_distal_lvs',
                'ring_distal_rvs','ring_medial_lvs','ring_medial_rvs',
                'ring_proximal_lvs','ring_proximal_rvs','r_bofoot','r_femur',
                'r_fibula','r_foot','r_patella','r_pelvis','r_talus','r_tibia',
                'sacrum','scaphoid_lvs','scaphoid_rvs','thumb_distal_lvs',
                'thumb_distal_rvs','thumb_proximal_lvs','thumb_proximal_rvs',
                'trapezium_lvs','trapezium_rvs','trapezoid_lvs','trapezoid_rvs',
                'triquetrum_lvs','triquetrum_rvs','ulna_lv','ulna_rv']
        else:
            raise ValueError("Geometries not available for this model")                
        for vtpName in vtpNames:
            url = 'https://mc-opencap-public.s3.us-west-2.amazonaws.com/geometries_vtp/{}/{}.vtp'.format(modelType, vtpName)
            filename = os.path.join(geometryFolder, '{}.vtp'.format(vtpName))                
            download_file(url, filename)
    except:
        pass
    
def import_metadata(filePath):
    myYamlFile = open(filePath)
    parsedYamlFile = yaml.load(myYamlFile, Loader=yaml.FullLoader)
    
    return parsedYamlFile
    
def download_kinematics(session_id, folder=None, trialNames=None):
    
    # Login to access opencap data from server. 
    
    # Create folder.
    if folder is None:
        folder = os.getcwd()    
    os.makedirs(folder, exist_ok=True)
    
    # Model and metadata.
    neutral_id = get_neutral_trial_id(session_id)
    get_motion_data(neutral_id, folder)
    modelName = get_model_and_metadata(session_id, folder)
    # Remove extension from modelName
    modelName = modelName.replace('.osim','')
    
    # Session trial names.
    sessionJson = get_session_json(session_id)
    sessionTrialNames = [t['name'] for t in sessionJson['trials']]
    if trialNames != None:
        [print(t + ' not in session trial names.') 
         for t in trialNames if t not in sessionTrialNames]
    
    # Motion data.
    loadedTrialNames = []
    for trialDict in sessionJson['trials']:
        if trialNames is not None and trialDict['name'] not in trialNames:
            continue        
        trial_id = trialDict['id']
        get_motion_data(trial_id,folder)
        loadedTrialNames.append(trialDict['name'])
        
    # Remove 'calibration' and 'neutral' from loadedTrialNames.    
    loadedTrialNames = [i for i in loadedTrialNames if i!='neutral' and i!='calibration']
        
    # Geometries.
    get_geometries(folder, modelName=modelName)
        
    return loadedTrialNames, modelName

# Download pertinent trial data.
def download_trial(trial_id, folder, session_id=None):
    
    trial = get_trial_json(trial_id)
    if session_id is None:
        session_id = trial['session_id']
        
    os.makedirs(folder,exist_ok=True)
    
    # download model
    get_model_and_metadata(session_id, folder)
    
    # download trc and mot
    get_motion_data(trial_id,folder)
    
    return trial['name']


# Get trial ID from name.
def get_trial_id(session_id,trial_name):
    session = get_session_json(session_id)
    
    trial_id = [t['id'] for t in session['trials'] if t['name'] == trial_name]
    
    return trial_id[0]

# %%  Storage file to numpy array.
def storage_to_numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
            skip_header=skip_header)

    return data

# %%  Storage file to dataframe.
def storage_to_dataframe(storage_file, headers):
    # Extract data
    data = storage_to_numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out

# %% Load storage and output as dataframe or numpy
def load_storage(file_path,outputFormat='numpy'):
    table = opensim.TimeSeriesTable(file_path)    
    data = table.getMatrix().to_numpy()
    time = np.asarray(table.getIndependentColumn()).reshape(-1, 1)
    data = np.hstack((time,data))
    headers = ['time'] + list(table.getColumnLabels())
    
    if outputFormat == 'numpy':
        return data,headers
    elif outputFormat == 'dataframe':
        return pd.DataFrame(data, columns=headers)
    else:
        return None    
    
# %%  Numpy array to storage file.
def numpy_to_storage(labels, data, storage_file, datatype=None):
    
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"
    
    f = open(storage_file, 'w')
    # Old style
    if datatype is None:
        f = open(storage_file, 'w')
        f.write('name %s\n' %storage_file)
        f.write('datacolumns %d\n' %data.shape[1])
        f.write('datarows %d\n' %data.shape[0])
        f.write('range %f %f\n' %(np.min(data[:, 0]), np.max(data[:, 0])))
        f.write('endheader \n')
    # New style
    else:
        if datatype == 'IK':
            f.write('Coordinates\n')
        elif datatype == 'ID':
            f.write('Inverse Dynamics Generalized Forces\n')
        elif datatype == 'GRF':
            f.write('%s\n' %storage_file)
        elif datatype == 'muscle_forces':
            f.write('ModelForces\n')
        f.write('version=1\n')
        f.write('nRows=%d\n' %data.shape[0])
        f.write('nColumns=%d\n' %data.shape[1])    
        if datatype == 'IK':
            f.write('inDegrees=yes\n\n')
            f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
            f.write("If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n")
        elif datatype == 'ID':
            f.write('inDegrees=no\n')
        elif datatype == 'GRF':
            f.write('inDegrees=yes\n')
        elif datatype == 'muscle_forces':
            f.write('inDegrees=yes\n\n')
            f.write('This file contains the forces exerted on a model during a simulation.\n\n')
            f.write("A force is a generalized force, meaning that it can be either a force (N) or a torque (Nm).\n\n")
            f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
            f.write('Angles are in degrees.\n\n')
            
        f.write('endheader \n')
    
    for i in range(len(labels)):
        f.write('%s\t' %labels[i])
    f.write('\n')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' %data[i, j])
        f.write('\n')
        
    f.close()

def download_videos_from_server(session_id,trial_id,
                             isCalibration=False, isStaticPose=False,
                             trial_name= None, session_path = None):
    
    if session_path is None:
        data_dir = os.getcwd() 
        session_path = os.path.join(data_dir,'Data', session_id)  
    if not os.path.exists(session_path): 
        os.makedirs(session_path, exist_ok=True)
    
    resp = requests.get("{}trials/{}/".format(API_URL,trial_id),
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    trial = resp.json()
    if trial_name is None:
        trial_name = trial['name']
    trial_name = trial_name.replace(' ', '')

    print("\nDownloading {}".format(trial_name))

    # The videos are not always organized in the same order. Here, we save
    # the order during the first trial processed in the session such that we
    # can use the same order for the other trials.
    if not os.path.exists(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle')):
        mappingCamDevice = {}
        for k, video in enumerate(trial["videos"]):
            os.makedirs(os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name), exist_ok=True)
            video_path = os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name, trial_name + ".mov")
            download_file(video["video"], video_path)                
            mappingCamDevice[video["device_id"].replace('-', '').upper()] = k
        with open(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle'), 'wb') as handle:
            pickle.dump(mappingCamDevice, handle)
    else:
        with open(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle'), 'rb') as handle:
            mappingCamDevice = pickle.load(handle) 
            # ensure upper on deviceID
            for dID in mappingCamDevice.keys():
                mappingCamDevice[dID.upper()] = mappingCamDevice.pop(dID)
        for video in trial["videos"]:            
            k = mappingCamDevice[video["device_id"].replace('-', '').upper()] 
            videoDir = os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name)
            os.makedirs(videoDir, exist_ok=True)
            video_path = os.path.join(videoDir, trial_name + ".mov")
            if not os.path.exists(video_path):
                if video['video'] :
                    download_file(video["video"], video_path)
              
    return trial_name
   
    
def get_calibration(session_id,session_path):
    calibration_id = get_calibration_trial_id(session_id)

    resp = requests.get("{}trials/{}/".format(API_URL,calibration_id),
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    trial = resp.json()
    calibResultTags = [res['tag'] for res in trial['results']]
   
    videoFolder = os.path.join(session_path,'Videos')
    os.makedirs(videoFolder, exist_ok=True)
    
    if trial['status'] != 'done':
        return
    
    mapURL = trial['results'][calibResultTags.index('camera_mapping')]['media']
    mapLocalPath = os.path.join(videoFolder,'mappingCamDevice.pickle')

    download_and_switch_calibration(session_id,session_path,calibTrialID=calibration_id)
    
    # Download mapping
    if len(glob.glob(mapLocalPath)) == 0:
        download_file(mapURL,mapLocalPath)
                        

def download_and_switch_calibration(session_id,session_path,calibTrialID = None):
    if calibTrialID == None:
        calibTrialID = get_calibration_trial_id(session_id)
    resp = requests.get("https://api.opencap.ai/trials/{}/".format(calibTrialID),
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    trial = resp.json()
       
    calibURLs = {t['device_id']:t['media'] for t in trial['results'] if t['tag'] == 'calibration_parameters_options'}
    calibImgURLs = {t['device_id']:t['media'] for t in trial['results'] if t['tag'] == 'calibration-img'}
    _,imgExtension = os.path.splitext(calibImgURLs[list(calibImgURLs.keys())[0]])
    lastIdx = imgExtension.find('?') 
    if lastIdx >0:
        imgExtension = imgExtension[:lastIdx]
    
    if 'meta' in trial.keys() and trial['meta'] is not None and 'calibration' in trial['meta'].keys():
        calibDict = trial['meta']['calibration']
        calibImgFolder = os.path.join(session_path,'CalibrationImages')
        os.makedirs(calibImgFolder,exist_ok=True)
        for cam,calibNum in calibDict.items():
            camDir = os.path.join(session_path,'Videos',cam)
            os.makedirs(camDir,exist_ok=True)
            file_name = os.path.join(camDir,'cameraIntrinsicsExtrinsics.pickle')
            img_fileName = os.path.join(calibImgFolder,'calib_img' + cam + imgExtension)
            if calibNum == 0:
                download_file(calibURLs[cam+'_soln0'], file_name)
                download_file(calibImgURLs[cam],img_fileName)
            elif calibNum == 1:
                download_file(calibURLs[cam+'_soln1'], file_name) 
                download_file(calibImgURLs[cam + '_altSoln'],img_fileName)
                
            
def post_file_to_trial(filePath,trial_id,tag,device_id):
    files = {'media': open(filePath, 'rb')}
    data = {
        "trial": trial_id,
        "tag": tag,
        "device_id" : device_id
    }

    requests.post("{}results/".format(API_URL), files=files, data=data,
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    files["media"].close()

def post_video_to_trial(filePath,trial_id,device_id,parameters):
    files = {'video': open(filePath, 'rb')}
    data = {
        "trial": trial_id,
        "device_id" : device_id,
        "parameters": parameters
    }

    requests.post("{}videos/".format(API_URL), files=files, data=data,
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    files["video"].close()

def delete_video_from_trial(video_id):

    requests.delete("{}videos/{}/".format(API_URL, video_id),
                        headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
def delete_results(trial_id, tag=None, resultNum=None):
    # Delete specific result number, or all results with a specific tag, or all results if tag==None
    if resultNum != None:
        resultNums = [resultNum]
    elif tag != None:
        trial = get_trial_json(trial_id)
        resultNums = [r['id'] for r in trial['results'] if r['tag']==tag]
        
    elif tag == None: 
        trial = get_trial_json(trial_id)
        resultNums = [r['id'] for r in trial['results']]

    for rNum in resultNums:
        requests.delete(API_URL + "results/{}/".format(rNum),
                        headers = {"Authorization": "Token {}".format(API_TOKEN)})
        
def set_trial_status(trial_id, status):

    # Available statuses: 'done', 'error', 'stopped', 'reprocess'
    # 'processing' and 'recording also exist, but it does not make sense to set them manually.
    # Throw error if status is not one of the above.
    if status not in ['done', 'error', 'stopped', 'reprocess']:
        raise ValueError('Invalid status. Available statuses: done, error, stopped, reprocess')

    requests.patch(API_URL+"trials/{}/".format(trial_id), data={'status': status},
                     headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
def set_session_subject(session_id, subject_id):
    requests.patch(API_URL+"sessions/{}/".format(session_id), data={'subject': subject_id},
                     headers = {"Authorization": "Token {}".format(API_TOKEN)})  

def get_syncd_videos(trial_id,session_path):
    trial = requests.get("{}trials/{}/".format(API_URL,trial_id),
                         headers = {"Authorization": "Token {}".format(API_TOKEN)}).json()
    trial_name = trial['name']
    
    if trial['results']:
        for result in trial['results']:
            if result['tag'] == 'video-sync':
                url = result['media']
                cam,suff = os.path.splitext(url[url.rfind('_')+1:])
                lastIdx = suff.find('?') 
                if lastIdx >0:
                    suff = suff[:lastIdx]
                
                syncVideoPath = os.path.join(session_path,'Videos',cam,'InputMedia',trial_name,trial_name + '_sync' + suff)
                download_file(url,syncVideoPath)
        
        
def download_session(session_id, sessionBasePath= None,
                     zipFolder=False,writeToDB=False, downloadVideos=True):
    print('\nDownloading {}'.format(session_id))
    
    if sessionBasePath is None:
        sessionBasePath = os.path.join(os.getcwd(),'Data')
    
    session = get_session_json(session_id)
    session_path = os.path.join(sessionBasePath,'OpenCapData_' + session_id) 
    
    calib_id = get_calibration_trial_id(session_id)
    neutral_id = get_neutral_trial_id(session_id)
    dynamic_ids = [t['id'] for t in session['trials'] if (t['name'] != 'calibration' and t['name'] !='neutral')]  
    
    # Calibration
    try:
        get_camera_mapping(session_id, session_path)
        if downloadVideos:
            download_videos_from_server(session_id,calib_id,
                                 isCalibration=True,isStaticPose=False,
                                 session_path = session_path) 

        get_calibration(session_id,session_path)
    except:
        pass
    
    # Neutral
    try:
        modelName = get_model_and_metadata(session_id,session_path)
        get_motion_data(neutral_id,session_path)
        if downloadVideos:
            download_videos_from_server(session_id,neutral_id,
                             isCalibration=False,isStaticPose=True,
                             session_path = session_path)

        get_syncd_videos(neutral_id,session_path)
    except:
        pass

    # Dynamic
    for dynamic_id in dynamic_ids:
        try:
            get_motion_data(dynamic_id,session_path)
            if downloadVideos:
                download_videos_from_server(session_id,dynamic_id,
                         isCalibration=False,isStaticPose=False,
                         session_path = session_path)

            get_syncd_videos(dynamic_id,session_path)
        except:
            pass
        
    repoDir = os.path.dirname(os.path.abspath(__file__))
    
    # Readme  
    try:        
        pathReadme = os.path.join(repoDir, 'Resources', 'README.txt')
        pathReadmeEnd = os.path.join(session_path, 'README.txt')
        shutil.copy2(pathReadme, pathReadmeEnd)
    except:
        pass
        
    # Geometry
    try:
        if 'Lai' in modelName:
            modelType = 'LaiArnold'
        else:
            raise ValueError("Geometries not available for this model, please contact us")
        if platform.system() == 'Windows':
            geometryDir = os.path.join(repoDir, 'tmp', modelType, 'Geometry')
        else:
            geometryDir = "/tmp/{}/Geometry".format(modelType)
        # If not in cache, download from s3.
        if not os.path.exists(geometryDir):
            os.makedirs(geometryDir, exist_ok=True)
            get_geometries(session_path, modelName=modelName)
        geometryDirEnd = os.path.join(session_path, 'OpenSimData', 'Model', 'Geometry')
        shutil.copytree(geometryDir, geometryDirEnd)
    except:
        pass
    
    # Zip   
    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), 
                                           os.path.join(path, '..')))    
    session_zip = '{}.zip'.format(session_path)
    if os.path.isfile(session_zip):
        os.remove(session_zip)  
    if zipFolder:
        zipf = zipfile.ZipFile(session_zip, 'w', zipfile.ZIP_DEFLATED)
        zipdir(session_path, zipf)
        zipf.close()
    
    # Write zip as a result to last trial for now
    if writeToDB:
        post_file_to_trial(session_zip,dynamic_ids[-1],tag='session_zip',
                           device_id='all')    
    
def cross_corr(y1, y2,multCorrGaussianStd=None,visualize=False):
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

def downsample(data,time,framerate_in,framerate_out):
    # Calculate the downsampling factor
    downsampling_factor = framerate_in / framerate_out
    
    # Create new indices for downsampling
    original_indices = np.arange(len(data))
    new_indices = np.arange(0, len(data), downsampling_factor)
    
    # Perform downsampling with interpolation
    downsampled_data = np.ndarray((len(new_indices), data.shape[1]))
    for i in range(data.shape[1]):
        downsampled_data[:,i] = np.interp(new_indices, original_indices, data[:,i])
    
    downsampled_time = np.interp(new_indices, original_indices, time)
    
    return downsampled_time, downsampled_data