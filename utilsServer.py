import os
import glob
import shutil
import requests
import json
import logging
import time
import random
import urllib

from main import main
from utils import getDataDirectory
from utils import getTrialName
from utils import getTrialJson
from utils import downloadVideosFromServer
from utils import switchCalibrationForCamera
from utils import deleteCalibrationFiles
from utils import deleteStaticFiles
from utils import writeMediaToAPI
from utils import postCalibrationOptions
from utils import getCalibration
from utils import getModelAndMetadata
from utils import writeCalibrationOptionsToAPI
from utils import postMotionData
from utils import getNeutralTrialID
from utils import getCalibrationTrialID
from utils import sendStatusEmail
from utils import importMetadata
from utils import checkAndGetPosePickles
from utils import getTrialNameIdMapping
from utils import makeRequestWithRetry
from utilsAuth import getToken
from utilsAPI import getAPIURL


API_URL = getAPIURL()
API_TOKEN = getToken()

def processTrial(session_id, trial_id, trial_type = 'dynamic',
                 imageUpsampleFactor = 4, poseDetector = 'OpenPose',
                 isDocker = True, resolutionPoseDetection = 'default',
                 bbox_thr = 0.8, extrinsicTrialName = 'calibration',
                 deleteLocalFolder = True,
                 hasWritePermissions = True,
                 use_existing_pose_pickle = False,
                 batchProcess = False,
                 cameras_to_use=['all']):

    # Get session directory
    session_name = session_id 
    data_dir = getDataDirectory(isDocker=isDocker)
    session_path = os.path.join(data_dir,'Data',session_name)    
    trial_url = "{}{}{}/".format(API_URL, "trials/", trial_id)
    metadata_path = os.path.join(session_path, 'sessionMetadata.yaml')        
       
    # Process the 3 different types of trials
    if trial_type == 'calibration':
        # delete extrinsic files if they exist.
        deleteCalibrationFiles(session_path)
        
        # download the videos
        trial_name = downloadVideosFromServer(session_id,trial_id,isDocker=isDocker,
                                 isCalibration=True,isStaticPose=False)
        
        # run calibration
        try:
            main(session_name, trial_name, trial_id, isDocker=isDocker, extrinsicsTrial=True,
                 imageUpsampleFactor=imageUpsampleFactor,genericFolderNames = True,
                 cameras_to_use=cameras_to_use)
        except Exception as e:
            error_msg = {}
            error_msg['error_msg'] = e.args[0]
            error_msg['error_msg_dev'] = e.args[1]
            _ = makeRequestWithRetry('PATCH',
                                     trial_url,
                                     data={"meta": json.dumps(error_msg)},
                                     headers = {"Authorization": "Token {}".format(API_TOKEN)}) 
            raise Exception('Calibration failed', e.args[0], e.args[1])
        
        if not hasWritePermissions:
            print('You are not the owner of this session, so do not have permission to write results to database.')
            return
            
        # Write calibration images to django
        images_path = os.path.join(session_path,'CalibrationImages')
        writeMediaToAPI(API_URL,images_path,trial_id,tag="calibration-img",deleteOldMedia=True)
        
        # Write calibration solutions to django
        writeCalibrationOptionsToAPI(session_path,session_id,calibration_id = trial_id,
                                     trialName = extrinsicTrialName)
        
    elif trial_type == 'static':
        # delete static files if they exist.
        deleteStaticFiles(session_path, staticTrialName = 'neutral')
        
        # Check for calibration to use on django, if not, check for switch calibrations and post result.
        calibrationOptions = getCalibration(session_id,session_path,trial_type=trial_type,getCalibrationOptions=True)
        
        # download the videos
        trial_name = downloadVideosFromServer(session_id,trial_id,isDocker=isDocker,
                                 isCalibration=False,isStaticPose=True)
        
        # Download the pose pickles to avoid re-running pose estimation.
        if batchProcess and use_existing_pose_pickle:
            checkAndGetPosePickles(trial_id, session_path, poseDetector, resolutionPoseDetection, bbox_thr)

        # If processTrial is run from app.py, poseDetector is set based on what
        # users select in the webapp, which is saved in metadata. Based on this,
        # we set resolutionPoseDetection or bbox_thr to the webapp defaults. If
        # processTrial is run from batchReprocess.py, then the settings used are
        # those passed as arguments to processTrial.
        if not batchProcess:
            sessionMetadata = importMetadata(metadata_path)
            poseDetector = sessionMetadata['posemodel']
            file_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(file_dir,'defaultOpenCapSettings.json')) as f:
                defaultOpenCapSettings = json.load(f)
            if poseDetector.lower() == 'openpose':
                resolutionPoseDetection = defaultOpenCapSettings['openpose']
            elif poseDetector.lower() == 'hrnet':
                bbox_thr = defaultOpenCapSettings['hrnet']            

        # run static
        try:
            main(session_name, trial_name, trial_id, isDocker=isDocker, extrinsicsTrial=False,
                 poseDetector=poseDetector,
                 imageUpsampleFactor=imageUpsampleFactor,
                 scaleModel = True,
                 resolutionPoseDetection = resolutionPoseDetection,
                 genericFolderNames = True,
                 bbox_thr = bbox_thr,
                 calibrationOptions = calibrationOptions,
                 cameras_to_use=cameras_to_use)
        except Exception as e:       
            # Try to post pose pickles so can be used offline. This function will 
            # error at kinematics most likely, but if pose estimation completed,
            # pickles will get posted
            try:
                # Write results to django
                if not batchProcess:
                    print('trial failed. posting pose pickles')
                    postMotionData(trial_id,session_path,trial_name=trial_name,isNeutral=True,
                                    poseDetector=poseDetector, 
                                    resolutionPoseDetection=resolutionPoseDetection,
                                    bbox_thr=bbox_thr)
            except:
                pass
            
            error_msg = {}
            error_msg['error_msg'] = e.args[0]
            error_msg['error_msg_dev'] = e.args[1]
            _ = makeRequestWithRetry('PATCH',
                                     trial_url,
                                     data={"meta": json.dumps(error_msg)},
                                     headers = {"Authorization": "Token {}".format(API_TOKEN)})
            raise Exception('Static trial failed', e.args[0], e.args[1])
        
        if not hasWritePermissions:
            print('You are not the owner of this session, so do not have permission to write results to database.')
            return
        
        # Write videos to django
        video_path = getResultsPath(session_id, trial_id,
                                    resultType='neutralVideo', isDocker=isDocker)
        writeMediaToAPI(API_URL,video_path,trial_id, tag='video-sync',deleteOldMedia=True)
        
        # Write neutral pose images to django
        images_path = os.path.join(session_path,'NeutralPoseImages')
        writeMediaToAPI(API_URL,images_path,trial_id,tag="neutral-img",deleteOldMedia=True)
        
        # Write visualizer jsons to django
        visualizerJson_path = getResultsPath(session_id, trial_id, 
                                             resultType='visualizerJson', 
                                             isDocker=isDocker)
        writeMediaToAPI(API_URL,visualizerJson_path,trial_id,
                        tag="visualizerTransforms-json",deleteOldMedia=True)
        
        # Write results to django
        postMotionData(trial_id,session_path,trial_name=trial_name,isNeutral=True,
                       poseDetector=poseDetector, 
                       resolutionPoseDetection=resolutionPoseDetection,
                       bbox_thr=bbox_thr)
        
        # Write calibration options to django
        postCalibrationOptions(session_path,session_id,overwrite=True)
        
    elif trial_type == 'dynamic':
        # download calibration, model, and metadata if not existing
        getCalibration(session_id,session_path,trial_type=trial_type)   
        getModelAndMetadata(session_id,session_path)
        
        # download the videos
        trial_name = downloadVideosFromServer(
            session_id, trial_id, isDocker=isDocker, isCalibration=False,
            isStaticPose=False)
        
        # Download the pose pickles to avoid re-running pose estimation.
        if batchProcess and use_existing_pose_pickle:
            checkAndGetPosePickles(trial_id, session_path, poseDetector, resolutionPoseDetection, bbox_thr)

        # If processTrial is run from app.py, poseDetector is set based on what
        # users select in the webapp, which is saved in metadata. Based on this,
        # we set resolutionPoseDetection or bbox_thr to the webapp defaults. If
        # processTrial is run from batchReprocess.py, then the settings used are
        # those passed as arguments to processTrial.
        if not batchProcess:
            sessionMetadata = importMetadata(metadata_path)
            poseDetector = sessionMetadata['posemodel']
            file_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(file_dir,'defaultOpenCapSettings.json')) as f:
                defaultOpenCapSettings = json.load(f)
            if poseDetector.lower() == 'openpose':
                resolutionPoseDetection = defaultOpenCapSettings['openpose']
            elif poseDetector.lower() == 'hrnet':
                bbox_thr = defaultOpenCapSettings['hrnet'] 
        
        # run dynamic
        try:
            main(session_name, trial_name, trial_id, isDocker=isDocker, extrinsicsTrial=False,
                 poseDetector=poseDetector,
                 imageUpsampleFactor=imageUpsampleFactor,
                 resolutionPoseDetection = resolutionPoseDetection,
                 genericFolderNames = True,
                 bbox_thr = bbox_thr,
                 cameras_to_use=cameras_to_use)
        except Exception as e:
            # Try to post pose pickles so can be used offline. This function will 
            # error at kinematics most likely, but if pose estimation completed,
            # pickles will get posted
            try:
                # Write results to django
                if not batchProcess:
                    print('trial failed. posting pose pickles')
                    postMotionData(trial_id,session_path,trial_name=trial_name,isNeutral=False,
                                    poseDetector=poseDetector, 
                                    resolutionPoseDetection=resolutionPoseDetection,
                                    bbox_thr=bbox_thr)
            except:
                pass
            
            error_msg = {}
            error_msg['error_msg'] = e.args[0]
            error_msg['error_msg_dev'] = e.args[1]
            _ = makeRequestWithRetry('PATCH',
                                     trial_url,
                                     data={"meta": json.dumps(error_msg)},
                                     headers = {"Authorization": "Token {}".format(API_TOKEN)})
            raise Exception('Dynamic trial failed.\n' + error_msg['error_msg_dev'], e.args[0], e.args[1])
        
        if not hasWritePermissions:
            print('You are not the owner of this session, so do not have permission to write results to database.')
            return
        
        # Write videos to django
        video_path = getResultsPath(session_id, trial_id,
                                    resultType='sync_video', isDocker=isDocker)
        writeMediaToAPI(API_URL,video_path,trial_id, tag='video-sync',deleteOldMedia=True)
        
        # Write visualizer jsons to django
        visualizerJson_path = getResultsPath(session_id, trial_id, 
                                             resultType='visualizerJson', 
                                             isDocker=isDocker)
        writeMediaToAPI(API_URL,visualizerJson_path,trial_id,
                        tag="visualizerTransforms-json",deleteOldMedia=True)
        
        # Write results to django
        postMotionData(trial_id,session_path,trial_name=trial_name,isNeutral=False,
                       poseDetector=poseDetector, 
                       resolutionPoseDetection=resolutionPoseDetection,
                       bbox_thr=bbox_thr)
        
    else:
        raise Exception('Wrong trial type. Options: calibration, static, dynamic.', 'TODO', 'TODO')
    
    # Remove data
    if deleteLocalFolder:
        shutil.rmtree(session_path)
        
        
def getCalibrationImagePath(session_id,isDocker=True):
    session_name = session_id # TODO We may want to name this on server side?
    data_dir = getDataDirectory(isDocker=isDocker)
    session_path = os.path.join(data_dir,'Data',session_name)
    
    image_path = os.path.join(session_path,'CalibrationImages')
    
    return image_path

def getResultsPath(session_id, trial_id, resultType='sync_video', isDocker=True):
    session_name = session_id 
    data_dir = getDataDirectory(isDocker=isDocker)
    session_path = os.path.join(data_dir,'Data',session_name)  
    trial_name = getTrialName(trial_id)
    
    if resultType == 'sync_video':
        result_path = os.path.join(session_path, 'VisualizerVideos', trial_name)
    elif resultType == 'visualizerJson':
        result_path = os.path.join(session_path, 'VisualizerJsons', trial_name)
    elif resultType == 'neutralVideo':
        result_path = os.path.join(session_path, 'NeutralPoseImages','Videos')
    else:
        raise Exception('Wrong result type.')
    
    return result_path


def switchCalibration(session_id,trial_id,camList,isDocker=True):  
    # Get session directory
    session_name = session_id # TODO We may want to name this on server side?
    data_dir = getDataDirectory(isDocker=isDocker)
    session_path = os.path.join(data_dir,'Data',session_name)
    
    for cam in camList:
        switchCalibrationForCamera(cam,trial_id,session_path)
    

def newSessionSameSetup(session_id_old,session_id_new,extrinsicTrialName='calibration',isDocker=True):
    # Get session directory
    session_name_new = session_id_new # TODO We may want to name this on server side?
    session_name_old = session_id_old
    data_dir = getDataDirectory(isDocker=isDocker)
    session_path_new = os.path.join(data_dir,'Data',session_name_new)
    session_path_old = os.path.join(data_dir,'Data',session_name_old)
    
    os.makedirs(session_path_new, exist_ok=True)
    
    # Get cameras
    camList = [os.path.split(name)[1] for name in glob.glob(os.path.join(session_path_old,'Videos','Cam*'))]
    for cam in camList:
        # Calibration trial
        newCalibFolder = os.path.join(session_path_new,'Videos',cam,'InputMedia',extrinsicTrialName)
        shutil.copytree(os.path.join(session_path_old,'Videos',cam,'InputMedia',extrinsicTrialName),
                        newCalibFolder)
        # Calibration parameters
        shutil.copyfile(os.path.join(session_path_old,'Videos',cam,'cameraIntrinsicsExtrinsics.pickle'),
                        os.path.join(session_path_new,'Videos',cam,'cameraIntrinsicsExtrinsics.pickle'))
           
    # Copy metadata and CalibrationImages
    shutil.copytree(os.path.join(session_path_old,'CalibrationImages'),
                     os.path.join(session_path_new,'CalibrationImages'))
    shutil.copyfile(os.path.join(session_path_old,'sessionMetadata.yaml'),
                     os.path.join(session_path_new,'sessionMetadata.yaml'))
            
    
def batchReprocess(session_ids,calib_id,static_id,dynamic_trialNames,poseDetector='OpenPose', 
                   resolutionPoseDetection='1x736',deleteLocalFolder=True,
                   isServer=False, use_existing_pose_pickle=True,
                   cameras_to_use=['all']):

    # extract trial ids from trial names
    if dynamic_trialNames is not None and len(dynamic_trialNames)>0:
        trialNames = getTrialNameIdMapping(session_ids[0])
        dynamic_ids = [trialNames[name]['id'] for name in dynamic_trialNames]
    else:
        dynamic_ids = dynamic_trialNames
    
    if (type(calib_id) == str or type(static_id) == str or type(dynamic_ids) == str or 
        (type(dynamic_ids)==list and len(dynamic_ids)>0)) and len(session_ids) >1:
        raise Exception('can only have one session number if hardcoding other trial ids')
        
    for session_id in session_ids:
        print('Processing ' + session_id)
        
        # check if write permissions (session owner or admin)
        response = makeRequestWithRetry('GET',
                                        API_URL + "sessions/{}/get_session_permission/".format(session_id),
                                        headers = {"Authorization": "Token {}".format(API_TOKEN)})
        permissions = response.json()
        hasWritePermissions = permissions['isAdmin'] or permissions['isOwner']


        if calib_id == None:
            calib_id_toProcess = getCalibrationTrialID(session_id)
        else:
            calib_id_toProcess = calib_id
        
        if len(calib_id_toProcess) > 0:
            try:
                processTrial(session_id,
                              calib_id_toProcess,
                              trial_type="calibration",
                              poseDetector = poseDetector,
                              deleteLocalFolder = deleteLocalFolder,
                              isDocker=isServer,
                              hasWritePermissions = hasWritePermissions,
                              cameras_to_use=cameras_to_use)
                statusData = {'status':'done'}
                _ = makeRequestWithRetry('PATCH',
                                         API_URL + "trials/{}/".format(calib_id_toProcess),
                                         data=statusData,
                                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
            except Exception as e:
                print(e)
                statusData = {'status':'error'}
                _ = makeRequestWithRetry('PATCH',
                                         API_URL + "trials/{}/".format(calib_id_toProcess),
                                         data=statusData,
                                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
        
        if static_id == None:
            static_id_toProcess = getNeutralTrialID(session_id)
        else:
            static_id_toProcess = static_id
        
        if len(static_id_toProcess)  > 0:
            try:
                processTrial(session_id,
                              static_id_toProcess,
                              trial_type="static",
                              resolutionPoseDetection = resolutionPoseDetection,
                              poseDetector = poseDetector,
                              deleteLocalFolder = deleteLocalFolder,
                              isDocker=isServer,
                              hasWritePermissions = hasWritePermissions,
                              use_existing_pose_pickle = use_existing_pose_pickle,
                              batchProcess = True,
                              cameras_to_use=cameras_to_use)
                statusData = {'status':'done'}
                _ = makeRequestWithRetry('PATCH',
                                         API_URL + "trials/{}/".format(static_id_toProcess),
                                         data=statusData,
                                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
            except Exception as e:
                print(e)
                statusData = {'status':'error'}
                _ = makeRequestWithRetry('PATCH',
                                         API_URL + "trials/{}/".format(static_id_toProcess),
                                         data=statusData,
                                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
        if dynamic_ids == None:
            response = makeRequestWithRetry('GET',
                                            API_URL + "sessions/{}/".format(session_id),
                                            headers = {"Authorization": "Token {}".format(API_TOKEN)})
            session = response.json()
            dynamic_ids_toProcess = [t['id'] for t in session['trials'] if (t['name'] != 'calibration' and t['name'] !='neutral')]
        else:
            if type(dynamic_ids) == str:
                dynamic_ids_toProcess = [dynamic_ids]
            elif type(dynamic_ids) == list:
                dynamic_ids_toProcess=dynamic_ids
        
        for dID in dynamic_ids_toProcess:
            try:
                processTrial(session_id,
                          dID,
                          trial_type="dynamic",
                          resolutionPoseDetection = resolutionPoseDetection,
                          poseDetector = poseDetector,
                          deleteLocalFolder = deleteLocalFolder,
                          isDocker=isServer,
                          hasWritePermissions = hasWritePermissions,
                          use_existing_pose_pickle = use_existing_pose_pickle,
                          batchProcess = True,
                          cameras_to_use=cameras_to_use)
                
                statusData = {'status':'done'}
                _ = makeRequestWithRetry('PATCH',
                                         API_URL + "trials/{}/".format(dID),
                                         data=statusData,
                                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
            except Exception as e:
                print(e)
                statusData = {'status':'error'}
                _ = makeRequestWithRetry('PATCH',
                                         API_URL + "trials/{}/".format(dID),
                                         data=statusData,
                                         headers = {"Authorization": "Token {}".format(API_TOKEN)})

def runTestSession(pose='all',isDocker=True,maxNumTries=3):
    # We retry test sessions because different sometimes when different
    # containers are processing the test trial, the API can change the
    # URL, causing 404 errors. 
    numTries = 0
    while numTries < maxNumTries:
        numTries += 1
        logging.info(f"Starting test trial attempt #{numTries} of {maxNumTries}")
        trials = {}
        
        if not any(s in API_URL for s in ['dev.opencap', '127.0']) : # prod trials
            trials['openpose'] = '3f2960c7-ca29-45b0-9be5-8d74db6131e5' # session ae2d50f1-537a-44f1-96a5-f5b7717452a3 
            trials['hrnet'] = '299ca938-8765-4a84-9adf-6bdf0e072451' # session faef80d3-0c26-452c-a7be-28dbfe04178e
            # trials['failure'] = '698162c8-3980-46e5-a3c5-8d4f081db4c4' # failed trial for testing
        else: # dev trials
            trials['openpose'] = '89d77579-8371-4760-a019-95f2c793622c' # session acd0e19c-6c86-4ba4-95fd-94b97229a926
            trials['hrnet'] = 'e0e02393-42ee-46d4-9ae1-a6fbb0b89c42' # session 3510c726-a1b8-4de4-a4a2-52b021b4aab2
        
        if pose == 'all':
            trialList = list(trials.values())
        else:
            try: 
                trialList = [trials[pose]]
            except:
                trialList = list(trials.values())
        
        try: 
            for trial_id in trialList:
                trial = getTrialJson(trial_id)
                logging.info("Running status check on trial name: " + trial['name'] + "_" + str(trial_id) + "\n\n")
                processTrial(trial["session"], trial_id, trial_type='static', isDocker=isDocker)

            logging.info("\n\n\nStatus check succeeded. \n\n")
            return
        
        # Catch and re-enter while loop if it's an HTTPError or URLError 
        # (could be more than just 404 errors). Wait between 30 and 60 seconds 
        # before retrying.
        except (requests.exceptions.HTTPError, urllib.error.URLError) as e:
            if numTries < maxNumTries:
                logging.info(f"test trial failed on try #{numTries} due to HTTPError or URLError. Retrying.")
                wait_time = random.randint(30,60)
                logging.info(f"waiting {wait_time} seconds then retrying...")
                time.sleep(wait_time)
                continue
            else:
                logging.info(f"test trial failed on try #{numTries} due to HTTPError or URLError.")
                # send email
                message = "A backend OpenCap machine failed the status check (HTTPError or URLError). It has been stopped."
                sendStatusEmail(message=message)
                raise Exception('Failed status check (HTTPError or URLError). Stopped.')
        
        # Catch other errors and stop
        except:
            logging.info("test trial failed. stopping machine.")
            # send email
            message = "A backend OpenCap machine failed the status check (not HTTPError or URLError). It has been stopped."
            sendStatusEmail(message=message)
            raise Exception('Failed status check. Stopped.')
            