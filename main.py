"""
    @authors: Scott Uhlrich, Antoine Falisse, Łukasz Kidziński
    
    This function calibrates the cameras, runs the pose detection algorithm, 
    reconstructs the 3D marker positions, augments the marker set,
    and runs the OpenSim pipeline.

"""

import os 
import glob
import numpy as np
import yaml
import traceback

import logging
logging.basicConfig(level=logging.INFO)

from utils import importMetadata, loadCameraParameters, getVideoExtension
from utils import getDataDirectory, getOpenPoseDirectory, getMMposeDirectory
from utilsChecker import saveCameraParameters
from utilsChecker import calcExtrinsicsFromVideo
from utilsChecker import isCheckerboardUpsideDown
from utilsChecker import autoSelectExtrinsicSolution
from utilsChecker import triangulateMultiviewVideo
from utilsChecker import writeTRCfrom3DKeypoints
from utilsChecker import popNeutralPoseImages
from utilsChecker import rotateIntrinsics
from utilsSync import synchronizeVideos
from utilsDetector  import runPoseDetector
from utilsAugmenter import augmentTRC
from utilsOpenSim import runScaleTool, getScaleTimeRange, runIKTool, generateVisualizerJson
from defaults import DEFAULT_SYNC_VER

def main(sessionName, trialName, trial_id, cameras_to_use=['all'],
         intrinsicsFinalFolder='Deployed', isDocker=False,
         extrinsicsTrial=False, alternateExtrinsics=None, 
         calibrationOptions=None,
         markerDataFolderNameSuffix=None, imageUpsampleFactor=4,
         poseDetector='OpenPose', resolutionPoseDetection='default', 
         scaleModel=False, bbox_thr=0.8, augmenter_model='v0.3',
         genericFolderNames=False, offset=True, benchmark=False,
         dataDir=None, overwriteAugmenterModel=False,
         filter_frequency='default', overwriteFilterFrequency=False,
         scaling_setup='upright_standing_pose', overwriteScalingSetup=False,
         overwriteCamerasToUse=False, syncVer=None,):

    # %% High-level settings.
    # Camera calibration.
    runCameraCalibration = True
    # Pose detection.
    runPoseDetection = True
    # Video Synchronization.
    runSynchronization = True
    # Triangulation.
    runTriangulation = True
    # Marker augmentation.
    runMarkerAugmentation = True
    # OpenSim pipeline.
    runOpenSimPipeline = True    
    # High-resolution for OpenPose.
    resolutionPoseDetection = resolutionPoseDetection
    # Set to False to only generate the json files (default is True).
    # This speeds things up and saves storage space.
    generateVideo = True
    # This is a hack to handle a mismatch between the use of mmpose and hrnet,
    # and between the use of OpenPose and openpose.
    if poseDetector == 'hrnet':
        poseDetector = 'mmpose'        
    elif poseDetector == 'openpose':
        poseDetector = 'OpenPose'
    if poseDetector == 'mmpose':
        outputMediaFolder = 'OutputMedia_mmpose' + str(bbox_thr)
    elif poseDetector == 'OpenPose':
        outputMediaFolder = 'OutputMedia_' + resolutionPoseDetection
    
    # %% Special case: extrinsics trial.
    # For that trial, we only calibrate the cameras.
    if extrinsicsTrial:
        runCameraCalibration = True
        runPoseDetection = False
        runSynchronization = False
        runTriangulation =  False
        runMarkerAugmentation = False
        runOpenSimPipeline = False
        
    # %% Paths and metadata. This gets defined through web app.
    baseDir = os.path.dirname(os.path.abspath(__file__))
    if dataDir is None:
        dataDir = getDataDirectory(isDocker)
    if 'dataDir' not in locals():
        sessionDir = os.path.join(baseDir, 'Data', sessionName)
    else:
        sessionDir = os.path.join(dataDir, 'Data', sessionName)
    sessionMetadata = importMetadata(os.path.join(sessionDir,
                                                  'sessionMetadata.yaml'))
    
    # If augmenter model defined through web app.
    # If overwriteAugmenterModel is True, the augmenter model is the one
    # passed as an argument to main(). This is useful for local testing.
    if 'augmentermodel' in sessionMetadata and not overwriteAugmenterModel:
        augmenterModel = sessionMetadata['augmentermodel']
    else:
        augmenterModel = augmenter_model
        
    # Lowpass filter frequency of 2D keypoints for gait and everything else.
    # If overwriteFilterFrequency is True, the filter frequency is the one
    # passed as an argument to main(). This is useful for local testing.
    if 'filterfrequency' in sessionMetadata and not overwriteFilterFrequency:
        filterfrequency = sessionMetadata['filterfrequency']
    else:
        filterfrequency = filter_frequency
    if filterfrequency == 'default':
        filtFreqs = {'gait':12, 'default':500} # defaults to framerate/2
    else:
        filtFreqs = {'gait':filterfrequency, 'default':filterfrequency}

    # If scaling setup defined through web app.
    # If overwriteScalingSetup is True, the scaling setup is the one
    # passed as an argument to main(). This is useful for local testing.
    if 'scalingsetup' in sessionMetadata and not overwriteScalingSetup:
        scalingSetup = sessionMetadata['scalingsetup']
    else:
        scalingSetup = scaling_setup

    # If camerastouse is in sessionMetadata, reprocess with specified cameras.
    # This allows reprocessing trials with missing videos. If
    # overwriteCamerasToUse is True, the camera selection is the one
    # passed as an argument to main(). This is useful for local testing.
    if 'camerastouse' in sessionMetadata and not overwriteCamerasToUse:
        camerasToUse = sessionMetadata['camerastouse']
    else:
        camerasToUse = cameras_to_use

    # We'll use syncVer if provided to this function. If not, try to use one 
    # from sessionMetadata, otherwise use the default one.
    syncVer = syncVer or sessionMetadata.get('sync_ver', DEFAULT_SYNC_VER)

    # %% Paths to pose detector folder for local testing.
    if poseDetector == 'OpenPose':
        poseDetectorDirectory = getOpenPoseDirectory(isDocker)
    elif poseDetector == 'mmpose':
        poseDetectorDirectory = getMMposeDirectory(isDocker)    
        
    # %% Create marker folders
    # Create output folder.
    if genericFolderNames:
        markerDataFolderName = os.path.join('MarkerData') 
    else:
        if poseDetector == 'mmpose':
            suff_pd = '_' + str(bbox_thr)
        elif poseDetector == 'OpenPose':
            suff_pd = '_' + resolutionPoseDetection
                
        markerDataFolderName = os.path.join('MarkerData', 
                                            poseDetector + suff_pd) 
        if not markerDataFolderNameSuffix is None:
            markerDataFolderName = os.path.join(markerDataFolderName,
                                                markerDataFolderNameSuffix)
    preAugmentationDir = os.path.join(sessionDir, markerDataFolderName,
                                      'PreAugmentation')
    os.makedirs(preAugmentationDir, exist_ok=True)
    
    # Create augmented marker folders as well
    if genericFolderNames:
        postAugmentationDir = os.path.join(sessionDir, markerDataFolderName, 
                                           'PostAugmentation')
    else:
        postAugmentationDir = os.path.join(
            sessionDir, markerDataFolderName, 
            'PostAugmentation_{}'.format(augmenterModel))
    os.makedirs(postAugmentationDir, exist_ok=True)
        
    # %% Dump settings in yaml.
    if not extrinsicsTrial:
        pathSettings = os.path.join(postAugmentationDir,
                                    'Settings_' + trial_id + '.yaml')
        settings = {
            'poseDetector': poseDetector, 
            'augmenter_model': augmenterModel, 
            'imageUpsampleFactor': imageUpsampleFactor,
            'openSimModel': sessionMetadata['openSimModel'],
            'scalingSetup': scalingSetup,
            'filterFrequency': filterfrequency,
            }
        if poseDetector == 'OpenPose':
            settings['resolutionPoseDetection'] = resolutionPoseDetection
        elif poseDetector == 'mmpose':
            settings['bbox_thr'] = bbox_thr
        with open(pathSettings, 'w') as file:
            yaml.dump(settings, file)

    # %% Camera calibration.
    if runCameraCalibration:    
        # Get checkerboard parameters from metadata.
        CheckerBoardParams = {
            'dimensions': (
                sessionMetadata['checkerBoard']['black2BlackCornersWidth_n'],
                sessionMetadata['checkerBoard']['black2BlackCornersHeight_n']),
            'squareSize': 
                sessionMetadata['checkerBoard']['squareSideLength_mm']}       
        # Camera directories and models.
        cameraDirectories = {}
        cameraModels = {}
        for pathCam in glob.glob(os.path.join(sessionDir, 'Videos', 'Cam*')):
            if os.name == 'nt': # windows
                camName = pathCam.split('\\')[-1]
            elif os.name == 'posix': # ubuntu
                camName = pathCam.split('/')[-1]
            cameraDirectories[camName] = os.path.join(sessionDir, 'Videos',
                                                      pathCam)
            cameraModels[camName] = sessionMetadata['iphoneModel'][camName]        
        
        # Get cameras' intrinsics and extrinsics.     
        # Load parameters if saved, compute and save them if not.
        CamParamDict = {}
        loadedCamParams = {}
        for camName in cameraDirectories:
            camDir = cameraDirectories[camName]
            # Intrinsics ######################################################
            # Intrinsics and extrinsics already exist for this session.
            if os.path.exists(
                    os.path.join(camDir,"cameraIntrinsicsExtrinsics.pickle")):
                logging.info("Load extrinsics for {} - already existing".format(
                    camName))
                CamParams = loadCameraParameters(
                    os.path.join(camDir, "cameraIntrinsicsExtrinsics.pickle"))
                loadedCamParams[camName] = True
                
            # Extrinsics do not exist for this session.
            else:
                logging.info("Compute extrinsics for {} - not yet existing".format(camName))
                # Intrinsics ##################################################
                # Intrinsics directories.
                intrinsicDir = os.path.join(baseDir, 'CameraIntrinsics',
                                            cameraModels[camName])
                permIntrinsicDir = os.path.join(intrinsicDir, 
                                                intrinsicsFinalFolder)            
                # Intrinsics exist.
                if os.path.exists(permIntrinsicDir):
                    CamParams = loadCameraParameters(
                        os.path.join(permIntrinsicDir,
                                      'cameraIntrinsics.pickle'))                    
                # Intrinsics do not exist throw an error. Eventually the
                # webapp will give you the opportunity to compute them.
                
                else:
                    exception = "Intrinsics don't exist for your camera model. OpenCap supports all iOS devices released in 2018 or later: https://www.opencap.ai/get-started."
                    raise Exception(exception, exception)
                        
                # Extrinsics ##################################################
                # Compute extrinsics from images popped out of this trial.
                # Hopefully you get a clean shot of the checkerboard in at
                # least one frame of each camera.
                useSecondExtrinsicsSolution = (
                    alternateExtrinsics is not None and 
                    camName in alternateExtrinsics)
                pathVideoWithoutExtension = os.path.join(
                    camDir, 'InputMedia', trialName, trial_id)
                extension = getVideoExtension(pathVideoWithoutExtension)
                extrinsicPath = os.path.join(camDir, 'InputMedia', trialName, 
                                             trial_id + extension) 
                                              
                # Modify intrinsics if camera view is rotated
                CamParams = rotateIntrinsics(CamParams,extrinsicPath)
                
                # for 720p, imageUpsampleFactor=4 is best for small board
                try:
                    CamParams = calcExtrinsicsFromVideo(
                        extrinsicPath,CamParams, CheckerBoardParams, 
                        visualize=False, imageUpsampleFactor=imageUpsampleFactor,
                        useSecondExtrinsicsSolution = useSecondExtrinsicsSolution)
                except Exception as e:
                    if len(e.args) == 2: # specific exception
                        raise Exception(e.args[0], e.args[1])
                    elif len(e.args) == 1: # generic exception
                        exception = "Camera calibration failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about camera calibration and https://www.opencap.ai/troubleshooting for potential causes for a failed calibration."
                        raise Exception(exception, traceback.format_exc())
                loadedCamParams[camName] = False
                
       
            # Append camera parameters.
            if CamParams is not None:
                CamParamDict[camName] = CamParams.copy()
            else:
                CamParamDict[camName] = None

        # Save parameters if not existing yet.
        if not all([loadedCamParams[i] for i in loadedCamParams]):
            for camName in CamParamDict:
                saveCameraParameters(
                    os.path.join(cameraDirectories[camName],
                                 "cameraIntrinsicsExtrinsics.pickle"), 
                    CamParamDict[camName])
            
    # %% 3D reconstruction
    
    # Set output file name.
    pathOutputFiles = {}
    if benchmark:
        pathOutputFiles[trialName] = os.path.join(preAugmentationDir,
                                                  trialName + ".trc")
    else:
        pathOutputFiles[trialName] = os.path.join(preAugmentationDir,
                                                  trial_id + ".trc")
    
    # Trial relative path
    trialRelativePath = os.path.join('InputMedia', trialName, trial_id)
    
    if runPoseDetection:
        # Get rotation angles from motion capture environment to OpenSim.
        # Space-fixed are lowercase, Body-fixed are uppercase. 
        checkerBoardMount = sessionMetadata['checkerBoard']['placement']
        if checkerBoardMount == 'backWall' or checkerBoardMount == 'Perpendicular':
            # Detect if checkerboard is upside down.
            upsideDownChecker = isCheckerboardUpsideDown(CamParamDict)
            if upsideDownChecker:
                rotationAngles = {'y':-90}
            else:
                rotationAngles = {'y':90, 'z':180}
        elif checkerBoardMount == 'ground' or checkerBoardMount == 'Lying':
            rotationAngles = {'x':90, 'y':90}
        else:
            raise Exception('checkerBoard placement value in\
             sessionMetadata.yaml is not currently supported')
             
        # Detect all available cameras (ie, cameras with existing videos).
        cameras_available = []
        for camName in cameraDirectories:
            camDir = cameraDirectories[camName]
            pathVideoWithoutExtension = os.path.join(camDir, 'InputMedia', trialName, trial_id)
            if len(glob.glob(pathVideoWithoutExtension + '*')) == 0:
                print(f"Camera {camName} does not have a video for trial {trial_id}")
            else:
                if os.path.exists(os.path.join(pathVideoWithoutExtension + getVideoExtension(pathVideoWithoutExtension))):
                    cameras_available.append(camName)
                else:
                    print(f"Camera {camName} does not have a video for trial {trial_id}")

        if camerasToUse[0] == 'all':
            cameras_all = list(cameraDirectories.keys())
            if not all([cam in cameras_available for cam in cameras_all]):
                exception = 'Not all cameras have uploaded videos; one or more cameras might have turned off or lost connection'
                raise Exception(exception, exception)
            else:
                camerasToUse_c = camerasToUse
        elif camerasToUse[0] == 'all_available':
            camerasToUse_c = cameras_available
            print(f"Using available cameras: {camerasToUse_c}")
        else:
            if not all([cam in cameras_available for cam in camerasToUse]):
                raise Exception('Not all specified cameras in camerasToUse have videos; verify the camera names or consider setting camerasToUse to ["all_available"]')
            else:
                camerasToUse_c = camerasToUse
                print(f"Using cameras: {camerasToUse_c}")
        settings['camerasToUse'] = camerasToUse_c
        if camerasToUse_c[0] != 'all' and len(camerasToUse_c) < 2:
            exception = 'At least two videos are required for 3D reconstruction, video upload likely failed for one or more cameras.'
            raise Exception(exception, exception)
            
        # For neutral, we do not allow reprocessing with not all cameras.
        # The reason is that it affects extrinsics selection, and then you can only process
        # dynamic trials with the same camera selection (ie, potentially not all cameras). 
        # This might be addressable, but I (Antoine) do not see an immediate need + this
        # would be a significant change in the code base. In practice, a data collection
        # will not go through neutral if not all cameras are available.
        if scaleModel:
            if camerasToUse_c[0] != 'all' and len(camerasToUse_c) < len(cameraDirectories):
                exception = 'All cameras are required for calibration and neutral pose.'
                raise Exception(exception, exception)
        
        # Run pose detection algorithm.
        try:        
            videoExtension = runPoseDetector(
                    cameraDirectories, trialRelativePath, poseDetectorDirectory,
                    trialName, CamParamDict=CamParamDict, 
                    resolutionPoseDetection=resolutionPoseDetection, 
                    generateVideo=generateVideo, cams2Use=camerasToUse_c,
                    poseDetector=poseDetector, bbox_thr=bbox_thr)
            trialRelativePath += videoExtension
        except Exception as e:
            if len(e.args) == 2: # specific exception
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1: # generic exception
                exception = """Pose detection failed. Verify your setup and try again. 
                    Visit https://www.opencap.ai/best-pratices to learn more about data collection
                    and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."""
                raise Exception(exception, traceback.format_exc())
      
    if runSynchronization:
        # Synchronize videos.
        try:
            keypoints2D, confidence, keypointNames, frameRate, nansInOut, startEndFrames, cameras2Use = (
                synchronizeVideos( 
                    cameraDirectories, trialRelativePath, poseDetectorDirectory,
                    undistortPoints=True, CamParamDict=CamParamDict,
                    filtFreqs=filtFreqs, confidenceThreshold=0.4,
                    imageBasedTracker=False, cams2Use=camerasToUse_c, 
                    poseDetector=poseDetector, trialName=trialName,
                    resolutionPoseDetection=resolutionPoseDetection,
                    syncVer=syncVer))
        except Exception as e:
            if len(e.args) == 2: # specific exception
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1: # generic exception
                exception = """Video synchronization failed. Verify your setup and try again. 
                    A fail-safe synchronization method is for the participant to
                    quickly raise one hand above their shoulders, then bring it back down. 
                    Visit https://www.opencap.ai/best-pratices to learn more about 
                    data collection and https://www.opencap.ai/troubleshooting for 
                    potential causes for a failed trial."""
                raise Exception(exception, traceback.format_exc())
                
    # Note: this should not be necessary, because we prevent reprocessing the neutral trial
    # with not all cameras, but keeping it in there in case we would want to.
    if calibrationOptions is not None:
        allCams = list(calibrationOptions.keys())
        for cam_t in allCams:
            if not cam_t in cameras2Use:
                calibrationOptions.pop(cam_t)
                
    if scaleModel and calibrationOptions is not None and alternateExtrinsics is None:
        # Automatically select the camera calibration to use
        CamParamDict = autoSelectExtrinsicSolution(sessionDir,keypoints2D,confidence,calibrationOptions)
     
    if runTriangulation:
        # Triangulate.
        try:
            keypoints3D, confidence3D = triangulateMultiviewVideo(
                CamParamDict, keypoints2D, ignoreMissingMarkers=False, 
                cams2Use=cameras2Use, confidenceDict=confidence,
                spline3dZeros = True, splineMaxFrames=int(frameRate/5), 
                nansInOut=nansInOut,CameraDirectories=cameraDirectories,
                trialName=trialName,startEndFrames=startEndFrames,trialID=trial_id,
                outputMediaFolder=outputMediaFolder)
        except Exception as e:
            if len(e.args) == 2: # specific exception
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1: # generic exception
                exception = "Triangulation failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."
                raise Exception(exception, traceback.format_exc())
        
        # Throw an error if not enough data
        if keypoints3D.shape[2] < 10:
            e1 = 'Error - less than 10 good frames of triangulated data.'
            raise Exception(e1,e1)
    
        # Write TRC.
        writeTRCfrom3DKeypoints(keypoints3D, pathOutputFiles[trialName],
                                keypointNames, frameRate=frameRate, 
                                rotationAngles=rotationAngles)
    
    # %% Augmentation.
    
    # Get augmenter model.
    augmenterModelName = (
        sessionMetadata['markerAugmentationSettings']['markerAugmenterModel'])
    
    # Set output file name.
    pathAugmentedOutputFiles = {}
    if genericFolderNames:
        pathAugmentedOutputFiles[trialName] = os.path.join(
                postAugmentationDir, trial_id + ".trc")
    else:
        if benchmark:
            pathAugmentedOutputFiles[trialName] = os.path.join(
                    postAugmentationDir, trialName + "_" + augmenterModelName +".trc")
        else:
            pathAugmentedOutputFiles[trialName] = os.path.join(
                    postAugmentationDir, trial_id + "_" + augmenterModelName +".trc")
    
    if runMarkerAugmentation:
        os.makedirs(postAugmentationDir, exist_ok=True)    
        augmenterDir = os.path.join(baseDir, "MarkerAugmenter")
        logging.info('Augmenting marker set')
        try:
            vertical_offset = augmentTRC(
                pathOutputFiles[trialName],sessionMetadata['mass_kg'], 
                sessionMetadata['height_m'], pathAugmentedOutputFiles[trialName],
                augmenterDir, augmenterModelName=augmenterModelName,
                augmenter_model=augmenterModel, offset=offset)
        except Exception as e:
            if len(e.args) == 2: # specific exception
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1: # generic exception
                exception = "Marker augmentation failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."
                raise Exception(exception, traceback.format_exc())
        if offset:
            # If offset, no need to offset again for the webapp visualization.
            # (0.01 so that there is no overall offset, see utilsOpenSim).
            vertical_offset_settings = float(np.copy(vertical_offset)-0.01)
            vertical_offset = 0.01   
        
    # %% OpenSim pipeline.
    if runOpenSimPipeline:
        openSimPipelineDir = os.path.join(baseDir, "opensimPipeline")        
        
        if genericFolderNames:
            openSimFolderName = 'OpenSimData'
        else:
            openSimFolderName = os.path.join('OpenSimData', 
                                             poseDetector + suff_pd)
            if not markerDataFolderNameSuffix is None:
                openSimFolderName = os.path.join(openSimFolderName,
                                                 markerDataFolderNameSuffix)
        
        openSimDir = os.path.join(sessionDir, openSimFolderName)        
        outputScaledModelDir = os.path.join(openSimDir, 'Model')

        # Check if shoulder model.
        if 'shoulder' in sessionMetadata['openSimModel']:
            suffix_model = '_shoulder'
        else:
            suffix_model = ''
        
        # Scaling.    
        if scaleModel:
            os.makedirs(outputScaledModelDir, exist_ok=True)
            # Path setup file.
            if scalingSetup == 'any_pose':
                genericSetupFile4ScalingName = 'Setup_scaling_LaiUhlrich2022_any_pose.xml'
            else: # by default, use upright_standing_pose
                genericSetupFile4ScalingName = 'Setup_scaling_LaiUhlrich2022.xml'

            pathGenericSetupFile4Scaling = os.path.join(
                openSimPipelineDir, 'Scaling', genericSetupFile4ScalingName)
            # Path model file.
            pathGenericModel4Scaling = os.path.join(
                openSimPipelineDir, 'Models', 
                sessionMetadata['openSimModel'] + '.osim')            
            # Path TRC file.
            pathTRCFile4Scaling = pathAugmentedOutputFiles[trialName]
            # Get time range.
            try:
                thresholdPosition = 0.003
                maxThreshold = 0.015
                increment = 0.001
                success = False
                while thresholdPosition <= maxThreshold and not success:
                    try:
                        timeRange4Scaling = getScaleTimeRange(
                            pathTRCFile4Scaling,
                            thresholdPosition=thresholdPosition,
                            thresholdTime=0.1, removeRoot=True)
                        success = True
                    except Exception as e:
                        logging.info(f"Attempt identifying scaling time range with thresholdPosition {thresholdPosition} failed: {e}")
                        thresholdPosition += increment  # Increase the threshold for the next iteration

                # Run scale tool.
                logging.info('Running Scaling')
                pathScaledModel = runScaleTool(
                    pathGenericSetupFile4Scaling, pathGenericModel4Scaling,
                    sessionMetadata['mass_kg'], pathTRCFile4Scaling, 
                    timeRange4Scaling, outputScaledModelDir,
                    subjectHeight=sessionMetadata['height_m'], 
                    suffix_model=suffix_model)
            except Exception as e:
                if len(e.args) == 2: # specific exception
                    raise Exception(e.args[0], e.args[1])
                elif len(e.args) == 1: # generic exception
                    exception = "Musculoskeletal model scaling failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed neutral pose."
                    raise Exception(exception, traceback.format_exc())
            # Extract one frame from videos to verify neutral pose.
            staticImagesFolderDir = os.path.join(sessionDir, 
                                                 'NeutralPoseImages')
            os.makedirs(staticImagesFolderDir, exist_ok=True)
            popNeutralPoseImages(cameraDirectories, cameras2Use, 
                                 timeRange4Scaling[0], staticImagesFolderDir,
                                 trial_id, writeVideo = True)   
            pathOutputIK = pathScaledModel[:-5] + '.mot'
            pathModelIK = pathScaledModel
        
        # Inverse kinematics.
        if not scaleModel:
            outputIKDir = os.path.join(openSimDir, 'Kinematics')
            os.makedirs(outputIKDir, exist_ok=True)
            # Check if there is a scaled model.
            pathScaledModel = os.path.join(outputScaledModelDir, 
                                            sessionMetadata['openSimModel'] + 
                                            "_scaled.osim")
            if os.path.exists(pathScaledModel):
                # Path setup file.
                genericSetupFile4IKName = 'Setup_IK{}.xml'.format(suffix_model)
                pathGenericSetupFile4IK = os.path.join(
                    openSimPipelineDir, 'IK', genericSetupFile4IKName)
                # Path TRC file.
                pathTRCFile4IK = pathAugmentedOutputFiles[trialName]
                # Run IK tool. 
                logging.info('Running Inverse Kinematics')
                try:
                    pathOutputIK, pathModelIK = runIKTool(
                        pathGenericSetupFile4IK, pathScaledModel, 
                        pathTRCFile4IK, outputIKDir)
                except Exception as e:
                    if len(e.args) == 2: # specific exception
                        raise Exception(e.args[0], e.args[1])
                    elif len(e.args) == 1: # generic exception
                        exception = "Inverse kinematics failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."
                        raise Exception(exception, traceback.format_exc())
            else:
                raise ValueError("No scaled model available.")
        
        # Write body transforms to json for visualization.
        outputJsonVisDir = os.path.join(sessionDir,'VisualizerJsons',
                                        trialName)
        os.makedirs(outputJsonVisDir,exist_ok=True)
        outputJsonVisPath = os.path.join(outputJsonVisDir,
                                         trialName + '.json')
        generateVisualizerJson(pathModelIK, pathOutputIK,
                               outputJsonVisPath, 
                               vertical_offset=vertical_offset,
                               roundToRotations=4, roundToTranslations=4)
        
    # %% Rewrite settings, adding offset  
    if not extrinsicsTrial:
        if offset:
            settings['verticalOffset'] = vertical_offset_settings 
        with open(pathSettings, 'w') as file:
            yaml.dump(settings, file)
