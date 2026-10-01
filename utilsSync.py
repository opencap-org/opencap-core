import copy
import cv2
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import pchip_interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, gaussian, find_peaks, sosfiltfilt
from scipy import signal
import scipy.linalg

from utils import (getOpenPoseMarkerNames, getOpenPoseFaceMarkers, 
                   delete_multiple_element)
from utilsChecker import (loadPklVideo, unpackKeypointList, 
                          keypointsToBoundingBox, calcReprojectionError,
                          triangulateMultiviewVideo)
from utilsCameraPy3 import Camera
from defaults import DEFAULT_SYNC_VER

# %%
def synchronizeVideos(CameraDirectories, trialRelativePath, pathPoseDetector,
                      undistortPoints=False, CamParamDict=None, 
                      confidenceThreshold=0.3, 
                      filtFreqs={'gait':12,'default':30},
                      imageBasedTracker=False, cams2Use=['all'],
                      poseDetector='OpenPose', trialName=None, bbox_thr=0.8,
                      resolutionPoseDetection='default', 
                      visualizeKeypointAnimation=False,
                      syncVer=DEFAULT_SYNC_VER):
    
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
            poseDetector=poseDetector,confidenceThresholdForBB=0.3)
        thisVideo = cv2.VideoCapture(videoFullPath.replace('.mov', '_rotated.avi'))
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

    # Creates a web animation for each camera's keypoints. For debugging.
    if visualizeKeypointAnimation:
        import plotly.express as px
        import plotly.io as pio
        pio.renderers.default = 'browser'
    
        for i,data in enumerate(pointList):
        
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
            fig = px.scatter(df,x='x', y='y', title="Cam " + str(i),
                              animation_frame='Frame',
                              range_x=[0, 1200], range_y=[1200,0],
                              color='Point', color_continuous_scale=px.colors.sequential.Viridis)
    
            # Show the animation
            fig.show()

    # Synchronize keypoints.
    pointList, confList, nansInOutList,startEndFrameList = synchronizeVideoKeypoints(
        pointList, confList, confidenceThreshold=confidenceThreshold,
        filtFreqs=filtFreqs, sampleFreq=frameRate, visualize=False,
        maxShiftSteps=2*frameRate, CameraParams=CamParamList_selectedCams,
        cameras2Use=cameras2Use, 
        CameraDirectories=CameraDirectories_selectedCams, trialName=trialName,
        syncVer=syncVer)
    
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
                              filtFreqs = {'gait':12,'default':500},
                              sampleFreq=30, visualize=False, maxShiftSteps=30,
                              isGait=False, CameraParams = None,
                              cameras2Use=['none'],CameraDirectories = None,
                              trialName=None, trialID='',
                              syncVer=DEFAULT_SYNC_VER):
    visualize2Dkeypoint = False # this is a visualization just for testing what filtered input data looks like
    
    # keypointList is a mCamera length list of (nmkrs,nTimesteps,2) arrays of camera 2D keypoints
    logging.info(f'Synchronizing Keypoints using version {syncVer}')
    
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
    cameras2NotUse = []
    for idxBadCamera in idxBadCameras:
        print('{} kicked out of synchronization'.format(
            c_cameras2Use[idxBadCamera]))
        cameras2NotUse.append(c_cameras2Use[idxBadCamera])
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
    inHandPunchVertPositionList = []
    inHandPunchConfidenceList = []
    allMarkerList = []
    for (keyRaw,conf) in zip(keypointList,confidenceList):
        keyRaw_clean, _, _, _ = clean2Dkeypoints(keyRaw,conf,confidenceThreshold=0.3,nCams=nCams,linearInterp=True)        
        keyRaw_clean_smooth = smoothKeypoints(keyRaw_clean, sdKernel=3) 
        inHandPunchVertPositionList.append(getPositions(keyRaw_clean_smooth,markers4HandPunch,direction=1))
        inHandPunchConfidenceList.append(conf[markers4HandPunch])
        vertVelList.append(getVertVelocity(keyRaw_clean_smooth)) # doing it again b/c these settings work well for synchronization
        mkrSpeedList.append(getMarkerSpeed(keyRaw_clean_smooth,markers4VertVel,confidence=conf,averageVels=False)) # doing it again b/c these settings work well for synchronization
        allMarkerList.append(keyRaw_clean_smooth)
    
    # Prepare hand punch data
    # Clip the end of the hand punch data to the shortest length of the hand punch data
    min_hand_frames = min(np.shape(p)[1] for p in inHandPunchVertPositionList)
    inHandPunchVertPositionList = [p[:, :min_hand_frames] for p in inHandPunchVertPositionList]
    inHandPunchConfidenceList = [c[:, :min_hand_frames] for c in inHandPunchConfidenceList]

    # For sync 1.1, we keep the original input (inHandPunchVertPositionList)
    # but make a copy that is clipped after checks below (for sync 1.0)
    clippedHandPunchVertPositionList = copy.deepcopy(inHandPunchVertPositionList)
    clippedHandPunchConfidenceList = copy.deepcopy(inHandPunchConfidenceList)
    
    # Prepare data for syncing without hand punch
    # Find indices with high confidence that overlap between cameras with
    # ankle markers.
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
    clippedHandPunchVertPositionList = [p[:,idxStart:idxEnd] for p in clippedHandPunchVertPositionList]
    clippedHandPunchConfidenceList = [c[:,idxStart:idxEnd] for c in clippedHandPunchConfidenceList]
    allMarkerList = [p[:,idxStart:idxEnd] for p in allMarkerList]
    confSyncList= [c[:,idxStart:idxEnd] for c in confidenceList]
    
    # We do this again, since it might have changed after finding the overlap period.
    keypointList = list(keypointList)
    confidenceList = list(confidenceList)
    badCamerasOverlap = []
    for icam, conf in enumerate(confSyncList):
        if np.mean(conf) <= 0.01: # Looser than sum=0 to disregard very few frames with data
            badCamerasOverlap.append(icam)
    idxbadCamerasOverlap = [badCamerasOverlap[i]-i for i in range(len(badCamerasOverlap))]
    for idxbadCameraOverlap in idxbadCamerasOverlap:
        print('{} kicked out of synchronization - after overlap check'.format(
            c_cameras2Use[idxbadCameraOverlap]))
        cameras2NotUse.append(c_cameras2Use[idxbadCameraOverlap])
        keypointList.pop(idxbadCameraOverlap)
        confidenceList.pop(idxbadCameraOverlap)
        c_CameraParams.pop(idxbadCameraOverlap)
        c_cameras2Use.pop(idxbadCameraOverlap)
        c_CameraDirectories.pop(idxbadCameraOverlap)
        
        vertVelList.pop(idxbadCameraOverlap)
        mkrSpeedList.pop(idxbadCameraOverlap)
        clippedHandPunchVertPositionList.pop(idxbadCameraOverlap)
        clippedHandPunchConfidenceList.pop(idxbadCameraOverlap)
        inHandPunchVertPositionList.pop(idxbadCameraOverlap)
        inHandPunchConfidenceList.pop(idxbadCameraOverlap)
        allMarkerList.pop(idxbadCameraOverlap)
        confSyncList.pop(idxbadCameraOverlap)        
    nCams = len(keypointList)
    
    # Detect activities, which determines sync function and filtering
    # that gets used

    # Gait trial: Input right and left ankle marker speeds. Gait should be
    # detected for all cameras (all but one camera is > 2 cameras) for the
    # trial to be considered a gait trial.
    try:
        isGait = detectGaitAllVideos(mkrSpeedList,allMarkerList,confSyncList,markers4Ankles,sampleFreq)
    except:
        isGait = False
        print('Detect gait activity algorithm failed.')
    
    # Hand punch: Input right and left wrist and shoulder positions.
    # Different sync versions use different input data, handled by
    # dispatcher function detectHandPunchAllVideos.
    isHandPunch, handForPunch, handPunchRange = \
        detectHandPunchAllVideos(syncVer, 
                                 clippedHandPunchVertPositionList=clippedHandPunchVertPositionList,
                                 clippedHandPunchConfidenceList=clippedHandPunchConfidenceList,
                                 inHandPunchVertPositionList=inHandPunchVertPositionList,
                                 inHandPunchConfidenceList=inHandPunchConfidenceList,
                                 sampleFreq=sampleFreq,
                                 )

    if isHandPunch:
        syncActivity = 'handPunch'
    elif isGait:
        syncActivity = 'gait'
    else:
        syncActivity = 'general'
    logging.info(f'Using {syncActivity} sync function.')
    
    # Select filtering frequency based on if it is gait
    if isGait: 
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
                    corVal,lag = cross_corr(vertVel,vertVelList[0],multCorrGaussianStd=maxShiftSteps/2,
                                            visualize=False,dataForReproj=dataForReproj,
                                            frameRate=sampleFreq) # gaussian curve gets multipled by correlation plot - helping choose the smallest shift value for periodic motions
                
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
                                            visualize=False,
                                            frameRate=sampleFreq)
            elif syncActivity == 'handPunch':
                corVal,lag = syncHandPunch(syncVer,
                                           clippedHandPunchVertPositionList=[clippedHandPunchVertPositionList[i] for i in [0,iCam]],
                                           inHandPunchVertPositionList=[inHandPunchVertPositionList[i] for i in [0,iCam]],
                                           clippedHandPunchConfidenceList=[clippedHandPunchConfidenceList[i] for i in [0,iCam]],
                                           inHandPunchConfidenceList=[inHandPunchConfidenceList[i] for i in [0,iCam]],
                                           handForPunch=handForPunch,
                                           maxShiftSteps=maxShiftSteps,
                                           handPunchRange=handPunchRange,
                                           frameRate=sampleFreq,
                                           )
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
    idxCameras2NotUse = [cameras2Use.index(cam) for cam in cameras2NotUse]
    for idxCamera2NotUse in idxCameras2NotUse:
        keypointsSync.insert(idxCamera2NotUse, np.zeros(keypointsSync[0].shape))
        confidenceSync.insert(idxCamera2NotUse, np.zeros(confidenceSync[0].shape))
        nansInOutSync.insert(idxCamera2NotUse, np.array([np.nan, np.nan]))
        startEndFrames.insert(idxCamera2NotUse, None)
 
    return keypointsSync, confidenceSync, nansInOutSync, startEndFrames


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

# %%
def detectHandPunchAllVideos_v1(handPunchPositionList,sampleFreq,punchDurationThreshold=3):
    
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
                    if (startLength+endLength)/sampleFreq<punchDurationThreshold:
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
def detectHandPunchAllVideos_v2(handPunchPositionList, confList, sampleFreq,
                                punchDurationLimits=(0.2, 3.0), confThresh=0.5, 
                                maxPosSimilarityThresh=0.5, maxConfGap=4):
    """
    Detects a hand punch event across all cameras, determines which hand
    (left or right) performed the punch, and finds the consensus frame
    range of the punch event. Uses wrist-over-shoulder position signals
    and confidence values.

    Args:
        handPunchPositionList (list of np.ndarray):
            List of arrays (one per camera), each of shape (4, nFrames),
            containing the vertical positions of the right wrist, left wrist,
            right shoulder, and left shoulder over time.
            Expected order: [r_wrist, l_wrist, r_shoulder, l_shoulder].
        confList (list of np.ndarray):
            List of arrays (one per camera), each of shape (4, nFrames),
            containing the confidence values for the corresponding markers in
            handPunchPositionList. Required for this function.
        sampleFreq (float):
            The sampling frequency (frames per second) of the video data.
        punchDurationLimits (tuple, optional):
            (min_duration, max_duration) in seconds for a punch event.
            Default is (0.2, 3).
        confThresh (float, optional):
            Confidence threshold multiplier for determining high-confidence
            stretches. Default is 0.5.
        maxPosSimilarityThresh (float, optional):
            Threshold for how similar two punch heights can be on the same arm
            before the punch is considered ambiguous. Default is 0.5.
        maxConfGap (int, optional):
            Maximum allowed gap (in frames) of low confidence within a
            high-confidence stretch. Default is 4.

    Returns:
        isTrialPunch (bool):
            True if a valid punch is detected in all cameras and by the same
            hand, False otherwise.
        hand (str or None):
            'r' if the right hand performed the punch, 'l' if the left hand,
            or None if no valid punch is found.
        handPunchRange (list or None):
            [start_idx, end_idx] of the possible punch event (frame indices) 
            across all cameras, or None if no valid punch is found.
    """
    if confList is None:
        raise Exception('list of confidences need to be passed in to "conf"')
    
    # all positions and confs should have the same number of markers and frames
    if any(np.shape(p) != np.shape(handPunchPositionList[0]) for p in handPunchPositionList):
        raise Exception('all positions should have the same number of markers')
    if any(np.shape(c) != np.shape(handPunchPositionList[0]) for c in confList):
        raise Exception('all confs should have the same number of frames')

    punch_duration_min, punch_duration_max = punchDurationLimits

    # Loop over each camera, adding cleanest possible punch event for each camera.
    cam_punch_list = []
    for pos, conf in zip(handPunchPositionList, confList):
        valid_stretches = []
       
        # Loop over each side, starting with right side of arrays in
        # handPunchPositionList.
        # Expected order: ['r_wrist', 'l_wrist', 'r_shoulder', 'l_shoulder']
        for side, indices in zip(['r', 'l'], [
            {'wrist': 0, 'shoulder': 2, 'other_wrist': 1, 'other_shoulder': 3},
            {'wrist': 1, 'shoulder': 3, 'other_wrist': 0, 'other_shoulder': 2}
        ]):
            # Find the highest relative position of the wrist over shoulder
            relPos = np.diff(pos[(indices['shoulder'], indices['wrist']), :], axis=0).squeeze()

            # Find all stretches when wrist is above shoulder
            positiveInd = relPos > 0
            starts = np.where((~positiveInd[:-1]) & (positiveInd[1:]))[0]
            ends = np.where((positiveInd[:-1]) & (~positiveInd[1:]))[0] + 1

            # Account for cases where hand starts or ends above shoulder
            if positiveInd[0]:
                starts = np.insert(starts, 0, 0)
            if positiveInd[-1]:
                ends = np.append(ends, len(relPos))

            positiveStretches = list(zip(starts, ends))
            
            # Ensure stretches are valid:
            # 1. punch duration is within time limits
            # 2. high confidence
            # 3. other wrist is below other shoulder
            for start, end in positiveStretches:
                duration = (end - start) / sampleFreq
                if duration < punch_duration_min or duration > punch_duration_max:
                    continue 

                # Require minimum confidence of wrist and shoulder to be
                # above 'confThresh' of the maximum of the minimum confidence 
                # values.
                # Gaps of 'maxConfGap' or fewer frames of low confidence are allowed.
                conf_wrist = conf[indices['wrist']]
                conf_shoulder = conf[indices['shoulder']]
                high_conf_indices = \
                    find_longest_confidence_stretch_in_range_with_gaps(
                        [conf_wrist, conf_shoulder], confThresh, maxConfGap,
                        rangeList=[start, end])
                # Check if the high confidence region covers the entire stretch
                if (
                    high_conf_indices is None or
                    high_conf_indices[0] > start or
                    high_conf_indices[1] < end
                ):
                    continue

                other_relPos = np.diff(pos[(indices['other_shoulder'], indices['other_wrist']), :], axis=0).squeeze()
                other_relPos_stretch = other_relPos[start:end]
                if np.any(other_relPos_stretch >= 0):
                    continue 

                # If all checks passed, add to valid stretches
                stretch_relPos = relPos[start:end]
                stretch_maxpos = np.max(stretch_relPos)
                valid_stretches.append({
                    'crossings': [start, end],
                    'maxPos': stretch_maxpos,
                    'hand': side
                })

        # Use the side with the higher punch. Do not use if there is
        # another valid punch with a similar height (maxPosSimilarityThresh) 
        # on the same arm.
        if valid_stretches:
            max_pos_idx = np.argmax([s['maxPos'] for s in valid_stretches])
            max_pos_hand = valid_stretches[max_pos_idx]['hand']
            max_pos_val = valid_stretches[max_pos_idx]['maxPos']
            other_max_pos_val = np.array([s['maxPos'] for i, s in enumerate(valid_stretches) if i != max_pos_idx and s['hand'] == max_pos_hand])
            if other_max_pos_val.size and np.any(other_max_pos_val / max_pos_val > maxPosSimilarityThresh):
                cam_punch_list.append(None)
            else:
                cam_punch_list.append(valid_stretches[max_pos_idx])
        else:
            cam_punch_list.append(None)
    
    # across all cameras check if: 
    # 1. there is a punch in all trials 
    # 2. it's the same hand each time.
    # 3. confidence is good across the widest range of indices of crossings
    isTrialPunch = all(cam_punch is not None for cam_punch in cam_punch_list)
    hand = None
    handPunchRange = None
    
    if isTrialPunch:
        hands = [cam_punch['hand'] for cam_punch in cam_punch_list]
        if len(set(hands)) == 1:
            crossingsList = [cam_punch['crossings'] for cam_punch in cam_punch_list]
            handPunchRange = [np.min(crossingsList), np.max(crossingsList)]

            confMeanList = [np.mean(c, axis=0) for c in confList]
            high_conf_indices = \
                find_longest_confidence_stretch_in_range_with_gaps(
                    confMeanList, confThresh, maxConfGap, rangeList=handPunchRange)
            # Check if the high confidence region covers the entire hand punch range
            if high_conf_indices is not None and high_conf_indices[0] == handPunchRange[0] and high_conf_indices[1] == handPunchRange[1]:
                hand = hands[0]
            else:
                isTrialPunch = False
                handPunchRange = None

        else:
            isTrialPunch = False
    
    return isTrialPunch, hand, handPunchRange

# %%
def detectHandPunchAllVideos(syncVer, **kwargs):
    """
    Dispatcher and version-specific logic for detecting hand punch.

    Args:
        syncVer (str):
            Version of the hand punch detection algorithm.
        **kwargs:
            Keyword arguments for the hand punch detection algorithm.

    Returns:
        isTrialPunch (bool):
            True if a valid punch is detected in all cameras and by the same
            hand, False otherwise.
        hand (str or None):
            'r' if the right hand performed the punch, 'l' if the left hand,
            or None if no valid punch is found.
        handPunchRange (list or None):
            Version 1.0: None
            Version 1.1: [start_idx, end_idx] of the possible punch event 
            (frame indices) across all cameras, or None if no valid punch is 
            found.

    Raises:
        ValueError: If the sync version is not supported.
    """
    if syncVer == '1.0':
        handPunchVertPositionList = kwargs.get('clippedHandPunchVertPositionList', None)
        sampleFreq = kwargs.get('sampleFreq', None)

        isTrialPunch, hand = detectHandPunchAllVideos_v1(handPunchVertPositionList,
                                                         sampleFreq,
                                                         punchDurationThreshold=3)
        return isTrialPunch, hand, None
    
    elif syncVer == '1.1':
        handPunchVertPositionList = kwargs.get('inHandPunchVertPositionList', None)
        confList = kwargs.get('inHandPunchConfidenceList', None)
        sampleFreq = kwargs.get('sampleFreq', None)

        punchDurationLimits = (0.2, 3.0)
        confThresh = 0.5
        maxPosSimilarityThresh = 0.7
        maxConfGap = 4

        return detectHandPunchAllVideos_v2(handPunchVertPositionList,
                                           confList,
                                           sampleFreq,
                                           punchDurationLimits=punchDurationLimits,
                                           confThresh=confThresh,
                                           maxPosSimilarityThresh=maxPosSimilarityThresh,
                                           maxConfGap=maxConfGap)
    else:
        raise ValueError(f'Unsupported sync version: {syncVer}')

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
def syncHandPunch_v1(positions,hand,maxShiftSteps=600):
    if hand == 'r':
        startInd = 0
    else:
        startInd = 1
        
    relVel = []
    for pos in positions:
        relPos = -np.diff(pos[(startInd, startInd+2),:],axis=0) # vertical position of wrist over shoulder
        relVel.append(np.squeeze(np.diff(relPos)))
        
    corr_val,lag = cross_corr(relVel[1],relVel[0],multCorrGaussianStd=maxShiftSteps,visualize=False)
    
    logging.debug(f'corr_val: {corr_val}, lag: {lag}')
    return corr_val, lag

# %% 
def syncHandPunch_v2(positionsList, hand, confList, handPunchRange, frameRate,
                     padTime=1.0, confThresh=0.5, maxConfGap=4,
                     signalType='velocity', signalFilterFreq=6.0):
    """
    Synchronize two hand punch signals from two cameras by aligning the punch
    event using correlation (and optionally reprojection error minimization).

    This function computes the lag (frame shift) that best aligns the punch
    event between two cameras, focusing on the period of the punch and
    optionally refining the alignment using reprojection error. It supports
    both velocity and position-based synchronization, and can restrict the
    search to high-confidence regions.

    Args:
        positionsList (list of np.ndarray):
            List of arrays (one per camera), each of shape (4, nFrames),
            containing the vertical positions of the right wrist, left wrist,
            right shoulder, and left shoulder over time. Expected order:
            [r_wrist, l_wrist, r_shoulder, l_shoulder].
        hand (str):
            'r' for right hand punch, 'l' for left hand punch.
        confList (list of np.ndarray):
            List of arrays (one per camera), each of shape (4, nFrames),
            containing the confidence values for the corresponding markers in
            positionsList.
        handPunchRange (list of int):
            [start_idx, end_idx] of the punch event (frame indices) to focus
            the synchronization.
        frameRate (float):
            Frame rate (frames per second) of the video.
        padTime (float, optional):
            Time (in seconds) to pad before and after the punch event when
            searching for the best lag. If None, uses the full range.
            Default is 1.0.
        confThresh (float, optional):
            Confidence threshold multiplier for determining high-confidence
            stretches. Default is 0.5.
        maxConfGap (int, optional):
            Maximum allowed gap (in frames) of low confidence within a
            high-confidence stretch. Default is 4.
        signalType (str, optional):
            'velocity' to use the derivative of the wrist-over-shoulder
            position, 'position' to use the position directly. Default is
            'velocity'.
        signalFilterFreq (float or None, optional):
            Frequency (in Hz) to filter the signals before correlation.
            None is no filtering. Default is 6.0.

    Returns:
        corr_val (float):
            Maximum correlation value found between the two signals.
        lag (int):
            The lag (in frames) that best aligns the punch event between the
            two cameras.

    Raises:
        Exception: If required arguments are missing or inconsistent.
    """
    if confList is None:
        raise Exception('list of confidences need to be passed in to "conf"')
    
    if handPunchRange is None:
        raise Exception('"handPunchRange" must be provided to give start and end indices')

    if frameRate is None:
        raise Exception('video frequency was not specified')

    if len(positionsList) != len(confList):
        raise Exception('length of "positions" and "conf" lists are not equal')

    # all positions and confs should have the same number of frames and markers
    if any(np.shape(p) != np.shape(positionsList[0]) for p in positionsList):
        raise Exception('all positions should have the same number of frames and markers')
    if any(np.shape(c) != np.shape(positionsList[0]) for c in confList):
        raise Exception('all confs should have the same number of frames and markers')

    # expected order of positions and conf lists:
    # [r_wrist, l_wrist, r_shoulder, l_shoulder]
    if hand == 'r':
        shoulderInd = 2
        wristInd = 0
    elif hand == 'l':
        shoulderInd = 3
        wristInd = 1
    else:
        raise Exception(f'hand must be either "r" or "l", but was "{hand}"')

    # Set initial search range based on padTime
    confMeanList = [np.mean(c,axis=0) for c in confList]
    if padTime is not None:
        # Use padded region around hand punch
        padFrames = int(padTime * frameRate)
        search_start = max(0, handPunchRange[0] - padFrames)
        search_end = min(positionsList[0].shape[1] - 1, handPunchRange[1] + padFrames)
        rangeList = [search_start, search_end]
    else:
        # Use the entire input range
        rangeList = None
    
    # Find the widest high confidence region that contains the hand punch range
    high_conf_indices = \
        find_longest_confidence_stretch_in_range_with_gaps(
            confMeanList, confThresh, maxConfGap, rangeList=rangeList)
    
    if high_conf_indices is not None:
        # Ensure the hand punch range is contained within the high confidence region
        if high_conf_indices[0] <= handPunchRange[0] and high_conf_indices[1] >= handPunchRange[1]:
            # High confidence region contains the hand punch range
            startInd = high_conf_indices[0]
            endInd = high_conf_indices[1]
        else:
            # High confidence region doesn't contain hand punch range, fall back
            startInd = handPunchRange[0]
            endInd = handPunchRange[1]
    else:
        # Fall back to hand punch range
        startInd = handPunchRange[0]
        endInd = handPunchRange[1]

    relPosList = []
    relVelList = []
    for pos in positionsList:
        relPos = np.diff(pos[(shoulderInd, wristInd),startInd:endInd],axis=0).squeeze() # vertical position of wrist over shoulder
        relPosList.append(relPos)
        relVelList.append(np.diff(relPos))
    
    if signalFilterFreq is not None:
        order = 4
        wn = signalFilterFreq / (frameRate/2)
        sos = signal.butter(order, wn, 'low', analog=False, output='sos')
        relPosList = [signal.sosfiltfilt(sos, pos) for pos in relPosList]
        relVelList = [signal.sosfiltfilt(sos, vel) for vel in relVelList]
        
    if signalType == 'velocity':
        corr = signal.correlate(relVelList[1], relVelList[0], mode='full', method='auto')
        lags = signal.correlation_lags(len(relVelList[1]), len(relVelList[0]), mode='full')
    elif signalType == 'position':
        corr = signal.correlate(relPosList[1], relPosList[0], mode='full', method='auto')
        lags = signal.correlation_lags(len(relPosList[1]), len(relPosList[0]), mode='full')
    else:
        raise ValueError(f'Invalid signalType: {signalType}. Must be "velocity" or "position"')
    lag = lags[np.argmax(corr)]
    corr_val = np.max(corr)
    
    logging.debug(f'Using signalType: {signalType}, signalFilterFreq: {signalFilterFreq}')
    logging.debug(f'corr_val: {corr_val}, lag: {lag}')
    return corr_val, lag

# %%
def syncHandPunch(syncVer, **kwargs):
    """Dispatcher and version-specific logic for synchronizing hand punch.

    Args:
        syncVer (str):
            Version of the hand punch synchronization algorithm.
        **kwargs:
            Keyword arguments for the hand punch synchronization algorithm.

    Returns:
        corr_val (float):
            Maximum correlation value found between the two signals.
        lag (int):
            The lag (in frames) that best aligns the punch event between the
            two cameras.

    Raises:
        ValueError: If the sync version is not supported.
    """
    if syncVer == '1.0':
        positions = kwargs.get('clippedHandPunchVertPositionList', None)
        hand = kwargs.get('handForPunch', None)
        maxShiftSteps = kwargs.get('maxShiftSteps', 600)

        return syncHandPunch_v1(positions,
                                hand,
                                maxShiftSteps=maxShiftSteps)
    elif syncVer == '1.1':
        positionsList = kwargs.get('inHandPunchVertPositionList', None)
        hand = kwargs.get('handForPunch', None)
        confList = kwargs.get('inHandPunchConfidenceList', None)
        handPunchRange = kwargs.get('handPunchRange', None)
        frameRate = kwargs.get('frameRate', None)

        padTime = 1.0
        confThresh = 0.5
        maxConfGap = 4
        signalType = 'velocity'
        signalFilterFreq = 6.0

        return syncHandPunch_v2(positionsList,
                                hand,
                                confList,
                                handPunchRange,
                                frameRate,
                                padTime=padTime,
                                confThresh=confThresh,
                                maxConfGap=maxConfGap,
                                signalType=signalType,
                                signalFilterFreq=signalFilterFreq,
                                )
    else:
        raise ValueError(f'Unsupported synchronization version: {syncVer}')

# %%
def cross_corr(y1, y2,multCorrGaussianStd=None,visualize=False, dataForReproj=None, frameRate=60):
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
        
        # inject no delay so it doesn't throw an error for static trials
        if len(peaks['peak_heights']) == 0:
            peaks['peak_heights'] = np.ndarray((1,1))
            peaks['peak_heights'][0] = corr[int(len(corr)/2)]
            print('There were no peaks in the vert vel cross correlation. Using 0 lag.')
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
            lag_corr = lags[np.argmin(reprojError)]
            max_corr = corr[lag+shift]
            
            if multCorrGaussianStd is not None:
                print('For {}, used reprojection error minimization to sync.'.format(dataForReproj['cameras2Use'][dataForReproj['cams2UseReproj'][1]]))
                 
            # Refine with reprojection error minimization
            # Calculate reprojection error with lags +/- .2 seconds around the selected lag. Select the lag with the lowest reprojection error.
            # This helps the fact that correlation peak is not always the best lag, esp for front-facing cameras

            # Create a list of lags to test that is +/- .2 seconds around the selected lag based on frameRate
            numFrames = int(.2*frameRate)
            lags = np.arange(lag_corr-numFrames,lag_corr+numFrames+1)
            reprojErrors = np.empty((len(lags),1))

            for iLag,lag in enumerate(lags):            
                reprojErrors[iLag,0], _ = calcReprojectionErrorForSync(
                    dataForReproj['CamParamList'], dataForReproj['keypointList'],
                    lag, dataForReproj['cams2UseReproj'], 
                    dataForReproj['confidence'], dataForReproj['cameras2Use'])
                
            # Select the lag with the lowest reprojection error
            lag = lags[np.argmin(reprojErrors)]

            # plot the reproj errors against lag and identify which was lag_corr
            if visualize:
                plt.figure()
                plt.plot(lags,reprojErrors)
                plt.plot(lag_corr,reprojErrors[list(lags).index(lag_corr)],marker='o',color='r')
                plt.plot(lag,reprojErrors[list(lags).index(lag)],marker='o',color='k')
                plt.xlabel('lag')
                plt.ylabel('reprojection error')
                plt.title('Reprojection error vs lag')
                plt.legend(['reprojection error','corr lag','refined lag'])
                plt.show()
                
                
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
def cross_corr_multiple_timeseries(Y1, Y2,multCorrGaussianStd=None,dataForReproj=None,visualize=False,frameRate=60):
    
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
            lag_corr = lags[np.argmin(reprojError)]
            max_corr = summedCorr[lag+shift]
            
            if multCorrGaussianStd is not None:
                print('For {}, used reprojection error minimization to sync.'.format(dataForReproj['cameras2Use'][dataForReproj['cams2UseReproj'][1]]))
            
            # Refine with reprojection error minimization
            # Calculate reprojection error with lags +/- .2 seconds around the selected lag. Select the lag with the lowest reprojection error.
            # This helps the fact that correlation peak is not always the best lag, esp for front-facing cameras

            # Create a list of lags to test that is +/- .2 seconds around the selected lag based on frameRate
            numFrames = int(.2*frameRate)
            lags = np.arange(lag_corr-numFrames,lag_corr+numFrames+1)
            reprojErrors = np.empty((len(lags),1))

            for iLag,lag in enumerate(lags):            
                reprojErrors[iLag,0], _ = calcReprojectionErrorForSync(
                    dataForReproj['CamParamList'], dataForReproj['keypointList'],
                    lag, dataForReproj['cams2UseReproj'], 
                    dataForReproj['confidence'], dataForReproj['cameras2Use'])
                
            # Select the lag with the lowest reprojection error
            lag = lags[np.argmin(reprojErrors)]

            # plot the reproj errors against lag and identify which was lag_corr
            if visualize:
                plt.figure()
                plt.plot(lags,reprojErrors)
                plt.plot(lag_corr,reprojErrors[list(lags).index(lag_corr)],marker='o',color='r')
                plt.plot(lag,reprojErrors[list(lags).index(lag)],marker='o',color='k')
                plt.xlabel('lag')
                plt.ylabel('reprojection error')
                plt.title('Reprojection error vs lag')
                plt.legend(['reprojection error','corr lag','refined lag'])
                plt.show()

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
        print('You tried to filter ' + str(int(sampleFreq)) + ' Hz signal with cutoff freq of ' + str(int(filtFreq)) + '. Will filter at ' + str(int(sampleFreq/2)) + ' instead.')
        wn=0.99
    elif wn==1:
        wn=0.99
        
    sos = butter(order/2,wn,btype='low',output='sos')
    
    for i in range(2):
        key2D_out[:,:,i] = sosfiltfilt(sos,key2D_out[:,:,i],axis=1)             
        
    return key2D_out

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
def getPositions(keypoints,indsMarkers,direction=1):
                     
    positions = np.max(keypoints.max(axis=1)[:,1])-keypoints[indsMarkers,:,:]
                    
    return positions[:,:,direction]

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
def find_longest_confidence_stretch_in_range_with_gaps(confList, confThresh, maxConfGap,
                                                       rangeList=None):
    """
    Check if confidence values meet threshold requirements, allowing for small gaps.
    Minimum confidence is calculated based on the whole range of confidence values.
    If rangeList is provided, the confidence is checked in the specified range.
    
    Args:
        confList: List of confidence arrays (e.g., different markers or cameras)
        confThresh: Threshold multiplier for minimum confidence
        maxConfGap: Maximum allowed gap in frames for low confidence
        rangeList: List of [start_idx, end_idx] of the range to check.
                   If None, the confidence is checked in the whole range.
        
    Returns:
        list: [start_idx, end_idx] of the widest good stretch that meets the gap requirements.
              If no valid stretch found, returns None.
    """

    if any(len(c) != len(confList[0]) or c.ndim != 1 for c in confList):
        raise Exception('all confs should be 1-D arrays of the same length')

    # find minimum confidence based on whole range
    # suppress warnings since it's OK if all cameras are nan on some frames
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        min_conf = np.nanmin(np.array(confList), axis=0)

    # but still check that min_conf is not all nan
    if np.all(np.isnan(min_conf)):
        return None
    
    min_conf_thresh = confThresh * np.nanmax(min_conf)

    # check confidence in specified range
    if rangeList is not None:
        conf_range_array = np.array([c[rangeList[0]:rangeList[1]] for c in confList])
    else:
        conf_range_array = np.array(confList)
    
    # Find the widest high confidence stretch that allows for small gaps
    # in conf_range_array
    min_conf_range_array = np.nanmin(conf_range_array, axis=0)
    high_conf_frames = min_conf_range_array > min_conf_thresh
    if np.any(high_conf_frames):
        # Find all stretches of high confidence
        diff_high_conf = np.diff(np.concatenate(([False], high_conf_frames, [False])).astype(int))
        high_conf_starts = np.where(diff_high_conf == 1)[0]
        high_conf_ends = np.where(diff_high_conf == -1)[0]
        
        if len(high_conf_starts) > 0:
            # Find the longest stretch that allows for gaps up to maxConfGap
            # We need to merge stretches that are close enough together
            merged_stretches = []
            current_start = high_conf_starts[0]
            current_end = high_conf_ends[0]
            
            for i in range(1, len(high_conf_starts)):
                gap = high_conf_starts[i] - current_end
                if gap <= maxConfGap:
                    # Merge this stretch with the current one
                    current_end = high_conf_ends[i]
                else:
                    # Gap is too large, save current stretch and start new one
                    merged_stretches.append((current_start, current_end))
                    current_start = high_conf_starts[i]
                    current_end = high_conf_ends[i]
            
            # Don't forget the last stretch
            merged_stretches.append((current_start, current_end))
            
            if merged_stretches:
                # Find the longest merged stretch
                longest_stretch = max(merged_stretches, key=lambda x: x[1] - x[0])
                start_idx, end_idx = longest_stretch
                if rangeList is not None:
                    start_idx += rangeList[0]
                    end_idx += rangeList[0]
                return [start_idx, end_idx]
    
    # If no valid stretches found, return None
    return None
