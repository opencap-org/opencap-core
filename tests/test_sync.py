import glob
import logging
import os
import pickle
import pytest
import sys

import numpy as np

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir,'../'))
sys.path.append(repoDir)
from utils import loadCameraParameters
from utilsSync import synchronizeVideos, detectHandPunchAllVideos, syncHandPunch, syncHandPunch_v2

# Helper functions

def generate_test_signal_with_peaks(
    length,
    peaks=None,
    baseline=0.0
):
    """
    Generate a deterministic signal with sine-shaped peaks defined by start/end.

    Parameters:
    - length: int, total length of output signal
    - peaks: list of dicts with:
        - 'start': index where the hump begins (can be before 0)
        - 'end': index where the hump ends (can be after length)
        - 'amplitude': peak height above the baseline
    - baseline: float, baseline level for signal

    Returns:
    - signal: 1D NumPy array of shape (length,)
    - metadata: list of actual start/end/peak info per hump
    """
    signal = np.full(length, baseline)
    metadata = []

    if peaks is None:
        return signal, metadata

    for peak in peaks:
        start = int(peak['start'])
        end = int(peak['end'])
        amplitude = peak['amplitude']

        # Original peak duration and bounds
        duration = end - start
        if duration <= 0:
            raise ValueError("Peak duration must be positive ('end' must be after 'start').")

        # Intended sine wave (full peak)
        x = np.linspace(0, np.pi, duration)
        peak_shape = (amplitude - baseline) * np.sin(x)

        # Determine where the peak maps in the signal
        clip_start = max(start, 0)
        clip_end = min(end, length)

        # Adjust indices for slicing into the sine wave
        sine_start = clip_start - start
        sine_end = sine_start + (clip_end - clip_start)

        # Only add if something remains after clipping
        if sine_end > sine_start:
            signal[clip_start:clip_end] = peak_shape[sine_start:sine_end]

        metadata.append({
            'requested_start': start,
            'requested_end': end,
            'actual_start': clip_start,
            'actual_end': clip_end,
            'amplitude': amplitude
        })

    return signal, metadata

# Integration / regression tests for the synchronizeVideos function.
dataDir = os.path.join(thisDir, 'opencap-test-data')
sessionName = 'sync_2-cameras'
sessionDir = os.path.join(dataDir, 'Data', sessionName)
videosDir = os.path.join(dataDir, 'Data', sessionName, 'Videos')
cameraDirectories = {'Cam1': os.path.join(videosDir, 'Cam1'), 
                     'Cam0': os.path.join(videosDir, 'Cam0')}

trial_name_list = ['squats', 'walk', 'squats-with-arm-raise']
sync_str_list = ['Using general sync function.',
                 'Using gait sync function.',
                 'Using handPunch sync function.']
sync_ver_list = ['1.0', '1.1']

@pytest.mark.parametrize("sync_ver", sync_ver_list)
@pytest.mark.parametrize("trial_name,sync_str", zip(trial_name_list, sync_str_list))
def test_synchronize_videos(trial_name, sync_str, sync_ver, caplog):
    caplog.set_level(logging.INFO)
    trialRelativePath = os.path.join('InputMedia', trial_name, f'{trial_name}.mov')
    CamParamDict = {}
    for camName in cameraDirectories:
        camDir = cameraDirectories[camName]
        CamParams = loadCameraParameters(os.path.join(camDir, "cameraIntrinsicsExtrinsics.pickle"))
        CamParamDict[camName] = CamParams.copy()

    poseDetectorDirectory = ''
    undistortPoints = True
    filtFreqs = {'gait':12, 'default':500} # defaults from main()
    confidenceThreshold = 0.4
    imageBasedTracker = False
    camerasToUse_c = ['all']
    poseDetector = 'mmpose'
    resolutionPoseDetection = 'default'

    keypoints2D, confidence, keypointNames, frameRate, nansInOut, startEndFrames, cameras2Use = (
        synchronizeVideos( 
            cameraDirectories, trialRelativePath, poseDetectorDirectory,
            undistortPoints=undistortPoints, CamParamDict=CamParamDict,
            filtFreqs=filtFreqs, confidenceThreshold=confidenceThreshold,
            imageBasedTracker=imageBasedTracker, cams2Use=camerasToUse_c, 
            poseDetector=poseDetector, trialName=trial_name,
            resolutionPoseDetection=resolutionPoseDetection,
            syncVer=sync_ver))

    assert sync_str in caplog.text

    ref_pkl = os.path.join(sessionDir, 'OutputReference', f'sync_{trial_name}_output.pkl')
    with open(ref_pkl, 'rb') as f:
        (ref_keypoints2D, ref_confidence, ref_keypointNames, ref_frameRate,
            ref_nansInOut, ref_startEndFrames, ref_cameras2Use) = pickle.load(f)
    
    # Check that keypoints2D, confidence, nansInOut, and startEndFrames
    # share the same keys as each other
    assert set(keypoints2D.keys()) == set(confidence.keys()) == set(nansInOut.keys()) == set(startEndFrames.keys())

    for key_cam in keypoints2D:
        np.testing.assert_array_almost_equal(keypoints2D[key_cam], ref_keypoints2D[key_cam], err_msg=f'keypoints2D: {key_cam}')
        np.testing.assert_array_almost_equal(confidence[key_cam], ref_confidence[key_cam], err_msg=f'confidence: {key_cam}')
        np.testing.assert_array_almost_equal(nansInOut[key_cam], ref_nansInOut[key_cam], err_msg=f'nansInOut: {key_cam}')
        np.testing.assert_array_almost_equal(startEndFrames[key_cam], ref_startEndFrames[key_cam], err_msg=f'startEndFrames: {key_cam}')
    assert keypointNames == ref_keypointNames
    assert frameRate == ref_frameRate
    assert cameras2Use == ref_cameras2Use

# Unit tests for the detectHandPunch function and synchronizeHandPunch function.
class TestDetectHandPunch:

    @pytest.mark.parametrize(
        "sync_ver, hand, punch_range, ref_handPunchRange",
        [
            ('1.0', 'r', [200, 440], None),
            ('1.1', 'r', [200, 440], [200, 440]),
            ('1.0', 'l', [1000, 1200], None),
            ('1.1', 'l', [1000, 1200], [1000, 1200]),
        ]
    )
    def test_punch_clean_right_or_left(
        self, sync_ver, hand, punch_range, ref_handPunchRange
    ):
        
        sampleFreq = 240
        peak_start = punch_range[0]
        peak_end = punch_range[1]

        # Create a signal with a single zero crossing and set to one of the wrists.
        # Set other signals to flat values (same shoulder as 0 to make calculations
        # easier, and other wrist as slightly negative to be below shoulder)
        length = 2000
        peaks = [
            {'start': peak_start, 'end': peak_end, 'amplitude': 100.0},
        ]
        baseline = -0.1

        if hand == 'r':
            right_wrist_signal, _ = generate_test_signal_with_peaks(
                length=length, 
                peaks=peaks,
                baseline=baseline
            )
            left_wrist_signal = np.ones(length) * baseline # keep left wrist below shoulder

        else:
            left_wrist_signal, _ = generate_test_signal_with_peaks(
                length=length, 
                peaks=peaks,
                baseline=baseline
            )
            right_wrist_signal = np.ones(length) * baseline # keep right wrist below shoulder

        right_shoulder_signal = np.zeros(length)
        left_shoulder_signal = np.zeros(length)
        cam_signal_array = np.array([right_wrist_signal, left_wrist_signal, 
                              right_shoulder_signal, left_shoulder_signal])
        
        # Mimic data for each camera
        num_cameras = 2
        handPunchVertPositionList = [cam_signal_array] * num_cameras

        # High confidence for all signals
        cam_confidence_array = np.array([np.ones(length)] * np.shape(cam_signal_array)[0])
        handPunchConfidenceList = [cam_confidence_array] * num_cameras

        isHandPunch, handForPunch, handPunchRange = \
            detectHandPunchAllVideos(sync_ver, 
                                     clippedHandPunchVertPositionList=handPunchVertPositionList,
                                     clippedHandPunchConfidenceList=handPunchConfidenceList,
                                     inHandPunchVertPositionList=handPunchVertPositionList,
                                     inHandPunchConfidenceList=handPunchConfidenceList,
                                     sampleFreq=sampleFreq,
                                    )
        
        assert isHandPunch is True
        assert handForPunch == hand
        assert handPunchRange == ref_handPunchRange

    @pytest.mark.parametrize("sync_ver", sync_ver_list)
    def test_no_punch(self, sync_ver):
        length = 2000
        sampleFreq = 240
        
        right_wrist_signal = np.ones(length) * -0.1 # keep wrist below shoulder
        left_wrist_signal = np.ones(length) * -0.1
        right_shoulder_signal = np.zeros(length)
        left_shoulder_signal = np.zeros(length)
        cam_signal_array = np.array([right_wrist_signal, left_wrist_signal, 
                              right_shoulder_signal, left_shoulder_signal])
        
        # Mimic data for each camera
        num_cameras = 2
        handPunchVertPositionList = [cam_signal_array] * num_cameras

        # High confidence for all signals
        cam_confidence_array = np.array([np.ones(length)] * np.shape(cam_signal_array)[0])
        handPunchConfidenceList = [cam_confidence_array] * num_cameras

        isHandPunch, handForPunch, handPunchRange = \
            detectHandPunchAllVideos(sync_ver, 
                                     clippedHandPunchVertPositionList=handPunchVertPositionList,
                                     clippedHandPunchConfidenceList=handPunchConfidenceList,
                                     inHandPunchVertPositionList=handPunchVertPositionList,
                                     inHandPunchConfidenceList=handPunchConfidenceList,
                                     sampleFreq=sampleFreq,
                                    )

        assert isHandPunch is False
        assert handForPunch is None
        assert handPunchRange is None

    @pytest.mark.parametrize("sync_ver", sync_ver_list)
    def test_punch_both_hands_same_time(self, sync_ver):
        sampleFreq = 240
        peak_start_end_l = [300, 500]
        peak_start_end_r = [350, 450]

        # Create signals with a single crossing on both wrists,
        # making sure they occur at the same time (specifically,
        # peak of higher wrist (right) occurs when left wrist
        # is above shoulder). This should not be detected as a punch.
        length = 2000
        baseline = -0.1

        right_wrist_signal, _ = generate_test_signal_with_peaks(
            length=length, 
            peaks=[{'start': peak_start_end_r[0], 
                    'end': peak_start_end_r[1], 
                    'amplitude': 100.0}],
            baseline=baseline
        )

        left_wrist_signal, _ = generate_test_signal_with_peaks(
            length=length, 
            peaks=[{'start': peak_start_end_l[0], 
                    'end': peak_start_end_l[1], 
                    'amplitude': 90.0}],
            baseline=baseline
        )

        right_shoulder_signal = np.zeros(length)
        left_shoulder_signal = np.zeros(length)
        cam_signal_array = np.array([right_wrist_signal, left_wrist_signal, 
                              right_shoulder_signal, left_shoulder_signal])
        
        # Mimic data for each camera
        num_cameras = 2
        handPunchVertPositionList = [cam_signal_array] * num_cameras

        # High confidence for all signals
        cam_confidence_array = np.array([np.ones(length)] * np.shape(cam_signal_array)[0])
        handPunchConfidenceList = [cam_confidence_array] * num_cameras

        isHandPunch, handForPunch, handPunchRange = \
            detectHandPunchAllVideos(sync_ver, 
                                     clippedHandPunchVertPositionList=handPunchVertPositionList,
                                     clippedHandPunchConfidenceList=handPunchConfidenceList,
                                     inHandPunchVertPositionList=handPunchVertPositionList,
                                     inHandPunchConfidenceList=handPunchConfidenceList,
                                     sampleFreq=sampleFreq,
                                    )

        assert isHandPunch is False
        assert handForPunch is None
        assert handPunchRange is None

    @pytest.mark.parametrize("sync_ver", sync_ver_list)
    def test_punch_both_hands_different_times(self, sync_ver):
        # Create signals with a punch in each hand at different times
        length = 2000
        sampleFreq = 240
        right_peak = {'start': 200, 'end': 300, 'amplitude': 100.0}
        left_peak = {'start': 800, 'end': 900, 'amplitude': 120.0}
        baseline = -0.1
        right_wrist_signal, _ = generate_test_signal_with_peaks(
            length=length, peaks=[right_peak], baseline=baseline
        )
        left_wrist_signal, _ = generate_test_signal_with_peaks(
            length=length, peaks=[left_peak], baseline=baseline
        )
        right_shoulder_signal = np.zeros(length)
        left_shoulder_signal = np.zeros(length)
        cam_signal_array = np.array([
            right_wrist_signal, left_wrist_signal,
            right_shoulder_signal, left_shoulder_signal
        ])
        num_cameras = 2
        handPunchVertPositionList = [cam_signal_array] * num_cameras
        cam_confidence_array = np.ones((np.shape(cam_signal_array)[0], length))
        handPunchConfidenceList = [cam_confidence_array] * num_cameras
        isHandPunch, handForPunch, handPunchRange = detectHandPunchAllVideos(
            sync_ver,
            clippedHandPunchVertPositionList=handPunchVertPositionList,
            clippedHandPunchConfidenceList=handPunchConfidenceList,
            inHandPunchVertPositionList=handPunchVertPositionList,
            inHandPunchConfidenceList=handPunchConfidenceList,
            sampleFreq=sampleFreq,
        )
        # Expect detection of the left hand punch (higher amplitude)
        assert isHandPunch is True
        if sync_ver == '1.0':
            assert handForPunch == 'r' # sync 1.0 preferred right hand
            assert handPunchRange is None
        elif sync_ver == '1.1':
            assert handForPunch == 'l'
            assert handPunchRange[0] == left_peak['start']
            assert handPunchRange[1] == left_peak['end']

    @pytest.mark.parametrize("sync_ver", sync_ver_list)
    def test_multiple_punches_different_height(self, sync_ver):
        # Create a signal with two distinct peaks of different heights
        length = 2000
        sampleFreq = 240
        peak1 = {'start': 200, 'end': 300, 'amplitude': 100.0}
        peak2 = {'start': 600, 'end': 700, 'amplitude': 200.0}
        peaks = [peak1, peak2]
        baseline = -0.1
        right_wrist_signal, _ = generate_test_signal_with_peaks(
            length=length, peaks=peaks, baseline=baseline
        )
        left_wrist_signal = np.ones(length) * baseline
        right_shoulder_signal = np.zeros(length)
        left_shoulder_signal = np.zeros(length)
        cam_signal_array = np.array([
            right_wrist_signal, left_wrist_signal,
            right_shoulder_signal, left_shoulder_signal
        ])

        num_cameras = 2
        handPunchVertPositionList = [cam_signal_array] * num_cameras
        cam_confidence_array = np.array([np.ones(length)] * np.shape(cam_signal_array)[0])
        handPunchConfidenceList = [cam_confidence_array] * num_cameras
        isHandPunch, handForPunch, handPunchRange = detectHandPunchAllVideos(
            sync_ver,
            clippedHandPunchVertPositionList=handPunchVertPositionList,
            clippedHandPunchConfidenceList=handPunchConfidenceList,
            inHandPunchVertPositionList=handPunchVertPositionList,
            inHandPunchConfidenceList=handPunchConfidenceList,
            sampleFreq=sampleFreq,
        )
        # Expect detection of the larger peak (peak2)
        assert isHandPunch is True
        assert handForPunch == 'r'
        if sync_ver == '1.0':
            assert handPunchRange is None
        elif sync_ver == '1.1':
            assert handPunchRange[0] == peak2['start']
            assert handPunchRange[1] == peak2['end']
        else:
            raise ValueError(f"Unexpected sync version: {sync_ver}")

    @pytest.mark.parametrize("sync_ver", sync_ver_list)
    def test_multiple_punches_similar_height(self, sync_ver):
        # Create a signal with two peaks of similar heights
        length = 2000
        sampleFreq = 240
        peak1 = {'start': 200, 'end': 300, 'amplitude': 100.0}
        peak2 = {'start': 600, 'end': 700, 'amplitude': 99.9}  # very close to peak1
        peaks = [peak1, peak2]
        baseline = -0.1
        right_wrist_signal, _ = generate_test_signal_with_peaks(
            length=length, peaks=peaks, baseline=baseline
        )
        left_wrist_signal = np.ones(length) * baseline
        right_shoulder_signal = np.zeros(length)
        left_shoulder_signal = np.zeros(length)
        cam_signal_array = np.array([
            right_wrist_signal, left_wrist_signal,
            right_shoulder_signal, left_shoulder_signal
        ])
        num_cameras = 2
        handPunchVertPositionList = [cam_signal_array] * num_cameras
        cam_confidence_array = np.array([np.ones(length)] * np.shape(cam_signal_array)[0])
        handPunchConfidenceList = [cam_confidence_array] * num_cameras
        isHandPunch, handForPunch, handPunchRange = detectHandPunchAllVideos(
            sync_ver,
            clippedHandPunchVertPositionList=handPunchVertPositionList,
            clippedHandPunchConfidenceList=handPunchConfidenceList,
            inHandPunchVertPositionList=handPunchVertPositionList,
            inHandPunchConfidenceList=handPunchConfidenceList,
            sampleFreq=sampleFreq,
        )
        # 1.0: Takes highest peak
        # 1.1: Similar height will not be detected as a punch
        if sync_ver == '1.0':
            assert isHandPunch is True
            assert handForPunch == 'r'
            assert handPunchRange is None
        elif sync_ver == '1.1':
            assert isHandPunch is False
            assert handForPunch is None
            assert handPunchRange is None

    @pytest.mark.parametrize("sync_ver", sync_ver_list)
    def test_punch_low_confidence(self, sync_ver):
        # Create a signal with a clear peak but low confidence only during the peak
        length = 2000
        sampleFreq = 240
        peak = {'start': 200, 'end': 300, 'amplitude': 100.0}
        peaks = [peak]
        baseline = -0.1
        right_wrist_signal, _ = generate_test_signal_with_peaks(
            length=length, peaks=peaks, baseline=baseline
        )
        left_wrist_signal = np.ones(length) * -0.1
        right_shoulder_signal = np.zeros(length)
        left_shoulder_signal = np.zeros(length)
        cam_signal_array = np.array([
            right_wrist_signal, left_wrist_signal,
            right_shoulder_signal, left_shoulder_signal
        ])
        num_cameras = 2
        handPunchVertPositionList = [cam_signal_array] * num_cameras
        
        # High confidence everywhere except for 10 contiguous 
        # frames roughly in middle of the peak
        num_low_confidence_frames = 10
        cam_confidence_array = np.ones((np.shape(cam_signal_array)[0], length))
        low_conf_start = peak['start'] + (peak['end'] - peak['start']) // 2 - 5
        low_conf_end = low_conf_start + num_low_confidence_frames
        cam_confidence_array[:, low_conf_start:low_conf_end] = 0.1
        handPunchConfidenceList = [cam_confidence_array] * num_cameras
        isHandPunch, handForPunch, handPunchRange = detectHandPunchAllVideos(
            sync_ver,
            clippedHandPunchVertPositionList=handPunchVertPositionList,
            clippedHandPunchConfidenceList=handPunchConfidenceList,
            inHandPunchVertPositionList=handPunchVertPositionList,
            inHandPunchConfidenceList=handPunchConfidenceList,
            sampleFreq=sampleFreq,
        )
        # 1.0 does not filter for low confidence, so it detects the punch
        # 1.1 expects no punch detected due to low confidence during the peak
        if sync_ver == '1.0':
            assert isHandPunch is True
            assert handForPunch == 'r'
            assert handPunchRange is None
        elif sync_ver == '1.1':
            assert isHandPunch is False
            assert handForPunch is None
            assert handPunchRange is None

    @pytest.mark.parametrize("sync_ver", sync_ver_list)
    def test_punch_too_short(self, sync_ver):
        # Create a signal with a very short peak
        length = 2000
        sampleFreq = 240
        peaks = [{'start': 200, 'end': 205, 'amplitude': 100.0}]
        baseline = -0.1
        right_wrist_signal, _ = generate_test_signal_with_peaks(
            length=length, peaks=peaks, baseline=baseline
        )
        left_wrist_signal = np.ones(length) * baseline
        right_shoulder_signal = np.zeros(length)
        left_shoulder_signal = np.zeros(length)
        cam_signal_array = np.array([
            right_wrist_signal, left_wrist_signal,
            right_shoulder_signal, left_shoulder_signal
        ])
        num_cameras = 2
        handPunchVertPositionList = [cam_signal_array] * num_cameras
        cam_confidence_array = np.array([np.ones(length)] * np.shape(cam_signal_array)[0])
        handPunchConfidenceList = [cam_confidence_array] * num_cameras
        isHandPunch, handForPunch, handPunchRange = detectHandPunchAllVideos(
            sync_ver,
            clippedHandPunchVertPositionList=handPunchVertPositionList,
            clippedHandPunchConfidenceList=handPunchConfidenceList,
            inHandPunchVertPositionList=handPunchVertPositionList,
            inHandPunchConfidenceList=handPunchConfidenceList,
            sampleFreq=sampleFreq,
        )
        # 1.0 detects this as a punch, but 1.1 does not
        if sync_ver == '1.0':
            assert isHandPunch is True
            assert handForPunch == 'r'
            assert handPunchRange is None
        elif sync_ver == '1.1':
            assert isHandPunch is False
            assert handForPunch is None
            assert handPunchRange is None

    @pytest.mark.parametrize("sync_ver", sync_ver_list)
    def test_punch_too_long(self, sync_ver):
        # Create a signal with a very long peak
        length = 2000
        sampleFreq = 240
        peak = {'start': 200, 'end': 1800, 'amplitude': 100.0}  # almost entire signal
        peaks = [peak]
        baseline = -0.1
        right_wrist_signal, _ = generate_test_signal_with_peaks(
            length=length, peaks=peaks, baseline=baseline
        )
        left_wrist_signal = np.ones(length) * baseline
        right_shoulder_signal = np.zeros(length)
        left_shoulder_signal = np.zeros(length)
        cam_signal_array = np.array([
            right_wrist_signal, left_wrist_signal,
            right_shoulder_signal, left_shoulder_signal
        ])
        num_cameras = 2
        handPunchVertPositionList = [cam_signal_array] * num_cameras
        cam_confidence_array = np.array([np.ones(length)] * np.shape(cam_signal_array)[0])
        handPunchConfidenceList = [cam_confidence_array] * num_cameras
        isHandPunch, handForPunch, handPunchRange = detectHandPunchAllVideos(
            sync_ver,
            clippedHandPunchVertPositionList=handPunchVertPositionList,
            clippedHandPunchConfidenceList=handPunchConfidenceList,
            inHandPunchVertPositionList=handPunchVertPositionList,
            inHandPunchConfidenceList=handPunchConfidenceList,
            sampleFreq=sampleFreq,
        )
        # long punch should not be detected
        assert isHandPunch is False
        assert handForPunch is None
        assert handPunchRange is None


class TestSyncHandPunch:

    @pytest.mark.parametrize("sync_ver,input_lag,expected_lag", [
        ("1.0", 5, 5),
        ("1.1", 5, 5),
        ("1.0", 120, 103), # gaussian blur shifts towards 0, especially for large lags
        ("1.1", 120, 120),
    ])
    def test_syncver_lag_between_cameras(self, sync_ver, input_lag, expected_lag):
        # Create a clean right hand punch for Cam0, shift for Cam1
        length = 2000
        frame_rate = 60
        punch_start = 200
        punch_end = 600
        amplitude = 100.0
        baseline = -0.1
        cam0_right_wrist_signal, _ = generate_test_signal_with_peaks(
            length=length, 
            peaks=[{"start": punch_start, "end": punch_end, "amplitude": amplitude}], 
            baseline=baseline
        )
        cam1_right_wrist_signal = np.roll(cam0_right_wrist_signal, input_lag)  # Shift Cam1 by lag

        # Other signals
        left_wrist_signal = np.ones(length) * -0.1
        right_shoulder_signal = np.zeros(length)
        left_shoulder_signal = np.zeros(length)

        cam0_array = np.array([cam0_right_wrist_signal, left_wrist_signal, 
                               right_shoulder_signal, left_shoulder_signal])
        cam1_array = np.array([cam1_right_wrist_signal, left_wrist_signal, 
                               right_shoulder_signal, left_shoulder_signal])
        handPunchVertPositionList = [cam0_array, cam1_array]
        cam_confidence_array = np.array([np.ones(length)] * np.shape(cam0_array)[0])
        handPunchConfidenceList = [cam_confidence_array] * len(handPunchVertPositionList)

        handForPunch = 'r'
        handPunchRange = [punch_start, punch_end + input_lag] # earliest start, latest end
        maxShiftSteps = 2 * frame_rate
        _, lag = syncHandPunch(
            sync_ver,
            clippedHandPunchVertPositionList=handPunchVertPositionList,
            inHandPunchVertPositionList=handPunchVertPositionList,
            clippedHandPunchConfidenceList=handPunchConfidenceList,
            inHandPunchConfidenceList=handPunchConfidenceList,
            handForPunch=handForPunch,
            maxShiftSteps=maxShiftSteps,
            handPunchRange=handPunchRange,
            frameRate=frame_rate,
        )
        
        assert lag == expected_lag

    @pytest.mark.parametrize("signal_type,filter_freq", [
        ("position", None),
        ("position", 6.0),
        ("velocity", None),
        ("velocity", 6.0),
    ])
    def test_sync_hand_punch_v2_stability(self, signal_type, filter_freq, caplog):
        caplog.set_level(logging.DEBUG)

        # Create a clean right hand punch for Cam0, shift for Cam1
        length = 2000
        frame_rate = 60
        punch_start = 200
        punch_end = 600
        input_lag = 120
        amplitude = 100.0
        baseline = -0.1
        cam0_right_wrist_signal, _ = generate_test_signal_with_peaks(
            length=length, 
            peaks=[{"start": punch_start, "end": punch_end, "amplitude": amplitude}], 
            baseline=baseline
        )
        cam1_right_wrist_signal = np.roll(cam0_right_wrist_signal, input_lag)  # Shift Cam1 by lag

        # Other signals
        left_wrist_signal = np.ones(length) * -0.1
        right_shoulder_signal = np.zeros(length)
        left_shoulder_signal = np.zeros(length)

        cam0_array = np.array([cam0_right_wrist_signal, left_wrist_signal, 
                                right_shoulder_signal, left_shoulder_signal])
        cam1_array = np.array([cam1_right_wrist_signal, left_wrist_signal, 
                                right_shoulder_signal, left_shoulder_signal])
        handPunchVertPositionList = [cam0_array, cam1_array]
        cam_confidence_array = np.array([np.ones(length)] * np.shape(cam0_array)[0])
        handPunchConfidenceList = [cam_confidence_array] * len(handPunchVertPositionList)

        handForPunch = 'r'
        handPunchRange = [punch_start, punch_end + input_lag] # earliest start, latest end

        _, lag = syncHandPunch_v2(
            handPunchVertPositionList,
            handForPunch,
            handPunchConfidenceList,
            handPunchRange,
            frame_rate,
            signalType=signal_type,
            signalFilterFreq=filter_freq,
        )
        assert f"signalType: {signal_type}" in caplog.text
        assert f"signalFilterFreq: {filter_freq}" in caplog.text
        assert lag == input_lag
        