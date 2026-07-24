import logging
import os
import pickle
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pytest

# Fixture paths and camera orders used by the main pipeline regression tests
from conftest import (
    CALIBRATION_FIXTURE_DIR,
    LAB_5CAM_DIR,
    REPO_DIR,
    SYNC_2CAM_DIR,
    TEST_DATA_ROOT,
)

sys.path.append(REPO_DIR)
from main import main

SYNC_2CAM_CALIB_VIDEOS = {
    f'Cam{cam_i}': os.path.join(
        CALIBRATION_FIXTURE_DIR,
        'sync_2-cameras',
        f'cam{cam_i}_calibration.qt',
    )
    for cam_i in range(2)
}
LAB_5CAM_CALIB_VIDEOS = {
    f'Cam{cam_i}': os.path.join(
        CALIBRATION_FIXTURE_DIR,
        'labvalidation',
        'five_camera',
        f'labvalidation_subject2_session0_cam{cam_i}_extrinsics.avi',
    )
    for cam_i in range(5)
}
# orders specifc to the runs done to build tests
LAB_5CAM_DYNAMIC_ORDER = ['Cam2', 'Cam4', 'Cam3', 'Cam1', 'Cam0']
SYNC_2CAM_NEUTRAL_ORDER = ['Cam1', 'Cam0']

# Helper functions to load and compare TRC, MOT, and OpenSim outputs
def load_trc(file, num_metadata_lines=5):
    with open(file, 'r') as f:
        lines = f.readlines()
        metadata = lines[:num_metadata_lines]
    df = pd.read_csv(file, sep='\t', skiprows=num_metadata_lines + 1, header=None)
    return df, metadata


def load_mot(file, num_metadata_lines=10):
    with open(file, 'r') as f:
        lines = f.readlines()
        metadata = lines[:num_metadata_lines]
    df = pd.read_csv(file, sep='\t', skiprows=num_metadata_lines)
    return df, metadata


def calc_rmse(series1, series2):
    return np.sqrt(((series1 - series2) ** 2).mean())


def compare_trc(output_trc, ref_trc, atol=1e-3):
    output_trc_df, _ = load_trc(output_trc)
    ref_trc_df, _ = load_trc(ref_trc)
    pd.testing.assert_frame_equal(
        output_trc_df, ref_trc_df, check_exact=False, atol=atol
    )
    assert output_trc_df.isna().sum().sum() == ref_trc_df.isna().sum().sum()


def load_osim_scales(file):
    root = ET.parse(file).getroot()
    scale_factors = {}
    for body in root.findall('.//Body'):
        body_name = body.attrib.get('name')
        if not body_name:
            continue
        for mesh_index, mesh in enumerate(body.findall('./attached_geometry/Mesh')):
            scale_element = mesh.find('scale_factors')
            if scale_element is None or not scale_element.text:
                continue
            mesh_name = mesh.attrib.get('name', f'Mesh{mesh_index}')
            scale_factors[(body_name, mesh_name)] = np.array(
                [float(value) for value in scale_element.text.split()]
            )
    return scale_factors


def compare_osim_scales(output_osim, ref_osim, atol=5e-3):
    output_scale_factors = load_osim_scales(output_osim)
    ref_scale_factors = load_osim_scales(ref_osim)
    assert output_scale_factors.keys() == ref_scale_factors.keys()
    for mesh_key in ref_scale_factors:
        np.testing.assert_allclose(
            output_scale_factors[mesh_key],
            ref_scale_factors[mesh_key],
            rtol=0,
            atol=atol,
            err_msg=str(mesh_key),
        )


def compare_mot(output_mot_df, ref_mot_df, t0, tf):
    '''Function to compare MOT dataframes within a time range [t0, tf].
    We use the specific time range to analyze the range with the motion
    of interest. In particular, the arm raise can create larger differences
    on single frames.

    - Time column is checked for equality (IK is frame-by-frame).
    - Translation error is checked within 2 mm max per frame, RMSE within 
      1 mm.
    - Rotation error for wrist pronation/supination (coordinates pro_sup_r
      and pro_sup_l) are checked within 5.0 degrees max per frame, RMSE
      within 1.0 degrees.
    - Rotation error for all other coordinates are tighter and checked 
      within 2.5 degrees max per frame, RMSE within 0.5 degrees.
    '''
    output_mot_df_slice = output_mot_df[(output_mot_df['time'] >= t0) & (output_mot_df['time'] <= tf)]
    ref_mot_df_slice = ref_mot_df[(ref_mot_df['time'] >= t0) & (ref_mot_df['time'] <= tf)]
    for col in ref_mot_df.columns:
        # time column should be equal since IK is frame-by-frame
        if col == 'time':
            pd.testing.assert_series_equal(output_mot_df[col], ref_mot_df[col])

        # check translational within 2 mm max error, rmse within 1 mm
        elif any(substr in col for substr in ['tx', 'ty', 'tz']):
            pd.testing.assert_series_equal(
                output_mot_df_slice[col], ref_mot_df_slice[col], atol=0.002
            )
            rmse = calc_rmse(output_mot_df_slice[col], ref_mot_df_slice[col])
            assert rmse <= 0.001

        elif 'pro_sup' in col:
            pd.testing.assert_series_equal(
                output_mot_df_slice[col], ref_mot_df_slice[col], atol=5.0
            )
            rmse = calc_rmse(output_mot_df_slice[col], ref_mot_df_slice[col])
            assert rmse <= 1.0

        # check rotational within 2.5 degrees max error, rmse within 0.5 degrees
        else:
            pd.testing.assert_series_equal(
                output_mot_df_slice[col], ref_mot_df_slice[col], atol=2.5
            )
            rmse = calc_rmse(output_mot_df_slice[col], ref_mot_df_slice[col])
            assert rmse <= 0.5


def compare_mot_files(output_mot, ref_mot, t0, tf):
    output_mot_df, _ = load_mot(output_mot)
    ref_mot_df, _ = load_mot(ref_mot)
    pd.testing.assert_index_equal(output_mot_df.columns, ref_mot_df.columns)
    compare_mot(output_mot_df, ref_mot_df, t0, tf)

# Calibration regression test
def test_main_calibration(tmp_path):
    sessionName = 'sync_2-cameras_calibration'
    trialName = 'calibration'
    trialID = 'calibration'
    dataDir = tmp_path
    sessionDir = os.path.join(dataDir, 'Data', sessionName)

    os.makedirs(sessionDir, exist_ok=True)
    shutil.copy2(
        os.path.join(SYNC_2CAM_DIR, 'sessionMetadata.yaml'),
        os.path.join(sessionDir, 'sessionMetadata.yaml'),
    )

    for camName, videoPath in SYNC_2CAM_CALIB_VIDEOS.items():
        mediaDir = os.path.join(
            sessionDir,
            'Videos',
            camName,
            'InputMedia',
            trialName,
        )
        os.makedirs(mediaDir, exist_ok=True)
        _, videoExt = os.path.splitext(videoPath)
        shutil.copy2(
            videoPath,
            os.path.join(mediaDir, f'{trialID}{videoExt}'),
        )

    main(
        sessionName,
        trialName,
        trialID,
        dataDir=dataDir,
        genericFolderNames=True,
        extrinsicsTrial=True,
        imageUpsampleFactor=2,
    )

    for camName in SYNC_2CAM_CALIB_VIDEOS:
        paramsPath = os.path.join(
            sessionDir,
            'Videos',
            camName,
            'cameraIntrinsicsExtrinsics.pickle',
        )
        assert os.path.exists(paramsPath)
        with open(paramsPath, 'rb') as f:
            cameraParams = pickle.load(f)
        assert np.all(np.isfinite(cameraParams['rotation']))
        assert np.all(np.isfinite(cameraParams['translation']))


# End to end tests with different sync methods (hand, gait, general).
# Also check that syncVer updates with main changes.
# Note: no pose detection, uses pre-scaled opensim model
@pytest.mark.parametrize("syncVer", ['1.0', '1.1'])
@pytest.mark.parametrize("trialName, t0, tf", [
    ('squats-with-arm-raise', 5.0, 10.0),
    ('squats', 3.0, 8.0),
    ('walk', 1.0, 5.0),
])
def test_main(trialName, t0, tf, syncVer, caplog):
    caplog.set_level(logging.INFO)

    sessionName = 'sync_2-cameras'
    trialID = trialName
    dataDir = TEST_DATA_ROOT
    main(
        sessionName,
        trialName,
        trialID,
        dataDir=dataDir,
        genericFolderNames=True,
        poseDetector='hrnet',
        syncVer=syncVer,
    )
    assert f"Synchronizing Keypoints using version {syncVer}" in caplog.text

    # Compare marker data
    output_trc = os.path.join(dataDir,
        'Data',
        sessionName,
        'MarkerData',
        'PostAugmentation',
        f'{trialName}.trc',
    )
    ref_trc = os.path.join(
        dataDir,
        'Data',
        sessionName,
        'OutputReference',
        f'{trialName}.trc',
    )
    compare_trc(output_trc, ref_trc)

    # Compare IK data
    output_mot = os.path.join(
        dataDir,
        'Data',
        sessionName,
        'OpenSimData',
        'Kinematics',
        f'{trialName}.mot',
    )
    ref_mot = os.path.join(
        dataDir,
        'Data',
        sessionName,
        'OutputReference',
        f'{trialName}.mot',
    )
    compare_mot_files(output_mot, ref_mot, t0, tf)

# Regression test for neutral scaling using existing pose detection outputs
def test_neutral_scaling(tmp_path):
    sessionName = 'sync_2-cameras'
    trialName = 'neutral'
    trialID = trialName
    dataDir = tmp_path
    sessionDir = os.path.join(dataDir, 'Data', sessionName)
    shutil.copytree(
        SYNC_2CAM_DIR,
        sessionDir,
        ignore=shutil.ignore_patterns('.DS_Store'),
    )

    for camName in ['Cam0', 'Cam1']:
        production_pickle = os.path.join(
            sessionDir,
            'Videos',
            camName,
            'OutputPkl',
            f'{trialName}_keypoints.pkl',
        )
        local_pickle_dir = os.path.join(
            sessionDir,
            'Videos',
            camName,
            'OutputPkl_default',
            trialName,
        )
        os.makedirs(local_pickle_dir, exist_ok=True)
        shutil.copy2(
            production_pickle,
            os.path.join(local_pickle_dir, f'{trialName}_rotated_pp.pkl'),
        )

    main(
        sessionName,
        trialName,
        trialID,
        cameras_to_use=SYNC_2CAM_NEUTRAL_ORDER,
        dataDir=dataDir,
        genericFolderNames=True,
        scaleModel=True,
        syncVer='1.1',
    )

    output_pre_augmentation_trc = os.path.join(
        sessionDir,
        'MarkerData',
        'PreAugmentation',
        f'{trialName}.trc',
    )
    assert os.path.exists(output_pre_augmentation_trc)

    output_post_augmentation_trc = os.path.join(
        sessionDir,
        'MarkerData',
        'PostAugmentation',
        f'{trialName}.trc',
    )
    ref_post_augmentation_trc = os.path.join(
        SYNC_2CAM_DIR,
        'MarkerData',
        f'{trialName}.trc',
    )
    compare_trc(output_post_augmentation_trc, ref_post_augmentation_trc, atol=1e-5)

    scaled_model = os.path.join(
        sessionDir,
        'OpenSimData',
        'Model',
        'LaiUhlrich2022_scaled.osim',
    )
    assert os.path.exists(scaled_model)
    ref_scaled_model = os.path.join(
        SYNC_2CAM_DIR,
        'OpenSimData',
        'Model',
        'LaiUhlrich2022_scaled.osim',
    )
    compare_osim_scales(scaled_model, ref_scaled_model, atol=1e-5)


# More than 2 camera tests.
def test_lab_5cam_calibration(tmp_path):
    sessionName = 'labvalidation_calibration_5-cameras'
    trialName = 'calibration'
    trialID = 'calibration'
    dataDir = tmp_path
    sessionDir = os.path.join(dataDir, 'Data', sessionName)

    os.makedirs(sessionDir, exist_ok=True)
    shutil.copy2(
        os.path.join(LAB_5CAM_DIR, 'sessionMetadata.yaml'),
        os.path.join(sessionDir, 'sessionMetadata.yaml'),
    )

    for camName, videoPath in LAB_5CAM_CALIB_VIDEOS.items():
        mediaDir = os.path.join(
            sessionDir,
            'Videos',
            camName,
            'InputMedia',
            trialName,
        )
        os.makedirs(mediaDir, exist_ok=True)
        shutil.copy2(
            videoPath,
            os.path.join(mediaDir, f'{trialID}.avi'),
        )

    main(
        sessionName,
        trialName,
        trialID,
        dataDir=dataDir,
        genericFolderNames=True,
        extrinsicsTrial=True,
        imageUpsampleFactor=2,
    )

    for camName in LAB_5CAM_CALIB_VIDEOS:
        paramsPath = os.path.join(
            sessionDir,
            'Videos',
            camName,
            'cameraIntrinsicsExtrinsics.pickle',
        )
        assert os.path.exists(paramsPath)
        with open(paramsPath, 'rb') as f:
            cameraParams = pickle.load(f)
        assert np.all(np.isfinite(cameraParams['rotation']))
        assert np.all(np.isfinite(cameraParams['translation']))


def test_lab_5cam_dynamic(tmp_path):
    sessionName = 'labvalidation_subject2_session0_5-cameras'
    trialName = 'squats1'
    trialID = trialName
    dataDir = tmp_path
    sessionDir = os.path.join(dataDir, 'Data', sessionName)
    shutil.copytree(LAB_5CAM_DIR, sessionDir)

    for camName in LAB_5CAM_DYNAMIC_ORDER:
        production_pickle = os.path.join(
            sessionDir,
            'Videos',
            camName,
            'OutputPkl',
            f'{trialName}_keypoints.pkl',
        )
        local_pickle_dir = os.path.join(
            sessionDir,
            'Videos',
            camName,
            'OutputPkl_1x736',
            trialName,
        )
        os.makedirs(local_pickle_dir, exist_ok=True)
        shutil.copy2(
            production_pickle,
            os.path.join(local_pickle_dir, f'{trialName}_rotated_pp.pkl'),
        )

    main(
        sessionName,
        trialName,
        trialID,
        cameras_to_use=LAB_5CAM_DYNAMIC_ORDER,
        dataDir=dataDir,
        genericFolderNames=True,
        poseDetector='openpose',
        resolutionPoseDetection='1x736',
    )

    output_pre_augmentation_trc = os.path.join(
        sessionDir,
        'MarkerData',
        'PreAugmentation',
        f'{trialName}.trc',
    )
    assert os.path.exists(output_pre_augmentation_trc)

    output_post_augmentation_trc = os.path.join(
        sessionDir,
        'MarkerData',
        'PostAugmentation',
        f'{trialName}.trc',
    )
    ref_post_augmentation_trc = os.path.join(
        LAB_5CAM_DIR,
        'MarkerData',
        f'{trialName}.trc',
    )
    compare_trc(output_post_augmentation_trc, ref_post_augmentation_trc)

    output_mot = os.path.join(
        sessionDir,
        'OpenSimData',
        'Kinematics',
        f'{trialName}.mot',
    )
    ref_mot = os.path.join(
        LAB_5CAM_DIR,
        'OpenSimData',
        'Kinematics',
        f'{trialName}.mot',
    )
    # excludes the setup motion
    compare_mot_files(output_mot, ref_mot, 2, 9)


# TODO: augmenter versions
