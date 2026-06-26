import os
import shutil
import sys

import cv2
import numpy as np
import pytest

os.environ.setdefault('API_TOKEN', 'test-token')

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir, '../'))
sys.path.append(repoDir)

from utilsChecker import (
    calcExtrinsicsFromVideo,
    generate3Dgrid,
    loadCameraParameters,
    rotateIntrinsics,
)


STANDARD_CHECKERBOARD_PARAMS = {
    'dimensions': (5, 4),
    'squareSize': 35.0,
}
LABVALIDATION_CHECKERBOARD_PARAMS = {
    'dimensions': (11, 8),
    'squareSize': 60.0,
}
MAX_MEAN_REPROJECTION_ERROR_PX = 0.5

ORIGINAL_FIND_CHESSBOARD_CORNERS = cv2.findChessboardCorners
ORIGINAL_FIND_CHESSBOARD_CORNERS_SB_WITH_META = cv2.findChessboardCornersSBWithMeta
ORIGINAL_CORNER_SUB_PIX = cv2.cornerSubPix

CALIBRATION_FIXTURE_DIR = os.path.join(
    thisDir,
    'opencap-test-data',
    'Data',
    'calibration-fixtures',
)
IPHONE13_MODEL = 'iPhone13,3'
IPHONE17_1_MODEL = 'iPhone17,1'
IPHONE17_3_MODEL = 'iPhone17,3'
INTRINSICS_FOLDER = 'Deployed'

ACL_EXHAUSTIVE_FALLBACK_VIDEO = os.path.join(
    CALIBRATION_FIXTURE_DIR,
    'acl',
    'exhaustive_fallback_success',
    'acl_exhaustive_only_success.qt',
)
UTAH_PRODUCTION_VIDEO = os.path.join(
    CALIBRATION_FIXTURE_DIR,
    'utah',
    'production_success',
    'utah_production_success.mov',
)

PRIMARY_SUCCESS_FIXTURES = [
    (
        'acl',
        os.path.join(
            CALIBRATION_FIXTURE_DIR,
            'acl',
            'primary_success',
            'acl_primary_success.qt',
        ),
        STANDARD_CHECKERBOARD_PARAMS,
        IPHONE13_MODEL,
    ),
    (
        'labvalidation',
        os.path.join(
            CALIBRATION_FIXTURE_DIR,
            'labvalidation',
            'primary_success',
            'labvalidation_subject5_session0_cam3_extrinsics.avi',
        ),
        LABVALIDATION_CHECKERBOARD_PARAMS,
        IPHONE13_MODEL,
    ),
]

NEGATIVE_FIXTURES = [
    (
        'no_checkerboard_cam0',
        os.path.join(
            CALIBRATION_FIXTURE_DIR,
            'comprehensive',
            'no_checkerboard',
            'no_checkerboard_cam0.mov',
        ),
        IPHONE17_3_MODEL,
    ),
    (
        'no_checkerboard_cam1',
        os.path.join(
            CALIBRATION_FIXTURE_DIR,
            'comprehensive',
            'no_checkerboard',
            'no_checkerboard_cam1.mov',
        ),
        IPHONE17_1_MODEL,
    ),
    (
        'partial_checkerboard_cam0',
        os.path.join(
            CALIBRATION_FIXTURE_DIR,
            'comprehensive',
            'partial_checkerboard',
            'partial_checkerboard_cam0.mov',
        ),
        IPHONE17_3_MODEL,
    ),
    (
        'partial_checkerboard_cam1',
        os.path.join(
            CALIBRATION_FIXTURE_DIR,
            'comprehensive',
            'partial_checkerboard',
            'partial_checkerboard_cam1.mov',
        ),
        IPHONE17_1_MODEL,
    ),
]


def load_intrinsics(video_path, iphone_model):
    intrinsics_path = os.path.join(
        repoDir,
        'CameraIntrinsics',
        iphone_model,
        INTRINSICS_FOLDER,
        'cameraIntrinsics.pickle',
    )
    camera_params = loadCameraParameters(intrinsics_path)
    return rotateIntrinsics(camera_params, str(video_path))


def input_media_dir(tmp_path):
    media_dir = (
        tmp_path
        / 'Data'
        / 'test_session'
        / 'Videos'
        / 'Cam0'
        / 'InputMedia'
        / 'calibration'
    )
    media_dir.mkdir(parents=True, exist_ok=True)
    return media_dir


def stage_video(tmp_path, video_path):
    staged_video_path = input_media_dir(tmp_path) / os.path.basename(video_path)
    shutil.copy2(video_path, staged_video_path)
    return staged_video_path


def assert_extrinsics(camera_params):
    assert camera_params is not None
    for key in ('rotation', 'translation', 'rotation_EulerAngles'):
        assert key in camera_params
        assert np.all(np.isfinite(camera_params[key]))
    assert camera_params['rotation'].shape == (3, 3)
    assert camera_params['translation'].shape in ((3, 1), (3,))


def sb_flags(exhaustive):
    flags = cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_LARGER
    if exhaustive:
        flags |= cv2.CALIB_CB_EXHAUSTIVE
    return flags


def mean_reproj_error(camera_params, checkerboard_params, corners, image_shape):
    if camera_params is None or corners is None or image_shape is None:
        return None

    image_width = image_shape[1]
    camera_image_width = float(np.squeeze(camera_params['imageSize'])[1])
    scale = image_width / camera_image_width
    observed_corners = corners / scale
    object_points = generate3Dgrid(checkerboard_params)
    projected_corners, _ = cv2.projectPoints(
        object_points,
        camera_params['rotation_EulerAngles'],
        camera_params['translation'],
        camera_params['intrinsicMat'],
        camera_params['distortion'],
    )
    corner_errors = np.linalg.norm(
        projected_corners.reshape(-1, 2) - observed_corners.reshape(-1, 2),
        axis=1,
    )
    return float(np.mean(corner_errors))


def run_video_calibration(
    video_path,
    checkerboard_params,
    iphone_model,
    tmp_path,
    monkeypatch,
    detector_mode,
):
    staged_video_path = stage_video(tmp_path, video_path)
    camera_params = load_intrinsics(staged_video_path, iphone_model)
    calls = {'primary': 0, 'fallback': 0}
    production_flags = []
    detector_flags = []
    detected = {'corners': None, 'image_shape': None}

    def primary_detector(*args, **kwargs):
        calls['primary'] += 1
        found, corners = ORIGINAL_FIND_CHESSBOARD_CORNERS(*args, **kwargs)
        if found:
            detected['corners'] = corners.copy()
            detected['image_shape'] = args[0].shape
        return found, corners

    def fallback_detector(image, pattern_size, flags):
        calls['fallback'] += 1
        production_flags.append(flags)
        if detector_mode == 'primary_only':
            return False, None, None
        if detector_mode == 'current_fallback':
            flags = sb_flags(exhaustive=False)
        elif detector_mode == 'exhaustive_fallback':
            flags = sb_flags(exhaustive=True)
        detector_flags.append(flags)
        found, corners, meta = ORIGINAL_FIND_CHESSBOARD_CORNERS_SB_WITH_META(
            image, pattern_size, flags
        )
        if found:
            detected['corners'] = corners.copy()
            detected['image_shape'] = image.shape
        return found, corners, meta

    def corner_subpix(image, corners, win_size, zero_zone, criteria):
        refined_corners = ORIGINAL_CORNER_SUB_PIX(
            image, corners, win_size, zero_zone, criteria
        )
        detected['corners'] = refined_corners.copy()
        detected['image_shape'] = image.shape
        return refined_corners

    monkeypatch.setattr(cv2, 'findChessboardCorners', primary_detector)
    monkeypatch.setattr(cv2, 'findChessboardCornersSBWithMeta', fallback_detector)
    monkeypatch.setattr(cv2, 'cornerSubPix', corner_subpix)

    try:
        result = calcExtrinsicsFromVideo(
            str(staged_video_path),
            camera_params,
            checkerboard_params,
            visualize=False,
            imageUpsampleFactor=2,
        )
    except Exception as exc:
        if 'checkerboard was not detected' not in str(exc):
            raise
        result = None
    error = mean_reproj_error(
        result, checkerboard_params, detected['corners'], detected['image_shape']
    )
    return result, calls, production_flags, detector_flags, error


@pytest.mark.parametrize(
    'fixture_name, video_path, checkerboard_params, iphone_model',
    PRIMARY_SUCCESS_FIXTURES,
    ids=[fixture[0] for fixture in PRIMARY_SUCCESS_FIXTURES],
)
# Good videos that should work with no fallback
def test_primary_fixtures_calibrate(
    fixture_name, video_path, checkerboard_params, iphone_model, tmp_path, monkeypatch
):
    result, calls, production_flags, detector_flags, mean_error = run_video_calibration(
        video_path,
        checkerboard_params,
        iphone_model,
        tmp_path,
        monkeypatch,
        detector_mode='primary_only',
    )

    assert_extrinsics(result)
    assert mean_error < MAX_MEAN_REPROJECTION_ERROR_PX, (
        fixture_name,
        mean_error,
    )
    assert calls['primary'] == 1
    assert calls['fallback'] == 0
    assert production_flags == []
    assert detector_flags == []


# Utah vid should succeed through the production fallback route (and in many cases via primary detector)
def test_utah_production_calibrates(tmp_path, monkeypatch):
    result, _, _, _, mean_error = run_video_calibration(
        UTAH_PRODUCTION_VIDEO,
        STANDARD_CHECKERBOARD_PARAMS,
        IPHONE13_MODEL,
        tmp_path,
        monkeypatch,
        detector_mode='exhaustive_fallback',
    )

    assert_extrinsics(result)
    assert mean_error < MAX_MEAN_REPROJECTION_ERROR_PX, mean_error


# A hard ACL vid that fails with the old fallback but succeeds with the new exhaustive fallback
def test_acl_exhaustive_recovery(tmp_path, monkeypatch):
    (
        current_result,
        current_calls,
        current_production_flags,
        current_flags,
        _,
    ) = run_video_calibration(
        ACL_EXHAUSTIVE_FALLBACK_VIDEO,
        STANDARD_CHECKERBOARD_PARAMS,
        IPHONE13_MODEL,
        tmp_path / 'current_fallback',
        monkeypatch,
        detector_mode='current_fallback',
    )
    assert current_result is None
    assert current_calls['fallback'] > 0
    assert current_production_flags
    assert current_flags
    assert not any(flags & cv2.CALIB_CB_EXHAUSTIVE for flags in current_flags)

    (
        exhaustive_result,
        exhaustive_calls,
        exhaustive_production_flags,
        exhaustive_flags,
        mean_error,
    ) = run_video_calibration(
        ACL_EXHAUSTIVE_FALLBACK_VIDEO,
        STANDARD_CHECKERBOARD_PARAMS,
        IPHONE13_MODEL,
        tmp_path / 'exhaustive_fallback',
        monkeypatch,
        detector_mode='exhaustive_fallback',
    )

    assert_extrinsics(exhaustive_result)
    assert mean_error < MAX_MEAN_REPROJECTION_ERROR_PX, mean_error
    assert exhaustive_calls['fallback'] > 0
    assert exhaustive_production_flags
    assert exhaustive_flags
    assert any(
        flags & cv2.CALIB_CB_EXHAUSTIVE for flags in exhaustive_production_flags
    )
    assert any(flags & cv2.CALIB_CB_EXHAUSTIVE for flags in exhaustive_flags)


@pytest.mark.parametrize(
    'fixture_name, video_path, iphone_model',
    NEGATIVE_FIXTURES,
    ids=[fixture[0] for fixture in NEGATIVE_FIXTURES],
)
# No board and partial board should fail even with exhaustive fallback
def test_negative_fixtures_reject(
    fixture_name, video_path, iphone_model, tmp_path, monkeypatch
):
    result, calls, production_flags, detector_flags, _ = run_video_calibration(
        video_path,
        STANDARD_CHECKERBOARD_PARAMS,
        iphone_model,
        tmp_path,
        monkeypatch,
        detector_mode='exhaustive_fallback',
    )

    assert result is None, fixture_name
    assert calls['fallback'] > 0
    assert production_flags
    assert detector_flags
    assert all(flags & cv2.CALIB_CB_EXHAUSTIVE for flags in production_flags)
    assert all(flags & cv2.CALIB_CB_EXHAUSTIVE for flags in detector_flags)
