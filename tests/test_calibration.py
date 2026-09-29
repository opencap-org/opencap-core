import os
import shutil
import sys

import cv2
import numpy as np
import pytest

os.environ.setdefault('API_TOKEN', 'test-token')

from conftest import CALIBRATION_FIXTURE_DIR, REPO_DIR

sys.path.append(REPO_DIR)

import utilsChecker
from utilsChecker import (
    calcExtrinsicsFromVideo,
    generate3Dgrid,
    loadCameraParameters,
    rotateIntrinsics,
)


# ---- Checkerboard / fixture constants ----

DEFAULT_CHECKERBOARD_PARAMS = {
    'dimensions': (5, 4),
    'squareSize': 35.0,
}
LABVALIDATION_CHECKERBOARD_PARAMS = {
    'dimensions': (11, 8),
    'squareSize': 60.0,
}
MAX_MEAN_REPROJECTION_ERROR_PX = 0.5


INTRINSICS_FOLDER = 'Deployed'

ACL_EXHAUSTIVE_FALLBACK_VIDEO = os.path.join(
    CALIBRATION_FIXTURE_DIR,
    'acl',
    'exhaustive_fallback_success',
    'acl_exhaustive_only_success.qt',
)
UTAH_FIXTURE_VIDEO = os.path.join(
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
        DEFAULT_CHECKERBOARD_PARAMS,
        'iPhone13,3',
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
        'iPhone13,3',
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
        'iPhone17,3',
    ),
    (
        'no_checkerboard_cam1',
        os.path.join(
            CALIBRATION_FIXTURE_DIR,
            'comprehensive',
            'no_checkerboard',
            'no_checkerboard_cam1.mov',
        ),
        'iPhone17,1',
    ),
    (
        'partial_checkerboard_cam0',
        os.path.join(
            CALIBRATION_FIXTURE_DIR,
            'comprehensive',
            'partial_checkerboard',
            'partial_checkerboard_cam0.mov',
        ),
        'iPhone17,3',
    ),
    (
        'partial_checkerboard_cam1',
        os.path.join(
            CALIBRATION_FIXTURE_DIR,
            'comprehensive',
            'partial_checkerboard',
            'partial_checkerboard_cam1.mov',
        ),
        'iPhone17,1',
    ),
]


# ---- Unpatched cv2 handles ----

UNPATCHED_CV2_FIND_CHESSBOARD_CORNERS = cv2.findChessboardCorners
UNPATCHED_CV2_FIND_CHESSBOARD_CORNERS_SB_WITH_META = (
    cv2.findChessboardCornersSBWithMeta
)
UNPATCHED_CV2_CORNER_SUB_PIX = cv2.cornerSubPix
UNPATCHED_ENSURE_CORNER_ORDERING = utilsChecker.ensureCornerOrdering


# ---- Helpers ----

def load_intrinsics(video_path, iphone_model):
    intrinsics_path = os.path.join(
        REPO_DIR,
        'CameraIntrinsics',
        iphone_model,
        INTRINSICS_FOLDER,
        'cameraIntrinsics.pickle',
    )
    camera_params = loadCameraParameters(intrinsics_path)
    return rotateIntrinsics(camera_params, str(video_path))


def input_media_dir(tmp_path):
    media_dir = tmp_path.joinpath(
        os.path.join(
            'Data',
            'test_session',
            'Videos',
            'Cam0',
            'InputMedia',
            'calibration',
        )
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
    # Need to scale as detected coreners are potentially from upsampled/downsampled image,
    # but checking agaisnt original camera intrinsics
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
    fallback_enabled=True,
    fallback_flag_override=None,
):
    staged_video_path = stage_video(tmp_path, video_path)
    camera_params = load_intrinsics(staged_video_path, iphone_model)
    calls = {'primary': 0, 'fallback': 0}
    fallback_flags = []
    # captured_corners stores corners detected by the different methods,
    # pre and post sub pixel refining for primary path and pre and 
    # post re-ordering for fallback path
    captured_corners = {
        'raw_primary': None,
        'refined_primary': None,
        'raw_sb': None,
        'ordered_sb': None,
        'image_shape': None,
    }

    def primary_detector(*args, **kwargs):
        calls['primary'] += 1
        found, corners = UNPATCHED_CV2_FIND_CHESSBOARD_CORNERS(*args, **kwargs)
        if found:
            captured_corners['raw_primary'] = corners.copy()
            captured_corners['image_shape'] = args[0].shape
        return found, corners

    # fallback_flags records flags across retries, the flags used are always those requested
    # by prod except for the case where we test the old fallback without exhaustive
    def fallback_detector(image, pattern_size, flags):
        calls['fallback'] += 1
        if not fallback_enabled:
            return False, None, None
        if fallback_flag_override is not None:
            flags = fallback_flag_override
        fallback_flags.append(flags)
        found, corners, meta = UNPATCHED_CV2_FIND_CHESSBOARD_CORNERS_SB_WITH_META(
            image, pattern_size, flags
        )
        if found:
            captured_corners['raw_sb'] = corners.copy()
            captured_corners['image_shape'] = image.shape
        return found, corners, meta

    def ensure_corner_ordering(image, corners, pattern, squareResolution=1):
        ordered_corners, ordering_success, ordering_error = (
            UNPATCHED_ENSURE_CORNER_ORDERING(
                image, corners, pattern, squareResolution=squareResolution
            )
        )
        if ordering_success:
            captured_corners['ordered_sb'] = ordered_corners.copy()
            captured_corners['image_shape'] = image.shape
        return ordered_corners, ordering_success, ordering_error

    def corner_subpix(image, corners, win_size, zero_zone, criteria):
        refined_corners = UNPATCHED_CV2_CORNER_SUB_PIX(
            image, corners, win_size, zero_zone, criteria
        )
        captured_corners['refined_primary'] = refined_corners.copy()
        captured_corners['image_shape'] = image.shape
        return refined_corners

    monkeypatch.setattr(cv2, 'findChessboardCorners', primary_detector)
    monkeypatch.setattr(cv2, 'findChessboardCornersSBWithMeta', fallback_detector)
    monkeypatch.setattr(cv2, 'cornerSubPix', corner_subpix)
    monkeypatch.setattr(utilsChecker, 'ensureCornerOrdering', ensure_corner_ordering)

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
    corners_for_reprojection = (
        captured_corners['refined_primary']
        if captured_corners['refined_primary'] is not None
        else captured_corners['ordered_sb']
    )
    error = mean_reproj_error(
        result,
        checkerboard_params,
        corners_for_reprojection,
        captured_corners['image_shape'],
    )
    return result, calls, fallback_flags, error


# ---- Tests ----

@pytest.mark.parametrize(
    'fixture_name, video_path, checkerboard_params, iphone_model',
    PRIMARY_SUCCESS_FIXTURES,
    ids=[fixture[0] for fixture in PRIMARY_SUCCESS_FIXTURES],
)
# Good videos that should work with no fallback
def test_primary_fixtures_calibrate(
    fixture_name, video_path, checkerboard_params, iphone_model, tmp_path, monkeypatch
):
    result, calls, fallback_flags, mean_error = (
        run_video_calibration(
            video_path,
            checkerboard_params,
            iphone_model,
            tmp_path,
            monkeypatch,
            fallback_enabled=False,
        )
    )

    assert_extrinsics(result)
    assert mean_error < MAX_MEAN_REPROJECTION_ERROR_PX, (
        fixture_name,
        mean_error,
    )
    # These fixtures should pass within the primary detector's resize attempts.
    assert 1 <= calls['primary'] <= 4
    assert calls['fallback'] == 0
    assert fallback_flags == []


# Utah fixture should calibrate through the exhaustive fallback route when needed.
def test_utah_fixture_exhaustive(tmp_path, monkeypatch):
    result, _, _, mean_error = run_video_calibration(
        UTAH_FIXTURE_VIDEO,
        DEFAULT_CHECKERBOARD_PARAMS,
        'iPhone13,3',
        tmp_path,
        monkeypatch,
    )

    assert_extrinsics(result)
    assert mean_error < MAX_MEAN_REPROJECTION_ERROR_PX, mean_error


# A hard ACL vid that fails with the old fallback but succeeds with the new exhaustive fallback
def test_acl_exhaustive_recovery(tmp_path, monkeypatch):
    (
        current_result,
        current_calls,
        current_fallback_flags,
        _,
    ) = run_video_calibration(
        ACL_EXHAUSTIVE_FALLBACK_VIDEO,
        DEFAULT_CHECKERBOARD_PARAMS,
        'iPhone13,3',
        tmp_path / 'current_fallback',
        monkeypatch,
        fallback_flag_override=sb_flags(exhaustive=False),
    )
    assert current_result is None
    assert current_calls['fallback'] > 0
    assert current_fallback_flags
    assert not any(
        flags & cv2.CALIB_CB_EXHAUSTIVE for flags in current_fallback_flags
    )

    (
        exhaustive_result,
        exhaustive_calls,
        exhaustive_fallback_flags,
        mean_error,
    ) = run_video_calibration(
        ACL_EXHAUSTIVE_FALLBACK_VIDEO,
        DEFAULT_CHECKERBOARD_PARAMS,
        'iPhone13,3',
        tmp_path / 'exhaustive_fallback',
        monkeypatch,
    )

    assert_extrinsics(exhaustive_result)
    assert mean_error < MAX_MEAN_REPROJECTION_ERROR_PX, mean_error
    assert exhaustive_calls['fallback'] > 0
    assert exhaustive_fallback_flags
    assert any(
        flags & cv2.CALIB_CB_EXHAUSTIVE
        for flags in exhaustive_fallback_flags
    )


@pytest.mark.parametrize(
    'fixture_name, video_path, iphone_model',
    NEGATIVE_FIXTURES,
    ids=[fixture[0] for fixture in NEGATIVE_FIXTURES],
)
# No board and partial board should fail even with exhaustive fallback
def test_negative_fixtures_reject(
    fixture_name, video_path, iphone_model, tmp_path, monkeypatch
):
    result, calls, fallback_flags, _ = (
        run_video_calibration(
            video_path,
            DEFAULT_CHECKERBOARD_PARAMS,
            iphone_model,
            tmp_path,
            monkeypatch,
        )
    )

    assert result is None, fixture_name
    assert calls['fallback'] > 0
    assert fallback_flags
    assert all(flags & cv2.CALIB_CB_EXHAUSTIVE for flags in fallback_flags)
