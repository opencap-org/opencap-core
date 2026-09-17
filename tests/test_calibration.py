import os
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock

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

# These are provisional values. We should discuss what values would be appropriate.
MAX_INTRINSICS_REPROJECTION_DIFFERENCE_PX = 0.25
MAX_INTRINSIC_MATRIX_DIFFERENCE_PX = 2.0


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
IPAD_A16_FIXTURE_DIR = os.path.join(
    CALIBRATION_FIXTURE_DIR,
    'ipad_a16',
)


INTRINSICS_FIXED_IMAGE_DIR = os.path.join(
    CALIBRATION_FIXTURE_DIR,
    'ipad_a16',
    'fixed_images',
)

EXPECTED_FIXED_IMAGE_COUNT = 49

INTRINSICS_FIXED_EXPECTED = os.path.join(
    INTRINSICS_FIXED_IMAGE_DIR,
    'cameraIntrinsics.pickle',
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
    # Need to scale as detected corners are potentially from upsampled/downsampled image,
    # but checking against original camera intrinsics
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


def project_intrinsics_grid(
    points_3d,
    camera_params,
    rvec=None,
    tvec=None,
):
    # Project a fixed 3D grid using a camera model.
    if rvec is None:
        rvec = np.zeros((3, 1), dtype=np.float64)

    if tvec is None:
        tvec = np.array(
            [[0.0], [0.0], [1000.0]],
            dtype=np.float64,
        )

    projected_points, _ = cv2.projectPoints(
        points_3d,
        rvec,
        tvec,
        camera_params['intrinsicMat'],
        camera_params['distortion'],
    )

    return projected_points


def mean_reprojection_difference(
    reference_params,
    calculated_params,
    checkerboard_params,
):
    # Compare the image-space projection produced by two camera models.
    assert np.array_equal(
        reference_params['imageSize'],
        calculated_params['imageSize'],
    )

    points_3d = generate3Dgrid(checkerboard_params).astype(np.float32)

    reference_points = project_intrinsics_grid(
        points_3d,
        reference_params,
    )
    calculated_points = project_intrinsics_grid(
        points_3d,
        calculated_params,
    )

    errors = np.linalg.norm(
        reference_points - calculated_points,
        axis=2,
    )

    return float(np.mean(errors)), float(np.max(errors))

def stage_ipad_a16_videos(tmp_path):
    # Copy the fixed iPad A16 calibration videos to an isolated directory.
    video_dir = tmp_path / 'ipad_a16_videos'
    video_dir.mkdir(parents=True, exist_ok=True)

    for capture_name in (
        'ipad-a16_1',
        'ipad-a16_2',
        'ipad-a16_3',
    ):
        source_dir = Path(IPAD_A16_FIXTURE_DIR) / capture_name
        destination_dir = video_dir / capture_name

        destination_dir.mkdir(parents=True, exist_ok=True)

        video_path = source_dir / f'{capture_name}.mov'

        assert video_path.is_file(), (
            f'Calibration video not found: {video_path}'
        )

        shutil.copy2(
            video_path,
            destination_dir / video_path.name,
        )

    return video_dir

def stage_fixed_intrinsics_images(tmp_path):
    # Copy the fixed calibration images to an isolated temporary directory.
    image_dir = tmp_path / 'fixed_intrinsics_images'
    image_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(Path(INTRINSICS_FIXED_IMAGE_DIR).glob('*.jpg'))
    assert len(image_paths) == EXPECTED_FIXED_IMAGE_COUNT

    for image_path in image_paths:
        shutil.copy2(image_path, image_dir / image_path.name)

    return image_dir

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


# ---- Intrinsics workflow regression tests ----

def make_intrinsics(focal_length):
    return {
        'intrinsicMat': np.array([
            [focal_length, 0.0, 320.0],
            [0.0, focal_length + 10.0, 240.0],
            [0.0, 0.0, 1.0],
        ]),
        'distortion': np.array([[0.1, -0.1, 0.01, 0.02, 0.03]]),
        'imageSize': np.array([[480.0], [640.0]]),
    }


def test_intrinsics_api(tmp_path, monkeypatch):
    video_url = 'https://example.test/trial-a.mov'
    response = Mock()
    response.json.return_value = {
        'name': 'null',
        'videos': [{
            'video': video_url,
            'parameters': {'model': 'iPhone13,3'},
        }],
    }
    request = Mock(return_value=response)
    download = Mock(side_effect=lambda _, path: Path(path).touch())
    extraction = Mock()
    params = make_intrinsics(1000.0)
    monkeypatch.setattr(utilsChecker, 'makeRequestWithRetry', request)
    monkeypatch.setattr(utilsChecker, 'download_file', download)
    monkeypatch.setattr(utilsChecker, 'video2Images', extraction)
    monkeypatch.setattr(utilsChecker, 'calcIntrinsics', Mock(return_value=params))

    average, _, _, model = utilsChecker.computeAverageIntrinsics(
        str(tmp_path), ['trial-a'], DEFAULT_CHECKERBOARD_PARAMS
    )

    expected_path = os.path.join(tmp_path, 'trial-a', 'trial-a.mov')
    request.assert_called_once()
    download.assert_called_once_with(video_url, expected_path)
    assert extraction.call_args.args[0] == expected_path
    assert model == 'iPhone13,3'
    np.testing.assert_allclose(average['intrinsicMat'], params['intrinsicMat'])


def test_intrinsics_local(tmp_path, monkeypatch):
    trial_ids = ['capture-a', 'capture-b']
    paths = []
    for name in trial_ids:
        path = tmp_path / name / f'{name}.avi'
        path.parent.mkdir()
        path.touch()
        paths.append(str(path))

    request = Mock()
    download = Mock()
    extraction = Mock()
    monkeypatch.setattr(utilsChecker, 'makeRequestWithRetry', request)
    monkeypatch.setattr(utilsChecker, 'download_file', download)
    monkeypatch.setattr(utilsChecker, 'video2Images', extraction)
    monkeypatch.setattr(
        utilsChecker,
        'calcIntrinsics',
        Mock(side_effect=[make_intrinsics(900.0), make_intrinsics(1100.0)]),
    )

    average, _, _, model = utilsChecker.computeAverageIntrinsics(
        str(tmp_path),
        trial_ids,
        DEFAULT_CHECKERBOARD_PARAMS,
        nImages=5,
        cameraModel='ResearchCamera',
        videoType='.avi',
    )

    request.assert_not_called()
    download.assert_not_called()
    assert [call.args[0] for call in extraction.call_args_list] == paths
    assert model == 'ResearchCamera'
    np.testing.assert_allclose(average['intrinsicMat'][0, 0], 1000.0)


@pytest.mark.skipif(
    not os.path.isdir(IPAD_A16_FIXTURE_DIR),
    reason='ipad a16 calibration fixture is not available',
)
def test_intrinsics_ipad_a16(tmp_path):
    video_dir = stage_ipad_a16_videos(tmp_path)

    average, captures, _, model = utilsChecker.computeAverageIntrinsics(
        str(video_dir),
        ['ipad-a16_1', 'ipad-a16_2', 'ipad-a16_3'],
        {'dimensions': (11, 8), 'squareSize': 60},
        cameraModel='iPad15,7',
        videoType='.mov',
        nImages=50,
    )

    expected = loadCameraParameters(os.path.join(
        REPO_DIR,
        'CameraIntrinsics',
        'iPad15,7',
        'Deployed',
        'cameraIntrinsics.pickle',
    ))

    assert len(captures) == 3
    assert model == 'iPad15,7'

    # The intrinsic matrix is expected to remain geometrically close.
    np.testing.assert_allclose(
        average['intrinsicMat'],
        expected['intrinsicMat'],
        atol=MAX_INTRINSIC_MATRIX_DIFFERENCE_PX,
        rtol=0,
    )

    # Distortion coefficients are deliberately not compared directly.
    # Different valid checkerboard samples can produce noticeably
    # different distortion coefficients while producing almost the
    # same image-space projection.
    mean_error, max_error = mean_reprojection_difference(
        expected,
        average,
        LABVALIDATION_CHECKERBOARD_PARAMS,
    )

    assert mean_error < MAX_INTRINSICS_REPROJECTION_DIFFERENCE_PX
    assert max_error < MAX_INTRINSICS_REPROJECTION_DIFFERENCE_PX



def test_intrinsics_missing(tmp_path, monkeypatch):
    request = Mock()
    download = Mock()
    monkeypatch.setattr(utilsChecker, 'makeRequestWithRetry', request)
    monkeypatch.setattr(utilsChecker, 'download_file', download)

    with pytest.raises(FileNotFoundError, match='capture-a\\.avi'):
        utilsChecker.computeAverageIntrinsics(
            str(tmp_path),
            ['capture-a'],
            DEFAULT_CHECKERBOARD_PARAMS,
            cameraModel='ResearchCamera',
            videoType='.avi',
        )
    request.assert_not_called()
    download.assert_not_called()

def test_intrinsics_fixed_images(tmp_path):
    image_dir = stage_fixed_intrinsics_images(tmp_path)

    calculated = utilsChecker.calcIntrinsics(
        str(image_dir),
        CheckerBoardParams=LABVALIDATION_CHECKERBOARD_PARAMS,
        filenames=['*.jpg'],
        visualize=False,
    )

    assert calculated is not None

    expected = loadCameraParameters(INTRINSICS_FIXED_EXPECTED)

    # The intrinsic matrix should remain geometrically close when calibration
    # is run on the exact fixed set of reference images.
    np.testing.assert_allclose(
        calculated['intrinsicMat'],
        expected['intrinsicMat'],
        atol=MAX_INTRINSIC_MATRIX_DIFFERENCE_PX,
        rtol=0,
    )

    # Distortion coefficients are deliberately not compared directly.
    # Different valid calibrations can produce different coefficients while
    # producing nearly identical image-space projections.
    mean_error, max_error = mean_reprojection_difference(
        expected,
        calculated,
        LABVALIDATION_CHECKERBOARD_PARAMS,
    )

    assert mean_error < MAX_INTRINSICS_REPROJECTION_DIFFERENCE_PX
    assert max_error < MAX_INTRINSICS_REPROJECTION_DIFFERENCE_PX