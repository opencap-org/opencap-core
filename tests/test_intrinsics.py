import copy
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

os.environ.setdefault('API_TOKEN', 'test-token')

from conftest import INTRINSICS_FIXTURE_DIR, REPO_DIR

sys.path.append(REPO_DIR)

import utilsChecker

from tests.calibration_test_utils import (
    DEFAULT_CHECKERBOARD_PARAMS,
    projection_difference,
    LABVALIDATION_CHECKERBOARD_PARAMS
)

from utilsChecker import (
    loadCameraParameters,
)

# ---- Checkerboard / fixture constants ----

# These are provisional values. We should discuss what values would be appropriate.
MAX_INTRINSICS_PROJECTION_DIFFERENCE_PX = 0.25
MAX_INTRINSIC_MATRIX_DIFFERENCE_PX = 2.0

IPAD_A16_FIXTURE_DIR = os.path.join(
    INTRINSICS_FIXTURE_DIR,
    'ipad-a16',
)

INTRINSICS_FIXED_IMAGE_DIR = os.path.join(
    INTRINSICS_FIXTURE_DIR,
    'ipad-a16',
    'ipad-a16_1',
    'fixed_images',
)

EXPECTED_FIXED_IMAGE_COUNT = 49

INTRINSICS_FIXED_EXPECTED = os.path.join(
    INTRINSICS_FIXED_IMAGE_DIR,
    'cameraIntrinsics.pickle',
)

# ---- Helpers ----

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

    # The intrinsic matrix should remain numerically close to the deployed calibration.
    np.testing.assert_allclose(
        average['intrinsicMat'],
        expected['intrinsicMat'],
        atol=MAX_INTRINSIC_MATRIX_DIFFERENCE_PX,
        rtol=0,
    )

    # Distortion coefficients are deliberately not compared directly.
    # Different valid calibrations can produce noticeably
    # different distortion coefficients while producing almost the
    # same image-space projection.
    max_error = projection_difference(expected, average)

    assert max_error < MAX_INTRINSICS_PROJECTION_DIFFERENCE_PX


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

    # The intrinsic matrix should remain numerically close when calibration
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
    max_error = projection_difference(expected, calculated)

    assert max_error < MAX_INTRINSICS_PROJECTION_DIFFERENCE_PX


@pytest.mark.parametrize('rotation', [0, 180, 270])
def test_rotate_intrinsics_matches_rotated_image(rotation, monkeypatch):
    # Intrinsics are stored upright (rotation 90). rotateIntrinsics must describe the same
    # camera after the image is rotated: projecting a 3D point with the rotated parameters
    # must land on the rotated pixel of its projection with the upright parameters. This
    # covers the focal lengths, the principal point and the tangential distortion (p1, p2).
    width, height = 720, 1280  # upright image; imageSize is [height, width]
    upright = {
        'intrinsicMat': np.array([[1100.0, 0.0, 362.3],
                                  [0.0, 1090.0, 641.7],
                                  [0.0, 0.0, 1.0]]),
        'distortion': np.array([[0.12, -0.35, -0.0069, -0.003, 0.2]]),
        'imageSize': np.array([[height], [width]]),
    }

    # Exact pixel mapping (pixel centres at integer coordinates) and the matching
    # camera-frame rotation, for each video rotation.
    pixel_map = {
        0: lambda u, v: (v, (width - 1) - u),
        180: lambda u, v: ((height - 1) - v, u),
        270: lambda u, v: ((width - 1) - u, (height - 1) - v),
    }[rotation]
    rotate_points = {
        0: lambda X: np.c_[X[:, 1], -X[:, 0], X[:, 2]],
        180: lambda X: np.c_[-X[:, 1], X[:, 0], X[:, 2]],
        270: lambda X: np.c_[-X[:, 0], -X[:, 1], X[:, 2]],
    }[rotation]

    def project(points, params):
        pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                      params['intrinsicMat'], params['distortion'])
        return pixels.reshape(-1, 2)

    # 3D points spread over the whole image, 1-5 m from the camera.
    rng = np.random.default_rng(0)
    u, v = rng.uniform(0, width - 1, 200), rng.uniform(0, height - 1, 200)
    depth = rng.uniform(1, 5, 200)
    K = upright['intrinsicMat']
    points = np.c_[(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth]

    monkeypatch.setattr(utilsChecker, 'getVideoRotation', lambda _: rotation)
    rotated = utilsChecker.rotateIntrinsics(copy.deepcopy(upright), 'video.mov')

    expected = np.c_[pixel_map(*project(points, upright).T)]
    np.testing.assert_allclose(project(rotate_points(points), rotated), expected, atol=1e-6)
