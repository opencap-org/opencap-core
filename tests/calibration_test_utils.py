import os
import shutil

import cv2
import numpy as np

import utilsChecker
from conftest import REPO_DIR
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



def projection_difference(
    reference_params,
    calculated_params,
    grid_spacing=20,
):
    """
    Compare two camera models over the full image domain.

    The reference camera model defines the normalized camera ray associated
    with each sampled image pixel. Those rays are then projected through both
    camera models, and the resulting image-space difference is measured.

    This checks whether the calculated calibration reproduces the reference
    camera's image geometry across the entire image, rather than only over a
    checkerboard-sized region.
    """
    reference_image_size = np.squeeze(reference_params['imageSize']).astype(int)
    calculated_image_size = np.squeeze(calculated_params['imageSize']).astype(int)

    assert np.array_equal(
        reference_image_size,
        calculated_image_size,
    ), (
        f"Image sizes differ: reference={reference_image_size}, "
        f"calculated={calculated_image_size}"
    )

    image_height, image_width = reference_image_size

    # Sample pixels across the entire image, including the boundaries.
    x = np.arange(0, image_width, grid_spacing, dtype=np.float32)
    y = np.arange(0, image_height, grid_spacing, dtype=np.float32)

    if x[-1] != image_width - 1:
        x = np.append(x, image_width - 1)
    if y[-1] != image_height - 1:
        y = np.append(y, image_height - 1)

    xx, yy = np.meshgrid(x, y)
    image_points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(
        np.float32
    )

    # Convert reference-image pixels into normalized camera coordinates.
    # These normalized points define the camera rays used for the comparison.
    reference_normalized = cv2.undistortPoints(
        image_points.reshape(-1, 1, 2),
        reference_params['intrinsicMat'],
        reference_params['distortion'],
    ).reshape(-1, 2)

    points_3d = np.column_stack(
        [
            reference_normalized,
            np.ones(len(reference_normalized)),
        ]
    ).astype(np.float32)

    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)

    reference_projected, _ = cv2.projectPoints(
        points_3d,
        rvec,
        tvec,
        reference_params['intrinsicMat'],
        reference_params['distortion'],
    )

    calculated_projected, _ = cv2.projectPoints(
        points_3d,
        rvec,
        tvec,
        calculated_params['intrinsicMat'],
        calculated_params['distortion'],
    )

    reference_projected = reference_projected.reshape(-1, 2)
    calculated_projected = calculated_projected.reshape(-1, 2)

    differences = np.linalg.norm(
        reference_projected - calculated_projected,
        axis=1,
    )

    return float(np.max(differences))



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