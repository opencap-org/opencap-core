import os
import sys

import cv2
import pytest

os.environ.setdefault('API_TOKEN', 'test-token')

from conftest import CALIBRATION_FIXTURE_DIR, REPO_DIR

sys.path.append(REPO_DIR)

from tests.calibration_test_utils import (
    DEFAULT_CHECKERBOARD_PARAMS,
    LABVALIDATION_CHECKERBOARD_PARAMS,
    run_video_calibration,
    assert_extrinsics,
    MAX_MEAN_REPROJECTION_ERROR_PX,
    sb_flags
)

# ---- Checkerboard / fixture constants ----

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
