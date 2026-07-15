import os
import sys
from unittest.mock import patch

import pytest


thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir,'../'))
sys.path.append(repoDir)
from utils import getCalibration, getNeutralTrialID


@patch('utils.getSessionJson')
def test_get_neutral_trial_id_from_session_trial(mock_get_session):
    mock_get_session.return_value = {
        'trials': [
            {'id': 'dynamic-id', 'name': 'squats'},
            {'id': 'neutral-id', 'name': 'neutral'},
        ],
        'meta': {},
    }

    assert getNeutralTrialID('session-id') == 'neutral-id'


@patch('utils.getSessionJson')
def test_get_neutral_trial_id_from_metadata_fallback(mock_get_session):
    mock_get_session.return_value = {
        'trials': [
            {'id': 'dynamic-id', 'name': 'squats'},
        ],
        'meta': {
            'neutral_trial': {'id': 'metadata-neutral-id'},
        },
    }

    assert getNeutralTrialID('session-id') == 'metadata-neutral-id'


@patch('utils.getSessionJson')
def test_get_neutral_trial_id_raises_clear_error_without_neutral_trial(mock_get_session):
    mock_get_session.return_value = {
        'trials': [
            {'id': 'dynamic-id', 'name': 'squats'},
        ],
        'meta': {},
    }

    with pytest.raises(Exception, match='No neutral trial in session'):
        getNeutralTrialID('session-id')


@patch('utils.getTrialJson')
@patch('utils.getCalibrationTrialID')
def test_get_calibration_raises_clear_error_without_camera_mapping(
        mock_get_calibration_trial_id, mock_get_trial):
    mock_get_calibration_trial_id.return_value = 'calibration-id'
    mock_get_trial.return_value = {
        'results': [
            {'tag': 'calibration_parameters', 'media': 'calibration-url'},
        ],
    }

    with pytest.raises(Exception, match='Redo calibration before processing dynamic trials'):
        getCalibration('session-id', '/tmp/session')
