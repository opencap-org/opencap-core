import os
import sys
from unittest.mock import Mock, patch


thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir,'../'))
sys.path.append(repoDir)
from utils import changeSessionMetadata


@patch('utils.getTrialJson')
@patch('utils.getNeutralTrialID')
@patch('utils.makeRequestWithRetry')
@patch('utils.getSessionJson')
def test_change_session_metadata_uses_settings_framerate_for_filterfrequency(
        mock_get_session, mock_make_request, mock_get_neutral, mock_get_trial):
    mock_get_session.return_value = {
        'meta': {
            'settings': {
                'framerate': 240,
            },
        },
    }
    mock_make_request.return_value = Mock(status_code=200)
    mock_get_neutral.return_value = 'neutral-id'
    mock_get_trial.return_value = {
        'results': [],
    }

    changeSessionMetadata(['session-id'], {'filterfrequency': 100})

    patched_meta = mock_make_request.call_args.kwargs['data']['meta']
    assert '"filterfrequency": "100"' in patched_meta
