import logging
import os
import sys
import pytest
from unittest.mock import patch, Mock, mock_open

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir,'../'))
sys.path.append(repoDir)
from utils import (
    DEFAULT_REQUEST_TIMEOUT,
    UPLOAD_REQUEST_TIMEOUT,
    makeRequestWithRetry,
    uploadFileToS3,
)

logging.getLogger('urllib3').setLevel(logging.DEBUG)

@patch("requests.Session.request")
def test_get(mock_request):
    status_code = 200
    mock_request.return_value.status_code = status_code

    response = makeRequestWithRetry('GET', 'https://test.com', retries=2)
    assert response.status_code == status_code
    mock_request.assert_called_once_with('GET', 'https://test.com',
                                         headers=None,
                                         data=None,
                                         params=None,
                                         files=None,
                                         timeout=DEFAULT_REQUEST_TIMEOUT)

@patch("requests.Session.request")
def test_put(mock_request):
    status_code = 201
    mock_request.return_value.status_code = status_code

    data = {
        "key1": "value1",
        "key2": "value2"
    }

    params = {
        "param1": "value1"
    }

    response = makeRequestWithRetry('POST',
                                    'https://test.com',
                                    data=data,
                                    headers={"Authorization": "my_token"},
                                    params=params,
                                    retries=2)

    assert response.status_code == status_code
    mock_request.assert_called_once_with('POST',
                                         'https://test.com',
                                         data=data,
                                         headers={"Authorization": "my_token"},
                                         params=params,
                                         files=None,
                                         timeout=DEFAULT_REQUEST_TIMEOUT)

@patch("builtins.open", new_callable=mock_open, read_data=b"file-data")
@patch("utils.makeRequestWithRetry")
def test_upload_timeout(mock_request, mock_file):
    mock_request.return_value.json.return_value = {
        'url': 'https://upload.test.com',
        'fields': {'key': 'uploaded-file'},
    }

    key = uploadFileToS3('/tmp/upload.mov')

    assert key == 'uploaded-file'
    mock_request.assert_any_call('POST',
                                 'https://upload.test.com',
                                 data={'key': 'uploaded-file'},
                                 files={'file': mock_file.return_value},
                                 timeout=UPLOAD_REQUEST_TIMEOUT)

@patch("urllib3.connectionpool.HTTPConnectionPool._get_conn")
def test_success_after_retries(mock_get_conn):
    def make_response(status):
        response = Mock(status=status, headers={})
        response.stream.return_value = []
        response._original_response = None
        return response

    mock_get_conn.return_value.getresponse.side_effect = [
        make_response(500),
        make_response(502),
        make_response(200),
        make_response(429),
    ]

    response = makeRequestWithRetry('GET',
                                    'https://test.com',
                                    retries=5,
                                    backoff_factor=0.1)

    assert response.status_code == 200
    assert mock_get_conn.call_count == 3

# The httpbin test remains commented out for stability reasons
# def test_httpbin():
#     response = makeRequestWithRetry('GET',
#                                     'https://httpbin.org/status/500',
#                                     retries=4,
#                                     backoff_factor=0.1)
