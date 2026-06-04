import logging
import os
import sys
import pytest
from unittest.mock import patch, Mock
from http.client import HTTPMessage

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir,'../'))
sys.path.append(repoDir)
from utils import makeRequestWithRetry

logging.getLogger('urllib3').setLevel(logging.DEBUG)

@patch("requests.Session.request")
def test_get(mock_response):
    status_code = 200
    mock_response.return_value.status_code = status_code

    response = makeRequestWithRetry('GET', 'https://test.com', retries=2)
    assert response.status_code == status_code
    mock_response.assert_called_once_with('GET', 'https://test.com',
                                          headers=None,
                                          data=None,
                                          params=None,
                                          files=None)

@patch("requests.Session.request")
def test_put(mock_response):
    status_code = 201
    mock_response.return_value.status_code = status_code

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
    mock_response.assert_called_once_with('POST',
                                         'https://test.com',
                                         data=data,
                                         headers={"Authorization": "my_token"},
                                         params=params,
                                         files=None)

@patch("urllib3.connectionpool.HTTPConnectionPool._get_conn")
def test_success_after_retries(mock_response):
    mock_response.return_value.getresponse.side_effect = [
        Mock(status=500, msg=HTTPMessage()),
        Mock(status=502, msg=HTTPMessage()),
        Mock(status=200, msg=HTTPMessage()),
        Mock(status=429, msg=HTTPMessage()),
    ]

    response = makeRequestWithRetry('GET',
                                    'https://test.com',
                                    retries=5,
                                    backoff_factor=0.1)

    assert response.status_code == 200
    assert mock_response.call_count == 3

# The httpbin test remains commented out for stability reasons
# def test_httpbin():
#     response = makeRequestWithRetry('GET',
#                                     'https://httpbin.org/status/500',
#                                     retries=4,
#                                     backoff_factor=0.1)