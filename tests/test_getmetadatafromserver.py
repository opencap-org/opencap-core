import os
import sys
from unittest.mock import patch

import pytest

thisDir = os.path.dirname(os.path.realpath(__file__))
repoDir = os.path.abspath(os.path.join(thisDir, '../'))
sys.path.append(repoDir)

from utils import getMetadataFromServer


class TestGetMetadataFromServer:
    """Test suite for getMetadataFromServer function."""

    @pytest.fixture
    def mock_default_metadata(self):
        default_metadata = {
            "subjectID": "default_subject",
            "mass_kg": 75.0,
            "height_m": 1.75,
            "gender_mf": "Prefer not to respond",
            "posemodel": "openpose",
            "openSimModel": "LaiUhlrich2022",
            "augmentermodel": "v0.2",
            "filterfrequency": "default",
            "scalingsetup": "upright_standing_pose",
            "sync_ver": "1.1",
            "checkerBoard": {
                "squareSideLength_mm": 25.0,
                "black2BlackCornersWidth_n": 7,
                "black2BlackCornersHeight_n": 5,
                "placement": "ground"
            }
        }

        with patch('utils.importMetadata', return_value=default_metadata):
            yield default_metadata

    @pytest.fixture
    def mock_session_json_with_legacy_meta(self):
        """Create a mock session JSON with legacy metadata structure (direct subject fields)."""
        return {
            'id': 'test-session-123',
            'name': 'Test Session',
            'meta': {
                'subject': {
                    'id': 'test-subject',
                    'mass': 80.5,
                    'height': 1.82,
                    'gender': 'man',
                    'posemodel': 'openpose'
                },
                'settings': {
                    'openSimModel': 'LaiUhlrich2022',
                    'augmentermodel': 'v0.2',
                    'filterfrequency': '12',
                    'scalingsetup': 'upright_standing_pose',
                    'sync_ver': '1.1'
                },
                'checkerboard': {
                    'square_size': 30.0,
                    'cols': 8,
                    'rows': 6,
                    'placement': 'floor'
                }
            },
            'subject': None
        }

    @pytest.fixture
    def mock_session_json_with_subject_ref(self):
        """Create a mock session JSON with subject reference structure."""
        return {
            'id': 'test-session-456',
            'name': 'Test Session 2',
            'meta': {
                'settings': {
                    'openSimModel': 'LaiUhlrich2022',
                    'augmentermodel': 'v0.2',
                    'filterfrequency': 'default',
                    'scalingsetup': 'upright_standing_pose',
                    'sync_ver': '1.1',
                    'posemodel': 'mmpose'
                },
                'checkerboard': {
                    'square_size': 25.0,
                    'cols': 7,
                    'rows': 5,
                    'placement': 'ground'
                }
            },
            'subject': 'subject-789'
        }

    @pytest.fixture
    def mock_subject_json(self):
        """Create a mock subject JSON."""
        return {
            'id': 'subject-789',
            'name': 'Test Subject',
            'weight': 70.0,
            'height': 1.68,
            'gender': 'woman'
        }

    @pytest.fixture
    def mock_session_json_with_calibration_ref(self):
        """Create a mock session JSON with reference to another session for calibration."""
        return {
            'id': 'test-session-456',
            'name': 'Test Session',
            'meta': {
                'sessionWithCalibration': {
                    'id': 'calibration-session-789'
                },
                'checkerboard': {
                    'square_size': 25.0,
                    'cols': 7,
                    'rows': 5,
                    'placement': 'ground'
                }
            },
            'subject': 'subject-789'
        }

    @pytest.fixture
    def mock_calibration_session_json(self):
        """Create a mock calibration session JSON."""
        return {
            'id': 'calibration-session-789',
            'name': 'Calibration Session',
            'meta': {
                'checkerboard': {
                    'square_size': 25.0,
                    'cols': 7,
                    'rows': 5,
                    'placement': 'ground'
                }
            },
            'subject': 'subject-789'
        }

    def test_get_metadata_with_legacy_structure(self, mock_default_metadata, mock_session_json_with_legacy_meta):
        """Test getMetadataFromServer with legacy metadata structure (direct subject fields)."""
        with patch('utils.getSessionJson', return_value=mock_session_json_with_legacy_meta):
            result = getMetadataFromServer('test-session-123')

            # Check that values from session metadata override defaults
            assert result['subjectID'] == 'test-subject'
            assert result['mass_kg'] == 80.5
            assert result['height_m'] == 1.82
            assert result['gender_mf'] == 'Man'
            assert result['posemodel'] == 'openpose'
            assert result['openSimModel'] == 'LaiUhlrich2022'
            assert result['augmentermodel'] == 'v0.2'
            assert result['filterfrequency'] == 12.0
            assert result['scalingsetup'] == 'upright_standing_pose'
            assert result['sync_ver'] == '1.1'

            # Check calibration parameters
            assert result['checkerBoard']['squareSideLength_mm'] == 30.0
            assert result['checkerBoard']['black2BlackCornersWidth_n'] == 8
            assert result['checkerBoard']['black2BlackCornersHeight_n'] == 6
            assert result['checkerBoard']['placement'] == 'floor'

    def test_get_metadata_with_subject_reference(self, mock_default_metadata,
                                                 mock_session_json_with_subject_ref,
                                                 mock_subject_json):
        """Test getMetadataFromServer with subject reference structure."""
        with patch('utils.getSessionJson', return_value=mock_session_json_with_subject_ref):
            with patch('utils.getSubjectJson', return_value=mock_subject_json):
                result = getMetadataFromServer('test-session-456')

                # Check that subject fields are loaded from referenced subject
                assert result['subjectID'] == 'Test Subject'
                assert result['mass_kg'] == 70.0
                assert result['height_m'] == 1.68
                assert result['gender_mf'] == 'Woman'

                # Check settings from session meta
                assert result['posemodel'] == 'mmpose'
                assert result['openSimModel'] == 'LaiUhlrich2022'
                assert result['augmentermodel'] == 'v0.2'
                assert result['filterfrequency'] == 'default'
                assert result['scalingsetup'] == 'upright_standing_pose'
                assert result['sync_ver'] == '1.1'

                # Check calibration parameters
                assert result['checkerBoard']['squareSideLength_mm'] == 25.0
                assert result['checkerBoard']['black2BlackCornersWidth_n'] == 7
                assert result['checkerBoard']['black2BlackCornersHeight_n'] == 5
                assert result['checkerBoard']['placement'] == 'ground'

    def test_get_metadata_with_just_checker_params(self, mock_default_metadata,
                                                   mock_session_json_with_legacy_meta):
        """Test getMetadataFromServer with justCheckerParams=True."""
        with patch('utils.getSessionJson', return_value=mock_session_json_with_legacy_meta):
            result = getMetadataFromServer('test-session-123', justCheckerParams=True)

            # Should only have checkerboard parameters and defaults
            # Subject-specific fields should still be defaults
            assert result['subjectID'] == 'default_subject'  # Not updated
            assert result['mass_kg'] == 75.0  # Not updated
            assert result['height_m'] == 1.75  # Not updated

            # Checkerboard should be updated from session
            assert result['checkerBoard']['squareSideLength_mm'] == 30.0
            assert result['checkerBoard']['black2BlackCornersWidth_n'] == 8
            assert result['checkerBoard']['black2BlackCornersHeight_n'] == 6
            assert result['checkerBoard']['placement'] == 'floor'

    def test_get_metadata_with_calibration_reference(self, mock_default_metadata,
                                                     mock_session_json_with_calibration_ref,
                                                     mock_calibration_session_json,
                                                     mock_subject_json):
        """Test getMetadataFromServer when session references another session for calibration."""
        # First call returns session with reference
        # Second call (for calibration session) returns the calibration session
        with patch('utils.getSessionJson') as mock_get_session:
            mock_get_session.side_effect = [
                mock_session_json_with_calibration_ref,  # First call for test session
                mock_calibration_session_json  # Second call for calibration session
            ]
            with patch('utils.getSubjectJson', return_value=mock_subject_json):
                result = getMetadataFromServer('test-session-456')

                # Check that checkerboard comes from calibration session
                assert result['checkerBoard']['squareSideLength_mm'] == 25.0
                assert result['checkerBoard']['black2BlackCornersWidth_n'] == 7
                assert result['checkerBoard']['black2BlackCornersHeight_n'] == 5
                assert result['checkerBoard']['placement'] == 'ground'

    def test_get_metadata_with_missing_checkerboard(self, mock_default_metadata):
        """Test getMetadataFromServer when checkerboard is missing from session meta."""
        session_json_no_checker = {
            'id': 'test-session-123',
            'name': 'Test Session',
            'meta': {
                'subject': {
                    'id': 'test-subject',
                    'mass': 80.5,
                    'height': 1.82
                }
                # No checkerboard field
            },
            'subject': None
        }

        with patch('utils.getSessionJson', return_value=session_json_no_checker):
            # This should raise an exception because checkerboard is required
            with pytest.raises(KeyError):
                result = getMetadataFromServer('test-session-123')

    def test_get_metadata_with_none_meta(self, mock_default_metadata, caplog):
        """Test getMetadataFromServer when session.meta is None."""
        session_json_no_meta = {
            'id': 'test-session-123',
            'name': 'Test Session',
            'meta': None,
            'subject': None
        }

        with patch('utils.getSessionJson', return_value=session_json_no_meta):
            result = getMetadataFromServer('test-session-123')

            # Should return default metadata
            assert result['subjectID'] == 'default_subject'
            assert result['mass_kg'] == 75.0
            assert result['height_m'] == 1.75

    def test_get_metadata_with_filterfrequency_default(self, mock_default_metadata,
                                                       mock_session_json_with_legacy_meta):
        """Test handling of filterfrequency = 'default'."""
        session_json = mock_session_json_with_legacy_meta.copy()
        session_json['meta']['settings']['filterfrequency'] = 'default'

        with patch('utils.getSessionJson', return_value=session_json):
            result = getMetadataFromServer('test-session-123')

            # Should remain as string 'default'
            assert result['filterfrequency'] == 'default'

    def test_get_metadata_with_filterfrequency_number(self, mock_default_metadata,
                                                      mock_session_json_with_legacy_meta):
        """Test handling of filterfrequency as number."""
        session_json = mock_session_json_with_legacy_meta.copy()
        session_json['meta']['settings']['filterfrequency'] = '15'

        with patch('utils.getSessionJson', return_value=session_json):
            result = getMetadataFromServer('test-session-123')

            # Should convert to float
            assert result['filterfrequency'] == 15.0

    def test_get_metadata_missing_settings(self, mock_default_metadata):
        """Test when settings are missing from session meta."""
        session_json_no_settings = {
            'id': 'test-session-123',
            'name': 'Test Session',
            'meta': {
                'subject': {
                    'id': 'test-subject',
                    'mass': 80.5,
                    'height': 1.82
                },
                'checkerboard': {
                    'square_size': 25.0,
                    'cols': 7,
                    'rows': 5,
                    'placement': 'ground'
                }
            },
            'subject': None
        }

        with patch('utils.getSessionJson', return_value=session_json_no_settings):
            result = getMetadataFromServer('test-session-123')

            # Should use default values for settings
            assert result['openSimModel'] == 'LaiUhlrich2022'
            assert result['augmentermodel'] == 'v0.2'
            assert result['filterfrequency'] == 'default'
            assert result['scalingsetup'] == 'upright_standing_pose'
            assert result['sync_ver'] == '1.1'

    def test_get_metadata_different_genders(self, mock_default_metadata):
        """Test handling of different gender values."""
        gender_cases = [
            ('woman', 'Woman'),
            ('man', 'Man'),
            ('transgender', 'Transgender'),
            ('non-binary', 'Non-Binary/Non-Conforming'),
            ('prefer-not-respond', 'Prefer not to respond')
        ]

        for input_gender, expected_output in gender_cases:
            session_json = {
                'id': 'test-session-123',
                'name': 'Test Session',
                'meta': {
                    'subject': {
                        'id': 'test-subject',
                        'mass': 80.5,
                        'height': 1.82,
                        'gender': input_gender
                    },
                    'checkerboard': {
                        'square_size': 25.0,
                        'cols': 7,
                        'rows': 5,
                        'placement': 'ground'
                    }
                },
                'subject': None
            }

            with patch('utils.getSessionJson', return_value=session_json):
                result = getMetadataFromServer('test-session-123')
                assert result['gender_mf'] == expected_output

    def test_get_metadata_unknown_gender(self, mock_default_metadata):
        """Test handling of unknown gender value."""
        session_json = {
            'id': 'test-session-123',
            'name': 'Test Session',
            'meta': {
                'subject': {
                    'id': 'test-subject',
                    'mass': 80.5,
                    'height': 1.82,
                    'gender': 'unknown'
                },
                'checkerboard': {
                    'square_size': 25.0,
                    'cols': 7,
                    'rows': 5,
                    'placement': 'ground'
                }
            },
            'subject': None
        }

        with patch('utils.getSessionJson', return_value=session_json):
            result = getMetadataFromServer('test-session-123')
            # Unknown gender should return None
            assert result['gender_mf'] is None

    def test_get_metadata_numeric_conversion(self, mock_default_metadata,
                                             mock_session_json_with_legacy_meta):
        """Test numeric conversions for mass, height, and filterfrequency."""
        session_json = mock_session_json_with_legacy_meta.copy()
        session_json['meta']['subject']['mass'] = '85.5'  # String input
        session_json['meta']['subject']['height'] = '1.75'  # String input

        with patch('utils.getSessionJson', return_value=session_json):
            result = getMetadataFromServer('test-session-123')

            assert result['mass_kg'] == 85.5
            assert result['height_m'] == 1.75

    def test_get_metadata_missing_posemodel_in_legacy(self, mock_default_metadata):
        """Test backward compatibility when posemodel missing from subject."""
        session_json = {
            'id': 'test-session-123',
            'meta': {
                'subject': {
                    'id': 'test-subject',
                    'mass': 80.5,
                    'height': 1.82,
                    # No posemodel field
                },
                'checkerboard': {
                    'square_size': 25.0,
                    'cols': 7,
                    'rows': 5,
                    'placement': 'ground'
                }
            }
        }

        with patch('utils.getSessionJson', return_value=session_json):
            result = getMetadataFromServer('test-session-123')
            assert result['posemodel'] == 'openpose'  # Should default to openpose
