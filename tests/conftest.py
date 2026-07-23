import os


TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.abspath(os.path.join(TESTS_DIR, '../'))
TEST_DATA_ROOT = os.path.join(TESTS_DIR, 'opencap-test-data')
TEST_DATA_DIR = os.path.join(TEST_DATA_ROOT, 'Data')
CALIBRATION_FIXTURE_DIR = os.path.join(TEST_DATA_DIR, 'calibration-fixtures')
