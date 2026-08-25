import os


TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.abspath(os.path.join(TESTS_DIR, '../'))
TEST_DATA_ROOT = os.path.join(TESTS_DIR, 'opencap-test-data')
TEST_DATA_DIR = os.path.join(TEST_DATA_ROOT, 'Data')
CALIBRATION_FIXTURE_DIR = os.path.join(TEST_DATA_DIR, 'calibration-fixtures')
SYNC_2CAM_DIR = os.path.join(TEST_DATA_DIR, 'sync_2-cameras')
LAB_5CAM_DIR = os.path.join(
    TEST_DATA_DIR,
    'labvalidation-fixtures',
    'subject2_session0_5-cameras',
)
