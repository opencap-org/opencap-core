"""
---------------------------------------------------------------------------
OpenCap: batchDownloadData.py
---------------------------------------------------------------------------

Copyright 2022 Stanford University and the Authors

Author(s): Scott Uhlrich, Antoine Falisse

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy
of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


This script dowloads data from multiple sessions from the database.


You will need login credentials from app.opencap.ai to run this script. You 
can quickly create credentials from the home page: https://app.opencap.ai. 
We recommend first running the Examples/createAuthenticationEnvFile.py script
prior to running this script. Otherwise, you will need to to login every time 
you run the script.
"""

# %% Imports

import sys
import os
sys.path.append(os.path.abspath('./..'))
import utils

# %% User inputs

# Sessions to download. List of strings of identifiers (36-character) string
# at the end of the session url app.opencap.ai/session/<session_id>
session_ids = [
               '23d52d41-69fe-47cf-8b60-838e4268dd50'
              ] # list of session identifiers as strings

# Local path to download sessions to
downloadFolder = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','Data'))

# Name downloaded folder with subject name from web app (True), or the session_id (False)
useSubjectIdentifierAsFolderName = True


#%% Processing

# Batch download. 
for session_id in session_ids:
    print(f'Downloading session id {session_id}')
    utils.downloadAndZipSession(session_id,justDownload=True,data_dir=downloadFolder,
                          useSubjectNameFolder=useSubjectIdentifierAsFolderName,
                          include_pose_pickles=False)
