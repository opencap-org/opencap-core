"""
---------------------------------------------------------------------------
OpenCap: changeSessionMetadata.py
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


This script allows you to change the metadata of the session. E.g., to change
the pose estimator used when reprocessing data in the cloud. This is mostly for
developer use.

The available options for metadata are:
    - scalingsetup:     upright_standing_pose
                        any_pose
    - openSimModel:     LaiUhlrich2022
                        LaiUhlrich2022_shoulder
    - posemodel:        openpose
                        hrnet
    - augmentermodel:   v0.2
                        v0.3
    - filterfrequency:  default
                        float number
    - datasharing:      Share processed data and identified videos
                        Share processed data and de-identified videos
                        Share processed data
                        Share no data


"""
import os
import sys
sys.path.append(os.path.abspath('./..'))

from utils import changeSessionMetadata

session_ids = ["0d46adef-62cb-455f-9ff3-8116717cc2fe"]

# Dictionary of metadata fields to change (see sessionMetadata.yaml).
newMetadata = {
    'openSimModel':'LaiUhlrich2022_shoulder',
    'posemodel':'hrnet',
    'augmentermodel':'v0.3',
    'filterfrequency':15,
    'datasharing':'Share processed data and identified videos',
    'scalingsetup': 'upright_standing_pose'
}
changeSessionMetadata(session_ids,newMetadata)