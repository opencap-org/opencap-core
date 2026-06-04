"""
---------------------------------------------------------------------------
OpenCap: checkDataForSubjects.py
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


This script checks if all videos are uploaded and not error for a subject 
across however many sessions are associated with them.

Note: this script does not run without updating the subject name and trialnames
for your own data.


You will need login credentials from app.opencap.ai to run this script. You 
can quickly create credentials from the home page: https://app.opencap.ai. 
We recommend first running the Examples/createAuthenticationEnvFile.py script
prior to running this script. Otherwise, you will need to login every time 
you run the script.
"""

import sys
import os
sys.path.append(os.path.abspath('./..'))

import utils

# %% Editable variables

subjectName = 'p029'
 
# Trials you expect in different sessions.
trialNames = [
              ['brooke', '10mwt', '10mwrt', 'tug_line', 'tug_cone'],
              ['5xsts', 'arm_rom', 'curls', 'toe_stand', 'jump'],
              ['stairs_up','stairs_down']
            ]


# %% main

subjectSessions = utils.getSubjectSessions(subjectName)
if len(subjectSessions) == 0:
    raise Exception('no sessions for this subject')
    
subjectTrialNames = [utils.getTrialNames(s) for s in subjectSessions]
sessionMapping = utils.findSessionWithTrials(subjectTrialNames,trialNames)

for i,sMap in enumerate(sessionMapping):
    if sMap is None:
        print('Session with ' + trialNames[i][0] + ' doesn''t have every trial.') 
        continue
    
    session = subjectSessions[sMap]
    subjectTrialNamesThisSession = subjectTrialNames[sMap]
    goodTrialNames = []
    for t in trialNames[i]:
        trials = [st for st in subjectTrialNamesThisSession if t in st]
        if len(trials) > 1:
            goodTrialNames.append(utils.get_entry_with_largest_number(trials))
        else:
            goodTrialNames.append(trials[0])
        
    # check for nNeutral videos, make sure all dynamic trials have the same and they are uploaded
    nNeutralVideos = len(session['trials'][subjectTrialNamesThisSession.index('neutral')]['videos'])
    
    anyBad=0
    for t in goodTrialNames:
        trial = session['trials'][subjectTrialNamesThisSession.index(t)]
        if trial['status'] == 'error':
            anyBad=1
            print(trial['name'] + ' is error.')
            continue
        nVideos = sum([1 for v in trial['videos'] if v != None])
        if nVideos != nNeutralVideos:
            anyBad = 1
            print(trial['name'] + ' only has ' + str(nVideos) + ' of ' + str(nNeutralVideos) + ' videos.')
    if anyBad == 0:
        print('All trials in the ' + trialNames[i][0] + ' session have ' + str(nNeutralVideos) + ' videos.')
    else:
        print('Problem trials in ' + trialNames[i][0] + ' session: ' + session['id'])
        

