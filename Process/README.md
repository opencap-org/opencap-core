These scripts process the video data from videos to kinemaitcs:

Kristen Steudel, working off of a fork of OpenCap-Core

Overview of the files:
- `labValidationVideosToKinematics.py`: loads the static trial videos and scales the model.

- `labValidationVideosToKinematicsNoStatic.py`: Takes in a scaled model and videos and computes the preaugmented markers, postaugmented markers, and kinematics for that trial.
    - Options: runOpenSimPipeline=False stops the script after calculating the post augmentation marker file. Then transfer to the process-opencap-sprinting repository to import the marker files for scaling and running kinematics. Then transfer to opencap-processing for calculating study outputs of interest.

- `GetAndPlotMTULengths.py`: Plots the MTU lengths for the bflh

