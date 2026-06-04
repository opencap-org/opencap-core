These scripts reproduce the analyses from the paper:

Uhlrich, SD\*, Falisse A\*, Kidzinski L\*, Muccini J, Ko M, Chaudhari AS, Hicks JL, Delp SL. 2022.
OpenCap: 3D human movement dynamics from smartphone videos. biorxiv. https://doi.org/10.1101/2022.07.07.499061. *contributed equally

To run them, download the dataset from [SimTK](https://simtk.org/projects/opencap), and copy it into your repository directory. When complete, you 
should have two folders: `<repoDirectory>\Data\LabValidation` and `<repoDirectory>\Data\FieldStudy`.

The data are processed in steps, and the results from each step are provided in the online dataset. Thus, once you download the data, you could run any of these scripts in isolation. E.g., if you just want to reproduce the plots, run `makePaperPlots.py`.

Overview of the files:
- `gatherSimulationResults.py`: loads the outputs from dynamic simulations and saves a file with kinematics and kinetics for each subject and activity.
- `computeScalars.py`: computes scalar values and averages time series data across activities for all participants.
- `makePaperPlots.py`: loads metrics averaged across subjects and generate plots for the paper.
- `labValidationVideosToKinematics.py`: computes kinematics from videos using the data from the paper. You do not need to download the dataset to run this script. Please read the documentation in the script for more details.
