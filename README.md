# OpenCap Core Forked
This code takes two or more videos and estimates 3D marker positions and human movement kinematics (joint angles) in an OpenSim format. You can just get to the augmented markers by setting runOpenSimPipeline=False in the main() function

2) Run this pipeline locally using mp4 videos recorded using a video camera and a recording of a checkerboard.

3) Run this pipeline locally using videos collected near-synchronously from another source (e.g., videos collected synchronously with marker-based motion capture). Easy-to-use utilities for this pipeline are under development and will be released soon.


## Publication
More information is available in [preprint](https://www.biorxiv.org/content/10.1101/2022.07.07.499061v1): <br> <br> 
Uhlrich SD*, Falisse A*, Kidzinski L*, Ko M, Chaudhari AS, Hicks JL, Delp SL, 2022. OpenCap: 3D human movement dynamics from smartphone videos. _biorxiv_. https://doi.org/10.1101/2022.07.07.499061. *contributed equally <br> <br> 
Archived code base accompanying the paper: [https://doi.org/10.5281/zenodo.7419967](https://doi.org/10.5281/zenodo.7419967).

## Running the pipeline locally
### Hardware and OS requirements:
This is the one OpenCap Repository that requires a NVIDIA GPU to run.
These instructions are for Windows 10. The pipeline also runs on Ubuntu. Minimum GPU requirements: CUDA-enabled GPU with at least 4GB memory. Not all of the OpenPose settings will run on small GPUs. To run the OpenPose settings we use in the cloud pipeline, you need a GPU with 8GB of memory. To run the high resolution settings, you need a GPU with at least 24GB memory. For local postprocessing, we use NVIDIA GeForce RTX 3090s (24GB).