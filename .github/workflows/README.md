# Workflows

The actions only push the opencap (-dev) image to ECR, not the openpose (-dev) and mmpose (-dev) images.
The mmpose (-dev) image is too big and the action fail. To push them to ECR, do it manually through the makefile by running make build and then make run.