import os
import time
import logging
import shutil
import ffmpeg
import json
import subprocess

logging.basicConfig(level=logging.INFO)

#%%
def check_cuda_device():
    try:
        # Run the nvidia-smi command and capture the output
        _ = subprocess.check_output(["nvidia-smi"])
        # If the command ran successfully, assume a CUDA device is present
        logging.info("A CUDA-capable device is detected.")
    except subprocess.CalledProcessError as e:
        # If the command fails, it means no CUDA device is detected
        logging.info("No CUDA-capable device is detected. Error:", e)
        raise Exception("No CUDA-capable device is detected.")

#%%
def getVideoOrientation(videoPath):
    
    meta = ffmpeg.probe(videoPath)
    try:
        rotation = meta['format']['tags']['com.apple.quicktime.video-orientation']
    except:
        # For AVI (after we rewrite video), no rotation paramter, so just using h and w. 
        # For now this is ok, we don't need leaning right/left for this, just need to know
        # how to orient the pose estimation resolution parameters.
        try: 
            if meta['format']['format_name'] == 'avi':
                if meta['streams'][0]['height']>meta['streams'][0]['width']:
                    rotation = 90
                else:
                    rotation = 0
            else:
                raise Exception('no rotation info')
        except:
            rotation = 90 # upright is 90, and intrinsics were captured in that orientation
            
    if int(rotation) in [0,180]: 
        horizontal = True
    else:
        horizontal = False
        
    return horizontal

#%%
def getResolutionCommand(resolutionPoseDetection, horizontal):
    
    # Adjust OpenPose call based on selected resolution.    
    if resolutionPoseDetection == 'default':
        cmd_hr = ' '
    elif resolutionPoseDetection == '1x1008_4scales':
        if horizontal:
            cmd_hr = ' --net_resolution "1008x-1" --scale_number 4 --scale_gap 0.25 '
        else:
            cmd_hr = ' --net_resolution "-1x1008" --scale_number 4 --scale_gap 0.25 '
    elif resolutionPoseDetection == '1x736':
        if horizontal:
            cmd_hr = ' --net_resolution "736x-1" '
        else:
            cmd_hr = ' --net_resolution "-1x736" '  
    elif resolutionPoseDetection == '1x736_2scales':
        if horizontal:
            cmd_hr = ' --net_resolution "-1x736" --scale_number 2 --scale_gap 0.75 '
        else:
            cmd_hr = ' --net_resolution "736x-1" --scale_number 2 --scale_gap 0.75 '
            
    return cmd_hr

#%% 

logging.info("Waiting for data...")

video_path = "/openpose/data/video_openpose.mov"
output_dir = "/openpose/data/output_openpose"

# Set resolution for OpenPose ('default', '1x736', or '1x1008_4scales').
with open('/openpose/defaultOpenCapSettings.json') as f:
    defaultOpenCapSettings = json.load(f)
resolutionPoseDetection = defaultOpenCapSettings['openpose']
    
if os.path.isfile(video_path):
    os.remove(video_path)

while True:    
    if not os.path.isfile(video_path):
        time.sleep(0.1)
        continue

    logging.info("Processing openpose...")

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    horizontal = getVideoOrientation(video_path)
    cmd_hr = getResolutionCommand(resolutionPoseDetection, horizontal)

    try: 
        check_cuda_device()
        command = "/openpose/build/examples/openpose/openpose.bin\
            --video {video_path}\
            --display 0\
            --write_json {output_dir}\
            --render_pose 0{cmd_hr}".format(video_path=video_path, output_dir=output_dir, cmd_hr=cmd_hr)
        os.system(command)

        logging.info("openpose: Done. Cleaning up")
        os.remove(video_path)

    except:
        logging.info("openpose: Pose detection failed.")
        os.remove(video_path)
