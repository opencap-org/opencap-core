import os
import time
import logging
import shutil
import json
import torch

from utilsMMpose import detection_inference, pose_inference

logging.basicConfig(level=logging.INFO)

logging.info("Waiting for data...")

def checkCudaPyTorch():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logging.info(f"Found {num_gpus} GPU(s).")
    else:
        logging.info("No GPU detected. Exiting.")
        raise Exception("No GPU detected. Exiting.")

video_path = "/mmpose/data/video_mmpose.mov"
output_dir = "/mmpose/data/output_mmpose"

generateVideo=False

with open('/mmpose/defaultOpenCapSettings.json') as f:
    defaultOpenCapSettings = json.load(f)
bbox_thr = defaultOpenCapSettings['hrnet']
model_config_person='/mmpose/faster_rcnn_r50_fpn_coco.py'
model_ckpt_person='/mmpose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
model_config_pose='/mmpose/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
model_ckpt_pose='/mmpose/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
    
if os.path.isfile(video_path):
    os.remove(video_path)

checkCudaPyTorch()
while True:    
    if not os.path.isfile(video_path):
        time.sleep(0.1)
        continue

    logging.info("Processing mmpose...")

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    try:
        checkCudaPyTorch()
        # Run human detection.
        pathModelCkptPerson = model_ckpt_person
        bboxPath = os.path.join(output_dir, 'box.pkl')
        full_model_config_person = model_config_person
        detection_inference(full_model_config_person, pathModelCkptPerson,
                            video_path, bboxPath)        
        
        # Run pose detection.     
        pathModelCkptPose = model_ckpt_pose
        pklPath = os.path.join(output_dir, 'human.pkl')
        videoOutPath = ''
        full_model_config_pose = model_config_pose
        pose_inference(full_model_config_pose, pathModelCkptPose, 
                       video_path, bboxPath, pklPath, videoOutPath, 
                       bbox_thr=bbox_thr, visualize=generateVideo)
        if os.path.isfile(video_path):
            os.remove(video_path)
        if os.path.isfile(bboxPath):
            os.remove(bboxPath)
        
        logging.info("mmpose: Done. Cleaning up")
        
    except:
        logging.info("mmpose: Pose detection failed.")
        os.remove(video_path)
