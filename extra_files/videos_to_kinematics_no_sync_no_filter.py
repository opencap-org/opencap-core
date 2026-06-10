"""
videos_to_kinematics_no_sync_no_filter.py

Full pipeline: pose detection → triangulation → augmentation → IK.
No synchronization algorithm, no median filter.

Edit the paths at the top and run:
    python videos_to_kinematics_no_sync_no_filter.py
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from main import main

# ---------------------------------------------------------------------------
# Edit these before each run.
# ---------------------------------------------------------------------------

# Root folder that contains your subject folders (e.g. subject5).
DATA_DIR = r"G:\Shared drives\Stanford Football\March_2"

# Subject folder name(s) to process.
SUBJECT_NAMES = ["subject4"]

# Camera names to use (must match folder names under Videos/).
CAMERAS_TO_USE = ["Cam1b", "Cam4b", "Cam7b"]

# Name of the intrinsics calibration folder inside each camera's directory.
INTRINSICS_FINAL_FOLDER = "Deployed_720_60fps"

# OpenPose resolution. "default" works for most cases.
RESOLUTION_POSE_DETECTION = '1x1008_4scales' #"default"

# Augmenter model version. "v0.3" is the latest; use "v0.2" to match older results.
AUGMENTER_MODEL = "v0.3"

# Set True to scale the OpenSim model from a static trial (only needed once per subject).
SCALE_MODEL = False

# ---------------------------------------------------------------------------


def process_trial(session_name, trial_name, cam2use,
                  intrinsics_folder, resolution, augmenter_model,
                  scale_model, data_dir, extrinsics_trial=False):
    main(
        session_name,
        trial_name,
        trial_name,          # trial_id same as name for local data
        cam2use,
        intrinsics_folder,
        isDocker=False,
        extrinsicsTrial=extrinsics_trial,
        markerDataFolderNameSuffix=f"{len(cam2use)}-cameras",
        poseDetector="OpenPose",
        resolutionPoseDetection=resolution,
        scaleModel=scale_model,
        augmenter_model=augmenter_model,
        dataDir=data_dir,
        overwriteCamerasToUse=True,
        # --- The two flags this script is built around ---
        runSynchronization=False,   # skip sync; treat videos as already aligned
        runMedianFilter=False,      # skip pre-augmentation median filter
    )


def main_runner():
    for subject_name in SUBJECT_NAMES:
        session_dir = os.path.join(DATA_DIR, subject_name)
        if not os.path.isdir(session_dir):
            print(f"[{subject_name}] Subject folder not found: {session_dir}")
            continue

        input_media_root = os.path.join(session_dir, "Videos", CAMERAS_TO_USE[0], "InputMedia")
        if not os.path.isdir(input_media_root):
            print(f"[{subject_name}] InputMedia not found: {input_media_root}")
            continue

        trials_all = [
            t for t in os.listdir(input_media_root)
            if os.path.isdir(os.path.join(input_media_root, t))
        ]

        # Process extrinsics trial first, then all others.
        extrinsics_trials = [t for t in trials_all if "extrinsics" in t.lower()]
        other_trials = [t for t in trials_all if "extrinsics" not in t.lower()]
        ordered_trials = extrinsics_trials + other_trials

        print(f"\n=== {subject_name} — {len(ordered_trials)} trial(s) ===")

        for trial in ordered_trials:
            is_extrinsics = "extrinsics" in trial.lower()
            cam2use = ["all_available"] if is_extrinsics else CAMERAS_TO_USE

            print(f"  Processing: {trial}  (extrinsics={is_extrinsics})")
            try:
                process_trial(
                    session_name=subject_name,
                    trial_name=trial,
                    cam2use=cam2use,
                    intrinsics_folder=INTRINSICS_FINAL_FOLDER,
                    resolution=RESOLUTION_POSE_DETECTION,
                    augmenter_model=AUGMENTER_MODEL,
                    scale_model=SCALE_MODEL,
                    data_dir=DATA_DIR,
                    extrinsics_trial=is_extrinsics,
                )
                print(f"  Done: {trial}")
            except Exception as e:
                print(f"  ERROR on {trial}: {e}")


if __name__ == "__main__":
    main_runner()
