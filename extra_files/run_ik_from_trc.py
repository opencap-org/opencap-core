import os
import sys
import glob
import faulthandler
from utilsOpenSim import runIKTool
import utilsDataman

# Dump Python tracebacks on hard crashes (best-effort).
faulthandler.enable()

# ---------------------------------------------------------------------------
# Edit these before each run.
# ---------------------------------------------------------------------------
# Generic InverseKinematicsTool setup XML (marker tasks + settings).
# It can contain Unassigned paths; runIKTool overrides model/TRC/output/results in code.
IK_SETUP_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "opensimPipeline",
    "IK",
    "Setup_IK.xml",
)

# Change the output directory and the IKFileName to match which method I am testing out.

# Scaled OpenSim model (.osim). runIKTool builds/uses *_no_patella.osim next to it for IK.
MODEL_PATH = r"G:\Shared drives\Stanford Football\March_2\subject5\OpenSimData\OpenPose_default\3-cameras\Model\LaiUhlrich2022_scaled.osim"

# Marker trial (.trc) — should match <marker_file> in the setup unless you rely on runIKTool to override it.
TRC_PATH = r"G:\Shared drives\Stanford Football\AnalysisCompare\PostaugmentationMarkerFiles\ID5_S7_sprintNoSync_medFilt_LSTM_postaug_filt15Hz.trc"
# Where to write the .mot and the saved Setup_IK_<trial>.xml (should match <results_directory> in your setup).
# The .mot *filename* comes from <output_motion_file> / tool name in the setup XML when present (not only the TRC stem).
OUTPUT_DIR = r"G:\Shared drives\Stanford Football\AnalysisCompare\MedFiltPostAugFiltLengthFilt"

# If True, pass an explicit finite time range from the TRC.
# This avoids relying on -Inf/Inf time_range in XML (which has caused native crashes in some OpenSim builds).
USE_TRC_TIME_RANGE = True

IKFileName = 'ID5_S7_sprint_LSTM_MedFiltPostAugFiltLengthFilt'


def main():
    ik_setup = IK_SETUP_PATH.strip()
    model_path = MODEL_PATH.strip()
    trc_path = TRC_PATH.strip()
    output_dir = os.path.abspath(os.path.normpath(OUTPUT_DIR.strip()))

    if not ik_setup or not model_path or not trc_path:
        print("Set IK_SETUP_PATH, MODEL_PATH, and TRC_PATH at the top of run_ik_from_trc.py.")
        sys.exit(1)

    if not os.path.isfile(ik_setup):
        print(f"Error: IK setup file not found: {ik_setup}")
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    if not os.path.isfile(trc_path):
        print(f"Error: TRC file not found: {trc_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("Running IK:", flush=True)
    print(f"  Setup: {ik_setup}", flush=True)
    print(f"  Model: {model_path}", flush=True)
    print(f"  TRC:   {trc_path}", flush=True)
    print(f"  Out:   {output_dir}", flush=True)
    print(flush=True)

    try:
        print("Calling runIKTool now...", flush=True)
        time_range = []
        if USE_TRC_TIME_RANGE:
            trc = utilsDataman.TRCFile(trc_path)
            if trc.time is None or len(trc.time) < 2:
                raise RuntimeError(f"TRC has no usable time vector: {trc_path}")
            time_range = [float(trc.time[0]), float(trc.time[-1])]
            print(f"  TRC time range: {time_range[0]} to {time_range[1]}", flush=True)

        path_output_mot, _ = runIKTool(
            ik_setup, model_path, trc_path, output_dir, timeRange=time_range, IKFileName=IKFileName
        )
        print(f"runIKTool returned: {path_output_mot}", flush=True)
        print(f"Returned path exists: {os.path.isfile(path_output_mot)}", flush=True)
        if os.path.isfile(path_output_mot):
            print(f"Returned path size (bytes): {os.path.getsize(path_output_mot)}", flush=True)

        mot_files = sorted(glob.glob(os.path.join(output_dir, "*.mot")))
        print(f".mot files in OUTPUT_DIR ({output_dir}):", flush=True)
        if mot_files:
            for p in mot_files:
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    sz = None
                print(f"  - {p} (bytes={sz})", flush=True)
        else:
            print("  (none found)", flush=True)

        if not os.path.isfile(path_output_mot):
            print(
                "No .mot found at the returned path. This points to an IK setup mismatch\n"
                "(output_motion_file/name) or OpenSim writing somewhere unexpected.",
                flush=True,
            )
    except Exception as e:
        print(f"Error running IK: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
