import os
import sys
from utilsOpenSim import runIKTool

def main():
    # ========================================
    # CONFIGURE YOUR PATHS HERE
    # ========================================
    model_path = r"G:\Shared drives\Stanford Football\March_2\subject5\OpenSimData\OpenPose_default\3-cameras\Model\LaiUhlrich2022_scaled.osim"

    trc_path = r"G:\Shared drives\Stanford Football\March_2\subject5\MarkerData\OpenPose_default\3-cameras\PreAugmentation\ID5_S7_sprintNoSync.trc"
    
    output_dir = r"G:\Shared drives\Stanford Football\AnalysisCompare\LengthFilt"

    ik_setup_path = None #r"C:/path/to/your/Setup_IK.xml"  # Set to None to use default

    
    # ========================================
    # END CONFIGURATION
    # ========================================

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(trc_path):
        print(f"Error: TRC file not found: {trc_path}")
        sys.exit(1)

    # If no custom IK setup, use default
    if ik_setup_path is None:
        # Assume default IK setup in opensimPipeline
        base_dir = os.path.dirname(os.path.abspath(__file__))
        opensim_pipeline_dir = os.path.join(base_dir, 'opensimPipeline')
        ik_setup_path = os.path.join(opensim_pipeline_dir, 'IK', 'Setup_IK.xml')

        if not os.path.exists(ik_setup_path):
            print(f"Error: Default IK setup file not found: {ik_setup_path}")
            print("Please provide a custom IK setup file path in the script")
            sys.exit(1)

    print(f"Running IK with:")
    print(f"  Model: {model_path}")
    print(f"  TRC: {trc_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  IK setup: {ik_setup_path}")

    try:
        # Run IK
        path_output_mot, path_model_ik = runIKTool(
            ik_setup_path, model_path, trc_path, output_dir
        )
        print(f"Success! IK results saved to: {path_output_mot}")

    except Exception as e:
        print(f"Error running IK: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()