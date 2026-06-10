import os
import sys

# ---------------------------------------------------------------------------
# Edit these before each run.
# ---------------------------------------------------------------------------

# Input TRC file to filter.
INPUT_TRC = r"G:\Shared drives\Stanford Football\March_2\subject5\MarkerData\OpenPose_default\3-cameras\PreAugmentation\ID5_S7_sprintNoSync.trc"

# Output TRC file (filtered). Leave empty to auto-name as <stem>_medFilt.trc next to input.
OUTPUT_TRC = r""

# Median filter window (frames). 7 = 3 frames each side of center.
# Larger = smoother, but more lag. 7 is a good starting point.
WINDOW_FRAMES = 7

# ---------------------------------------------------------------------------

def main():
    inp = INPUT_TRC.strip()
    if not inp:
        print("Set INPUT_TRC at the top of median_filter_trc.py.")
        sys.exit(1)
    if not os.path.isfile(inp):
        print(f"Error: Input TRC not found: {inp}")
        sys.exit(1)

    out = OUTPUT_TRC.strip()
    if not out:
        stem, ext = os.path.splitext(inp)
        out = stem + "_medFilt" + ext

    print(f"Input:  {inp}")
    print(f"Output: {out}")
    print(f"Window: {WINDOW_FRAMES} frames")

    from utilsMedian import median_filter_trc_file
    median_filter_trc_file(inp, out, window=WINDOW_FRAMES)

    print(f"\nDone. Filtered TRC saved to: {out}")


if __name__ == "__main__":
    main()
