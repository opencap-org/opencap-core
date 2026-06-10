# visualizeOpenPoseMarkers.py
import numpy as np
import matplotlib.pyplot as plt
import utilsTRC
import utilsMedian
import utilsPlotting

# This script is used to visualize markers from a pre-augmentation trc file

# ============================================================================
# CONFIGURATION - ADJUST FOR EACH RUN

date = 'March_2'
subject_num = 5
session = 7
trial_type = 'sprint'
trc_file_name = f'ID{subject_num}_S{session}_{trial_type}NoSync.trc' # Try out the normal trc and the NoSync trc files
# ============================================================================
# END OF CONFIGURATION - ADJUST FOR EACH RUN


# Path to trc file
trc_file_path = fr"G:\Shared drives\Stanford Football\{date}\subject{subject_num}\MarkerData\OpenPose_default\3-cameras\PreAugmentation\{trc_file_name}"

# Read the trc file
data = utilsTRC.trc_2_dict(trc_file_path)
time = data['time']
markers = data['markers']
marker_list = list(markers.keys())

print("Time shape:", time.shape) # Ends up as (337,) in first example trial
print("Markers keys:", markers.keys()) 
print("Marker Example Shape:", markers[marker_list[0]].shape) # Ends up as (337, 3) in first example trial

filtered_all = utilsMedian.median_filter_all_markers(markers, window=7, verbose=True)

# Save filtered data to new TRC file
output_path = trc_file_path.replace('.trc', '_median_filtered.trc')
utilsTRC.dict_2_trc(trc_file_path, filtered_all, time, output_path)

# Plot single marker
utilsPlotting.plot_single_marker_xyz(
    'LBigToe', 
    markers['LBigToe'], 
    filtered_all['LBigToe'], 
    time,
    save_path='LBigToe_filtered.png'
)

# Plot all markers - X coordinate
utilsPlotting.plot_all_markers_by_coordinate(
    markers, filtered_all, time, marker_list,
    coord_idx=0,  # X
    save_path='all_markers_X.png'
)

# Plot all coordinates
utilsPlotting.plot_all_markers_all_coords(
    markers, filtered_all, time, marker_list,
    save_path=True
)

# # Plot all the coordinates for a single marker
# utilsPlotting.plot_all_markers_all_coords_one(
#     markers, 
#     filtered_all, 
#     time, 
#     marker_list,
#     save_path='All_markers_all_coords.png'
# )