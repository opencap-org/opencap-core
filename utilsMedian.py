# utilsMedian.py
"""
Utility functions for median filtering of OpenPose marker trajectories.
"""

import numpy as np


 ############################################################
# Use this function below if you want to use the MAD threshold to identify outliers
############################################################

# def median_window_filter(x, window, mad_threshold=1):
#     """
#     Moving median filter that can take np.nan as entries.
#     Outliers are identified using the median absolute deviation.
    
#     Inputs:
#         x             ->  signal to filter,
#         window        ->  number of samples to use for median estimation (must be odd),
#         mad_threshold  ->  threshold for identifying outliers using MAD (default is 3)

#     Output:
#         y             <-  median filtered signal
#     """
    
#     # Ensure the window size is odd
#     if window % 2 == 0:
#         window += 1
#     win2 = window // 2  # Now win2 is always an integer
#     n = len(x)
#     y = np.array(x, dtype=float)
    
#     for ii in range(win2, n - win2):
#         idx = np.arange(ii - win2, ii + win2 + 1, dtype=int)
#         valid_data = x[idx][~np.isnan(x[idx])]  # Exclude NaN values
        
#         if len(valid_data) > 0:
#             median = np.nanmedian(valid_data)
#             mad = np.nanmedian(np.abs(valid_data - median))  # Median Absolute Deviation
            
#             # Determine which points are considered outliers
#             is_outlier = np.abs(valid_data - median) > mad_threshold * mad
            
#             # Replace outlier points with NaN in valid_data
#             filtered_data = valid_data[~is_outlier]
            
#             if len(filtered_data) > 0:
#                 y[ii] = np.nanmedian(filtered_data)
#             else:
#                 y[ii] = median  # If all points are outliers, use the raw median

#     return y

##################################################################################################
# Use this function below to avoid the use of MAD

def median_window_filter(x, window):
    """
    Moving median filter that can take np.nan as entries.
    Note that the filter is non-causal, output of sample ii is the median
    of samples of the corresponding window centered around ii.
    
    Inputs:
        x       ->  signal to filtered,
        window  ->  number of samples to use for median estimation.

    Output:
        y       <-  median filtered signal
    """

    # To take care of an even window size if you accidentally put one in. But choosing an odd window size is better.
    if window % 2:
        window = window - 1 # Making the window even for equal points on left and right 
    win2 = int(window / 2) # Now win2 is always an integer
    n = len(x)
    y = np.array(x, dtype=float)
    
    for ii in range(win2, n - win2):
        idx = np.arange(ii - win2, ii + win2, dtype=int)
        y[ii] = np.nanmedian(x[idx])
    
    return y
##################################################################################################

def median_filter_trajectory(marker_data, window=7, return_outliers=False):
    """
    Apply median filter to 3D marker trajectory.
    
    Parameters:
    -----------
    marker_data : numpy.ndarray
        Array of shape (n_frames, 3) containing XYZ coordinates
    window : int
        Window size for median filter (will be made even if odd)
    return_outliers : bool
        If True, returns original data as well (for comparison)
    
    Returns:
    --------
    filtered_data : numpy.ndarray
        Median filtered trajectory
    """
    marker_data = np.asarray(marker_data, dtype=float)

    print(marker_data.ndim, marker_data.shape)

    if marker_data.ndim != 2 or marker_data.shape[1] != 3:
        raise ValueError(f"Expected shape (n_frames, 3), got {marker_data.shape}")
    
    filtered_data = np.zeros_like(marker_data)
    
    # Apply median filter to each coordinate
    filtered_data[:, 0] = median_window_filter(marker_data[:, 0], window)  # X
    filtered_data[:, 1] = median_window_filter(marker_data[:, 1], window)  # Y
    filtered_data[:, 2] = median_window_filter(marker_data[:, 2], window)  # Z
    
    if return_outliers:
        # Return both for visualization purposes
        return filtered_data, marker_data
    
    return filtered_data


def median_filter_all_markers(markers_dict, window=7, verbose=False):
    """
    Apply median filter to all markers in a dictionary.
    
    Parameters:
    -----------
    markers_dict : dict
        Dictionary with marker names as keys and (n_frames, 3) arrays as values
    window : int
        Window size for median filter
    verbose : bool
        If True, print progress
    
    Returns:
    --------
    filtered_markers : dict
        Dictionary with same structure, containing filtered trajectories
    """
    filtered_markers = {}
    
    for marker_name, trajectory in markers_dict.items():
        filtered_markers[marker_name] = median_filter_trajectory(trajectory, window)
        
        if verbose:
            print(f"Filtered {marker_name}")
    
    return filtered_markers


import utilsTRC

def median_filter_trc_file(input_trc_path, output_trc_path, window=7,
                           show_plot=False):
    """
    Apply median filter to all markers in a TRC file and save to a new file.
    
    Parameters:
    -----------
    input_trc_path : str
        Path to input TRC file
    output_trc_path : str
        Path to output filtered TRC file
    window : int
        Window size for median filter
    show_plot : bool
        If True, display the diagnostic figure (blocks until closed).
        If False, save the PNG and close the figure automatically.
    """
    # Load TRC data
    data = utilsTRC.trc_2_dict(input_trc_path)
    time = data['time']
    markers = data['markers']
    
    # Apply median filter to all markers
    filtered_markers = median_filter_all_markers(markers, window=window, verbose=False)
    
    # Save filtered data to new TRC file
    utilsTRC.dict_2_trc(input_trc_path, filtered_markers, time, output_trc_path)
    import matplotlib.pyplot as plt

    # Create a plot to visualize the effect of the median filter on a sample marker
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    sample_marker = list(markers.keys())[12]  # Just take the first marker for visualization
    original_traj = markers[sample_marker]
    filtered_traj = filtered_markers[sample_marker]
    for i, coord in enumerate(['X', 'Y', 'Z']):
        axes[i].plot(original_traj[:, i], 'o-', alpha=0.4, markersize=3, 
                    label='Original', color='lightblue')
        axes[i].plot(filtered_traj[:, i], '-', linewidth=2, 
                    label='Filtered', color='darkblue')
        axes[i].set_ylabel(f'{coord}')
        axes[i].set_xlabel('Frame')
        axes[i].legend(loc='upper left')
        axes[i].grid(True, alpha=0.3)
        axes[0].set_title(f'Median Filter (Window={window}) on Marker: {sample_marker}')
    plt.tight_layout()
    output_figure_path = output_trc_path.replace('.trc', '_median_filter_effect.png')
    fig.savefig(output_figure_path, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # print("Testing median_window_filter...")
    
    # # Test 1: Simple signal with spikes
    # signal = np.array([1, 2, 3, 100, 5, 6, 7, 8, -50, 10], dtype=float)
    # filtered = median_window_filter(signal, window=5)
    
    # print(f"Original: {signal}")
    # print(f"Filtered: {filtered}")
    
    # # Test 2: 3D trajectory with spikes
    # print("\nTesting median_filter_trajectory...")
    # np.random.seed(42)
    # trajectory = np.random.randn(100, 3) * 0.1
    # trajectory += np.linspace(0, 1, 100)[:, np.newaxis]
    
    # # Add spikes
    # trajectory[25, :] = [10, 10, 10]
    # trajectory[50, 0] = -5
    # trajectory[75, :] = [-8, -8, -8]
    
    # filtered_trajectory = median_filter_trajectory(trajectory, window=7)
    
    # # Visualize
    # fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    # for i, coord in enumerate(['X', 'Y', 'Z']):
    #     axes[i].plot(trajectory[:, i], 'o-', alpha=0.4, markersize=3, 
    #                 label='Original', color='lightblue')
    #     axes[i].plot(filtered_trajectory[:, i], '-', linewidth=2, 
    #                 label='Filtered', color='darkblue')
    #     axes[i].set_ylabel(f'{coord}')
    #     axes[i].legend(loc='upper left')
    #     axes[i].grid(True, alpha=0.3)
    
    # axes[0].set_title('Median Filter (Window=7) on 3D Trajectory')
    # axes[2].set_xlabel('Frame')
    # plt.tight_layout()
    # plt.show()