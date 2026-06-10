import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load the intrinsics from YOUR pickle file ---
def load_intrinsics_from_pickle(filepath):
    """Loads camera intrinsics from a pickle file."""
    try:
        # Use 'latin1' encoding for pickles created with Python 2.
        # This seems to be working for you, so we'll keep it.
        with open(filepath, 'rb') as f:
            intrinsics = pickle.load(f, encoding='latin1')
            print("\nSuccessfully loaded intrinsics:")
            for key, value in intrinsics.items():
                print(f"  {key}:")
                # Use np.array2string for pretty printing numpy arrays
                print(f"    {np.array2string(value, prefix='    ')}\n")
            return intrinsics
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

# --- Step 2: CORRECTED Visualization function ---
def visualize_camera_intrinsics(intrinsics, scale=1.0):
    """Visualizes camera intrinsics by plotting the camera frustum in 3D."""
    if not intrinsics:
        print("Intrinsics data is not valid. Cannot visualize.")
        return

    print("--- Extracting parameters for visualization ---")
    
    try:
        K = intrinsics['intrinsicMat']
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        print(f"Found fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

        H = int(intrinsics['imageSize'][0, 0])
        W = int(intrinsics['imageSize'][1, 0])
        print(f"Found width={W}, height={H}")

    except KeyError as e:
        print(f"\n--- ERROR ---")
        print(f"Could not find the expected key {e} in your pickle data.")
        print("The structure might be different. Please check the printout above.")
        return

    image_corners_2d = np.array([[0, 0], [W, 0], [W, H], [0, H]])
    Z = scale
    x_coords = (image_corners_2d[:, 0] - cx) * Z / fx
    y_coords = (image_corners_2d[:, 1] - cy) * Z / fy
    image_corners_3d = np.vstack([x_coords, y_coords, np.full(4, Z)]).T

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- PLOTTING CODE RESTORED ---
    # Plot camera center
    ax.scatter(0, 0, 0, c='r', marker='o', s=50, label="Camera Center")

    # Plot the frustum base (image plane)
    ax.plot(image_corners_3d[[0, 1, 2, 3, 0], 0],
            image_corners_3d[[0, 1, 2, 3, 0], 1],
            image_corners_3d[[0, 1, 2, 3, 0], 2], 'b-')

    # Plot lines from camera center to frustum corners
    for i in range(4):
        ax.plot([0, image_corners_3d[i, 0]], [0, image_corners_3d[i, 1]], [0, image_corners_3d[i, 2]], 'g--')
    
    # Add text labels for corners
    ax.text(image_corners_3d[0,0], image_corners_3d[0,1], image_corners_3d[0,2], " (0,0)")
    ax.text(image_corners_3d[1,0], image_corners_3d[1,1], image_corners_3d[1,2], f" ({W},0)")
    ax.text(image_corners_3d[2,0], image_corners_3d[2,1], image_corners_3d[2,2], f" ({W},{H})")
    ax.text(image_corners_3d[3,0], image_corners_3d[3,1], image_corners_3d[3,2], f" (0,{H})")
    # --- END OF RESTORED PLOTTING CODE ---


    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis (Depth)')
    ax.set_title('Camera Intrinsics Visualization (Frustum)')
    
    # --- FIX FOR THE 'Singular Matrix' ERROR ---
    # Create a cubic plotting volume to ensure equal aspect ratio
    all_points = np.vstack([image_corners_3d, [0,0,0]])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    center = np.array([(x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2])
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    # --- END OF FIX ---

    ax.legend()
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    # Use a raw string (r'...') to prevent unicode errors on Windows
    YOUR_PICKLE_FILE = r'C:\Users\steudelkri\Documents\opencap-core\CameraIntrinsics\SONYRX0-II-Cam1\Deployed\cameraIntrinsics.pickle'

    # 1. Load the data from your specific file
    intrinsics_data = load_intrinsics_from_pickle(YOUR_PICKLE_FILE)

    # 2. Visualize the loaded data
    if intrinsics_data:
        # This scale is in meters. Since fx is large, a smaller scale looks better.
        visualize_camera_intrinsics(intrinsics_data, scale=0.5)