import pickle
import os

# Load the intrinsics pickle file for each camera and view them

pickle_dir = 'C:/Users/steudelkri/Documents/opencap-core/CameraIntrinsics'

cameras = ['SONYRX0-II-Cam1','SONYRX0-II-Cam2','SONYRX0-II-Cam3','SONYRX0-II-Cam4','SONYRX0-II-Cam5','SONYRX0-II-Cam6','SONYRX0-II-Cam7','SONYRX0-II-Cam8']

for camera in cameras:
    with open(os.path.join(pickle_dir, camera, 'Deployed_720_60fps/cameraIntrinsics.pickle'), 'rb') as f:
        intrinsics = pickle.load(f)
    print(f'Intrinsics for {camera}:')
    print(intrinsics)
    print('-----------------------------------')

extrinsics_pickle_path = 'G:\Shared drives\Stanford Football Prototyping\December_10\subject2\Videos\Cam4\cameraIntrinsicsExtrinsics.pickle'
extrinsics = pickle.load(open(extrinsics_pickle_path, 'rb'))
print(f'Extrinsics for Cam4:')
print(extrinsics)
print('-----------------------------------')

extrinsics_pickle_path = 'G:\Shared drives\Stanford Football Prototyping\December_10\subject2\Videos\Cam5\cameraIntrinsicsExtrinsics.pickle'
extrinsics = pickle.load(open(extrinsics_pickle_path, 'rb'))
print(f'Extrinsics for Cam5:')
print(extrinsics)