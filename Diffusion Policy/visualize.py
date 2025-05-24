import h5py
import matplotlib.pyplot as plt
import numpy as np
from diffusion_policy.real_world.load_hdf5 import load_hdf5_replay_buffer

# Basic usage
buffer = load_hdf5_replay_buffer('/home/daiying/diffusion_policy_with_kovis/diffusion_policy/output.h5')

# Get all data for specific keys
data = buffer.get_all_data(['timestamp', 'camera_0'])

# Get data for a specific episode
episode_data = buffer.get_episode(0, ['camera_0'])

# Get image dimensions for a camera
height, width = buffer.get_image_size('camera_0')

# Get all camera keys
camera_keys = buffer.get_camera_keys()

# Get all non-camera keys
lowdim_keys = buffer.get_lowdim_keys()

# Use with context manager
with load_hdf5_replay_buffer('path/to/dataset.h5') as buffer:
    data = buffer.get_all_data()

# Direct access to specific keys
timestamps = buffer['timestamp']