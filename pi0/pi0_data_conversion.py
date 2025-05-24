#!/usr/bin/env python
"""
Conversion script for converting H5PY datasets to LeRobot format.

Usage:
uv run convert_h5_to_lerobot.py --data_dir /path/to/h5_dataset --output_repo your_username/dataset_name

If you want to push your dataset to the Hugging Face Hub, you can use:
uv run convert_h5_to_lerobot.py --data_dir /path/to/h5_dataset --output_repo your_username/dataset_name --push_to_hub
"""
import os
import glob
import h5py
import multiprocessing
import shutil
import tyro
from tqdm import tqdm
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main(
    data_dir: str,
    output_repo: str,
    *,
    robot_type: str = "UR5",
    fps: int = 10,
    push_to_hub: bool = False,
    image_size: tuple = (256, 256),
    resize_images: bool = True
):
    """
    Convert H5PY dataset to LeRobot format.
    
    Args:
        data_dir: Path to the directory containing H5PY files
        output_repo: Name of the output dataset on HuggingFace (e.g., "username/dataset_name")
        robot_type: Type of robot used in the dataset
        fps: Frames per second of the video data
        push_to_hub: Whether to push the dataset to HuggingFace Hub
        image_size: Size to resize images to (height, width) if resize_images is True
        resize_images: Whether to resize images to image_size
    """
    # Find all h5 files in the data directory
    h5_files = glob.glob(os.path.join(data_dir, "**", "*.hdf5"), recursive=True)
    
    if not h5_files:
        h5_files = glob.glob(os.path.join(data_dir, "**", "*.h5"), recursive=True)
    
    if not h5_files:
        raise ValueError(f"No H5PY files found in {data_dir}")
    
    print(f"Found {len(h5_files)} H5 files")
    
    # Sample the first file to determine structure
    with h5py.File(h5_files[0], 'r') as f:
        # Get action dimension
        action_dim = f['action'].shape[1]
        
        # Get state dimension from qpos
        state_dim = f['observations']['qpos'].shape[1]
        
        # Get available cameras
        camera_names = list(f['observations']['images'].keys())
        
        # Get image dimensions from the first camera
        first_camera = camera_names[0]
        orig_height, orig_width, channels = f['observations']['images'][first_camera].shape[1:]
    
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / output_repo
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create feature dictionary for LeRobot dataset
    features = {
        "image": {
            "dtype": "image",
            "shape": (*image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (*image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        # "velocity": {
        #     "dtype": "float32",
        #     "shape": (state_dim,),
        #     "names": ["velocity"],
        # },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
    }
    
    # Add camera features
    # for camera_name in camera_names:
    #     if resize_images:
    #         img_height, img_width = image_size
    #     else:
    #         img_height, img_width = orig_height, orig_width
            
    #     features[camera_name] = {
    #         "dtype": "image",
    #         "shape": (img_height, img_width, 3),
    #         "names": ["height", "width", "channel"],
    #     }
    
    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=output_repo,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=multiprocessing.cpu_count() // 2,
    )
    
    # Process each h5 file (each representing an episode)
    for episode_idx, h5_path in enumerate(tqdm(h5_files, desc="Processing episodes")):
        with h5py.File(h5_path, 'r') as f:
            # Get number of frames in this episode
            num_frames = f['action'].shape[0]
            
            # Process each frame
            for frame_idx in range(num_frames):
                # Prepare frame data
                frame_data = {
                    "state": f['observations']['qpos'][frame_idx],
                    #"velocity": f['observations']['qvel'][frame_idx],
                    "action": f['action'][frame_idx],
                }
                
                # Add camera frames
                for camera_name in camera_names:
                    # Get the raw image data
                    img = f['observations']['images'][camera_name][frame_idx]
                    
                    # Resize if needed
                    if resize_images:
                        # You would need a resize function here
                        # For simplicity, I'm using a placeholder that assumes you have OpenCV
                        import cv2
                        img = cv2.resize(img, (image_size[1], image_size[0]))
                        # Convert BGR to RGB if needed (depends on your dataset)
                        if img.shape[-1] == 3:  # Only if it's a color image
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    frame_data['image'] = img
                    frame_data['wrist_image'] = img
                
                # Add frame to dataset
                dataset.add_frame(frame_data)
            
            # Extract task name if available, otherwise use default
            task_name = f"task_{episode_idx}"
            # If the h5 file has a task attribute, use that instead
            if 'task' in f.attrs:
                task_name = f.attrs['task']
                
            # Save the episode
            dataset.save_episode(task=task_name)
    
    # Consolidate the dataset
    dataset.consolidate(run_compute_stats=True)
    
    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=[robot_type, "h5py_converted"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
    
    print(f"Successfully converted H5 dataset to LeRobot format at {output_path}")
    print(f"Total episodes: {len(h5_files)}")

if __name__ == "__main__":
    tyro.cli(main)