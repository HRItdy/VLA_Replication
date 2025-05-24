# import os
# import h5py
# from typing import Optional, Dict, Any

# def convert_replay_to_h5py(
#     replay_buffer: Any,
#     data_dir: str,
#     task_name: str,
#     camera_names: Optional[list] = None,
# ) -> None:
#     """
#     Convert replay buffer data to ACT's HDF5 dataset format.
    
#     Args:
#         replay_buffer: Source replay buffer object
#         output_dir: Base directory to save the dataset
#         task_name: Name of the task (will be used as subdirectory)
#         config: Configuration dictionary containing dataset parameters
#         camera_names: List of camera names to include (defaults to all cameras in replay buffer)
#     """
#     # Create output directory structure
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
    
#     # Count existing episodes
#     idx = len([name for name in os.listdir(data_dir) 
#                if os.path.isfile(os.path.join(data_dir, name))])
    
#     # Get episode information
#     episode_lengths = replay_buffer.episode_lengths[:]
#     episode_ends = replay_buffer.episode_ends[:]
#     episode_starts = episode_ends - episode_lengths
    
#     # Process each episode
#     for episode_idx, (start_idx, length) in enumerate(zip(episode_starts, episode_lengths)):
#         # Create output path for this episode
#         dataset_path = os.path.join(data_dir, f'episode_{idx + episode_idx}')
        
#         # Extract episode data
#         data_dict = {
#             'qpos': replay_buffer['robot_eef_pose'][start_idx:start_idx + length],
#             'qvel': replay_buffer['robot_eef_pose_vel'][start_idx:start_idx + length],
#             'action': replay_buffer['action'][start_idx:start_idx + length],
#         }
        
#         # Determine camera names if not provided
#         if camera_names is None:
#             camera_names = [key.split('/')[1] for key in replay_buffer.data.keys() if key.startswith("camera_")]

#         # Save the data
#         with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
#             root.attrs['sim'] = True  # Add simulation attribute
            
#             # Create observation group
#             obs = root.create_group('observations')
#             image = obs.create_group('images')
            
#             max_timesteps = length
#             for cam_name in camera_names:
#                 _ = image.create_dataset(
#                     cam_name,
#                     (max_timesteps, 240, 320, 3),
#                     dtype='uint8',
#                     chunks=(1, 240, 320, 3),
#                 )
            
#             # Create datasets for qpos, qvel, and action
#             qpos = obs.create_dataset('qpos', (max_timesteps, 6))
#             qvel = obs.create_dataset('qvel', (max_timesteps, 6))
#             action = root.create_dataset('action', (max_timesteps, 6))
            
#             # Populate datasets with data
#             for name, array in data_dict.items():
#                 if name in ['qpos', 'qvel']:
#                     # Access these through the observations group
#                     obs[name][...] = array
#                 else:
#                     # Action is directly under root
#                     root[name][...] = array
    
#     print(f"Successfully converted and saved episodes to {data_dir}.")


from typing import Optional, Dict, Any, Sequence, Union, Tuple
import os
import pathlib
import numpy as np
import av
import zarr
import h5py
import numcodecs
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.real_world.video_recorder import read_video
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k

#register_codecs()

def convert_replay_to_h5py(
        replay_buffer: ReplayBuffer,
        output_dir: str,
        task_name: str,
        dataset_path: str,
        camera_names: Optional[list] = None,
        n_decoding_threads: int = multiprocessing.cpu_count(),
        dt: float = 0.1
    ) -> None:
    """
    Convert replay buffer data to ACT's HDF5 dataset format including image data.
    
    Args:
        replay_buffer: Source replay buffer object
        output_dir: Base directory to save the dataset
        task_name: Name of the task (will be used as subdirectory)
        config: Configuration dictionary containing dataset parameters
        dataset_path: Path to the original dataset containing video files
        camera_names: List of camera names to include
        n_decoding_threads: Number of threads for video decoding
        dt: Time step between frames
    """
    # Create output directory structure
    data_dir = os.path.join(output_dir, task_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Count existing episodes
    idx = len([name for name in os.listdir(data_dir) 
               if os.path.isfile(os.path.join(data_dir, name))])

    # Get episode information
    episode_lengths = replay_buffer.episode_lengths[:]
    episode_ends = replay_buffer.episode_ends[:]
    episode_starts = episode_ends - episode_lengths

    # Setup video reading
    input_path = pathlib.Path(os.path.expanduser(dataset_path))
    in_video_dir = input_path.joinpath('videos')
    assert in_video_dir.is_dir()

    # Process each episode
    for episode_idx, (start_idx, length) in enumerate(zip(episode_starts, episode_lengths)):
        # Create output path for this episode
        h5_path = os.path.join(data_dir, f'episode_{idx + episode_idx}.hdf5')
        
        # Extract episode data
        data_dict = {
            'qpos': replay_buffer['robot_eef_pose'][start_idx:start_idx + length],
            'qvel': replay_buffer['robot_eef_pose_vel'][start_idx:start_idx + length],
            'action': replay_buffer['action'][start_idx:start_idx + length],
        }

        # Open HDF5 file for this episode
        with h5py.File(h5_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = False
            
            # Create observation group
            obs = root.create_group('observations')
            image = obs.create_group('images')
            
            # Create datasets for non-image data
            qpos = obs.create_dataset('qpos', 
                                    (length, 6),
                                    dtype='float32')
            qvel = obs.create_dataset('qvel', 
                                    (length, 6),
                                    dtype='float32')
            action = root.create_dataset('action', 
                                       (length, 6),
                                       dtype='float32')

            # Populate datasets with data
            for name, array in data_dict.items():
                if name in ['qpos', 'qvel']:
                    # Access these through the observations group
                    obs[name][...] = array
                else:
                    # Action is directly under root
                    root[name][...] = array

            # Process video data
            episode_video_dir = in_video_dir.joinpath(str(episode_idx))
            episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), 
                                      key=lambda x: int(x.stem))

            # Create image datasets for each camera
            for video_path in episode_video_paths:
                camera_idx = int(video_path.stem)
                camera_name = f'camera_{camera_idx}'
                
                if camera_names is not None and camera_name not in camera_names:
                    continue

                # Get video resolution
                with av.open(str(video_path.absolute())) as container:
                    video = container.streams.video[0]
                    vcc = video.codec_context
                    height, width = vcc.height, vcc.width

                # Create dataset for this camera
                img_dataset = image.create_dataset(
                    camera_name,
                    (length, height, width, 3),
                    dtype='uint8',
                    chunks=(1, height, width, 3)
                )

                # Read and store video frames
                image_tf = get_image_transform(
                    input_res=(width, height),
                    output_res=(width, height),
                    bgr_to_rgb=False
                )

                for step_idx, frame in enumerate(read_video(
                    video_path=str(video_path),
                    dt=dt,
                    img_transform=image_tf,
                    thread_type='FRAME',
                    thread_count=n_decoding_threads
                )):
                    if step_idx >= length:
                        break
                    img_dataset[step_idx] = frame

        print(f"Successfully saved episode {idx + episode_idx} to {h5_path}")

    print(f"Successfully converted and saved all episodes to {data_dir}")