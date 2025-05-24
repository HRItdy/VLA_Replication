import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class RobotDatasetPreprocessor:
    def __init__(self, dataset_dir, target_length=None, pad_mode='repeat', truncate_mode='end'):
        """
        Initialize the dataset preprocessor.
        
        Args:
            dataset_dir (str): Directory containing the dataset files
            target_length (int, optional): Target length for all episodes. If None, will use the max length.
            pad_mode (str): How to pad shorter sequences ('zero' or 'repeat')
            truncate_mode (str): How to truncate longer sequences ('start', 'end', or 'middle')
        """
        self.dataset_dir = dataset_dir
        self.pad_mode = pad_mode
        self.truncate_mode = truncate_mode
        self.target_length = target_length
        
    def analyze_dataset(self):
        """
        Analyze the dataset to find episode lengths and determine target length if not specified.
        
        Returns:
            dict: Statistics about the dataset
        """
        episode_lengths = []
        episode_files = []
        
        for file in os.listdir(self.dataset_dir):
            if file.endswith('.hdf5'):
                file_path = os.path.join(self.dataset_dir, file)
                with h5py.File(file_path, 'r') as f:
                    # Get the length of the episode
                    length = f['observations/qpos'].shape[0]
                    episode_lengths.append(length)
                    episode_files.append(file_path)
        
        stats = {
            'min_length': min(episode_lengths),
            'max_length': max(episode_lengths),
            'mean_length': sum(episode_lengths) / len(episode_lengths),
            'num_episodes': len(episode_lengths),
            'episode_lengths': episode_lengths,
            'episode_files': episode_files
        }
        
        # If target_length is not specified, use the max length
        if self.target_length is None:
            self.target_length = max(episode_lengths)
            print(f"Setting target length to {self.target_length}")
        
        return stats
    
    def preprocess_episode(self, episode_path, output_path=None):
        """
        Preprocess a single episode file.
        
        Args:
            episode_path (str): Path to the episode file
            output_path (str, optional): Path to save the preprocessed file. If None, will overwrite.
            
        Returns:
            str: Path to the preprocessed file
        """
        if output_path is None:
            output_path = episode_path.replace('.hdf5', '_preprocessed.hdf5')
        
        with h5py.File(episode_path, 'r') as src:
            curr_length = src['observations/qpos'].shape[0]
            
            # Determine indices for slicing
            if curr_length > self.target_length:
                if self.truncate_mode == 'start':
                    start_idx = 0
                    end_idx = self.target_length
                elif self.truncate_mode == 'end':
                    start_idx = curr_length - self.target_length
                    end_idx = curr_length
                elif self.truncate_mode == 'middle':
                    start_idx = (curr_length - self.target_length) // 2
                    end_idx = start_idx + self.target_length
                else:
                    raise ValueError(f"Invalid truncate mode: {self.truncate_mode}")
            else:
                start_idx = 0
                end_idx = curr_length
            
            with h5py.File(output_path, 'w') as dst:
                # Copy attributes
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                
                # Process groups and datasets
                self._process_group(src, dst, start_idx, end_idx)
        
        return output_path
    
    def _process_group(self, src_group, dst_group, start_idx, end_idx):
        """
        Process a group in the HDF5 file recursively.
        
        Args:
            src_group: Source group
            dst_group: Destination group
            start_idx: Start index for slicing
            end_idx: End index for slicing
        """
        for key, item in src_group.items():
            if isinstance(item, h5py.Group):
                # Create the group in the destination file
                dst_subgroup = dst_group.create_group(key)
                # Process the subgroup recursively
                self._process_group(item, dst_subgroup, start_idx, end_idx)
            elif isinstance(item, h5py.Dataset):
                # Process the dataset
                self._process_dataset(item, dst_group, key, start_idx, end_idx)
    
    def _process_dataset(self, src_dataset, dst_group, key, start_idx, end_idx):
        """
        Process a dataset in the HDF5 file.
        
        Args:
            src_dataset: Source dataset
            dst_group: Destination group
            key: Dataset key
            start_idx: Start index for slicing
            end_idx: End index for slicing
        """
        shape = list(src_dataset.shape)
        dtype = src_dataset.dtype
        
        if len(shape) > 0:  # Skip scalar datasets
            curr_length = shape[0]
            shape[0] = self.target_length
            
            # Create the dataset in the destination file
            dst_dataset = dst_group.create_dataset(key, shape=tuple(shape), dtype=dtype)
            
            # Copy attributes
            for attr_key, attr_value in src_dataset.attrs.items():
                dst_dataset.attrs[attr_key] = attr_value
            
            # Handle truncation
            if curr_length > self.target_length:
                dst_dataset[:] = src_dataset[start_idx:end_idx]
            
            # Handle padding
            elif curr_length < self.target_length:
                dst_dataset[:curr_length] = src_dataset[:]
                
                if self.pad_mode == 'zero':
                    # Zero padding for the rest - default behavior for newly created datasets
                    pass
                elif self.pad_mode == 'repeat':
                    # Repeat the last timestep for the rest
                    for i in range(curr_length, self.target_length):
                        dst_dataset[i] = src_dataset[curr_length - 1]
                else:
                    raise ValueError(f"Invalid pad mode: {self.pad_mode}")
            else:
                # Same length, just copy
                dst_dataset[:] = src_dataset[:]
        else:
            # For scalar datasets, just copy
            dst_dataset = dst_group.create_dataset(key, data=src_dataset[()])
    
    def preprocess_dataset(self, output_dir=None):
        """
        Preprocess the entire dataset.
        
        Args:
            output_dir (str, optional): Directory to save the preprocessed files. If None, will create a new directory.
            
        Returns:
            str: Path to the preprocessed dataset
        """
        if output_dir is None:
            output_dir = self.dataset_dir + '_preprocessed'
        
        os.makedirs(output_dir, exist_ok=True)
        
        stats = self.analyze_dataset()
        
        for i, file_path in enumerate(tqdm(stats['episode_files'], desc="Preprocessing episodes")):
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)
            self.preprocess_episode(file_path, output_path)
        
        return output_dir
