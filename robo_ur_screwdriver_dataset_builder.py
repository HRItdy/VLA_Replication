from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py


class RoboUrScrewdriver(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for screwdriver dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(720, 1280, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'qpos': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='robot_eef_pose',
                        ),
                        'qvel': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='robot_eef_pose_vel',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='Robot action'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/home/daiying/diffusion_policy_with_kovis/diffusion_policy/data/hdf5/screwdriver/episode_*.hdf5'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            with h5py.File(episode_path, "r") as f:
                observations = f["observations"]
                actions = f["action"][:]
                num_steps = actions.shape[0]
                print(f"num_steps is {num_steps}...")

                for i in range(num_steps):
                        # compute Kona language embedding
                    language_embedding = self._embed(["pick up the screwdriver"])[0].numpy()

                    episode.append({
                        'observation': {
                            'image': observations["images"]["camera_0"][i],
                            'qpos': observations["qpos"][i],
                            'qvel': observations["qvel"][i],
                        },
                        'action': actions[i],
                        'discount': 1.0,
                        'reward': float(i == (num_steps - 1)),
                        'is_first': i == 0,
                        'is_last': i == (num_steps - 1),
                        'is_terminal': i == (num_steps - 1),
                        'language_instruction': "pick up the screwdriver",
                        'language_embedding': language_embedding,
                    })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)
        #print(f"Found files: {episode_paths}")  # Debug print
        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )


