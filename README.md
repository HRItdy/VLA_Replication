# VLA_Replication
The UR5 script for data collection and evaluation, with the record of common issues encountered


## Dataset conversion
We used 

## RDT-1b

## $\pi_0$

## Diffusion Policy

## Action Chunking Transformer
https://github.com/Shaka-Labs/ACT
1. First align the dimension of `state_dim` of vae and the training dataet.
   When installed ACT you should also installed `detr`, go to `detr->models->detr_vae.py`, change line 230 to the demension of your training dataset.

   Another way to open this is directing to `training->policy.py', go to definition of `build_ACT_model_and_optimizer->build_ACT_model->build_vae`.

2. Second need to padding all the episodes to the fixed length, which is defined in `ACT/config/config.py` as `episode_len` in `TASK_CONFIG`. Please run the code in `dataset_preproc.py` under ACT folder, which will pad all of your episodes in the dataset to the predefined `episode_len`.

3. Put the dataset with **hdf5** format under `data` folder
```
├── data
│   └── pick_screwdriver
│       ├── screwdriver
│       │   ├── episode_0.hdf5
│       │   ├── episode_1.hdf5
│       │   ├── episode_2.hdf5
│       │   ├── ...
│       └── text_embed
│           └── embed_0.pt
```
