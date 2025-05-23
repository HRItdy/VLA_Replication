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

2. 
