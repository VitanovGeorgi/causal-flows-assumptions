#!/bin/bash

PARTITION=batch
NO_TASKS=1
NO_CPU=1
MEMORY=8GB
CONTAINER_MOUNTS=/netscratch:/netscratch,/ds:/ds,pwd:pwd
CONTAINER_IMAGE=/netscratch/gvitanov/envs/cnf_env.sqsh
# CONTAINER_SAVE=/netscratch/gvitanov/envs/trial.sqsh
MAIL_USER=gevi03@dfki.de
MAIL_TYPE=ALL # BEGIN, END
JOB_NAME=cnf
TIME=1:00:00
LOG_FILE=/netscratch/gvitanov/causal-flows-assumptions/output_general/trial.log
ERR_FILE=/netscratch/gvitanov/causal-flows-assumptions/output_general/trial.err
#pty is for interactive seesion, only used in srun

#SBATCH -p $PARTITION 
#SBATCH --job-name=$JOB_NAME
#SBATCH --ntasks=$NO_TASKS 
#SBATCH --cpus-per-task=$NO_CPU 
#SBATCH --mem=$MEMORY 
#SBATCH --container-mounts=/netscratch:/netscratch,/ds:/ds,pwd:pwd 
#SBATCH --container-image=$CONTAINER_IMAGE 
#SBATCH --container-workdir=pwd 
#SBATCH --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=$MAIL_USER 
#SBATCH --time=$TIME 
#SBATCH --array=1,2
#SBATCH --output=/netscratch/gvitanov/causal-flows-assumptions/output_general

source ~/.bashrc


# eval "$(conda shell.bash hook)"



# /opt/conda/bin:/opt/conda/condabin:/usr/local/nvm/versions/node/v15.0.1/bin:/opt/conda/bin:/opt/cmake-3.14.6-Linux-x86_64/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin:

# /opt/conda/envs/cnf/bin/python
# echo $PATH
srun bash ./grids/causal_nf/ablation_u_x/base/jobs_sh/jobs_${SLURM_ARRAY_TASK_ID}.sh

