#!/bin/bash
# From https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
# Submit this job with the command: sbatch job.sh

#SBATCH --job-name=slurm-jax-test
#SBATCH --time=3:00

# Request 4 exclusive nodes with 2 gpus each.
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive
NUM_NODES=4

### get the first node name as master address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR="$master_addr

# Load conda environment.
# See https://github.com/conda/conda/issues/7980#issuecomment-441358406
# for details on why we do it this way.
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2021b/etc/profile.d/conda.sh
conda activate slurm-jax
echo "Conda environment loaded! id="$SLURM_PROCID

### the command to run
srun python run.py \
    --server_addr="$master_addr:1456" \
    --num_hosts=$NUM_NODES
