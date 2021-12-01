#!/bin/sh
#SBATCH --job-name=run_permuted_v

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1

export OMP_NUM_THREADS=1
source /users/s/a/sahahn/.bashrc

cd ${SLURM_SUBMIT_DIR}

python run_permuted_v.py $1 $2 $3 ${SLURM_ARRAY_TASK_ID}