#!/bin/sh
#SBATCH --partition=short
#SBATCH --time=3:00:00
#SBATCH --mem=12G

#SBATCH --job-name=submit_permuted_v
#SBATCH --output=Job_Logs/%x_%j.out
#SBATCH --error=Job_Logs/%x_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1

export OMP_NUM_THREADS=1
source /users/s/a/sahahn/.bashrc

cd ${SLURM_SUBMIT_DIR}

python submit_permuted_v.py ${SLURM_MEM_PER_NODE}