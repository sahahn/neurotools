#!/bin/sh
#SBATCH --partition=bluemoon
#SBATCH --time=30:00:00
#SBATCH --job-name=runMM
#SBATCH --output=Job_Logs/%x_%j.out
#SBATCH --error=Job_Logs/%x_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1

export OMP_NUM_THREADS=1
source /users/s/a/sahahn/.bashrc

cd ${SLURM_SUBMIT_DIR}

python run_mm.py $1 0