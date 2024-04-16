#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --partition=gpu # Pod cluster's GPU queue
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=10U_tau_15mer
#SBATCH --mail-user=lobo@ucsb.edu # uncomment these two lines and include email if desired
#SBATCH --mail-type=END,FAIL    # Send email at begin and end of job

cd $SLURM_SUBMIT_DIR
module load cuda/11.2
export PATH="/sw/alphafold/localcolabfold/colabfold-conda/bin:$PATH"
srun --gres=gpu:1 colabfold_batch 10U.fasta 10U --save-recycles
