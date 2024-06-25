#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=4:00:00
#SBATCH --job-name=Kepler
#SBATCH --output=openmp_output_%j.txt
#SBATCH --mail-type=FAIL
#SBATCH --array=0-54
 
cd $SLURM_SUBMIT_DIR
 
module load python/3.11.5

source venv/bin/activate
 
python fixed.py $SLURM_ARRAY_TASK_ID
