#!/bin/sh
#SBATCH --time=1-12:00:00
#SBATCH --mem=32G
#SBATCH --job-name=nonpre
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --constraint=gpu_32gb
#SBATCH --error=job.nonpreempt.%J.err
#SBATCH --output=job.nonpreempt.%J.out
#SBATCH --licenses=common
module load anaconda
conda activate editnts
python -u $@
