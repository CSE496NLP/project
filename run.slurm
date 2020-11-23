#!/bin/sh
#SBATCH --time=2-12:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --constraint=gpu_32gb
#SBATCH --job-name=train-editnts-ppdb
#SBATCH --error=ppdb-tldr-train.%J.err
#SBATCH --output=ppdb-tldr-train.%J.out
#SBATCH --licenses=common

module load anaconda
conda activate editnts-ppdb

echo "sbatch run.slurm $@"

python -u $@
