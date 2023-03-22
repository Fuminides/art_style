#!/bin/bash
#SBATCH --partition=power_std
#SBATCH --account=acc_ure_power_std
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
# Activate the conda environment named "pytorch"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

srun python wikiart_pytorch.py --batch_size 128 --data_path ../wikiart/ --train_file style_train.csv --val_file style_val.csv --samples 0.2


