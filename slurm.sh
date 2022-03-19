#!/bin/bash

#SBATCH --account=gpu-s5-data_augmentation-0
#SBATCH --partition=gpu-core-0
#SBATCH --gres=gpu:1
#SBATCH --time=320:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

#starting from 91 epoch
pip install tensorflow matplotlib
python3 crfcnn_combine.py -md crf
