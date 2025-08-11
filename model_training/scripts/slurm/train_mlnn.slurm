#!/bin/bash
#SBATCH --job-name=mlnn_neutrino
#SBATCH --account=isaac-utk0307
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=condo-slagergr
#SBATCH --qos=condo
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/nfs/home/jmcguig1/Rhea/model_training
#SBATCH -o /nfs/home/jmcguig1/Rhea/model_training/logs/train_mlnn_%j.out
#SBATCH -e /nfs/home/jmcguig1/Rhea/model_training/logs/train_mlnn_%j.err

module load gcc/10.2.0
module load openmpi/4.0.3-gcc
module load hdf5/1.10.7-gcc
module load Python/3.9.10-gcc
module load cuda/11.4.2-gcc
source /nfs/home/jmcguig1/venv_Emu/bin/activate

# Make the package importable (no need to pip install -e on the cluster)
export PYTHONPATH=$PWD/src:$PYTHONPATH

# If NPZ files live under model_training/train_data, make sure configs point there.
# Single run (one config):
# srun python -m rhea_train.train_binary --config configs/config.yaml \
#      --seed 43 --num_layers 6 --hidden_size 384 --dropout 0.01 \
#      --lr 5e-3 --wd 1e-4 --bs 32768 --random_multiplier 10

# Full grid search (replicates your full_nn.py loops; writes results9.npy)
srun python -m rhea_train.grid_search --config configs/config.yaml
