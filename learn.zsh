#!/bin/zsh
#PBS -q default
#PBS -l nodes=groupF:ppn=32:gpus=1
#PBS -l walltime=72:00:00
#PBS -m abe
#PBS -j oe
#PBS -N training
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
# run the environment module
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH

source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
export PATH=$PATH:/home/ruigengji/mambaforge/bin
cd /home/ruigengji/ABFE_IBS/Atenolol-rank11
mamba activate openmm_dev

ABFE_RANDOM_SEED=20260907 IBS_RANDOM_SEED=20260907 OPENMM_RANDOM_SEED=20260907 PYTHONHASHSEED=20260907 \
  python runabfe.py --config abfe_config.json --output ./output_lrc_fix_repeat03_seed20260907 --reset --platform CUDA
