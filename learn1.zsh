#!/bin/zsh
#PBS -q default
#PBS -l select=1:ncpus=32:ngpus=1:mem=50gb:host=yayoi32
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -N training
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
# run the environment module
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH

export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev
cd /home/ruigengji/ABFE_IBS/Atenolol-rank11
bash ./scripts/exp030_run_node1_repeat1.sh

