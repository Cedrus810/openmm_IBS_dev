#!/bin/zsh
#PBS -q default
#PBS -l nodes=groupG:ppn=32:gpus=1
#PBS -l walltime=72:00:00
#PBS -m abe
#PBS -j oe
#PBS -N TEST
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
# run the environment module
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH

source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
export PATH=$PATH:/home/ruigengji/mambaforge/bin
#mamba activate omm_torch_124
mamba activate openmm_dev
cd /home/ruigengji/ABFE_IBS/Atenolol-rank11
python runabfe.py --config abfe_config.json --ligand MOL --output ./output_lrc_fix --reset