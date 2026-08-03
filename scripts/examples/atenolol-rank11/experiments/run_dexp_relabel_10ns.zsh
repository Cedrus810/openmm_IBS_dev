#!/bin/zsh
#PBS -q default
#PBS -l nodes=groupG:ppn=32:gpus=1
#PBS -l walltime=6:00:00
#PBS -m abe
#PBS -j oe
#PBS -N DEXP_relabel10ns
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
# run the environment module
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH

source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
export PATH=$PATH:/home/ruigengji/mambaforge/bin
mamba activate openmm_dev
cd /home/ruigengji/ABFE_IBS/Atenolol-rank11

# 用修好的坐标/环境一致性代码，在 10ns 诊断轨迹（output/dexp_experiment_10ns_diag）上重跑 relabel。
# 两处关键修复（已在 dexp_experiment.py 里改好，这里不需要额外参数）：
#   1. 形状分箱坐标从"全原子(含H)最近距离"改成"限制在 [fit_r_min,fit_r_max]=[0.20,0.45]nm 内的最近距离"
#      ——之前一直在用 DEXP 公式根本看不见的坐标给它打分。
#   2. DEXP/MM 两条轨迹的 relabel 现在共用 fit 阶段的固定 255 个环境原子集合（从
#      fit_label_cache_meta.json 读取），不再各自按最后一帧重选（之前 DEXP=219/MM=242，
#      两边 MACE 局部能量分解建立在不同原子集合上）。
#
# 1ns 基线（output/dexp_experiment）用同样两处修复重跑后，结论从"未过地板"反转成
# "通过地板"（DEXP 形状 RMSE 5.88 vs MM 7.46），但只有 100 帧/2-3 个可信 bin，统计力度弱。
# 这次在 10ns（500 帧/条腿）上验证是否稳定成立。
#
# 先重新 fit-only（用改过的坐标定义重新做 PMF matching，用缓存的 MACE 标签，很快），
# 再 relabel（500+500 帧 MACE 单点，这才是耗时的部分，之前在共享节点上跑到一半被叫停）。
python dexp_experiment.py --fit-only --reuse-fit-labels --device cuda \
  --output-dir output/dexp_experiment_10ns_diag

python dexp_experiment.py \
  --output-dir output/dexp_experiment_10ns_diag \
  --relabel-traj output/dexp_experiment_10ns_diag/dexp_surrogate/traj.dcd \
  --relabel-baseline-traj output/dexp_experiment_10ns_diag/original_baseline/traj.dcd \
  --relabel-max-frames 500 --relabel-pmf-bins 24 --device cuda
