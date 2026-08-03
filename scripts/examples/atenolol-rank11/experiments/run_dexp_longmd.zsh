#!/bin/zsh
#PBS -q default
#PBS -l nodes=groupG:ppn=32:gpus=1
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -j oe
#PBS -N DEXP_longmd
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
# run the environment module
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH

source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
export PATH=$PATH:/home/ruigengji/mambaforge/bin
mamba activate openmm_dev
cd /home/ruigengji/ABFE_IBS/Atenolol-rank11

# 诊断目的：DEXP/MM 在 1ns 里 min-dist 都卡在一条 <0.05nm 的窄缝里，且落在 DEXP 拟合窗口
# [0.20,0.45]nm 的边缘/以下。这里先跑一版更长的*无偏* MD（不加任何采样偏置力），看这条窄缝
# 是否会随时间自然变宽。用独立的 --output-dir，不会碰到已经验证过的 output/dexp_experiment
# 这套 1ns 基线（switch 修复 + offset_c0/锚点估计量修复都在那套基线里，留作对照）。
#
# --reuse-fit-labels 指向新 output-dir，那里没有缓存，会重新对 500 帧做一次 MACE 标注再拟合
# （几分钟量级，一次性成本，和 MD 时长无关），拟合出的 DEXP 参数应与现有基线里的完全一致
# （同一条 pre_equilibration 轨迹、同一个 seed，确定性拟合）。
#
# 时长/磁盘量级参考：1ns×2 条腿（含 fit + 全套后处理）大约 45 分钟、traj.dcd 各 ~88MB。
# production MD 部分随 --sim-ns 大致线性增长，其余 fit/后处理开销基本固定。10ns×2 条腿
# 预计几小时量级、traj.dcd 各 ~0.9GB（两条共 ~1.8GB，磁盘剩余 1.7TB，无压力）。
# 如果 walltime 不够或想先看个更便宜的信号，把下面 --sim-ns 改成 5.0 甚至 3.0 也可以。
python dexp_experiment.py \
  --reuse-fit-labels \
  --device cuda \
  --platform CUDA \
  --output-dir output/dexp_experiment_10ns_diag \
  --sim-ns 10.0 \
  --traj-interval 5000 \
  --seed 20260526
