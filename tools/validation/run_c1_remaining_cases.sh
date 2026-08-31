#!/usr/bin/env bash
# 跑完 C1 剩下的 5 个 case（Na_small 已经跑过）+ 两次必须的 compare-box（Na/Cl）
# + 一次额外的 compare-box（Ca，不在原稿硬验收里，但既然做了就一起记录）。
#
# 用法：
#   cd /home/ruigengji/ABFE_IBS/Atenolol-rank11
#   source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
#   mamba activate openmm_dev
#   bash tools/validation/run_c1_remaining_cases.sh
#
# 任一 case 的 dynamics/ukn 失败会让脚本停在那一步（set -e），不会带着坏结果
# 继续往下跑。跑完把 validation/c1_waterbox/*/report.json 和最后两条 compare-box
# 的输出发回来。

set -euo pipefail
cd "$(dirname "$0")/../.."
SCRIPT=tools/validation/validate_charge_transfer_waterbox.py
BASE=validation/c1_waterbox

run_case () {
  local ion=$1 size=$2
  local out="${BASE}/${ion}_${size}"
  echo "########## ${ion}_${size} ##########"
  python "$SCRIPT" build --ion "$ion" --box-size "$size" --output-dir "$out"
  python "$SCRIPT" static-check --output-dir "$out"
  python "$SCRIPT" dynamics --output-dir "$out"
  python "$SCRIPT" ukn --output-dir "$out"
  python "$SCRIPT" report --output-dir "$out"
}

run_case Cl small
run_case Ca small
run_case Na large
run_case Cl large
run_case Ca large

echo "########## compare-box：§13.4 盒长敏感性 ##########"
python "$SCRIPT" compare-box \
  --small-report "${BASE}/Na_small/report.json" \
  --large-report "${BASE}/Na_large/report.json" \
  --output "${BASE}/compare_box_Na.json"

python "$SCRIPT" compare-box \
  --small-report "${BASE}/Cl_small/report.json" \
  --large-report "${BASE}/Cl_large/report.json" \
  --output "${BASE}/compare_box_Cl.json"

python "$SCRIPT" compare-box \
  --small-report "${BASE}/Ca_small/report.json" \
  --large-report "${BASE}/Ca_large/report.json" \
  --output "${BASE}/compare_box_Ca.json"

echo "✅ 全部跑完。"
