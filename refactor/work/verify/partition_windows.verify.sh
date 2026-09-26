#!/bin/bash
# 刀2 内部细分·刀P-C _partition_windows 抽取。before=partition_windows.before.py, after=resume_preoptimized.before.py
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_extract.py refactor/work/partition_windows.before.py refactor/work/resume_preoptimized.before.py --moved _partition_windows --src _run_dual_lambda_scheme --block fn --slice 26:41
