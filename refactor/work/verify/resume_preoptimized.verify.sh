#!/bin/bash
# 刀2 内部细分·刀P-B _resume_preoptimized 抽取。before=resume_preoptimized.before.py, after=当前文件
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_extract.py refactor/work/resume_preoptimized.before.py refactor/domains/pipeline_orchestration/abfe_pipeline_class.py --moved _resume_preoptimized --src _run_dual_lambda_scheme --block fn --slice 18:26
