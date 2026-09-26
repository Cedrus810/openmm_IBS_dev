#!/bin/bash
# 刀1-4 B① 3标量收编。before=absorb3.before.py, after=run_warmup_loop.before.py
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_rename.py refactor/work/absorb3.before.py refactor/work/run_warmup_loop.before.py --fn run_all_windows --bind warm --names frozen_f_k_snapshot,mode,ema_mean_p_values --keep-file refactor/work/warmup_state_names.txt --dataclass WarmupState
