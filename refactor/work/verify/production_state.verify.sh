#!/bin/bash
# 刀1-7 B② ProductionState 收编。before=production_state.before.py, after=finish_production.before.py
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_rename.py refactor/work/production_state.before.py refactor/work/finish_production.before.py --fn _produce_window --bind prod --names-file refactor/work/production_state_names.txt --keep-file refactor/work/warmup_state_names.txt --dataclass ProductionState --in-place --construction 1
