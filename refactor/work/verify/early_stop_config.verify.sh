#!/bin/bash
# D线-1 EarlyStopConfig 归组(三函数三道门)。before=early_stop_config.before.py, after=window_context.before.py(下一刀留底)
cd "$(dirname "$0")/../../.."
N="early_stop_check_interval_steps,early_stop_max_delta_g_drift_kJ_mol,early_stop_max_uncertainty_kJ_mol,early_stop_min_absolute_ess,early_stop_min_decorrelated_samples,early_stop_min_ess_ratio,early_stop_min_steps,early_stop_required_consecutive_passes,enable_early_stop"
python3 refactor/tools/verify_rename.py refactor/work/early_stop_config.before.py refactor/work/window_context.before.py --fn _finish_production --bind es --names "$N" --dataclass EarlyStopConfig --in-place
python3 refactor/tools/verify_rename.py refactor/work/early_stop_config.before.py refactor/work/window_context.before.py --fn _produce_window --bind es --names "$N" --dataclass EarlyStopConfig --in-place --expand-kwarg
python3 refactor/tools/verify_rename.py refactor/work/early_stop_config.before.py refactor/work/window_context.before.py --fn run_all_windows --bind es --names "$N" --dataclass EarlyStopConfig --in-place --construction 1 --expand-kwarg
