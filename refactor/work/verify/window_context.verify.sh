#!/bin/bash
# D线-2 WindowContext 归组(六道门)。before=window_context.before.py, after=当前文件
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_rename.py refactor/work/window_context.before.py refactor/domains/windows_sampling/window_manager_dual_lambda.py --fn run_all_windows --bind wc --names "lc_win,lv_win,sampler,sim,win_sys,win_sys_xml" --dataclass WindowContext --block for --in-place --construction 1 --expand-kwarg
python3 refactor/tools/verify_rename.py refactor/work/window_context.before.py refactor/domains/windows_sampling/window_manager_dual_lambda.py --fn _run_warmup_loop --bind wc --names "sampler,sim,lc_win,lv_win" --dataclass WindowContext --in-place
python3 refactor/tools/verify_rename.py refactor/work/window_context.before.py refactor/domains/windows_sampling/window_manager_dual_lambda.py --fn _summarize_warmup --bind wc --names "sampler,sim,lc_win,lv_win" --dataclass WindowContext --in-place
python3 refactor/tools/verify_rename.py refactor/work/window_context.before.py refactor/domains/windows_sampling/window_manager_dual_lambda.py --fn _commit_frozen_f_k --bind wc --names "sampler,sim,lc_win,lv_win" --dataclass WindowContext --in-place
python3 refactor/tools/verify_rename.py refactor/work/window_context.before.py refactor/domains/windows_sampling/window_manager_dual_lambda.py --fn _produce_window --bind wc --names "lc_win,lv_win,sampler,sim,win_sys,win_sys_xml" --dataclass WindowContext --in-place --expand-kwarg
python3 refactor/tools/verify_rename.py refactor/work/window_context.before.py refactor/domains/windows_sampling/window_manager_dual_lambda.py --fn _finish_production --bind wc --names "sampler,sim,win_sys,lc_win,lv_win" --dataclass WindowContext --in-place
