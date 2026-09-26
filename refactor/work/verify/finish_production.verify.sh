#!/bin/bash
# 刀1-8 刀d _finish_production 联合抽取。before=finish_production.before.py, after=early_stop_config.before.py(下一刀留底)
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_extract.py refactor/work/finish_production.before.py refactor/work/early_stop_config.before.py --moved _finish_production --src _produce_window --block fn --slice 58:83
