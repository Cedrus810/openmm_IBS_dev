# EXP-012 完整 MM ledger 预检（2026-08-03）

## 实现

- `exp012_xed/mm_ledger.py` 重建 scratch IBS window，并逐帧记录：
  - base groups `{0,2,3,5}`；
  - 五态 raw softcore CV；
  - 五态解析 LRC 与完整 target energies；
  - Group 1 IBS bias、Group 4 WCA sampling bias；
  - target/sample reduced potentials、相邻态 gap 和未归一 importance log-weight。
- `scripts/run_exp012_mm_ledger.py` 是单 run 入口。
- `run_outer_lambda_exp012_mm_ledger.sh` 是可恢复的三 run 节点入口；已完成且数组哈希正确的
  run 会跳过，非完整目录会 fail closed。

## CPU 结果

- 纯数值/schema 测试：13 passed。
- run1 frame 0 smoke：
  - rebuilt System SHA 与 scratch SHA `be34fd38...8144` 完全一致；
  - force-group closure `4.66e-10 kJ/mol`；
  - IBS LSE closure `0 kJ/mol`。
- run1 全 500 帧 CPU reference 已完成：
  - arrays SHA-256 `5f5582473d3a12edd6be3592c776e2268d740192fc12675749b721135644a58d`；
  - 最大 force-group closure `9.77e-4 kJ/mol`；
  - 最大 IBS LSE closure `2.84e-14 kJ/mol`；
  - 302.83 s，`0.606 s/frame`；
  - `production_data_mutated=false`，`scientific_qualification=false`。

本地 run2 在用户选择转到 GPU 节点后被中止；中止时尚未发布数组，空输出目录已移除。

## GPU 节点入口

先运行独立单帧 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/ruigengji/mambaforge/envs/openmm_dev/bin/python \
  scripts/run_exp012_mm_ledger.py \
  --run-id hard_window0_run1 \
  --output-dir output/outer_lambda_exp012/mm_ledger_cuda_smoke_run1 \
  --platform CUDA --device-index 0 \
  --frame-start 0 --frame-stop 1
```

smoke 通过后，以单一 CUDA backend 运行全部三条轨迹：

```bash
CUDA_VISIBLE_DEVICES=0 \
EXP012_PLATFORM=CUDA \
EXP012_DEVICE_INDEX=0 \
EXP012_LEDGER_ROOT="$PWD/output/outer_lambda_exp012/mm_ledger_cuda" \
bash run_outer_lambda_exp012_mm_ledger.sh
```

不要把 CPU run1 和 CUDA run2/run3 混成正式三 run dataset；CPU run1 只作独立 reference，
正式三 run 使用同一 `mm_ledger_cuda` 根目录。CUDA 结果返回后还需执行 CPU/CUDA 公共帧
能量一致性检查，才能把三个 ledger SHA 写入 sealed preregistration。
