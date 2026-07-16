# 当前仍未完成项（复核于 2026-07-16，P2 批量修复后更新）

本文件只保留当前源码里仍然存在、需要继续修改代码或采取工程行动的问题。代码已经
修复但尚未获得完整 CPU/GPU/依赖环境证据的项目统一迁入 `VALIDATION_MATRIX.md`；修复
历史和审计结论保留在 `AUDIT_STATUS.md`。

**当前结论：没有剩余的已确认默认生产路径 P0/P1，此前列出的 13 项 P2 已全部
修复；16:08 版本仍列出的主窗口 dynamics Context 续算缺口也已由
`MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION=1` 修复。** 当前只剩下列性能加固项；待运行
验证不在本文件重复维护，请查看 `VALIDATION_MATRIX.md`。

## 性能/恢复能力加固（不影响当前数值定义）

- [ ] **production ESS 自动修复的第二调用点仍使用旧 per-edge 探针。**
      GitHub：[`openmm_IBS_dev#1`](https://github.com/Cedrus810/openmm_IBS_dev/issues/1)，
      Project 状态 `Ready`，Priority `P2`。
      `abfe_pipeline.py::_probe_vdw_window_fixed_overlap` 仍逐边调用
      `probe_bidirectional_overlap` 和
      `probe_bidirectional_overlap_for_bias_calibration`，没有接入已实现的 per-state
      trajectory bank；中间态重复采样、校准重试重新 burn-in。若迁移，manifest 必须
      加调用点/起始构型与 box 哈希，避免和 `run_all_windows` 的活体起点缓存混用；
      现有 per-edge 结果缓存应保持"先查结果缓存，miss 后再补 bank 轨迹"的层次。

## 验证监控入口

当前待验证项目、验收条件、目标环境和证据要求见 `VALIDATION_MATRIX.md`。本文件不再
复制验证清单，以免同一项目在两个地方出现不同状态。
