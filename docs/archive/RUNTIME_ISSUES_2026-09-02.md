> 🗄️ **已归档（2026-09-02）。原文一字未改。**
>
> 本文当时自述「只记录，不分析。分析等使用者发话」。分析已完成并搬到正式文档：
>
> | 本文条目 | 结论去了哪 |
> |---|---|
> | BUG-1（`--openmm-cache-only` 仍报 GROMACS include 找不到） | [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) 同名小节（完整因果链）+ [TODO.md](../TODO.md) `CACHE-01`（未做的一行修法） |
> | 附带发现：`abfe_config.json` 的 `gmx_path` 不存在 | [TODO.md](../TODO.md) `CFG-01`（与 BUG-1 是两件事） |
>
> **结论**：BUG-1 在审计通过的缓存上是**纯噪音**，不影响任何数值。
> 本文保留为原始运行记录，不是待办。

---

# 运行期问题记录 —— 2026-09-02 WCA 壳退役后的重跑

> 只记录，不分析。分析等使用者发话。

运行上下文：λ-WCA 防护壳退役（`ibs_engine.WCA_SHIELD_RETIRED = True`，
`WCA_ACCOUNTING_VERSION 2→3`）后，对 `4W53/output_v3_seed20260908` 两腿重采。

命令：

```
cd /home/ruigengji/ABFE_IBS/4W53
ABFE_RANDOM_SEED=20260908 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
<openmm_dev>/python <repo>/runabfe.py --output output_v3_seed20260908 \
  --config <repo>/abfe_config.json --resume --openmm-cache-only \
  --platform CUDA --allow-untrusted-stage-results
```

---

## BUG-1（2026-09-02 06:31:09）`--openmm-cache-only` 下仍报 GROMACS include 目录找不到

原文：

```
2026-09-02 06:31:09 | WARNING | find_gmx_include_dir: 找不到 GROMACS 力场 include 目录
（--gmx-path 未给 / 未命中，GMXLIB、GMXDATA 未设置，PATH 里也没有 gmx）。
如果拓扑里有需要它才能解析的 #include，请用 --gmx-path 显式指定，
或设置 GMXLIB/GMXDATA 环境变量。
```

已知事实（不作结论）：

- 本次带了 `--openmm-cache-only`，其 `--help` 自述职责是
  「验证并复用 output 中现有 OpenMM XML/CIF/index/box 缓存，
  **不解析已不可用的 GROMACS include 树**」。
- `abfe_config.json` 里 `gmx_path = /home/ruigengji/gmx26.0C`，该路径在本机**不存在**。
- 09-01 那次实跑的 provenance 记的是
  `gmx_path = /home/ruigengji/gmx26.3/share/gromacs/top`（与 config 里的值不同）。

状态：**未分析。** 待定的问题至少有两个 ——
(a) `--openmm-cache-only` 是否本就应该跳过 `find_gmx_include_dir`；
(b) 这条只是噪音警告，还是后续真有 `#include` 解析失败。
