# 常见问题与排障

> 先确认运行环境、配置文件和输入文件确实属于同一体系。不要用另一个体系的
> checkpoint 来“修复”当前错误。

[返回项目首页](../README.md) · [安装与运行](GETTING_STARTED.md) ·
[输出与续跑](OUTPUTS_AND_RESUME.md)

## 常见问题

### `ModuleNotFoundError: No module named 'openmm'`

当前 Python 环境没有 OpenMM，或没有激活正确环境：

```bash
python -c "import openmm; print(openmm.__version__)"
```

### GROMACS include 文件找不到

检查 `--gmx-path` 是否指向包含 `.ff` 文件夹的目录，例如：

```text
/path/to/gromacs/share/gromacs/top
```

也可设置 `GMXDATA`，让代码自动尝试 `$GMXDATA/top`。

### 找不到配体残基

确认 `--ligand MOL` 与 `.gro/.top` 中的残基名一致。当前目录使用 `MOL`。

### 溶剂腿构建失败

常见原因是配体 XML/FFXML 不完整或 GROMACS 拓扑抽取失败。可尝试显式提供 `--ligand-xml`，或检查 `output/ligand_only.xml` 是否生成。

### Boresch 自动估计失败

当前推荐先用不依赖 ML 的：

```bash
--boresch --boresch-source simple
```

若锚点不稳定，检查 `output/boresch_simple.json`、`pre_equilibration.dcd` 和 Boresch harmonicity diagnostics。

### `--analyze-only` 缺少能量文件

至少需要保留阶段 checkpoint 或窗口级 `.npy` 能量文件。不要随意删除：

```text
output/checkpoints/stage1_decharging.json
output/checkpoints/stage2_vanishing.json
output/decharging/decharging_pme_u_kn.npy
output/vanishing/dual_window_*_energies.npy
```

### 结果里的 `thermodynamic_cycle` 和缺陷清单冲突

这是历史 provenance 文本缓存造成的已知问题。当前 README 和 `status/AUDIT_STATUS.md` 的结论优先（`PHYSICS_DEFECTS.md` 已被 `status/AUDIT_STATUS.md` 取代，文件已不存在）：APBS 不替代 LJ tail correction，手动 PME self `+C*lambda^2` 不作为生产修正项，`Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS`（不是 `Delta G_complex - Delta G_solvent`）。如果手头的 `output/final_binding_results.json`/`thermodynamic_cycle.md` 是在这几处文档修正之前生成的，其中的 `delta_G_bind_kJ_mol` 符号可能是反的，`thermodynamic_cycle` 字段文本也会是旧版本——重新跑一遍复合物腿+溶剂腿的最终汇总（不需要重新采样，只要 `complex_results`/`solv_results` 能从缓存加载）即可刷新成当前约定。
