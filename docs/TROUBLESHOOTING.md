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

### `--openmm-cache-only` 下仍然警告「GROMACS 力场 include 目录找不到」

**已修（2026-09-02，`CACHE-01`）** —— `find_gmx_include_dir()` 现在只在**非** cache-only
分支里调用。逐条核实「cache-only 确实用不到 `include_dir`」的原始记录在
[archive/RUNTIME_ISSUES_2026-09-02.md](archive/RUNTIME_ISSUES_2026-09-02.md)。

⚠️ 另一个**独立**的坑：本仓 `abfe_config.json` 的 `gmx_path` 自 2026-09-07 起**留空**
（机器本地路径，写死谁的都是错的），留空时回退到 `GMXLIB`/`GMXDATA`/`PATH` 自动探测。
跑非 cache-only 就自己填本机 GROMACS 安装前缀（见下节）。

### GROMACS include 文件找不到

`--gmx-path`（或 `config.gmx_path`）**两种写法都接受**：

```text
/path/to/gromacs                      # 安装前缀 —— 会自动往下找 share/gromacs/top、top
/path/to/gromacs/share/gromacs/top    # 直接给力场目录
```

给前缀时的自动定位是 `runabfe.py:557` `find_gmx_include_dir` 第 1 步做的，
只在「前缀本身不是一个已填充的 include 目录」时才往下找，所以不会改变任何
原本就能解析成功的输入。

四条显式入口，先到先得：`--gmx-path` → `$GMXLIB`（值本身就是力场目录，不拼 `top`）
→ `$GMXDATA`（力场在其 `top/` 下）→ PATH 上的 `gmx` 反推 `share/gromacs/top`。

### `缺少 LocalManyBodyResidual 插件库: …/plugins/LocalManyBodyResidual/build/lib….so`

仓库带了一份预编译 `.so`，没拿到（或被 `.gitignore` 拦了）就自己编一份，秒级：

```bash
conda activate <装有 OpenMM 的环境>
bash plugins/LocalManyBodyResidual/g0_build.sh
```

带的那份 `.so` 是对 `openmm=8.5.2`（环境文件钉的版本）编的。
**没走环境文件、装了别的 OpenMM 版本**时也重编一次 —— 插件吃的是 OpenMM
私有平台头，跨版本不保证可用。
完整说明见 [GETTING_STARTED.md](GETTING_STARTED.md)《CUDA 插件》一节。

只在用 `--outer-lambda-local-residual-ibs` 时会撞上；默认生产路径不加载插件。

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

### 结果里的 `thermodynamic_cycle` 字段和当前口径冲突

这是历史 provenance 文本缓存造成的已知问题。**当前口径以本仓库的代码为准**
（下面三条都能在源码里查到，不依赖任何已迁出本分支的文档）：

- `Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS`
  ——**不是** `Delta G_complex - Delta G_solvent`；
- APBS 修正**不替代** LJ tail correction，两者各自独立记账；
- 手动 PME self 项 `+C*lambda^2` **不作为生产修正项**。

如果手头的 `output/final_binding_results.json` 是在这几处口径修正之前生成的，
其中 `delta_G_bind_kJ_mol` 的符号可能是反的、`thermodynamic_cycle` 字段文本也是
旧版本。**不需要重新采样**：只要 `complex_results`/`solv_results` 能从缓存加载，
重跑一次最终汇总即可刷新成当前约定。

> 早期版本的本节曾让读者去查 `status/AUDIT_STATUS.md`、`PHYSICS_DEFECTS.md`、
> `thermodynamic_cycle.md`。**这三份都不在本工程区分支**（原文在
> `Atenolol-rank11`，登记在 [HISTORY_LOG.md](HISTORY_LOG.md)），且其中
> `PHYSICS_DEFECTS.md` 当时就已被取代。上面三条口径已经把需要的结论直接写在这里。
