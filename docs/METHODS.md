# 方法学与文献 / Methods and references

[项目入口](../README.md) · [文档导航](README.md) · [当前科学状态](STATUS.md)

本页把管线的每个环节对应到**它实现的是哪篇文献的方法**，以及**代码在哪**。
写论文的方法学一节、或者审稿人问"这一步依据是什么"时看这里。

DOI 逐条核对过（2026-09-05）。这里只列**本管线真正执行**的方法；计划中、
实验中和已退役的路线不在此列（见 [STATUS.md](STATUS.md) 的状态词）。

## 核心方法

| 环节 | 实现位置 | 文献 |
|---|---|---|
| **IBS 少态采样与重加权** | `ibs_engine.py`（`IBS_BIAS_PROTOCOL_VERSION`） | Lin, Xia, Zhang, Gao. *Integrated Boltzmann Sampling: A Few-State Approach for Efficient Multistate Free Energy Calculations.* **J. Chem. Theory Comput. 2026, 22, 818–830.** [doi:10.1021/acs.jctc.5c01240](https://doi.org/10.1021/acs.jctc.5c01240) |
| **MBAR 多态自由能估计** | `ibs_engine.py`（MBAR/TMBAR，经 PyMBAR） | Shirts, M. R.; Chodera, J. D. *Statistically optimal analysis of samples from multiple equilibrium states.* **J. Chem. Phys. 2008, 129, 124105.** [doi:10.1063/1.2978177](https://doi.org/10.1063/1.2978177) |
| **Boresch 位形约束与解析释放项** | `ibs_engine.py`（attachment / release 记账） | Boresch, S.; Tettinger, F.; Leitgeb, M.; Karplus, M. *Absolute Binding Free Energies: A Quantitative Approach for Their Calculation.* **J. Phys. Chem. B 2003, 107, 9535–9551.** [doi:10.1021/jp0217839](https://doi.org/10.1021/jp0217839) |
| **软核 vdW（避免 λ→0 端点奇点）** | `abfe_core.py` 的 `BeutlerSoftcoreBuilder`；`_create_softcore_force` | Beutler, T. C.; Mark, A. E.; van Schaik, R. C.; Gerber, P. R.; van Gunsteren, W. F. *Avoiding singularities and numerical instabilities in free energy calculations based on molecular simulations.* **Chem. Phys. Lett. 1994, 222, 529–539.** [doi:10.1016/0009-2614(94)00397-1](https://doi.org/10.1016/0009-2614(94)00397-1) |
| **带电体系有限尺寸静电修正** | `apbs_correction.py`（Rocklin NET/USV、RIP、EMP、DSC 各项） | Rocklin, G. J.; Mobley, D. L.; Dill, K. A.; Hünenberger, P. H. **J. Chem. Phys. 2013, 139, 184103.** [doi:10.1063/1.4826261](https://doi.org/10.1063/1.4826261) —— 实现按 Wu, Z.; Biggin, P. C. **J. Chem. Theory Comput. 2022**, [doi:10.1021/acs.jctc.1c01251](https://doi.org/10.1021/acs.jctc.1c01251) 的 APBS/RIP 口径 |

## 软件依赖

| 组件 | 用途 | 文献 |
|---|---|---|
| **OpenMM 8** | 全部 MD 与 alchemical Context | Eastman, P. *et al.* *OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials.* **J. Phys. Chem. B 2024, 128, 109–116.** [doi:10.1021/acs.jpcb.3c06662](https://doi.org/10.1021/acs.jpcb.3c06662) |
| **GROMACS** | 输入体系（`.gro`/`.top`）与力场 include 树 | Abraham, M. J. *et al.* **SoftwareX 2015, 1–2, 19–25.** [doi:10.1016/j.softx.2015.06.001](https://doi.org/10.1016/j.softx.2015.06.001) |
| **PyMBAR** | MBAR 数值求解与不确定度 | 见上 Shirts & Chodera 2008；实现 <https://github.com/choderalab/pymbar> |
| **MDTraj** | 轨迹读写与几何分析 | McGibbon, R. T. *et al.* **Biophys. J. 2015, 109, 1528–1532.** [doi:10.1016/j.bpj.2015.08.015](https://doi.org/10.1016/j.bpj.2015.08.015) |
| **APBS** | `apbs_correction.py` 的连续介质电势格点 | Jurrus, E. *et al.* **Protein Sci. 2018, 27, 112–128.** [doi:10.1002/pro.3280](https://doi.org/10.1002/pro.3280) |

## 本仓库的 IBS 变体与原文的差别

**必须在方法学里写清楚**，否则读者会按原文去理解本管线的行为：

1. **converge-then-freeze 单一冻结参考态**：原文的权重 `n_k` 随采样在线平衡到
   `⟨r_k⟩ → π_k`；本管线在预热阶段收敛偏置后**冻结** `f_k`，production 全程用固定
   偏置采样。MBAR 对任意**冻结**的 `f_k` 无偏，这是这样做的依据。
2. **可微偏置**：本管线的 `Ũ_IBS` 直接进 OpenMM 的力计算，因此 log-sum-exp 混合
   必须可微，而不是只在分析阶段做重加权。
3. **收敛门不是相邻窗口重叠度**：这条是本仓库早期的设计错误，已纠正——判据是
   局部滑动窗口 MBAR 的相邻 ΔF 与 ESS，不是相邻重叠。

原文的参考实现（SPONGE）在 <https://github.com/issacAzazel/Integrated-Boltzmann-Sampling>，
与本仓库的 OpenMM 实现是**两套独立代码**，不共享数值路径。

## 热力学循环与长程修正

符号约定与各项定义见 [OUTPUTS_AND_RESUME.md](OUTPUTS_AND_RESUME.md)：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

LJ 长程（尾）修正在 `ibs_engine.py` 里按 **pair-specific `sigma_ij` 分组积分**
（`TRADITIONAL_LJ_LRC_PROTOCOL_VERSION = 3`），与软核分母
`D_ij(r) = alpha_lj * sigma_ij^6 * (1-λ)^m + r^6` 的口径严格一致——这一点必须在
方法学里说明，因为把尾修正按 pair 无关的标量处理（本仓库 v2 的做法）与采样
哈密顿量不自洽。

## 参照真值

生产结果只与**独立于本管线**算出的靶子比对，不拿生产自己的数互相比。
现有参照真值及其 provenance 见 [reference_data/README.md](reference_data/README.md)。
