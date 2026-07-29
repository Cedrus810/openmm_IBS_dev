# 已移除：`_refine_lambda_path_with_medium_probe`

移除日期：2026-07-27　　对应待办：**ATT-27**

## 为什么移除

这是 `enable_lambda_refine` 的实现半边，**从未被调用**（全仓只有定义处一处引用）。
唯一会走到它的入口在 `abfe_pipeline.py::run_full_pipeline` 里已经被硬性拒绝：

```python
if kwargs.get("enable_lambda_refine", False):
    raise RuntimeError(
        "enable_lambda_refine 的旧实现按 |Δf| 重排，会覆盖新的 "
        "beta^2 Var[dU/dlambda] 双物理子区间路径，并重新引入 "
        "refine_overlap=2；vanishing v12 明确禁止启用。"
    )
```

那个 `raise` **保留**（它是防止静默重新启用的守卫），但它守着的实现已经没有存在意义：
按 |Δf| 重排 λ 路径会覆盖 v21 的 Fisher-度规混合布点，并重新引入已被否决的
`refine_overlap=2`。

需要重新做 λ 路径精修时，应基于当前的 `abfe_preoptimizer`
（`blended_metric_vanishing_lambdas` / `_refine_pilot_grid_in_steep_segments`）
重写，而不是复活这段。相关评估项见 `docs/TODO.md` 的 **R-03**。

## 移除的内容（83 行）

非可执行归档，仅供查阅。

```python
    def _refine_lambda_path_with_medium_probe(
        self,
        stage_name: str,
        fixed_lam_coul: float,
        fixed_lam_vdw: float,
        lambdas_var: List[float],
        window_ranges: List[Tuple[int, int]],
        preopt_path: str,
        potential_type: str,
        dexp_params: Optional[Dict],
        boresch_params: Optional[Dict],
        refine_n_steps_per_window: int,
        refine_steps_per_update: int,
        max_window_span_kJ: float,
        overlap: int,
        resume: bool = False,
    ) -> Tuple[List[float], List[Tuple[int, int]]]:
        """
        用"中等步数"探针（比粗扫 optimize_stageN 贵、比正式生产便宜得多）在独立
        scratch 目录里把当前 λ 路径实采一遍，基于真实测得的 f(λ) 曲线精修 λ 分布
        与窗口边界，写回 preopt_path。

        scratch 目录必须与正式生产的 stage_output_dir 完全隔离：
        IBSWindowManagerDualLambda.run_all_windows 的 resume 断点续传只按"能量数组
        形状是否匹配当前窗口"判断是否跳过采样，不检查实际步数/样本量是否够——如果
        中等步数探针直接写进生产目录，后续生产阶段会误把这些样本量不足的数据当成
        "已采样完成"而跳过，真正的生产步数永远不会被执行。
        """
        n_states = len(lambdas_var)
        lambdas_fix = [fixed_lam_vdw if stage_name == "decharging" else fixed_lam_coul] * n_states
        stage_type = "coul" if stage_name == "decharging" else "vdw"

        scratch_dir = os.path.join(self.output_dir, f"{stage_name}_refine_probe")
        os.makedirs(scratch_dir, exist_ok=True)

        alchemical_params = _resolve_alchemical_params(
            potential_type, dexp_params, self.ligand_indices
        )
        manager = IBSWindowManagerDualLambda(
            system_template=self.system,
            topology=self.topology,
            perturbed_atom_indices=self.ligand_indices,
            lambdas_coul=lambdas_var if stage_name == "decharging" else lambdas_fix,
            lambdas_vdw=lambdas_fix if stage_name == "decharging" else lambdas_var,
            temperature=self.temperature,
            window_ranges=window_ranges,
            alchemical_params=alchemical_params,
            potential_type=potential_type,
            restraint_params=boresch_params,
            prefix="abfe_dual_refine_probe",
            platform_name=self.platform_name,
            output_dir=scratch_dir,
            checkpoint_dir=self.checkpoint_dir,
        )
        manager.output_dir = scratch_dir

        self._log(
            f"  🔬 [精修探针] {stage_name}: 中等步数采样 "
            f"({refine_n_steps_per_window} 步/窗口，独立 scratch 目录，不影响生产数据)..."
        )
        manager.run_all_windows(
            positions=self.positions,
            box_vectors=self.box_vectors,
            n_steps_per_window=refine_n_steps_per_window,
            steps_per_update=refine_steps_per_update,
            stage_type=stage_type,
            resume=resume,
        )

        result = refine_stage_lambda_path_from_data(
            stage_dir=scratch_dir,
            preopt_path=preopt_path,
            temperature_k=self.temperature.value_in_unit(unit.kelvin),
            n_states=n_states,
            max_window_span_kJ=max_window_span_kJ,
            overlap=overlap,
            stage_type=stage_type,
        )
        self._log(
            f"  ✅ [精修探针] {stage_name} λ 路径已按实测 |Δf| 精修："
            f"{result['n_states']} 个状态，{len(result['window_ranges'])} 个窗口"
        )
        return result["lambdas_var"], [tuple(r) for r in result["window_ranges"]]
```
