# 已移除：`_retired_overlapping_vdw_schedule_design`

移除日期：2026-07-27　　对应待办：**E-03** / **ATT-27**

## 为什么移除

这个函数的 docstring 第一句就是它自己的判决书：

> Retired v9 scratch designer; retained only for failure-record archaeology.

也就是说它留在源文件里的唯一理由是「留个失败记录」。**归档成本文件完全满足这个
目的，不需要留成可执行代码。** 全仓引用 1 次（仅定义处），从未被调用。

更重要的是它记录的那套设计正是被 `non_mutating_v1` 否决的路线：

> K>=3 failures split the ensemble using only existing nodes. A K=2 failure is
> irreducible, so and only so may one thermodynamic-length midpoint be inserted;
> its two replacement IBS ensembles must be tested again.

这与同日移除的 `_run_stage_with_overlap_autorepair` 里那 872 行变异循环
（`docs/archive/removed_overlap_autorepair_mutation_loop.md`）是同一族逻辑——
把相邻态 fixed-H overlap 当成 IBS 收敛仲裁标准。那条路线曾**烧掉约一周 GPU
而没产出任何 ΔG**。留着可被误激活的实现，风险不对称：复活的代价远大于保留的收益。

## 与另两个"写完未接"子系统的区别

同批扫描还发现 `OnlineConvergenceMonitor`（230 行）与 `ChunkedMBARAnalyzer`
（98 行）也完全无人调用，但**它们没有被移除**，因为性质不同：那两个是"写完但没接
上流水线"，接错了只是不工作，不会产出错误的自由能；而本函数与 overlap 变异循环
属于"已被证伪的错误路线"，复活会造成实际损害。ATT-27 要清理的是后者。

## 移除的内容（188 行，`abfe_pipeline.py` 原 2318-2505 行）

非可执行归档，仅供查阅。不要复制回生产代码。

```python
    def _retired_overlapping_vdw_schedule_design(
        self,
        lambdas_var: List[float],
        window_ranges: List[Tuple[int, int]],
        path_diagnostics: Dict,
        potential_type: str,
        dexp_params: Optional[Dict],
        boresch_params: Optional[Dict],
        probe_max_bias_updates: int = 15,
        probe_max_warmup_steps: int = 150000,
        probe_required_consecutive: int = 2,
        lse_log_residual_tolerance: float = 0.5,
        max_window_splits: int = 32,
        max_lambda_insertions: int = 4,
        max_insertions_per_initial_edge: int = 2,
    ) -> Tuple[List[float], List[Tuple[int, int]], Dict]:
        """Retired v9 scratch designer; retained only for failure-record archaeology.

        Lambda placement and IBS ensemble width are separate decisions.  The
        pilot thermodynamic metric supplies the initial lambda grid.  Short
        runs of the actual production IBS Hamiltonian then test the paper's
        Log-Sum-Exp fixed point.  K>=3 failures split the ensemble using only
        existing nodes.  A K=2 failure is irreducible, so and only so may one
        thermodynamic-length midpoint be inserted; its two replacement IBS
        ensembles must be tested again.  All probes live under a scratch tree
        and never mutate production data or invoke fixed-H/MBAR overlap.
        """
        raise RuntimeError(
            "retired in thermodynamic-path protocol v10: vanishing is one "
            "integrated [0:K] IBS ensemble and cannot enter recursive window design"
        )
        lambdas = [float(x) for x in lambdas_var]
        ranges = [(int(s), int(e)) for s, e in window_ranges]
        pilot_lambdas = path_diagnostics.get("pilot_lambdas")
        pilot_cumulative = path_diagnostics.get(
            "pilot_cumulative_thermodynamic_length"
        )
        if not pilot_lambdas or not pilot_cumulative:
            raise RuntimeError(
                "Stage-2 LSE schedule design 缺少 pilot lambda/累计热力学长度，"
                "拒绝用算术 lambda 中点替代。"
            )

        edge_roots = list(range(len(lambdas) - 1))
        insertions_by_root = [0] * len(edge_roots)
        total_insertions = 0
        total_splits = 0
        probe_counter = 0
        validated_signatures = set()
        history = []
        scratch_root = os.path.join(
            self.output_dir, "schedule_design", "vanishing_ibs_lse"
        )
        os.makedirs(scratch_root, exist_ok=True)
        alchemical_params = _resolve_alchemical_params(
            potential_type, dexp_params, self.ligand_indices
        )

        def _signature(start: int, end: int) -> Tuple[float, ...]:
            return tuple(round(float(x), 14) for x in lambdas[start:end])

        while True:
            pending = [
                (s, e) for s, e in ranges
                if _signature(s, e) not in validated_signatures
            ]
            if not pending:
                break
            start, end = pending[0]
            probe_counter += 1
            probe_dir = os.path.join(scratch_root, f"probe_{probe_counter:03d}")
            probe_checkpoint_dir = os.path.join(probe_dir, "checkpoints")
            os.makedirs(probe_checkpoint_dir, exist_ok=True)
            manager = IBSWindowManagerDualLambda(
                system_template=self.system,
                topology=self.topology,
                perturbed_atom_indices=self.ligand_indices,
                lambdas_coul=[0.0] * len(lambdas),
                lambdas_vdw=lambdas,
                temperature=self.temperature,
                window_ranges=[(start, end)],
                alchemical_params=alchemical_params,
                potential_type=potential_type,
                restraint_params=boresch_params,
                prefix=f"abfe_dual_design_{probe_counter}",
                platform_name=self.platform_name,
                output_dir=probe_dir,
                checkpoint_dir=probe_checkpoint_dir,
            )
            try:
                result = manager.run_all_windows(
                    positions=self.positions,
                    box_vectors=self.box_vectors,
                    n_steps_per_window=0,
                    steps_per_update=500,
                    stage_type="vdw",
                    resume=False,
                    enable_gradual_warmup=True,
                    warmup_steps=int(probe_max_warmup_steps),
                    min_bias_updates=min(6, int(probe_max_bias_updates)),
                    max_bias_updates=int(probe_max_bias_updates),
                    required_consecutive_bias_updates=int(
                        probe_required_consecutive
                    ),
                    max_bias_warmup_steps=int(probe_max_warmup_steps),
                    mbar_calibration_reserved_steps=0,
                    repair_policy="non_mutating_v1",
                    lse_log_residual_tolerance=float(
                        lse_log_residual_tolerance
                    ),
                    warmup_only=True,
                )
            except IBSWarmupConvergenceError as error:
                diagnostics = dict(error.diagnostics or {})
                diagnostics["global_state_range"] = [int(start), int(end)]
                history.append({
                    "probe": int(probe_counter),
                    "range": [int(start), int(end)],
                    "lambdas_vdw": [float(x) for x in lambdas[start:end]],
                    "passed": False,
                    "lse_balance": diagnostics.get("lse_balance"),
                })
                if end - start >= 3:
                    if total_splits >= int(max_window_splits):
                        raise RuntimeError(
                            "IBS LSE schedule design 达到拆窗上限仍未稳定；"
                            "拒绝提交 schedule，请检查构象/pose。"
                        ) from error
                    ranges, feedback = split_window_from_ibs_lse_failure(
                        ranges, diagnostics, len(lambdas)
                    )
                    total_splits += 1
                    history[-1]["action"] = feedback
                    continue

                root = int(edge_roots[start])
                if (
                    total_insertions >= int(max_lambda_insertions)
                    or insertions_by_root[root]
                    >= int(max_insertions_per_initial_edge)
                ):
                    raise RuntimeError(
                        "不可再拆的两态 IBS ensemble 在热力学中点加密达到上限后仍未满足 "
                        "Log-Sum-Exp 自洽方程；这不是继续增加 lambda 能可靠修复的问题，"
                        "拒绝提交 schedule，请转 structural diagnosis / pose audit。"
                    ) from error
                lambdas, ranges, feedback = (
                    insert_thermodynamic_midpoint_from_ibs_lse_failure(
                        lambdas,
                        ranges,
                        diagnostics,
                        pilot_lambdas,
                        pilot_cumulative,
                    )
                )
                edge_roots[start:start + 1] = [root, root]
                insertions_by_root[root] += 1
                total_insertions += 1
                history[-1]["action"] = feedback
                continue

            if not result or len(result) != 1:
                raise RuntimeError(
                    f"IBS LSE design probe 未返回唯一窗口诊断: {result}"
                )
            validated_signatures.add(_signature(start, end))
            history.append({
                "probe": int(probe_counter),
                "range": [int(start), int(end)],
                "lambdas_vdw": [float(x) for x in lambdas[start:end]],
                "passed": True,
                "lse_balance": result[0].get("lse_balance"),
            })

        return lambdas, ranges, {
            "status": "passed",
            "criterion": "ibs_log_sum_exp_fixed_point",
            "lse_log_residual_tolerance": float(lse_log_residual_tolerance),
            "initial_n_states": int(len(edge_roots) + 1 - total_insertions),
            "final_n_states": int(len(lambdas)),
            "total_window_splits": int(total_splits),
            "total_lambda_insertions": int(total_insertions),
            "insertions_by_initial_edge": [int(x) for x in insertions_by_root],
            "final_window_ranges": [list(r) for r in ranges],
            "probe_history": history,
            "scratch_root": os.path.abspath(scratch_root),
            "fixed_h_overlap_used": False,
        }
```
