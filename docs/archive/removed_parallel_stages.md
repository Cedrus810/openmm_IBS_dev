# 已移除：`--parallel-stages` 并行阶段执行

移除日期：2026-07-27　　对应待办：**ATT-27** 后续

## 为什么移除

`--parallel-stages` 曾用于把 Stage 1（去电荷）与 Stage 2（去VDW）放进两个 spawn
子进程同时跑。但它**早已被无条件禁用**——`run_full_pipeline` 里读到该参数后直接：

```python
_parallel_stages = kwargs.get("parallel_stages", False)
if _parallel_stages:
    self._log("⚠️ 热力学路径协议 v1 需要在父进程捕获结构化 warmup 失败并立即"
              "反馈重切路径；跨进程异常反馈尚未序列化，当前自动回退串行阶段执行。")
    _parallel_stages = False
```

于是 `if _parallel_stages and ...:` 恒为假，**149 行并行分支**与
**87 行 spawn worker（`_run_stage_worker_process`）** 全部不可达。

留着它有实际代价，而且刚刚就付过一次：2026-07-27 排查「REMD 默认永远回退 CPU」时，
`max_resident_contexts` 需要透传到 **4 个** `_run_dual_lambda_stage` 调用点，
其中一个就在这条不可达的 worker 路径里——为一条没人能跑的代码路径做了plumbing。
它还是 ATT-04（spawn 子进程 import 期 CUDA 初始化）唯一的现实动因。

## 用户决定（2026-07-27）

> 「多 gpu 的直接给归档就行了，暂时用不上多 gpu」
> 「而且多 gpu 会导致各种调度问题，所以直接归档」

即：**不打算恢复 stage 级多 GPU**。除了当初禁用它的根因（跨进程 warmup 失败反馈
未序列化）之外，把两个 stage 钉在两张卡上本身就带来调度问题。若将来真要用多卡，
更合理的方向是**在 REMD 内部按 replica 分卡**（12 个副本分到 N 张卡；单卡上 12 个
Context 本来就是时分复用），那与本文件归档的 stage 级并行是两码事，
也不需要跨进程异常序列化。

## 现在的行为

只有一条顺序执行路径。若仍传 `parallel_stages=True`，`run_full_pipeline` **入口**
即 fail closed（与 `enable_lambda_refine` 同一模式、同一位置），不会跑到一半才告警。

要重新实现并行阶段的话，先解决它当初被禁用的根因：**跨进程结构化 warmup 失败反馈的
序列化**（父进程需要捕获 `IBSWarmupConvergenceError` 并据此重切 λ 路径）。
不要复活下面这段。

## 移除的内容

### 1. `run_full_pipeline` 里的并行分支（149 行）

```python
            if _parallel_stages and should_run_stage1 and should_run_stage2:
                self._log("\n[双λ] 🚀 并行执行 Stage 1 (去电荷) + Stage 2 (去VDW)")
                state_dir = os.path.join(self.checkpoint_dir, "parallel_state")
                self._save_state_to_dir(state_dir)

                _res_dir = os.path.join(self.checkpoint_dir, "parallel_results")
                os.makedirs(_res_dir, exist_ok=True)
                _res1 = os.path.join(_res_dir, "stage1.json")
                _res2 = os.path.join(_res_dir, "stage2.json")
                # ✅ 清空上一轮遗留的 stage1.json/stage2.json：若本轮 worker 子进程
                # 崩溃/被杀而未写出新结果，下面 open(_res1)/open(_res2) 必须直接报
                # FileNotFoundError，而不是静默读到上一次运行的旧结果当作本轮成功。
                for _stale in (_res1, _res2):
                    if os.path.exists(_stale):
                        os.remove(_stale)

                _temp_k = self.temperature.value_in_unit(unit.kelvin)
                _common = dict(
                    # 供 spawn worker 透传给 REMDManager，见 _run_stage_worker_process。
                    charging_max_resident_contexts=kwargs.get(
                        "charging_max_resident_contexts"
                    ),
                    n_states_stage1=stage1_states,
                    n_states_stage2=stage2_states,
                    n_steps_per_window=n_steps_per_window,
                    steps_per_update=steps_per_update,
                    system_type=system_type,
                    potential_type=potential_type,
                    dexp_params=dexp_params,
                    enable_early_stop=enable_early_stop,
                    boresch_params=boresch_params,
                    enable_gradual_warmup=kwargs.get("enable_gradual_warmup", True),
                    warmup_steps=kwargs.get("warmup_steps", 500000),
                    min_bias_updates=kwargs.get("min_bias_updates", 12),
                    max_bias_updates=kwargs.get("max_bias_updates", 50),
                    required_consecutive_bias_updates=kwargs.get(
                        "required_consecutive_bias_updates", 3
                    ),
                    max_bias_warmup_steps=kwargs.get("max_bias_warmup_steps", 500000),
                    resume=resume,
                )
                stage1_platform = self.platform_name
                stage2_platform = self.platform_name
                if str(self.platform_name).upper().startswith("CUDA"):
                    env_stage1 = os.environ.get("IBS_STAGE1_CUDA_DEVICE")
                    env_stage2 = os.environ.get("IBS_STAGE2_CUDA_DEVICE")
                    if env_stage1 is not None and env_stage2 is not None and env_stage1 != env_stage2:
                        stage1_platform = f"CUDA:{env_stage1}"
                        stage2_platform = f"CUDA:{env_stage2}"
                        self._log(f"  🔀 并行阶段将分别使用 CUDA 设备 {env_stage1} 和 {env_stage2}")
                    else:
                        self._log("  ⚠️ 检测到并行双阶段 + CUDA，但未提供两个不同 GPU；为避免上下文冲突，回退为串行执行。")
                        _parallel_stages = False

                if _parallel_stages:
                    ctx = mp.get_context("spawn")
                    p1 = ctx.Process(
                        target=_run_stage_worker_process,
                        args=(state_dir, _temp_k, stage1_platform, self.output_dir,
                              "decharging", 1.0, 1.0,
                              _common["n_states_stage1"], _common["n_steps_per_window"],
                              _common["steps_per_update"], _common["system_type"],
                              _common["potential_type"], _common["dexp_params"],
                              optimized_lambdas_1, window_ranges_1, _common["enable_early_stop"],
                              _common["boresch_params"], _common["enable_gradual_warmup"],
                              _common["warmup_steps"], _common["min_bias_updates"],
                              _common["max_bias_updates"], _common["required_consecutive_bias_updates"],
                              _common["max_bias_warmup_steps"], _common["resume"], _res1),
                        kwargs={"max_resident_contexts": _common.get("charging_max_resident_contexts")},
                    )
                    p2 = ctx.Process(
                        target=_run_stage_worker_process,
                        args=(state_dir, _temp_k, stage2_platform, self.output_dir,
                              "vanishing", 0.0, 1.0,
                              _common["n_states_stage2"], _common["n_steps_per_window"],
                              _common["steps_per_update"], _common["system_type"],
                              _common["potential_type"], _common["dexp_params"],
                              optimized_lambdas_2, window_ranges_2, _common["enable_early_stop"],
                              _common["boresch_params"], _common["enable_gradual_warmup"],
                              _common["warmup_steps"], _common["min_bias_updates"],
                              _common["max_bias_updates"], _common["required_consecutive_bias_updates"],
                              _common["max_bias_warmup_steps"], _common["resume"], _res2),
                        kwargs={"max_resident_contexts": _common.get("charging_max_resident_contexts")},
                    )
                    p1.start()
                    p2.start()
                    p1.join()
                    p2.join()
                else:
                    _run_stage_worker_process(
                        state_dir, _temp_k, stage1_platform, self.output_dir,
                        "decharging", 1.0, 1.0,
                        _common["n_states_stage1"], _common["n_steps_per_window"],
                        _common["steps_per_update"], _common["system_type"],
                        _common["potential_type"], _common["dexp_params"],
                        optimized_lambdas_1, window_ranges_1, _common["enable_early_stop"],
                        _common["boresch_params"], _common["enable_gradual_warmup"],
                        _common["warmup_steps"], _common["min_bias_updates"],
                        _common["max_bias_updates"], _common["required_consecutive_bias_updates"],
                        _common["max_bias_warmup_steps"], _common["resume"], _res1,
                        max_resident_contexts=_common.get("charging_max_resident_contexts"),
                    )
                    _run_stage_worker_process(
                        state_dir, _temp_k, stage2_platform, self.output_dir,
                        "vanishing", 0.0, 1.0,
                        _common["n_states_stage2"], _common["n_steps_per_window"],
                        _common["steps_per_update"], _common["system_type"],
                        _common["potential_type"], _common["dexp_params"],
                        optimized_lambdas_2, window_ranges_2, _common["enable_early_stop"],
                        _common["boresch_params"], _common["enable_gradual_warmup"],
                        _common["warmup_steps"], _common["min_bias_updates"],
                        _common["max_bias_updates"], _common["required_consecutive_bias_updates"],
                        _common["max_bias_warmup_steps"], _common["resume"], _res2,
                        max_resident_contexts=_common.get("charging_max_resident_contexts"),
                    )

                # Check for errors
                for _rf, _label in [(_res1, "Stage 1"), (_res2, "Stage 2")]:
                    with open(_rf) as f:
                        _r = json.load(f)
                    if "error" in _r:
                        raise RuntimeError(f"{_label} 子进程失败: {_r['error']}")

                with open(_res1) as f:
                    stage1 = json.load(f)
                with open(_res2) as f:
                    stage2 = json.load(f)

                # Save checkpoint files
                self._assert_stage_result_sane("Stage 1 (decharging)", stage1)
                _s1 = self._build_stage_cache_payload(
                    "decharging", stage1, stage1_states, _stage1_protocol_key,
                    optimized_lambdas_1, window_ranges_1,
                )
                with open(stage1_file, "w") as f:
                    json.dump(_s1, f, indent=2)
                self._update_stage_status(stage1_key, "completed",
                                          {"total_delta_G": stage1["total_delta_G"]})

                self._assert_stage_result_sane("Stage 2 (vanishing)", stage2)
                _s2 = self._build_stage_cache_payload(
                    "vanishing", stage2, stage2_states, _stage2_protocol_key,
                    optimized_lambdas_2, window_ranges_2,
                )
                with open(stage2_file, "w") as f:
                    json.dump(_s2, f, indent=2)
                self._update_stage_status(stage2_key, "completed",
                                          {"total_delta_G": stage2["total_delta_G"]})

```

### 2. `_run_stage_worker_process`（87 行）

```python
def _run_stage_worker_process(
    state_dir: str,
    temperature_k: float,
    platform_name: str,
    output_dir: str,
    stage_name: str,
    fixed_lam_coul: float,
    fixed_lam_vdw: float,
    n_states: int,
    n_steps_per_window: int,
    steps_per_update: int,
    system_type: str,
    potential_type: str,
    dexp_params: Optional[Dict],
    optimized_lambdas: Optional[List[float]],
    window_ranges: Optional[List[Tuple[int, int]]],
    enable_early_stop: bool,
    boresch_params: Optional[Dict],
    enable_gradual_warmup: bool,
    warmup_steps: int,
    min_bias_updates: int,
    max_bias_updates: int,
    required_consecutive_bias_updates: int,
    max_bias_warmup_steps: int,
    resume: bool,
    result_file: str,
    # 🔑 [2026-07-27] 必须一路传到 REMDManager，否则 worker 里会拿默认上限、
    # 预防性回退 CPU（慢 ~100×）。`--parallel-stages` 目前被无条件禁用、
    # 这条路径不可达，但先接上：重新启用它时不该再踩回同一个坑。
    max_resident_contexts: Optional[int] = None,
):
    """子进程工作函数：加载保存的Pipeline状态并执行一个双λ阶段"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import json as _json
    import numpy as _np
    from openmm import app as _app, unit as _unit, Vec3 as _Vec3, XmlSerializer as _XmlSerializer

    with open(os.path.join(state_dir, "system.xml")) as _f:
        _system = _XmlSerializer.deserialize(_f.read())
    _pdbx = app.PDBxFile(os.path.join(state_dir, "topology.cif"))
    _topology = _pdbx.topology
    _pos_np = _np.load(os.path.join(state_dir, "positions.npy"))
    _positions = [_Vec3(float(_v[0]), float(_v[1]), float(_v[2])) for _v in _pos_np] * _unit.nanometer
    _bv_np = _np.load(os.path.join(state_dir, "box_vectors.npy"))
    _box_vectors = [_Vec3(float(_v[0]), float(_v[1]), float(_v[2])) for _v in _bv_np] * _unit.nanometer
    with open(os.path.join(state_dir, "ligand_indices.json")) as _f:
        _ligand_indices = _json.load(_f)

    from abfe_pipeline import ABFEPipeline as _Pipeline
    _stage_ckpt_dir = os.path.join(output_dir, "checkpoints", stage_name)
    _pipeline = _Pipeline(
        system=_system,
        topology=_topology,
        positions=_positions,
        box_vectors=_box_vectors,
        ligand_indices=_ligand_indices,
        temperature=temperature_k,
        output_dir=output_dir,
        checkpoint_dir=_stage_ckpt_dir,
        platform_name=platform_name,
    )
    _result = _pipeline._run_dual_lambda_stage(
        stage_name=stage_name,
        fixed_lam_coul=fixed_lam_coul,
        fixed_lam_vdw=fixed_lam_vdw,
        n_states=n_states,
        n_steps_per_window=n_steps_per_window,
        steps_per_update=steps_per_update,
        system_type=system_type,
        resume=resume,
        potential_type=potential_type,
        dexp_params=dexp_params,
        optimized_lambdas=optimized_lambdas,
        window_ranges=window_ranges,
        enable_early_stop=enable_early_stop,
        boresch_params=boresch_params,
        enable_gradual_warmup=enable_gradual_warmup,
        warmup_steps=warmup_steps,
        min_bias_updates=min_bias_updates,
        max_bias_updates=max_bias_updates,
        required_consecutive_bias_updates=required_consecutive_bias_updates,
        max_bias_warmup_steps=max_bias_warmup_steps,
        remd_max_resident_contexts=max_resident_contexts,
    )
    with open(result_file, "w") as _f:
        _json.dump(_result, _f, indent=2)
```

### 3. `_save_state_to_dir`（28 行，随并行路径一并孤立）

它唯一的用途是把父进程状态写到 `checkpoints/parallel_state`，供 spawn worker
反序列化。并行路径移除后全仓只剩定义处一处引用，故一并删除。

```python
    def _save_state_to_dir(self, state_dir: str):
        """将 Pipeline 状态序列化至磁盘，供子进程加载"""
        os.makedirs(state_dir, exist_ok=True)
        with open(os.path.join(state_dir, "system.xml"), "w") as f:
            f.write(XmlSerializer.serialize(self.system))
        with open(os.path.join(state_dir, "topology.cif"), "w") as f:
            app.PDBxFile.writeFile(self.topology, self.positions, f)

        pos = self.positions
        if hasattr(pos, "value_in_unit"):
            pos_np = np.array([[float(v[i]) for i in range(3)] for v in pos.value_in_unit(unit.nanometer)])
        else:
            pos_np = np.asarray(pos, dtype=np.float64)
        np.save(os.path.join(state_dir, "positions.npy"), pos_np)

        if self.box_vectors is not None:
            bv = self.box_vectors
            if hasattr(bv, "value_in_unit"):
                bv_np = np.array([[float(v[i]) for i in range(3)] for v in bv.value_in_unit(unit.nanometer)])
            else:
                bv_np = np.asarray(bv, dtype=np.float64)
        else:
            bv_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        np.save(os.path.join(state_dir, "box_vectors.npy"), bv_np)

        with open(os.path.join(state_dir, "ligand_indices.json"), "w") as f:
            json.dump(self.ligand_indices, f)
        self._log(f"  💾 Pipeline 状态已保存至 {state_dir}")
```
