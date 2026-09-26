# ============================================================================
# 8. 传统 ABFE-REMD 流水线 (从 traditional_abfe_remd.py 迁移)
# ============================================================================
class TraditionalABFEPipeline:
    """传统 REMD + 离线 MBAR 双阶段 ABFE 流水线"""
    def __init__(
        self,
        system: openmm.System,
        topology: app.Topology,
        positions,
        box_vectors,
        ligand_indices: List[int],
        temperature: float = 300.0,
        platform_name: str = "CUDA",
        output_dir: str = "./traditional_abfe",
        repeat_seed: Optional[int] = None,
        leg_name: Optional[str] = None,
    ):
        self.system = system
        self.topology = topology
        self.positions = positions
        self.box_vectors = box_vectors
        self.ligand_indices = ligand_indices
        self.temperature = temperature
        self.platform_name = platform_name
        self.output_dir = output_dir
        self.repeat_seed = int(repeat_seed) if repeat_seed is not None else None
        self.leg_name = str(leg_name) if leg_name is not None else "traditional"
        if self.repeat_seed is not None:
            if self.repeat_seed <= 0:
                raise ValueError("traditional repeat_seed must be positive")
            if self.leg_name not in {"complex", "solvent"}:
                raise ValueError(
                    "seeded traditional pipelines require leg_name='complex' or 'solvent'"
                )
            self.seed_ledger = Exp019SeedLedger(self.repeat_seed, self.leg_name)
        else:
            self.seed_ledger = None

        # Traditional PME-REMD has no implemented co-alchemical ion
        # Hamiltonian.  Validate that limitation during construction, before
        # REMDManager can create any Context.  This keeps neutral fresh/resume
        # runs working while charged routes fail with an actionable protocol
        # error instead of missing-method AttributeError.
        raw_charge = float(_compute_ligand_net_charge(system, self.ligand_indices))
        if not np.isfinite(raw_charge):
            raise ValueError(f"traditional 配体净电荷不是有限数：{raw_charge!r}")
        integer_charge = int(round(raw_charge))
        if abs(raw_charge - integer_charge) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
            raise ValueError(
                f"traditional 配体净电荷 {raw_charge:+.8f} e 不接近整数"
                f"（容差 {LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:g} e）"
            )
        if integer_charge != 0:
            raise RuntimeError(
                "traditional PME-REMD 当前只支持中性配体；检测到配体净电荷 "
                f"{integer_charge:+d} e。该路径尚未实现 co-alchemical ion，"
                "已在创建任何 OpenMM Context 前拒绝运行。请使用 IBS dual_lambda "
                "并显式配置 charge_treatment。"
            )
        os.makedirs(output_dir, exist_ok=True)

    def resolve_co_alchemical_ion_spec(self) -> None:
        """Traditional is constructor-qualified as neutral, so no co-ion exists."""
        return None

    def _seed_for(
        self,
        phase: str,
        stage: str,
        window: Any,
        stream: str,
        attempt: int = 0,
    ) -> Optional[int]:
        if self.seed_ledger is None:
            return None
        return self.seed_ledger.derive(phase, stage, window, stream, attempt)

    def pre_equilibration_identity_fingerprint(
        self, requested_steps: Optional[int]
    ) -> str:
        """预平衡缓存身份。**两侧必须都调它**，否则时序不同 ⇒ 永远不匹配。

        坐标取构造期冻结的那份并量化，因此不受 PBC 修复/居中/再平衡影响，
        也不受 GRO↔mmCIF 往返浮点噪声影响。
        """
        return _pre_equilibration_fingerprint(
            self.system,
            self.ligand_indices,
            self.temperature,
            self.pressure,
            positions=getattr(self, "_identity_positions", self.positions),
            box_vectors=getattr(self, "_identity_box_vectors", self.box_vectors),
            requested_steps=requested_steps,
            barostat_protocol=self.barostat_protocol,
        )

    def seed_contract_identity(self) -> Optional[Dict[str, Any]]:
        """种子契约的**稳定身份**，供缓存/协议指纹使用。

        🔑 [2026-09-01] 与 `seed_contract_snapshot()` 的区别是**不含 derived_seed_map**。
        那张表记的是"本进程到此刻为止派生过哪些种子"，随执行路径增长：首跑时预平衡
        真跑过、派生了 equilibration/* 的种子；resume 时预平衡走 checkpoint 没派生，
        同一份配置下快照就不同。实测事故（4W53 resume_v3.log）：Stage 1 结果缓存因
        `seed_contract.derived_seed_map.equilibration/pre_equilibration/...
        缓存=285721177 -> 当前='<missing>'` 判失配，整段 Stage 1 重算。
        缓存键必须只依赖**配置身份**，不能依赖**执行历史**。
        """
        _ledger = getattr(self, "seed_ledger", None)
        if _ledger is None:
            return None
        # 优先用 ledger 自己的稳定身份；但 ledger 是 duck-typed 的（测试里有只实现
        # snapshot() 的替身），所以没有 identity() 时从 snapshot 里剥掉那张
        # 随执行路径增长的 derived_seed_map —— 语义保证与实现无关。
        _identity = getattr(_ledger, "identity", None)
        if callable(_identity):
            return _identity()
        _snap = _ledger.snapshot()
        if isinstance(_snap, dict):
            return {k: v for k, v in _snap.items() if k != "derived_seed_map"}
        return _snap

    def seed_contract_snapshot(self) -> Optional[Dict[str, Any]]:
        # `seed_ledger` 在 __init__ 里无条件赋值（有 repeat_seed 时是 ledger、
        # 否则 None），所以生产路径上一定存在。用 getattr 兜底是为了让
        # `object.__new__(ABFEPipeline)` 造的测试 stub 也能安全调用——与同类的
        # `residual_sampling_protocol_payload()` 用 getattr 读 flag 同一惯用法。
        # 2026-08-31 起本方法进了 stage/顶层/single·2D 三个协议键，调用面变宽了。
        _ledger = getattr(self, "seed_ledger", None)
        return _ledger.snapshot() if _ledger is not None else None

    @classmethod
    def from_gromacs(
        cls,
        gro_file: str,
        top_file: str,
        ligand_resname: str,
        temperature: float = 300.0,
        platform_name: str = "CUDA",
        output_dir: str = "./traditional_abfe",
        gmx_include_dir: str = None,
        repeat_seed: Optional[int] = None,
        leg_name: Optional[str] = None,
    ):
        gro = app.GromacsGroFile(gro_file)
        # 走唯一入口（见 abfe_core.load_gromacs_topology_for_openmm 的说明）。
        top = load_gromacs_topology_for_openmm(
            top_file,
            periodicBoxVectors=gro.getPeriodicBoxVectors(),
            includeDir=gmx_include_dir,
        )
        system = top.createSystem(
            nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer,
            constraints=app.HBonds, rigidWater=True,
        )
        topology = top.topology
        positions = gro.positions
        box_vectors = gro.getPeriodicBoxVectors()
        ligand_indices = [a.index for a in topology.atoms() if a.residue.name == ligand_resname]
        return cls(
            system=system, topology=topology,
            positions=positions, box_vectors=box_vectors,
            ligand_indices=ligand_indices,
            temperature=temperature, platform_name=platform_name,
            output_dir=output_dir,
            repeat_seed=repeat_seed, leg_name=leg_name,
        )

    def run_leg(
        self,
        stage_name: str,
        lambdas_coul: List[float],
        lambdas_vdw: List[float],
        n_steps: int = 500000,
        exchange_interval: int = 1000,
        resume: bool = False,
        boresch_params: Optional[Dict] = None,
        potential_type: str = "softcore",
    ) -> Dict:
        print(f"\n{'='*60}\n开始 {stage_name} 腿解耦\n{'='*60}")
        if len(lambdas_coul) != len(lambdas_vdw):
            raise ValueError("传统 REMD 腿的 lambdas_coul/lambdas_vdw 长度必须一致。")
        if potential_type == "dexp":
            raise NotImplementedError(
                "traditional / PME-REMD 路径当前未实现 DEXP 或混合 softcore 替代势；"
                "如需 DEXP，请使用 IBS dual_lambda。"
            )
        stage_output_dir = os.path.join(self.output_dir, stage_name)
        os.makedirs(stage_output_dir, exist_ok=True)
        traj_files = _expected_remd_traj_files(stage_output_dir, stage_name, len(lambdas_coul))
        u_kn_path = os.path.join(self.output_dir, f"{stage_name}_u_kn.npy")
        n_k_path = u_kn_path + ".n_k.npy"
        u_kn_meta_path = u_kn_path + ".meta.json"
        remd_meta_path = os.path.join(stage_output_dir, f"{stage_name}_remd.meta.json")
        sampling_fingerprint = _protocol_fingerprint({
            "kind": "traditional_remd_sampling",
            "pme_decharge_model_version": PME_DECHARGE_MODEL_VERSION,
            "stage_name": stage_name,
            "system_xml_sha256": _system_xml_hash(self.system),
            "topology_sha256": _topology_hash(self.topology),
            "initial_positions_sha256": _positions_hash(self.positions),
            "ligand_indices": [int(i) for i in self.ligand_indices],
            "constraint_identity": constraint_identity_fingerprint(
                self.system, self.ligand_indices
            ),
            "lambdas_coul": _lambda_signature(lambdas_coul),
            "lambdas_vdw": _lambda_signature(lambdas_vdw),
            "temperature_K": float(self.temperature),
            "n_steps": int(n_steps),
            "exchange_interval": int(exchange_interval),
            # 🔑 [2026-09-01] 同 `_stage_protocol_key`，见那里的说明。
            "boresch_params": _preopt_boresch_protocol_payload(boresch_params),
            "potential_type": potential_type,
            # 🔑 [P1-18] 同 `_remd_sampling_fingerprint`：REM 动力学消费的
            # repeat-seed contract 身份必须进 sampling fingerprint，否则换
            # repeat_seed 的独立重复会复用旧 DCD/u_kn。
            "seed_contract": self.seed_contract_identity(),
            # 🔑 [2026-08-24 修] 同 `_stage_protocol_key`/`_build_top_level_protocol_key`：
            # 曾经这里还有 `"code_sha256": _code_hash()`，删掉——同一个"不相关文件改动
            # 强制整段 REMD 重跑"的问题，只是这次是 traditional/PME-REMD 路径。
        })
        analysis_fingerprint = _protocol_fingerprint({
            "kind": "traditional_mbar_u_kn",
            "sampling_fingerprint_sha256": sampling_fingerprint["sha256"],
            "traditional_lj_lrc_protocol_version": (
                TRADITIONAL_LJ_LRC_PROTOCOL_VERSION
            ),
            "pme_decharge_model_version": PME_DECHARGE_MODEL_VERSION,
        })

        if resume and os.path.exists(u_kn_path):
            cached_meta = None
            if os.path.exists(u_kn_meta_path):
                try:
                    with open(u_kn_meta_path, "r", encoding="utf-8") as handle:
                        cached_meta = json.load(handle)
                except Exception as exc:
                    print(f"  [WARN] u_kn 元数据不可读 ({exc})，拒绝复用")
            if (
                cached_meta is not None
                and cached_meta.get("analysis_fingerprint") == analysis_fingerprint
                and os.path.exists(n_k_path)
            ):
                u_kn = np.load(u_kn_path)
                n_k = np.load(n_k_path)
                if (
                    u_kn.ndim == 2
                    and u_kn.shape[0] == len(lambdas_coul)
                    and int(np.sum(n_k)) == u_kn.shape[1]
                    and np.all(np.isfinite(u_kn))
                ):
                    print("  u_kn 完整协议指纹一致，跳过 REMD 与能量重算")
                    analyzer = TraditionalMBARAnalyzer(temperature=self.temperature)
                    analyzer._last_n_k = n_k
                    analyzer._last_lj_lrc_metadata = cached_meta.get(
                        "traditional_lj_lrc", {}
                    )
                    result = analyzer.solve(u_kn)
                    result.setdefault("diagnostics", {})["traditional_lj_lrc"] = (
                        analyzer._last_lj_lrc_metadata
                    )
                    return result
                print("  [WARN] u_kn/n_k 内容或形状无效，拒绝复用")
            else:
                print("  [WARN] u_kn 缺少或不匹配完整协议指纹，拒绝复用")

        expected_frames = max(1, _expected_remd_frame_count(n_steps))
        remd_fingerprint_matches = False
        if resume and os.path.exists(remd_meta_path):
            try:
                with open(remd_meta_path, "r", encoding="utf-8") as handle:
                    remd_fingerprint_matches = _protocol_fingerprint_ignoring_code_hash(
                        json.load(handle).get("sampling_fingerprint")
                    ) == _protocol_fingerprint_ignoring_code_hash(sampling_fingerprint)
            except Exception as exc:
                print(f"  [WARN] REMD 元数据不可读 ({exc})，拒绝复用轨迹")
        if resume and remd_fingerprint_matches and _all_remd_trajs_valid(
            stage_output_dir,
            stage_name,
            len(lambdas_coul),
            min_frames=expected_frames,
        ):
            print("  检测到完整 REMD DCD，视为采样已完成，跳过 REMD 继续离线 MBAR")
        else:
            remd = REMDManager(
                system_template=self.system,
                topology=self.topology,
                positions=self.positions,
                box_vectors=self.box_vectors,
                ligand_indices=self.ligand_indices,
                lambdas_coul=lambdas_coul,
                lambdas_vdw=lambdas_vdw,
                temperature=self.temperature,
                platform_name=self.platform_name,
                output_dir=stage_output_dir,
                boresch_params=boresch_params,
                # [MEM-00c] 所有 replica 共用同一份冻结身份。
                co_alchemical_ion_spec=self.resolve_co_alchemical_ion_spec(),
                random_seed=self._seed_for(
                    "charging", stage_name, "exchange", "numpy"
                ),
                seed_ledger=self.seed_ledger,
                seed_stage=stage_name,
                seed_leg=self.leg_name,
            )
            traj_files = remd.run(
                n_steps=n_steps,
                exchange_interval=exchange_interval,
                stage_name=stage_name,
            )
            with open(remd_meta_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {"sampling_fingerprint": sampling_fingerprint},
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

        analyzer = TraditionalMBARAnalyzer(temperature=self.temperature)
        u_kn = analyzer.compute_u_kn(
            traj_files=traj_files,
            system_template=self.system,
            ligand_indices=self.ligand_indices,
            lambdas_coul=lambdas_coul,
            lambdas_vdw=lambdas_vdw,
            platform_name="CPU",
            topology=self.topology,
            reference_positions=self.positions,
            reference_box_vectors=self.box_vectors,
            boresch_params=boresch_params,
            # [MEM-00c] 重算用动力学当时冻结的粒子，不按 reference_positions 重选。
            co_alchemical_ion_spec=self.resolve_co_alchemical_ion_spec(),
        )
        np.save(u_kn_path, u_kn)
        np.save(n_k_path, analyzer._last_n_k)
        with open(u_kn_meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "analysis_fingerprint": analysis_fingerprint,
                    "sampling_fingerprint": sampling_fingerprint,
                    "traditional_lj_lrc": analyzer._last_lj_lrc_metadata,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        result = analyzer.solve(u_kn)
        result.setdefault("diagnostics", {})["traditional_lj_lrc"] = (
            analyzer._last_lj_lrc_metadata
        )
        return result

    def run_full(
        self,
        n_lambda: int = 12,
        n_steps_per_leg: int = 500000,
        boresch_correction: float = 0.0,
        boresch_params: Optional[Dict] = None,
        potential_type: str = "softcore",
        resume: bool = False,
        attachment_result: Optional[Dict] = None,
        allow_untrusted_stage_results: bool = False,
    ) -> Dict:
        # 🔑 [2026-09-10] 这个参数原来不存在，而 runabfe 的两个 traditional 调用点
        # （runabfe.py:5172 / 5261）一直在传它 ⟹ `--mode traditional` 必抛
        # `TypeError: got an unexpected keyword argument`，且是在复合物腿基线预平衡 +
        # Boresch 估算跑完之后才抛，白烧几小时 GPU。
        #
        # 语义与 `ABFEPipeline.run_full_pipeline` 的同名参数对齐：它放行的是
        # **收敛质量门**，不是身份校验，默认仍然 fail-closed。
        if attachment_result is not None and boresch_params is None:
            raise RuntimeError(
                "attachment_result 不能脱离 boresch_params 使用；"
                "否则会把不对应当前 Hamiltonian 的 attachment 自由能并入传统腿。"
            )
        if boresch_params is not None and attachment_result is None:
            raise RuntimeError(
                "Traditional+Boresch 必须先完成 A'→A attachment leg；"
                "缺少 attachment_result，拒绝在仅有解析 release 的不闭合循环上采样。"
            )
        if attachment_result is not None:
            dg_attach, err_attach = validate_boresch_attachment_result(attachment_result)
        else:
            dg_attach = 0.0
            err_attach = 0.0
        try:
            boresch_correction = float(boresch_correction)
        except (TypeError, ValueError) as exc:
            raise ValueError("boresch_correction 必须是有限数值") from exc
        if not np.isfinite(boresch_correction):
            raise ValueError("boresch_correction 必须是有限数值")
        lambdas_coul = np.linspace(1.0, 0.0, n_lambda).tolist()
        lambdas_vdw = [1.0] * n_lambda
        res_coul = self.run_leg(
            "decharging",
            lambdas_coul,
            lambdas_vdw,
            n_steps_per_leg,
            resume=resume,
            boresch_params=boresch_params,
            potential_type=potential_type,
        )

        _untrusted_coul = _assert_or_warn_sampling_converged(
            {"total_delta_G": res_coul.get("delta_G"), "total_error": res_coul.get("error"),
             "converged": res_coul.get("converged")},
            context="traditional decharging",
            allow_untrusted=allow_untrusted_stage_results,
            log=self._log if hasattr(self, "_log") else print,
        )

        lambdas_coul = [0.0] * n_lambda
        lambdas_vdw = np.linspace(1.0, 0.0, n_lambda).tolist()
        res_vdw = self.run_leg(
            "vanishing",
            lambdas_coul,
            lambdas_vdw,
            n_steps_per_leg,
            resume=resume,
            boresch_params=boresch_params,
            potential_type=potential_type,
        )

        _untrusted_vdw = _assert_or_warn_sampling_converged(
            {"total_delta_G": res_vdw.get("delta_G"), "total_error": res_vdw.get("error"),
             "converged": res_vdw.get("converged")},
            context="traditional vanishing",
            allow_untrusted=allow_untrusted_stage_results,
            log=self._log if hasattr(self, "_log") else print,
        )

        dg_leg = res_coul["delta_G"] + res_vdw["delta_G"]
        err_leg = np.sqrt(res_coul["error"]**2 + res_vdw["error"]**2)
        dg_total = dg_leg + dg_attach + boresch_correction
        err_total = float(np.sqrt(err_leg**2 + err_attach**2))

        _untrusted = [r for r in (_untrusted_coul, _untrusted_vdw) if r]
        final = {
            # 放行过质量门的结果不许自称 converged —— 否则它和一次真正收敛的
            # 运行在产物里长得一模一样。
            "converged": not _untrusted,
            "stage_quality_failures": _untrusted,
            "allow_untrusted_stage_results": bool(allow_untrusted_stage_results),
            "stage_decharging": res_coul,
            "stage_vanishing": res_vdw,
            "delta_G_leg_kJ_mol": dg_leg,
            # 🔑 [0831issue P2] 字段名与内容确实不配对：`delta_G_leg_kJ_mol` **不含**
            # attachment，而这里是含 attachment 的 err_total。
            # **但值不能改成 err_leg**：`abfe_core.py` 的 schema 把
            # ("delta_G_total_kJ_mol", "error_leg_kJ_mol") 声明为**同一对**，
            # 下游（runabfe.py 两处 traditional 汇总）也确实是拿它配 delta_G_total 用的
            # ——那个组合在数值上是对的。按 issue 原建议改成 err_leg 会把 err_attach
            # 从 delta_G_total 的 ± 里悄悄丢掉，是引入新 bug 而不是修 bug。
            # 因此保留历史数值语义（= error_total_kJ_mol，配 delta_G_total），
            # 另外补一个名副其实的字段供需要"不含 attachment"那一半的消费者使用。
            "error_leg_kJ_mol": err_total,
            "error_leg_excluding_attachment_kJ_mol": float(err_leg),
            "error_leg_kJ_mol_pairs_with": "delta_G_total_kJ_mol",
            "error_total_kJ_mol": err_total,
            "boresch_correction_kJ_mol": boresch_correction,
            "boresch_attachment_delta_G_kJ_mol": dg_attach,
            "boresch_attachment_error_kJ_mol": err_attach,
            "boresch_attachment_result": attachment_result,
            "constraint_identity": constraint_identity_fingerprint(
                self.system, self.ligand_indices
            ),
            # ✅ dg_total = dg_leg + boresch_correction：本函数的 delta_G_total_kJ_mol
            # 恒已把传入的 boresch_correction 加总在内。调用方（如 runabfe.py 的
            # traditional 模式）若显式传 boresch_correction=0.0 并在外层单独扣减
            # Boresch，则本文件里的 boresch_correction_kJ_mol 会是 0，如实反映"未包含"；
            # 但只要调用方传入非零值，此标记即为 True，不应再对 delta_G_total_kJ_mol 二次扣减。
            "boresch_correction_already_included_in_total_delta_G": True,
            "delta_G_total_kJ_mol": dg_total,
            "delta_G_total_kcal_mol": dg_total / 4.184,
        }
        with open(os.path.join(self.output_dir, "final_results.json"), "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
        print(f"\n[OK] 传统腿完成 | ΔG_leg = {dg_total:.2f} ± {err_total:.2f} kJ/mol")
        return final
