#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""膜体系协议配置期自检：把 runabfe 在建任何 Context 之前会做的检查全跑一遍。

为什么单独有这个脚本：`runabfe.py` 的那批 fail-closed 检查（memtodolist §6.1
"在创建任何 Context 前完成协议组合的检查"）都在 `main()` 里，跑到它们之前会先
构建/加载 System。这个脚本只调同一批**纯校验函数**，秒级、不建 Context、不碰 GPU，
所以改完配置可以先用它确认协议这一关能过，再去节点上烧时间。

它复用生产实现，不另写一套判据——判据分叉正是这套代码反复吃过的亏。

    python check_membrane_protocol.py [abfe_config.json]
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, ".."))))

import abfe_core as core


def _fail(message):
    print(f"  ❌ {message}")
    return False


def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "abfe_config.json"
    with open(config_path, encoding="utf-8") as handle:
        config = {k: v for k, v in json.load(handle).items() if not k.startswith("_")}
    top = config["top"]
    print(f"配置: {config_path}\n拓扑: {top}\n")

    ok = True

    # ---- §1.1 力场族 ----
    print("[1/8] 力场族识别（§1.1）")
    try:
        family = core.resolve_forcefield_family(
            top_path=top, explicit_family=config.get("forcefield_family")
        )
        detection = family["detection"]
        print(f"  ✅ {family['family']}（来源 {family['source']}，依据 {detection['reason']}）")
        print(f"     [ defaults ]: {detection['defaults_row']}")
    except Exception as exc:
        ok = _fail(f"{type(exc).__name__}: {exc}")
        family = None

    # ---- 体系组成 ----
    print("\n[2/8] 体系组成（来自 .top 的 [ molecules ]，不靠残基名）")
    declaration = None
    composition = None
    try:
        decl_path = config.get("membrane_input_declaration")
        if decl_path:
            with open(decl_path, encoding="utf-8") as handle:
                declaration = {
                    k: v for k, v in json.load(handle).items() if not k.startswith("_")
                }
        parsed = core.parse_gromacs_topology(top)
        composition = core.classify_system_composition(
            parsed,
            ligand_molecule_name=(declaration or {}).get("ligand_molecule_name"),
            declared_roles=(declaration or {}).get("molecule_roles"),
        )
        for name, role in composition["roles"].items():
            count = composition["molecule_counts"][name]
            print(f"  ✅ {name:20s} {role:8s} ×{count}"
                  f"  ({composition['role_evidence'][name]})")
        atoms = composition["atom_indices_by_role"]
        print(f"     原子数: 蛋白 {len(atoms['protein'])}, 脂质 {len(atoms['lipid'])}"
              f"（{len(composition['molecules_by_role']['lipid'])} 个分子）, "
              f"水 {len(atoms['water'])}, 离子 {len(atoms['ion'])}, 配体 {len(atoms['ligand'])}")
        print(f"     合计 {composition['n_atoms_total']}")
    except Exception as exc:
        ok = _fail(f"{type(exc).__name__}: {exc}")

    # ---- 与坐标文件原子数一致 ----
    print("\n[3/8] .top 展开原子数 vs 坐标文件")
    try:
        with open(config["gro"], encoding="utf-8") as handle:
            handle.readline()
            n_gro = int(handle.readline())
        n_top = composition["n_atoms_total"]
        if n_gro == n_top:
            print(f"  ✅ {n_gro} == {n_top}")
        else:
            ok = _fail(f".top 展开 {n_top} 原子，{config['gro']} 有 {n_gro} 个——"
                       "索引会静默错位")
    except Exception as exc:
        ok = _fail(f"{type(exc).__name__}: {exc}")

    # ---- §3.1/§3.2 环境类型与恒压器 ----
    print("\n[4/8] 环境类型与膜恒压协议（§3.1/§3.2）")
    barostat = None
    try:
        # 这里不传 topology：脂质残基交叉检查需要 OpenMM topology，留给 runabfe 做。
        barostat = core.resolve_membrane_protocol(
            config.get("system_type"), membrane_config=config.get("membrane")
        )
        print(f"  ✅ {barostat['system_type']} → {barostat['barostat_class']}"
              f"（频率 {barostat['barostat_frequency']}）")
        if barostat["membrane"]:
            print(f"     {barostat['membrane']}")
    except Exception as exc:
        ok = _fail(f"{type(exc).__name__}: {exc}")

    # ---- §1.2 电荷路线 ----
    print("\n[5/8] 净电荷处理路线（§1.2）")
    try:
        charge = core.resolve_charge_treatment(
            config.get("charge_treatment"),
            # 配体净电荷由 runabfe 从 System 实算；这里按声明的中性做协议自检，
            # 若实际带电，runabfe 会在 §1.2 的 fail-closed 上报错。
            ligand_net_charge_e=0.0,
            apbs_correction_kJ_mol=config.get("apbs_correction_kJ_mol", 0.0),
            environment_type=config.get("system_type"),
        )
        print(f"  ✅ {charge['charge_treatment']}"
              f"（APBS {'适用' if charge['apbs_applicable'] else '不适用'}，"
              f"{charge['apbs_not_applicable_reason']}）")
        print("     ⚠️ 配体净电荷此处按 0 假设；真值由 runabfe 从 System 实算并交叉核对")
    except Exception as exc:
        ok = _fail(f"{type(exc).__name__}: {exc}")

    # ---- §1.3 色散路线 ----
    print("\n[6/8] LJ/色散路线（§1.3）")
    try:
        dispersion = core.resolve_dispersion_protocol(
            config.get("dispersion_protocol"),
            environment_type=config.get("system_type"),
            forcefield_family=(family or {}).get("family"),
        )
        print(f"  ✅ {dispersion['dispersion_protocol']}"
              f"（炼金均匀密度 LRC "
              f"{'生效' if dispersion['uniform_density_lrc_active'] else '已关闭'}）")
    except Exception as exc:
        ok = _fail(f"{type(exc).__name__}: {exc}")

    # ---- OpenMM 兼容性：[ pairs ] funct 2 ----
    print("\n[7/8] OpenMM 拓扑兼容性（[ pairs ] funct 2）")
    try:
        needs = core.gromacs_topology_has_funct2_pairs(top)
        if not needs:
            print("  ✅ 无 funct-2 pairs，OpenMM 可直接读")
        else:
            # 干跑一次转换（写到临时目录），确认逐对等价校验能过。
            import tempfile, shutil
            tmp = tempfile.mkdtemp(prefix="pairs_funct2_check_")
            try:
                res = core.convert_gromacs_pairs_funct2(top, tmp)
                print(f"  ✅ 检测到 funct-2 pairs，等价转换可行："
                      f"{res['n_pairs_converted']} 条 / 改写 "
                      f"{[os.path.basename(f['source']) for f in res['patched_files']]}")
                print(f"     逐对已校验：fudgeQQ == 全局 {res['global_fudge_qq']}，"
                      "且 q1/q2 == [ atoms ] 真实电荷 → 哈密顿量不变")
                print("     正式跑时转换产物写到 output_dir/gromacs_openmm_compat/，"
                      "原始输入不动")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    except Exception as exc:
        ok = _fail(f"{type(exc).__name__}: {exc}")

    # ---- 溶剂腿水模型（必须与复合物腿一致）----
    print("\n[8/8] 溶剂腿水模型识别")
    try:
        xml, source = core.resolve_water_model_xml(top)
        how = "参数指纹" if str(source).startswith("parameter_match:") else "include 文件名"
        print(f"  ✅ {xml}（依据：{how}，来源 {source}）")
        if str(source).startswith("parameter_match:"):
            ident = core.identify_water_model_by_parameters(top)
            pp = ident["topology_params"]
            print(f"     拓扑水: {pp['moleculetype']} / {pp['n_sites']} 位点 / "
                  f"q_O={pp['charge_o_e']:+.6f} q_H={pp['charge_h_e']:+.6f} "
                  f"σ_O={pp['sigma_o_nm']:.9f} nm ε_O={pp['epsilon_o_kj_mol']:.6f} kJ/mol")
    except Exception as exc:
        ok = _fail(f"{type(exc).__name__}: {exc}")

    # ---- §3.3 声明完整性 + §9/§15 平衡时长 ----
    print("\n[附] 膜输入声明与平衡时长（§3.3 / §9 / §15）")
    if declaration is None:
        ok = _fail("配置里没有 membrane_input_declaration")
    else:
        missing = [
            f for f in core.MEMBRANE_INPUT_REQUIRED_PROVENANCE_FIELDS
            if not declaration.get(f)
        ]
        if missing:
            ok = _fail(f"声明缺必填字段（或值为 null/空）：{missing}")
        else:
            print("  ✅ §3.3 必填字段齐全")
        for field in ("source_structure_id", "conformational_state"):
            value = str(declaration.get(field) or "")
            if value.startswith("请填"):
                ok = _fail(f"{field} 还是占位文字：{value!r}")
        if str(declaration.get("conformational_state") or "").strip().lower() == (
            core.MEMBRANE_CONFORMATIONAL_STATE_UNSPECIFIED
        ):
            print("  ⚠️ conformational_state=unspecified：构象态未声明，已如实记录。"
                  "跨构象态比较前必须补上（本次运行的构象由输入指纹唯一确定）")
        upstream = declaration.get("upstream_equilibration_ns")
        status = str(declaration.get("upstream_equilibration_status") or "").strip().lower()
        if upstream is None and status == core.MEMBRANE_UPSTREAM_STATUS_COMPLETED_UNRECORDED:
            if declaration.get("final_equilibration_job"):
                print("  ✅ 上游生产已完成、时长不可考 → 标称时长预检不适用；"
                      "§9 实测质量门仍为硬门")
            else:
                ok = _fail("声明为 completed_length_unrecorded 时必须给 "
                           "final_equilibration_job 指向证据")
        elif upstream is None:
            ok = _fail("上游平衡未说明：给 upstream_equilibration_ns（正数），或声明 "
                       "upstream_equilibration_status=completed_length_unrecorded"
                       "（需同时给 final_equilibration_job）")
        else:
            own_ns = int(config.get("n_equil_steps", 5_000_000)) * float(
                config.get("timestep_ps", 0.002)
            ) / 1000.0
            total = float(upstream) + own_ns
            limit = core.MEMBRANE_MIN_EQUILIBRATION_NS
            if total >= limit:
                print(f"  ✅ 总平衡时长 {total:.1f} ns（上游 {upstream} + 本流程 {own_ns:.1f}）≥ {limit:.0f}")
            elif declaration.get("equilibration_shortfall_justification"):
                print(f"  ⚠️ 总平衡时长 {total:.1f} ns < {limit:.0f}，按声明理由放行；"
                      "§9 的实测质量门仍是硬门")
            else:
                ok = _fail(f"总平衡时长 {total:.1f} ns < {limit:.0f} ns，且没有 "
                           "equilibration_shortfall_justification")
        if declaration.get("literature_apl_nm2"):
            ok = _fail("本体系含跨膜蛋白，raw APL 不可比纯脂文献值——"
                       "设 literature_apl_nm2 会把正常体系判失败（见 README）")
        pocket = declaration.get("pocket_atom_indices") or []
        if not pocket:
            ok = _fail("pocket_atom_indices 为空，膜质量门跑不起来")
        else:
            print(f"  ✅ 口袋 {len(pocket)} 个重原子 / "
                  f"{len(declaration.get('pocket_residues') or [])} 个残基（已冻结）")

    print("\n" + ("✅ 协议自检全部通过，可以上节点。" if ok else "❌ 协议自检未通过，先修上面标 ❌ 的项。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
