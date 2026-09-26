#=============================================================================
# 修复 5: 溶剂化能闭环支持 (Ligand-in-Water)
#=============================================================================
# 与溶剂腿 createSystem 的 nonbondedCutoff 保持一致；盒长校验要用它。
SOLVENT_NONBONDED_CUTOFF_NM = 1.0

# GROMACS 水模型 itp 名 → 对应的 OpenMM 水模型 XML。
# 这张表只用来"翻译"复合物腿实际用的那个水模型，不是可选项列表：认不出来就
# fail closed，绝不回退到某个默认值。
GMX_TO_OPENMM_WATER_XML = {
    "tip3p": "amber14/tip3p.xml",
    "tip3pfb": "amber14/tip3pfb.xml",
    "tip4pew": "amber14/tip4pew.xml",
    "tip4pfb": "amber14/tip4pfb.xml",
    "spce": "amber14/spce.xml",
    "opc": "amber14/opc.xml",
    "opc3": "amber14/opc3.xml",
}


# 参数匹配水模型时的容差。O 的 σ/ε 与电荷在各模型间的差别远大于此
# （例如 TIP3P σ=0.315075 vs SPC/E σ=0.316572），所以这个容差既能吸收
# 力场文件与 XML 的有效位差异，又不会把两个模型混为一谈。
WATER_MODEL_MATCH_CHARGE_TOLERANCE_E = 1.0e-4
WATER_MODEL_MATCH_SIGMA_TOLERANCE_NM = 1.0e-6
WATER_MODEL_MATCH_EPSILON_TOLERANCE_KJ_MOL = 1.0e-4


def water_model_parameters_from_topology(
    parsed: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """从已解析的 GROMACS 拓扑里取出水的 O/H 电荷与 O 的 σ/ε。

    找不到水 moleculetype 或缺 atomtype 参数时返回 None（由调用方 fail closed）。
    """
    water_name = next(
        (
            name
            for name in parsed["moleculetypes"]
            if str(name).strip().upper() in WATER_MOLECULE_NAMES
        ),
        None,
    )
    if water_name is None:
        return None
    atoms = parsed["moleculetypes"][water_name].get("atoms") or []
    atomtypes = parsed.get("atomtypes") or {}

    oxygen = next(
        (a for a in atoms if str(a.get("atom_name", "")).strip().upper().startswith("O")),
        None,
    )
    hydrogens = [
        a for a in atoms if str(a.get("atom_name", "")).strip().upper().startswith("H")
    ]
    if oxygen is None or not hydrogens:
        return None
    o_type = atomtypes.get(oxygen["type"])
    if o_type is None:
        return None
    return {
        "moleculetype": water_name,
        "n_sites": len(atoms),
        "charge_o_e": float(oxygen["charge"]),
        "charge_h_e": float(hydrogens[0]["charge"]),
        "sigma_o_nm": float(o_type["sigma_nm"]),
        "epsilon_o_kj_mol": float(o_type["epsilon_kj_mol"]),
    }


def openmm_water_model_parameters(xml_relative_path: str) -> Optional[Dict[str, Any]]:
    """从 OpenMM 自带的水模型 XML 里取同一组参数，用于与拓扑比对。

    刻意**读 XML 而不是硬编码一张参数表**：这样"选出的 XML"与"比对用的参数"
    构造性地来自同一处，不会出现表抄错或版本漂移。
    """
    import xml.etree.ElementTree as ET

    data_dir = os.path.join(os.path.dirname(app.__file__), "data")
    path = os.path.join(data_dir, xml_relative_path)
    if not os.path.isfile(path):
        return None
    root = ET.parse(path).getroot()

    residue = next(
        (r for r in root.iter("Residue") if r.get("name") in ("HOH", "WAT")), None
    )
    if residue is None:
        return None
    site_types: Dict[str, str] = {}
    charges: Dict[str, float] = {}
    for atom in residue:
        name, atom_type, charge = atom.get("name"), atom.get("type"), atom.get("charge")
        if name is None or atom_type is None or charge is None:
            continue
        site_types[name] = atom_type
        charges[name] = float(charge)
    if not site_types:
        return None

    sigma_epsilon: Dict[str, Tuple[float, float]] = {}
    for atom in root.iter("Atom"):
        atom_type = atom.get("type")
        sigma, epsilon = atom.get("sigma"), atom.get("epsilon")
        if atom_type in set(site_types.values()) and sigma is not None and epsilon is not None:
            sigma_epsilon[atom_type] = (float(sigma), float(epsilon))

    oxygen_name = next((n for n in site_types if n.upper().startswith("O")), None)
    hydrogen_name = next((n for n in site_types if n.upper().startswith("H")), None)
    if oxygen_name is None or hydrogen_name is None:
        return None
    o_params = sigma_epsilon.get(site_types[oxygen_name])
    if o_params is None:
        return None
    return {
        "xml": xml_relative_path,
        "n_sites": len(site_types),
        "charge_o_e": charges[oxygen_name],
        "charge_h_e": charges[hydrogen_name],
        "sigma_o_nm": o_params[0],
        "epsilon_o_kj_mol": o_params[1],
    }


def _water_models_match(topology_params, xml_params) -> bool:
    return (
        int(topology_params["n_sites"]) == int(xml_params["n_sites"])
        and abs(topology_params["charge_o_e"] - xml_params["charge_o_e"])
        <= WATER_MODEL_MATCH_CHARGE_TOLERANCE_E
        and abs(topology_params["charge_h_e"] - xml_params["charge_h_e"])
        <= WATER_MODEL_MATCH_CHARGE_TOLERANCE_E
        and abs(topology_params["sigma_o_nm"] - xml_params["sigma_o_nm"])
        <= WATER_MODEL_MATCH_SIGMA_TOLERANCE_NM
        and abs(topology_params["epsilon_o_kj_mol"] - xml_params["epsilon_o_kj_mol"])
        <= WATER_MODEL_MATCH_EPSILON_TOLERANCE_KJ_MOL
    )


def identify_water_model_by_parameters(
    top_file: str,
    gmx_include_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """按**实际参数**（O/H 电荷 + O 的 σ/ε + 位点数）识别水模型。

    为什么需要：原实现只看 `#include` 的文件名词干。那对
    `amber14sb_OL15_fs1.ff/tip3p.itp` 有效，但 CHARMM-GUI 的 AMBER 转换器把
    TIP3P 的水叫 **`TP3`**（`toppar/TP3.itp`），文件名匹配直接 fail closed。

    这是同一类根因的第三次出现（前两次是脂质按残基计数、水/离子计数静默为 0）：
    **靠名字判身份在换一套体系时就会错**。参数是权威的，且比对用的候选参数直接
    从 OpenMM 自带 XML 读出，不硬编码。

    返回 `{"xml", "matched", "topology_params", "candidates"}`；
    `matched=False` 时由调用方决定报错措辞。
    """
    parsed = parse_gromacs_topology(top_file, gmx_include_dir)
    topology_params = water_model_parameters_from_topology(parsed)
    if topology_params is None:
        return {
            "xml": None,
            "matched": False,
            "reason": "topology_water_parameters_unavailable",
            "topology_params": None,
            "candidates": {},
        }

    candidates: Dict[str, Any] = {}
    matches: List[str] = []
    for key, xml_relative_path in GMX_TO_OPENMM_WATER_XML.items():
        xml_params = openmm_water_model_parameters(xml_relative_path)
        candidates[key] = xml_params
        if xml_params and _water_models_match(topology_params, xml_params):
            matches.append(key)

    if len(matches) == 1:
        return {
            "xml": GMX_TO_OPENMM_WATER_XML[matches[0]],
            "matched": True,
            "reason": "parameter_fingerprint",
            "model_key": matches[0],
            "topology_params": topology_params,
            "candidates": candidates,
        }
    return {
        "xml": None,
        "matched": False,
        "reason": (
            f"ambiguous_parameter_match:{sorted(matches)}"
            if matches
            else "no_parameter_match"
        ),
        "topology_params": topology_params,
        "candidates": candidates,
    }


def resolve_water_model_xml(top_file: str) -> Tuple[str, str]:
    """从复合物 ``.top`` 的 ``#include`` 里解出水模型，返回 ``(OpenMM XML, 命中的 itp)``。

    溶剂腿的水必须和复合物腿是同一个模型。以前这里两边是各自硬编码的
    （复合物走 GROMACS ``amber14sb_OL15_fs1.ff/tip3p.itp``，而
    ``SolventLegRunner`` 写死 ``amber14/tip3pfb.xml``），TIP3P 与 TIP3P-FB 的
    σ/ε/电荷都不同，循环里本该抵消的水化项就对不上了。现在一律从复合物
    ``.top`` 反推，认不出来直接报错而不是默认成 TIP3P。
    """
    if not top_file or not os.path.isfile(top_file):
        raise FileNotFoundError(f"解析水模型需要有效的复合物 .top：{top_file!r}")
    with open(top_file, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    hits: Dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line.startswith("#include"):
            continue
        rest = line[len("#include"):].strip()
        if len(rest) < 2 or rest[0] not in '"<':
            continue
        closing = '"' if rest[0] == '"' else ">"
        end = rest.find(closing, 1)
        if end < 0:
            continue
        include_path = rest[1:end]
        stem = os.path.splitext(os.path.basename(include_path))[0]
        key = stem.lower().replace("-", "").replace("_", "")
        if key in GMX_TO_OPENMM_WATER_XML:
            hits[key] = include_path

    if not hits:
        # 文件名词干认不出来时，按**实际参数**识别（O/H 电荷 + O 的 σ/ε + 位点数）。
        # 实测理由：CHARMM-GUI 的 AMBER 转换器把 TIP3P 的水命名为 `TP3`
        # （`toppar/TP3.itp`），文件名匹配必然落空，但它的参数
        # （q_O=-0.834, q_H=+0.417, σ_O=0.315075240658 nm, ε_O=0.635968 kJ/mol）
        # 与 `amber14/tip3p.xml` 逐位吻合。参数是权威的，名字不是。
        identification = identify_water_model_by_parameters(top_file)
        if identification["matched"]:
            params = identification["topology_params"]
            logger.info(
                "水模型按参数识别为 %s（moleculetype=%s, %d 位点, "
                "q_O=%+.6f, q_H=%+.6f, σ_O=%.9f nm, ε_O=%.6f kJ/mol）——"
                "`#include` 文件名词干认不出，故用参数判定。",
                identification["xml"], params["moleculetype"], params["n_sites"],
                params["charge_o_e"], params["charge_h_e"],
                params["sigma_o_nm"], params["epsilon_o_kj_mol"],
            )
            return identification["xml"], f"parameter_match:{params['moleculetype']}"
        detail = ""
        if identification["topology_params"]:
            p = identification["topology_params"]
            detail = (
                f" 拓扑里的水是 moleculetype={p['moleculetype']!r}、{p['n_sites']} 位点、"
                f"q_O={p['charge_o_e']:+.6f}, q_H={p['charge_h_e']:+.6f}, "
                f"σ_O={p['sigma_o_nm']:.9f} nm, ε_O={p['epsilon_o_kj_mol']:.6f} kJ/mol，"
                f"与已知模型都不匹配（{identification['reason']}）。"
            )
        raise ValueError(
            f"在 {top_file} 的 #include 里没认出任何水模型，按参数也没匹配上；"
            f"已知的有 {sorted(GMX_TO_OPENMM_WATER_XML)}。{detail}"
            "拒绝为溶剂腿猜一个水模型——它必须和复合物腿一致。"
        )
    if len(hits) > 1:
        raise ValueError(
            f"{top_file} 同时 include 了多个水模型 {sorted(hits)}，无法确定复合物腿用的是哪个"
        )
    key, include_path = next(iter(hits.items()))
    return GMX_TO_OPENMM_WATER_XML[key], include_path


def solvent_box_edge_nm(
    lig_coords_nm,
    padding_nm: float,
    cutoff_nm: float = SOLVENT_NONBONDED_CUTOFF_NM,
) -> Tuple[float, float]:
    """按 ``gmx editconf -d`` 语义算立方溶剂盒边长：配体最长轴 + 2*padding。

    绝不依赖 ``addSolvent(padding=...)`` 自己推盒子——那条路径在本仓库产出过
    ``box = 2*padding`` 的 3.000 nm 立方盒（溶质尺寸对盒长的贡献是 0），配体
    最长轴 1.257 nm，每侧只剩 0.87 nm 溶剂；而且 OpenMM 7.7+ 的 padding 分支
    默认给的是菱形十二面体，不是立方。

    返回 ``(盒边 nm, 配体最长轴 nm)``。
    """
    coords = np.asarray(lig_coords_nm, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.all(np.isfinite(coords)):
        raise ValueError("配体坐标必须是有限的 (N, 3) nm 数组")
    padding = float(padding_nm)
    if not np.isfinite(padding) or padding <= 0.0:
        raise ValueError(f"padding_nm 必须是有限正数，收到 {padding_nm!r}")
    extent_nm = float(np.max(coords.max(axis=0) - coords.min(axis=0)))
    if not np.isfinite(extent_nm) or extent_nm <= 0.0:
        raise ValueError(f"配体最长轴计算异常: {extent_nm}")
    edge_nm = extent_nm + 2.0 * padding
    min_image_floor = 2.0 * float(cutoff_nm)
    if edge_nm <= min_image_floor:
        raise ValueError(
            f"溶剂盒边长 {edge_nm:.4f} nm 不满足最小镜像约定"
            f"（需 > 2×cutoff = {min_image_floor:.4f} nm）"
        )
    return edge_nm, extent_nm


class SolventLegRunner:
    """自动构建并运行 Ligand-in-Water 解耦腿"""
    def __init__(
        self,
        ligand_resname: str,
        box_size_nm: Optional[float] = None,
        platform_name: str = "CUDA",
        padding_nm: float = 1.5,
    ):
        self.ligand_resname = ligand_resname
        if box_size_nm is not None:
            warnings.warn(
                "SolventLegRunner.box_size_nm 已废弃且不再作为完整盒边；"
                "请使用 padding_nm 指定配体每侧的溶剂厚度。",
                DeprecationWarning,
                stacklevel=2,
            )
        self.padding_nm = float(padding_nm)
        if not np.isfinite(self.padding_nm) or self.padding_nm <= 0.0:
            raise ValueError("padding_nm 必须是有限正数")
        self.platform_name = platform_name
        self._cached_system = None
        self._cached_topology = None
        self._cached_positions = None
        self._cached_box_vectors = None

    def build_solvent_system(self, gro_file: str, top_file: str, gmx_include_dir: str = None):
        """从 GROMACS 文件提取配体并水盒化"""
        from openmm.app import Modeller, ForceField
        gro = app.GromacsGroFile(gro_file)
        # 走唯一入口：拓扑若含 `[ pairs ]` funct 2（OpenMM 不支持）会先做等价转换。
        top = load_gromacs_topology_for_openmm(top_file, includeDir=gmx_include_dir)
        modeller = Modeller(top.topology, gro.positions)
        
        lig_indices = [atom.index for atom in gro.topology.atoms() if atom.residue.name == self.ligand_resname]
        if not lig_indices:
            raise ValueError(f"未在 GRO 中找到配体残基 {self.ligand_resname}")

        # 🔑 水模型不再硬编码，一律从复合物 .top 反推，保证两腿同模型。
        water_xml, water_itp = resolve_water_model_xml(top_file)
        logger.info("  溶剂腿水模型继承自复合物 .top: %s → %s", water_itp, water_xml)
        ff = ForceField("amber14-all.xml", water_xml)

        # 🔑 盒子显式给出，不用 padding=。理由见 solvent_box_edge_nm 的 docstring。
        pos_nm = np.asarray(
            gro.positions.value_in_unit(unit.nanometer), dtype=np.float64
        )
        box_edge_nm, lig_extent_nm = solvent_box_edge_nm(
            pos_nm[lig_indices], padding_nm=self.padding_nm
        )
        modeller.addSolvent(
            ff,
            boxSize=openmm.Vec3(box_edge_nm, box_edge_nm, box_edge_nm) * unit.nanometer,
        )

        # fail closed：确认 OpenMM 真的按我们给的盒子建，而不是又自己推了一个。
        realized_vecs = modeller.topology.getPeriodicBoxVectors()
        if realized_vecs is None:
            raise RuntimeError("addSolvent 之后拓扑没有周期盒向量")
        realized_edges = np.linalg.norm(
            np.array([v.value_in_unit(unit.nanometer) for v in realized_vecs], dtype=float),
            axis=1,
        )
        if not np.allclose(realized_edges, box_edge_nm, atol=1.0e-6):
            raise RuntimeError(
                f"溶剂盒构建结果与请求不符：请求 {box_edge_nm:.6f} nm 立方"
                f"（配体最长轴 {lig_extent_nm:.4f} nm + 2×{self.padding_nm:.2f} nm），"
                f"实际边长 {[round(float(x), 6) for x in realized_edges]} nm"
            )

        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=SOLVENT_NONBONDED_CUTOFF_NM * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
        )
        # ✅ 缓存构建结果
        self._cached_system = system
        self._cached_topology = modeller.topology
        self._cached_positions = modeller.positions
        self._cached_box_vectors = modeller.topology.getPeriodicBoxVectors()
        return self._cached_system, self._cached_topology, self._cached_positions, self._cached_box_vectors
        
    def run_solvent_decoupling(self, positions, topology, ligand_indices, **pipeline_kwargs):
        """运行溶剂相解耦计算（委托给 ABFEPipeline）"""
        from abfe_pipeline import ABFEPipeline
        if self._cached_system is None:
            raise RuntimeError("请先调用 build_solvent_system 构建溶剂系统")
        
        # 🔑 [0831issue P2] repeat-seed contract 是**构造参数**（ABFEPipeline 在
        # __init__ 里据此建 Exp019SeedLedger），不是 run_full_pipeline 的参数。
        # 调用方（run_full_abfe_loop）把它放在 pipeline_kwargs 里传过来，所以这里
        # 必须 pop 出来交给构造函数，否则会以未知 kwarg 撞进 run_full_pipeline。
        pipeline_kwargs = dict(pipeline_kwargs)
        _repeat_seed = pipeline_kwargs.pop("repeat_seed", None)
        _leg_name = pipeline_kwargs.pop("leg_name", None)
        pipe = ABFEPipeline(
            system=self._cached_system,          # ✅ 修复：传入有效 system
            topology=self._cached_topology,
            positions=self._cached_positions,
            box_vectors=self._cached_box_vectors,
            ligand_indices=ligand_indices,
            temperature=pipeline_kwargs.get('temperature', 300.0),
            output_dir=pipeline_kwargs.get('output_dir', './solvent_output'),
            platform_name=self.platform_name,
            charge_treatment=pipeline_kwargs.get("charge_treatment"),
            charge_transfer_reservoir_correction=pipeline_kwargs.get(
                "charge_transfer_reservoir_correction"
            ),
            repeat_seed=_repeat_seed,
            leg_name=_leg_name,
        )
        # ✅ 移除 decoupling_scheme 硬编码，透传用户配置
        return pipe.run_full_pipeline(**pipeline_kwargs)
