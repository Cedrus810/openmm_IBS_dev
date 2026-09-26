

# ============================================================================
# 4. 幽灵离子 & 2D路径规划 & 替身系统构建器
# ============================================================================
class GhostIonHandler:
    def __init__(self, ghost_ion_distance=10.0, ghost_ion_scale_factor=1.0):
        self.ghost_ion_distance = ghost_ion_distance
        self.ghost_ion_scale_factor = ghost_ion_scale_factor

    def _resolve_ghost_anchor(self, box_vectors=None, reference_positions=None, ligand_indices=None):
        if box_vectors is None:
            return (
                float(self.ghost_ion_distance),
                0.0,
                0.0,
            )

        box_lengths = np.array(
            [
                np.linalg.norm(np.asarray(vec.value_in_unit(unit.nanometer) if hasattr(vec, "value_in_unit") else vec, dtype=float))
                for vec in box_vectors
            ],
            dtype=float,
        )
        box_lengths = np.where(box_lengths > 1.0e-6, box_lengths, 3.0)
        safe_margin = np.minimum(0.2, 0.1 * box_lengths)

        if reference_positions is not None and ligand_indices:
            lig_xyz = []
            for idx in ligand_indices:
                pos = reference_positions[idx]
                if hasattr(pos, "value_in_unit"):
                    lig_xyz.append(np.asarray(pos.value_in_unit(unit.nanometer), dtype=float))
                else:
                    lig_xyz.append(np.asarray(pos, dtype=float))
            if lig_xyz:
                lig_com = np.mean(np.asarray(lig_xyz, dtype=float), axis=0)
                anchor = np.mod(lig_com + 0.5 * box_lengths - safe_margin, box_lengths)
                return tuple(float(x) for x in anchor)

        anchor = 0.5 * box_lengths - safe_margin
        return tuple(float(x) for x in anchor)

    def create_ghost_ion_force(
        self,
        ligand_indices,
        ligand_charges,
        lambda_param="lam_coul",
        box_vectors=None,
        reference_positions=None,
    ):
        total = sum(ligand_charges)
        ghost = -total * self.ghost_ion_scale_factor
        if abs(ghost) < 1.0e-12 or not ligand_indices:
            return None

        xg, yg, zg = self._resolve_ghost_anchor(
            box_vectors=box_vectors,
            reference_positions=reference_positions,
            ligand_indices=ligand_indices,
        )
        force = openmm.CustomExternalForce(
            f"{lambda_param} * 138.935456 * ghost_charge * ligand_charge / "
            f"max(periodicdistance(x, y, z, ghost_x, ghost_y, ghost_z), 0.05)"
        )
        force.addGlobalParameter(lambda_param, 1.0)
        force.addGlobalParameter("ghost_charge", float(ghost))
        force.addGlobalParameter("ghost_x", float(xg))
        force.addGlobalParameter("ghost_y", float(yg))
        force.addGlobalParameter("ghost_z", float(zg))
        force.addPerParticleParameter("ligand_charge")
        for idx, charge in zip(ligand_indices, ligand_charges):
            force.addParticle(int(idx), [float(charge)])
        return force


class TwoDimensionalLambdaPathPlanner:
    def __init__(self, n_points=20, path_type="decoupling"):
        self.n_points = n_points
        self.path_type = path_type

    def generate_path(self):
        if self.path_type == "diagonal":
            return [
                (1.0 - i / self.n_points, 1.0 - i / self.n_points)
                for i in range(self.n_points + 1)
            ]
        elif self.path_type == "decoupling":
            n_half = self.n_points // 2
            lambdas = []
            for i in range(n_half + 1):
                lambdas.append((1.0 - i / n_half, 1.0))
            for i in range(1, self.n_points - n_half + 1):
                lambdas.append((0.0, 1.0 - i / (self.n_points - n_half)))
            return lambdas
        else:
            return [(1.0, 1.0 - i / self.n_points) for i in range(self.n_points + 1)]


class SurrogateSystemBuilder:
    def __init__(
        self,
        surrogate_params,
        ghost_handler=None,
        sigma_gauss_nm: float = GAUSS_COUL_SIGMA_NM,
    ):
        self.surrogate_potential = DEXPSurrogatePotential.from_dict(
            surrogate_params or {}
        )
        self.ghost_handler = ghost_handler
        self.sigma_gauss_nm = float(sigma_gauss_nm)

    def build_surrogate_system(
        self,
        original_system,
        ligand_indices,
        environment_indices,
        lambda_names=("lam_coul", "lam_vdw"),
        force_group=1,
        reference_positions=None,
        box_vectors=None,
    ):
        # 必须强制深拷贝：ensure_owned_system 在 thisown==1 时会原样返回同一个对象
        # （XmlSerializer.deserialize 出来的 System 默认就是 thisown==1），如果不
        # 先 serialize/deserialize 一次，下面对 nb_force 的原地修改和 addForce 会
        # 直接污染调用者传进来的 original_system —— 这会导致外部同时持有的“原始
        # 力场”引用实际上已经被替换成了这个 surrogate system。
        new_system = ensure_owned_system(
            XmlSerializer.clone(original_system)
        )
        resolved_box_vectors = (
            box_vectors
            if box_vectors is not None
            else new_system.getDefaultPeriodicBoxVectors()
        )
        _validate_minimum_image(resolved_box_vectors, DEXP_VDW_CUTOFF_NM)
        nb_force = next(
            (f for f in new_system.getForces() if isinstance(f, openmm.NonbondedForce)),
            None,
        )
        if not nb_force:
            raise ValueError("未找到 NonbondedForce")

        lig_set = {int(idx) for idx in ligand_indices}
        env_set = {int(idx) for idx in environment_indices if int(idx) not in lig_set}
        if not lig_set:
            raise ValueError("ligand_indices 为空，无法构建 surrogate decoupling system")
        if not env_set:
            raise ValueError("environment_indices 为空，无法构建 ligand-environment surrogate force")
        original_params = [
            nb_force.getParticleParameters(i) for i in range(new_system.getNumParticles())
        ]
        reference_exclusions = []
        for i in range(nb_force.getNumExceptions()):
            p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
            reference_exclusions.append((int(p1), int(p2)))

        # 1) 先恢复 ligand-ligand 内部 nonbonded / 1-4，保留原始 MM 内部拓扑语义。
        ll_force, ll_14_force = create_ligand_internal_force(
            nb_force=nb_force,
            perturbed_indices=sorted(lig_set),
            particle_params=original_params,
            reference_exclusions=reference_exclusions,
            num_particles=new_system.getNumParticles(),
            system=new_system,
        )
        ll_force.setForceGroup(force_group)
        new_system.addForce(ll_force)
        if ll_14_force is not None:
            ll_14_force.setForceGroup(force_group)
            new_system.addForce(ll_14_force)

        # 2) 主 NonbondedForce 中将 ligand 完全去耦，避免原始 MM L-E 项始终全开。
        for idx in sorted(lig_set):
            q, sigma, epsilon = original_params[idx]
            nb_force.setParticleParameters(
                idx,
                0.0 * unit.elementary_charge,
                sigma,
                0.0 * unit.kilojoule_per_mole,
            )
        for exc_idx in range(nb_force.getNumExceptions()):
            p1, p2, _, sigma, _ = nb_force.getExceptionParameters(exc_idx)
            p1, p2 = int(p1), int(p2)
            if p1 in lig_set or p2 in lig_set:
                nb_force.setExceptionParameters(
                    exc_idx,
                    p1,
                    p2,
                    0.0 * unit.elementary_charge * unit.elementary_charge,
                    sigma,
                    0.0 * unit.kilojoule_per_mole,
                )

        # 3) L-E Gaussian electrostatics：用平滑库仑核替代点电荷奇点，并与 DEXP 共用
        #    0.50~0.70 nm 的 switching/cutoff 缝合区。这样 0.45~0.50 nm 仍由
        #    MACE/DEXP 核心描述区承担，switch shell 只作为 surrogate 平滑退出区，
        #    不应当被当作核心近程
        #    势能面/RDF/PMF 判据区解释。
        sigma_gauss_nm = max(float(self.sigma_gauss_nm), 1.0e-6)
        gamma_eff = 1.0 / max(math.sqrt(2.0) * sigma_gauss_nm, 1.0e-6)
        # 修复 switch 伪影：erf(γr)/r 在 0.5~0.7 nm 仍是 ~1/r 长程尾巴，用能量 switching 截断会
        # 引入 -S'(r)·U(r) 假力，在 cutoff 内侧堆出假的 RDF 峰（~0.63 nm）。改用 shifted-force：
        # U_sf(r)=U(r)-U(rc)-(r-rc)U'(rc)，使势与力在 cutoff 处都连续归零，不再需要 switching。
        rc_nm = GAUSS_COUL_CUTOFF_NM
        g = float(gamma_eff)
        fc = math.erf(g * rc_nm) / rc_nm                                  # U(rc)/(k q1 q2)
        fpc = (2.0 * g / math.sqrt(math.pi)) * math.exp(-(g * rc_nm) ** 2) / rc_nm \
            - math.erf(g * rc_nm) / rc_nm ** 2                            # U'(rc)/(k q1 q2)
        gauss_expr = (
            f"{lambda_names[0]} * 138.935456*q1*q2*("
            f"erf({g}*r_safe)/r_safe - ({fc}) - (r_safe - {rc_nm})*({fpc})"
            "); r_safe = max(r, 1e-6)"
        )
        coul_force = openmm.CustomNonbondedForce(gauss_expr)
        coul_force.addPerParticleParameter("q")
        coul_force.addGlobalParameter(lambda_names[0], 1.0)
        for i in range(new_system.getNumParticles()):
            q, _, _ = original_params[i]
            coul_force.addParticle([q.value_in_unit(unit.elementary_charge)])
        coul_force.addInteractionGroup(sorted(lig_set), sorted(env_set))
        coul_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        coul_force.setCutoffDistance(GAUSS_COUL_CUTOFF_NM * unit.nanometer)
        # shifted-force 已保证 cutoff 处势/力归零，关闭 switching（否则又引入 -S'·U 假力）
        coul_force.setUseSwitchingFunction(False)
        coul_force.setForceGroup(force_group)
        for p1, p2 in reference_exclusions:
            coul_force.addExclusion(int(p1), int(p2))
        new_system.addForce(coul_force)

        # 4) L-E DEXP：短程排斥/色散替身，与 Gaussian electrostatics 拼成完整 surrogate。
        dexp_expr = (
            f"{self.surrogate_potential.build_expression(lam_vdw=lambda_names[1])}"
        )
        dexp_force = openmm.CustomNonbondedForce(dexp_expr)
        dexp_force.addGlobalParameter(lambda_names[1], 1.0)
        dexp_force.addPerParticleParameter("sigma")
        dexp_force.addPerParticleParameter("epsilon")
        for i in range(new_system.getNumParticles()):
            # 用去耦前捕获的 original_params，而不是已被步骤2清零 epsilon 的 nb_force：
            # DEXP 核需要 ligand/environment 双方各自真实的原始 LJ sigma/epsilon 才能
            # 按组合律解析出 pair-specific r0_ij/eps_ij，不能沿用去耦后的占位值。
            _, sigma_i, epsilon_i = original_params[i]
            dexp_force.addParticle([
                sigma_i.value_in_unit(unit.nanometer),
                epsilon_i.value_in_unit(unit.kilojoule_per_mole),
            ])
        dexp_force.addInteractionGroup(sorted(lig_set), sorted(env_set))
        dexp_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        dexp_force.setCutoffDistance(DEXP_VDW_CUTOFF_NM * unit.nanometer)
        dexp_force.setUseSwitchingFunction(True)
        dexp_force.setSwitchingDistance(
            (DEXP_VDW_CUTOFF_NM - DEXP_VDW_SWITCH_WIDTH_NM)
            * unit.nanometer
        )
        dexp_force.setForceGroup(force_group)

        for p1, p2 in reference_exclusions:
            dexp_force.addExclusion(int(p1), int(p2))

        new_system.addForce(dexp_force)
        sync_all_exclusions(new_system)

        if self.ghost_handler and ligand_indices:
            charges = [
                original_params[i][0].value_in_unit(unit.elementary_charge)
                for i in sorted(ligand_indices)
            ]
            ghost_force = self.ghost_handler.create_ghost_ion_force(
                ligand_indices=sorted(ligand_indices),
                ligand_charges=charges,
                lambda_param=lambda_names[0],
                box_vectors=box_vectors if box_vectors is not None else new_system.getDefaultPeriodicBoxVectors(),
                reference_positions=reference_positions,
            )
            if ghost_force is not None:
                new_system.addForce(ghost_force)
        return new_system


# ============================================================================
# 5. Orb 扫描器 & 混合工厂 & Surrogate 流水线
# ============================================================================

class OrbMMHybridFactory:
    def get_rotatable_torsions_rdkit(self, mol):
        import rdkit.Chem as Chem

        pattern = Chem.MolFromSmarts("[!#1;!D1]-[!#1;!D1]")
        if pattern is None:
            return []
        torsions = []
        for bond in mol.GetSubstructMatches(pattern):
            a1, a2 = bond[0], bond[1]
            for n1 in [x.GetIdx() for x in mol.GetAtomWithIdx(a1).GetNeighbors()]:
                if n1 != a2:
                    for n2 in [
                        x.GetIdx() for x in mol.GetAtomWithIdx(a2).GetNeighbors()
                    ]:
                        if n2 != a1 and mol.GetBondBetweenAtoms(a1, a2) is not None:
                            torsions.append((n1, a1, a2, n2))
        return torsions






# ============================================================================
# 6. 势能路由工厂 & 幽灵离子快捷函数
# ============================================================================
class AlchemicalPotentialFactory:
    @staticmethod
    def build(potential_type, params, lam_coul, lam_vdw):
        if isinstance(params, (ACESoftcorePotential, DEXPSurrogatePotential)):
            obj = params
        elif potential_type == "dexp":
            obj = DEXPSurrogatePotential.from_dict(params or {})
        elif potential_type == "softcore":  # ✅ 显式识别
            obj = ACESoftcorePotential.from_dict(params or {})
        else:
            obj = ACESoftcorePotential.from_dict(params or {})
        if isinstance(obj, DEXPSurrogatePotential):
            return obj.build_expression(lam_vdw=lam_vdw), obj.get_parameters_dict()
        return obj.build_expression(lam_coul, lam_vdw), obj.get_parameters_dict()

def create_ghost_ion_force(
    lig_indices,
    lig_charges,
    lam_param="lam_coul",
    dist=10.0,
    scale=0.1,
    box_vectors=None,
    reference_positions=None,
):
    return GhostIonHandler(dist, scale).create_ghost_ion_force(
        ligand_indices=lig_indices,
        ligand_charges=lig_charges,
        lambda_param=lam_param,
        box_vectors=box_vectors,
        reference_positions=reference_positions,
    )
