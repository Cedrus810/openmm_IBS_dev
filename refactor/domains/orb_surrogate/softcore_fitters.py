# ============================================================================
# 1. ACES 软核势 & DEXP 替身势 & Orb 拟合器
# ============================================================================
class ACESoftcorePotential:
    def __init__(
        self, alpha_lj=0.5, alpha_coul=0.2, power_lj=(2, 2), power_coul=(1, 1)
    ):
        self.alpha_lj, self.alpha_coul = float(alpha_lj), float(alpha_coul)
        self.m_lj, self.n_lj = power_lj
        self.m_coul, self.n_coul = power_coul

    def build_expression(self, lam_coul, lam_vdw):
        COUL = 138.935456
        lc, lv = f"({lam_coul}^{self.n_coul})", f"({lam_vdw}^{self.n_lj})"

        # 必须整体加括号，否则会被解析成 0.5*(sigma1+sigma2)^n，
        # 而不是 ((sigma1+sigma2)/2)^n，短程排斥会被严重放大。
        sigma12 = "(0.5*(sigma1+sigma2))"

        # 🔑 [SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2] alpha_lj /
        # alpha_coul 是【无量纲】Beutler 系数，必须乘 sigma_ij^6 / sigma_ij^2 才构成
        # 长度^6 / 长度^2 的软核偏移（openmmtools 等实现用约化距离 (r/sigma)，等价于
        # 隐式乘了 sigma^6，所以文献里的 alpha=0.5 是无量纲的）。
        #
        # 之前这里把 alpha_lj 当成【绝对 nm^6】直接加在 r^6 上。sigma_ij≈0.3 nm ⇒
        # sigma^6≈7.3e-4 nm^6，0.5 nm^6 相当于 ~685 sigma^6：软核偏移在
        # (1-lambda)=0.038（即 lambda≈0.962）处就已越过 sigma^6，于是
        #   lambda>0.962  → 偏移 ≪ sigma^6，几乎是硬 LJ 核；
        #   lambda<0.962  → 偏移 ≫ sigma^6，相互作用被整体抹平。
        # 硬核→全软的整个过程被压缩进 lambda_vdw∈[0.96,1]，实测 Fisher 度规在该段
        # 峰值 ~2e6，vanishing 路径 87.5% 的热力学长度堆在那 4 条边上（单边长 8.8~11.8，
        # 其余 18 条边合计仅 5.9），窗口 0 因此零重叠、IBS 占据退化成硬 argmax。
        #
        # 两个端点与 alpha 无关（整体乘 lambda_vdw：lambda=0 恒为 0；lambda=1 偏移恒为
        # 0 即精确 LJ），所以这条修正不改变精确平衡态 ΔG，只改变路径效率——但采样结果、
        # LRC 与所有旧缓存都属于旧哈密顿量，不能与新路径混用（见
        # get_parameters_dict 里写入指纹的 alpha_convention）。
        #
        # 1e-6 兜底只在 r<0.1 nm（r^6<1e-6）时才生效，是 NaN 防护；该处真实 LJ 同样
        # 是 ~1e6 kJ/mol 量级的排斥墙，不污染正常物理区间。
        dlj = (
            f"max({self.alpha_lj}*{sigma12}^6*(1.0-{lam_vdw})^{self.m_lj} + r^6, 1e-6)"
        )
        dc = (
            f"sqrt(max(r^2 + {self.alpha_coul}*{sigma12}^2"
            f"*(1.0-{lam_coul})^{self.m_coul}, 1e-6))"
        )
        lj = f"{lv} * 4 * sqrt(epsilon1*epsilon2) * ({sigma12}^12/({dlj}^2) - {sigma12}^6/{dlj})"
        coul = f"{lc} * {COUL} * q1 * q2 / {dc}"
        
        return f"{lj} + {coul}"

    # 🔑 软核 alpha 语义标签。写进 get_parameters_dict() → 进入协议指纹
    # (aces_softcore_params)，使得旧的「绝对 nm^6 alpha」缓存与本版本自动指纹不匹配、
    # fail closed。alpha 的【数值】没变（仍是 0.5/0.3），变的是它在表达式里乘不乘
    # sigma_ij^6 / sigma_ij^2，因此必须靠这个显式标签来区分，不能靠数值。
    ALPHA_CONVENTION = "dimensionless_sigma_scaled_v2"

    @staticmethod
    def _normalize_alpha_units(alpha_lj, alpha_coul):
        """alpha 是无量纲 Beutler 系数，缩放在 build_expression 里由 sigma_ij 完成，
        这里不做任何单位换算（保留此钩子是为了让「不换算」这件事显式可见）。"""
        return float(alpha_lj), float(alpha_coul)

    @staticmethod
    def optimize_alpha(n, alpha_coul_nm2=None):
        """返回无量纲 Beutler 软核系数。

        ⚠️ 历史教训：这里曾写着「OpenMM 软核 alpha 标准单位即为 nm⁶/nm²，文献值 0.5
        已对应 nm 尺度，无需额外乘以 1e-6/1e-2」——推理方向是反的。openmmtools 一类
        实现用的是约化距离 (r/sigma)，alpha 隐式乘了 sigma^6，所以文献值 0.5 恰恰是
        【无量纲】的。把它当绝对 nm^6 用，等于把软核偏移放大了 ~685 倍（sigma≈0.3 nm），
        后果见 build_expression 的注释。缩放现在由 build_expression 乘 sigma_ij^6 /
        sigma_ij^2 完成，这里只返回无量纲系数本身。
        """
        if n > 50:
            alpha_lj, alpha_coul = 0.5, (alpha_coul_nm2 or 0.2)
        else:
            alpha_lj, alpha_coul = 0.5, (alpha_coul_nm2 or 0.3)

        # 无量纲区间：文献常用 alpha_lj=0.5、alpha_coul=0.2~0.3。
        # 用 raise 而不是 assert——assert 在 `python -O`/`PYTHONOPTIMIZE` 下会被整段
        # 剥掉，届时这里就变成对 alpha_coul_nm2 零校验，一个荒谬的调用方传入值
        # （负数/巨大/NaN）会直接流进软核 Hamiltonian，静默产生错误采样却不报错。
        if not (0.1 < alpha_lj < 2.0):
            raise ValueError(f"alpha_lj 超出安全范围: {alpha_lj} (无量纲，预期 0.1~2.0)")
        if not (0.05 < alpha_coul < 1.0):
            raise ValueError(f"alpha_coul 超出安全范围: {alpha_coul} (无量纲，预期 0.05~1.0)")

        return {
            "alpha_lj": alpha_lj,        # 无量纲，表达式内乘 sigma_ij^6
            "alpha_coul": alpha_coul,    # 无量纲，表达式内乘 sigma_ij^2
            "power_lj": [2, 2],
            "power_coul": [1, 1],
            "alpha_convention": ACESoftcorePotential.ALPHA_CONVENTION,
        }

    def get_parameters_dict(self):
        return {
            "alpha_lj": self.alpha_lj,
            "alpha_coul": self.alpha_coul,
            "power_lj": list([self.m_lj, self.n_lj]),
            "power_coul": list([self.m_coul, self.n_coul]),
            "alpha_convention": self.ALPHA_CONVENTION,
        }

    @classmethod
    def from_dict(cls, p):
        # fail closed：显式给了 alpha 却没带（或带错）convention 标签的 dict，只可能来自
        # 「绝对 nm^6 alpha」旧协议的缓存/配置。数值相同但物理不同，静默复用会让采样
        # 哈密顿量与本版本不一致，必须拒绝。空 dict 走默认值，是合法的新建路径。
        if p and "alpha_lj" in p:
            convention = p.get("alpha_convention")
            if convention != cls.ALPHA_CONVENTION:
                raise ValueError(
                    "软核 alpha 协议不匹配："
                    f"alpha_convention={convention!r}，本版本要求 "
                    f"{cls.ALPHA_CONVENTION!r}（alpha 为无量纲、表达式内乘 sigma_ij^6）。"
                    "旧协议把 alpha_lj 当绝对 nm^6 使用，两者数值相同但哈密顿量不同，"
                    "旧的 lambda 路径/pilot 度规/能量缓存一律不可复用，需重新 pilot。"
                )
        alpha_lj, alpha_coul = cls._normalize_alpha_units(
            p.get("alpha_lj", 0.5),
            p.get("alpha_coul", 0.2),
        )
        return cls(
            alpha_lj,
            alpha_coul,
            tuple(p.get("power_lj", [2, 2])),
            tuple(p.get("power_coul", [1, 1])),
        )


class BeutlerSoftcoreBuilder:
    """传统 Beutler 式软核势构建器 (CustomNonbondedForce + interaction group)
    与 ACESoftcorePotential 区别：显式 L-E 对过滤 + 传统 alpha*(1-lambda)^power 表达式
    """
    @staticmethod
    def build(
        nb_force: openmm.NonbondedForce,
        ligand_indices: List[int],
        env_indices: List[int],
        alpha_lj: float = 0.5,
        alpha_coul: float = 0.5,
        power_lj: int = 1,
        power_coul: int = 1,
        particle_params_override=None,
    ) -> openmm.CustomNonbondedForce:
        # 🔑 [SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2] 与
        # ACESoftcorePotential.build_expression 保持同一约定：alpha_lj / alpha_coul 无量纲，
        # 表达式内乘 sigma12^6 / sigma12^2。原式把它们当绝对 nm^6 / nm^2 直接相加（详细
        # 后果见 ACESoftcorePotential.build_expression 的注释）。
        # 同理，原来那两个绝对数值兜底项也必须一起缩放：1e-4 nm^6 相对 sigma12^6≈7.3e-4
        # 已占 ~14%，缩放 alpha 之后它自己就会变成一个不该存在的额外软化项；1e-3 nm^2 相对
        # sigma12^2≈0.09 占 ~1%，同样按 sigma12^2 缩放才是纯数值兜底。
        d_lj = (
            f"(r^6 + {alpha_lj}*sigma12^6*(1-lambda_vdw)^{power_lj}"
            f" + 1e-4*sigma12^6*(1-lambda_vdw))"
        )
        d_coul = (
            f"sqrt(r^2 + {alpha_coul}*sigma12^2*(1-lambda_coul)^{power_coul}"
            f" + 1e-3*sigma12^2)"
        )
        expr = (
            f"lambda_vdw * 4*sqrt(epsilon1*epsilon2)*("
            f"(sigma12^12 / {d_lj}^2) - "
            f"(sigma12^6 / {d_lj})"
            f") + "
            f"lambda_coul * 138.935456 * q1*q2 / {d_coul}; "
            f"sigma12=(0.5*(sigma1+sigma2))"
        )
        sc_force = openmm.CustomNonbondedForce(expr)
        for p in ["q", "sigma", "epsilon"]:
            sc_force.addPerParticleParameter(p)
        sc_force.addGlobalParameter("lambda_coul", 1.0)
        sc_force.addGlobalParameter("lambda_vdw", 1.0)

        for i in range(nb_force.getNumParticles()):
            if particle_params_override is not None and i < len(particle_params_override):
                q, sig, eps = particle_params_override[i]
            else:
                q, sig, eps = nb_force.getParticleParameters(i)
            sc_force.addParticle([
                q.value_in_unit(unit.elementary_charge),
                sig.value_in_unit(unit.nanometer),
                eps.value_in_unit(unit.kilojoule_per_mole)
            ])

        sc_force.addInteractionGroup(set(ligand_indices), set(env_indices))
        sc_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        # [MEM-00h，2026-08-06] 统一到基础力场的 1.0 nm、无 switching——理由见
        # BEUTLER_SOFTCORE_CUTOFF_NM 定义处。switchingDistance 仍显式设成等于
        # cutoff（而不是干脆不调用）：保持跟 ibs_engine._create_softcore_force
        # 同样的写法，任何读 getSwitchingDistance() 的下游代码都不会读到一个
        # 语义不明的哨兵值。
        sc_force.setCutoffDistance(BEUTLER_SOFTCORE_CUTOFF_NM * unit.nanometer)
        sc_force.setUseSwitchingFunction(False)
        sc_force.setSwitchingDistance(BEUTLER_SOFTCORE_CUTOFF_NM * unit.nanometer)

        for i in range(nb_force.getNumExceptions()):
            p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
            sc_force.addExclusion(int(p1), int(p2))

        return sc_force


class DEXPSurrogatePotential:
    """
    LJ-matched、pair-specific 的双指数(DEXP) softcore 替身。

    每个 ligand-environment 原子对不再共用一套全局 (A_fit, B_fit, r0_vdw)，而是从
    该 pair 两端原子各自原有的 LJ sigma/epsilon，用标准 Lorentz-Berthelot 组合律
    解析给出：

        eps_ij = sqrt(eps_i * eps_j)
        sigma_ij = 0.5 * (sigma_i + sigma_j)
        r0_ij = 2^(1/6) * sigma_ij

    核函数：

        U_ij(r) = eps_ij * [ (beta/(a-b))*exp(-a*x) - (a/(a-b))*exp(-b*x) ],  x = r/r0_ij - 1

    对任意 alpha>beta>0，在 r=r0_ij 处都与标准 12-6 LJ 有完全相同的井深(-eps_ij)、井位、
    一阶导(=0)；r->0 时能量和力都有限，天然没有 LJ 的 r^-12 奇点。

    默认 alpha_vdw=14, beta_vdw=5（不是标准 LJ 的 12/6）——这是在本项目结合态局部
    anchor-relative 扰动云（--perturb-scan + --perturb-fit，20 anchor x 74 扰动/anchor，
    按扰动档等权 + leave-one-anchor-out 交叉验证）上验证过的经验值：19/20 折独立选中同一
    点，held-out 加权 RMSE 比 12/6 改善约 13%。**但要注意**：数据主要约束的是
    alpha+beta≈19 这个组合(2D score surface 在 (alpha,beta) 里是沿此方向的对角谷，
    PCA 验证过)，不是分别独立钉死 alpha=14 和 beta=5——(14,5) 是这条谷上最优的、
    好看的整数代表，换成谷上邻近的 (13,6) 几乎同样好。换配体/环境化学组成时，不要
    默认沿用这两个数字，应该重新跑一次 perturb-scan/perturb-fit。
    alpha_vdw/beta_vdw 仍是可调字段，不是硬编码常量。

    不再存在需要自由拟合的 A_fit/B_fit/r0_vdw：它们现在由 alpha_vdw、beta_vdw 和
    每个 pair 自己的 sigma/epsilon 解析给出。
    """

    def __init__(self, alpha_vdw=14.0, beta_vdw=5.0):
        alpha_vdw, beta_vdw = float(alpha_vdw), float(beta_vdw)
        if not (math.isfinite(alpha_vdw) and math.isfinite(beta_vdw)):
            raise ValueError(
                "alpha_vdw 和 beta_vdw 必须是有限数值；"
                f"得到 alpha_vdw={alpha_vdw}, beta_vdw={beta_vdw}"
            )
        if not (alpha_vdw > beta_vdw > 0.0):
            raise ValueError(
                f"alpha_vdw({alpha_vdw}) 必须大于 beta_vdw({beta_vdw})，且二者都为正，"
                "否则 LJ-matched 井深/井位系数公式无定义"
            )
        self.alpha_vdw, self.beta_vdw = alpha_vdw, beta_vdw

    def build_expression(self, lam_vdw="lam_vdw"):
        """
        pair-specific DEXP 核心表达式：r0_ij/eps_ij 由 sigma1/epsilon1/sigma2/epsilon2
        （OpenMM CustomNonbondedForce 每个 pair 自动提供的两端 per-particle 参数）
        经组合律解析给出。Switch、Gaussian-Coulomb 与传统 PME 解耦由外层 Builder 统一接管。
        """
        a, b = self.alpha_vdw, self.beta_vdw
        c_a = b / (a - b)
        c_b = a / (a - b)
        two_pow_1_6 = 2.0 ** (1.0 / 6.0)
        rs = "max(r, 1e-6)"
        # 注意：Lepton 的 `;`-分隔子表达式必须在整条表达式的顶层给出，不能包在外层
        # 圆括号里当成算术分组的一部分——所以主表达式和 `;` 定义列表是拼接关系，
        # 不是 f"{lam_vdw} * ({sub_expr_with_semicolons})" 这种嵌套关系。
        main_expr = f"{lam_vdw} * eps_ij * ({c_a}*exp(-{a}*x_ij) - {c_b}*exp(-{b}*x_ij))"
        defs = (
            f"x_ij = {rs}/r0_ij - 1.0; "
            f"r0_ij = {two_pow_1_6}*sigma_ij_safe; "
            f"sigma_ij_safe = max(sigma_ij, 1e-6); "
            f"sigma_ij = 0.5*(sigma1+sigma2); "
            f"eps_ij = sqrt(epsilon1*epsilon2)"
        )
        return f"{main_expr}; {defs}"

    def get_parameters_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, p):
        if not isinstance(p, dict):
            raise TypeError(f"DEXP 参数必须是字典，得到 {type(p).__name__}")
        allowed = {"alpha_vdw", "beta_vdw"}
        legacy_keys = sorted(
            key
            for key in p
            if key in DEXP_LEGACY_FIT_KEYS
            or key.startswith("diagnostic_")
            or key.startswith("optimizer_")
        )
        if legacy_keys:
            raise ValueError(
                "旧版全局 DEXP 拟合参数与 pair-specific LJ-matched 生产势不兼容："
                + ", ".join(legacy_keys)
                + "。生产 DEXP 配置只接受 alpha_vdw/beta_vdw。"
            )
        unknown = sorted(set(p) - allowed)
        if unknown:
            raise ValueError("未知 DEXP 生产参数：" + ", ".join(unknown))
        return cls(**{key: p[key] for key in allowed if key in p})
