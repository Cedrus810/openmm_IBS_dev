

# ============================================================================
# 0. 单位常量与验证器
# ============================================================================
class UnitConstants:
    NM_PER_ANGSTROM = 0.1
    KJ_PER_KCAL = 4.184
    KJ_PER_NM2_PER_KCAL_PER_A2 = 418.4
    RAD_PER_DEG = np.pi / 180.0
    DISTANCE_RANGE_NM = (0.1, 10.0)
    ANGLE_RANGE_RAD = (0.0, math.pi)
    FORCE_CONSTANT_KR_RANGE = (100.0, 100000.0)
    FORCE_CONSTANT_KANGLE_RANGE = (10.0, 1000.0)


class UnitValidator:
    @staticmethod
    def validate_distance(v, n="dist"):
        if v <= 0:
            raise ValueError(f"{n}必须>0")
        if v > 100:
            warnings.warn(f"{n}={v} nm 可能过大")

    @staticmethod
    def validate_force_constant(v, n="k"):
        if v <= 0:
            raise ValueError(f"{n}必须为正值")

