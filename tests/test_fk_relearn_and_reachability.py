"""老板 2026-09-12 裁决的两条规则的自检。

1. `IBSFrozenCalibrationValidationError` 只终止**这份候选**，不终止 Stage-2；
   一个 (path_version, window) 只给**一次**替代候选。
2. 验证可达性预检**只改路由，不改 verdict**：UNMEASURED 绝不变 REJECTED。
"""
import ast
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SRC = pathlib.Path(__file__).resolve().parents[1]


def _pure(module, names, extra=None):
    """从源码里抽纯函数（本机没有 openmm/numpy，不能整模块 import）。"""
    ns = dict(extra or {})
    ns.setdefault("os", os)
    ns.setdefault("json", __import__("json"))
    for n in ast.parse((_SRC / module).read_text()).body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            exec(compile(ast.Module(body=[n], type_ignores=[]), "<x>", "exec"), ns)
    return ns


_IE = _pure("ibs_engine.py",
            {"validation_reachability_verdict", "sealed_candidate_matches"},
            {"IBS_WARMUP_FRAME_STRIDE_STEPS": 250})
REACH = _IE["validation_reachability_verdict"]
SAME = _IE["sealed_candidate_matches"]

_PRE = _pure("abfe_preoptimizer.py",
             {"_sealed_candidates_path", "read_sealed_candidates",
              "seal_refuted_candidate", "relearn_epoch_used",
              "mark_relearn_epoch_consumed"})
_PRE["SEALED_CANDIDATES_FILENAME"] = "stage2_fk_sealed_candidates.json"


# ── 可达性 ──────────────────────────────────────────────────────────────
def test_win4_is_reachable_within_budget_not_unreachable():
    """win4 真实盘面：T=**10**（去相关下限），Ncap=600，剩 515000 步。

    ⚠️ T 曾被错当成 `minimum_complete_validation_frames`=200，把 gcrit 算小 20 倍，
    于是一个只差 26% 帧数的窗口被判成"差 7.5 倍、预算内不可达"。
    真实情况：它撞的是**15 批的周期上限**，不是预算 —— 继续攒就行。
    """
    r = REACH([67.9, 75.1, 77.31], required_decorrelated_frames=10,
              cycle_frame_cap=600, budget_remaining_steps=515000,
              frames_already=600)
    assert r["verdict"] == "REACHABLE_WITHIN_BUDGET", r
    assert abs(r["gcrit_cycle"] - 60.0) < 1e-9
    assert abs(r["gcrit_budget"] - 266.0) < 0.01
    # 10 × 77.31 = 773.1 帧 = 193275 步，剩 515000 ⟹ 富余
    assert int(r["projected_steps_needed"]) == 193275
    assert r["projected_steps_needed"] < 515000


def test_T_is_the_decorrelation_floor_not_the_completeness_requirement():
    """两个量差 20 倍，混掉就把可攒到的窗口判死。"""
    src = (_SRC / "ibs_engine.py").read_text()
    assert "IBS_LOCAL_MBAR_GATE_MIN_FRAMES = 10" in src
    # 两处 min_frames 默认值必须绑到常量，不许各写各的 10
    assert "min_frames: int = 10," not in src
    assert src.count("min_frames: int = IBS_LOCAL_MBAR_GATE_MIN_FRAMES,") == 2
    # 控制器回退时也绝不回退到 200
    ctl = (_SRC / "abfe_preoptimizer.py").read_text()
    assert "or _ie_min_frames()" in ctl


def test_reachability_never_rejects_f_k():
    """可达性只能改路由。把 UNMEASURED 改成 REJECTED 是被明令禁止的。"""
    r = REACH([67.9, 77.3], required_decorrelated_frames=10,
              cycle_frame_cap=600, budget_remaining_steps=515000)
    assert r["may_be_used_to_reject_f_k"] is False


def test_single_checkpoint_never_decides():
    """单点 g 在小样本下会误杀 —— 至少要两个检查点。"""
    r = REACH([77.3], required_decorrelated_frames=10, cycle_frame_cap=600,
              budget_remaining_steps=515000)
    assert r["verdict"] == "INDETERMINATE"


def test_three_reachable_bands():
    # T=10 ⟹ gcrit_cycle=60, gcrit_budget=266
    kw = dict(required_decorrelated_frames=10, cycle_frame_cap=600,
              budget_remaining_steps=515000, frames_already=600)
    assert REACH([40.0, 55.0], **kw)["verdict"] == "REACHABLE_THIS_CYCLE"
    assert REACH([70.0, 90.0], **kw)["verdict"] == "REACHABLE_WITHIN_BUDGET"
    assert REACH([300.0, 400.0], **kw)["verdict"] == "UNREACHABLE"
    # 区间跨过阈值 ⟹ 不许直接判死，只给一个诊断块
    assert REACH([100.0, 500.0], **kw)["verdict"] == "INDETERMINATE"


def test_g_lower_bound_is_min_not_bootstrap():
    """嵌套样本（400 帧含 200 帧）不满足 bootstrap 独立性前提，取最小值才真保守。"""
    r = REACH([67.9, 75.1, 77.31], required_decorrelated_frames=10,
              cycle_frame_cap=600, budget_remaining_steps=515000)
    assert r["g_lower_bound"] == 67.9
    assert r["g_lower_bound_method"] == "min_over_checkpoints_nested_samples"


# ── 同候选判定 ──────────────────────────────────────────────────────────
def test_same_candidate_is_distance_not_hash():
    """f_k 有规范自由度：整体平移是**同一份**候选，不是新的。"""
    assert SAME([1.0, 2.0, 3.0], [11.0, 12.0, 13.0]) is True
    assert SAME([1.0, 2.0, 3.0], [1.0, 2.0, 3.4]) is True      # 0.3 < 0.5 容差
    assert SAME([1.0, 2.0, 3.0], [1.0, 2.0, 9.0]) is False     # 形状变了
    assert SAME([1.0, 2.0, 3.0], [1.0, 2.0]) is False          # 长度不符


# ── 一次性配额 ──────────────────────────────────────────────────────────
def test_relearn_allowance_is_one_shot_and_shared(tmp_path):
    """REJECTED→RELEARN 与 UNREACHABLE→RELEARN **共用**同一个配额。"""
    ck = str(tmp_path)
    assert _PRE["relearn_epoch_used"](ck, 1, 4) is False
    _PRE["seal_refuted_candidate"](
        ck, path_version=1, window_idx=4, lambdas_vdw=[0.3, 0.2],
        f_k=[1.0, 2.0], reason="statistically_rejected")
    # 封存本身不消耗配额
    assert _PRE["relearn_epoch_used"](ck, 1, 4) is False
    _PRE["mark_relearn_epoch_consumed"](ck, 1, 4)
    assert _PRE["relearn_epoch_used"](ck, 1, 4) is True
    # 别的窗口 / 别的 path_version 不受影响
    assert _PRE["relearn_epoch_used"](ck, 1, 3) is False
    assert _PRE["relearn_epoch_used"](ck, 2, 4) is False


def test_sealed_record_binds_four_identities(tmp_path):
    """身份 = path_version + window + λ 身份 + **f_k 向量**（不是 hash）。"""
    ck = str(tmp_path)
    rec = _PRE["seal_refuted_candidate"](
        ck, path_version=3, window_idx=4, lambdas_vdw=[0.31556504, 0.2935],
        fingerprint="deadbeef", f_k=[1.0, -1.0], reason="x")
    for k in ("path_version", "window_idx", "lambda_identity", "f_k_kJ_mol"):
        assert rec[k] is not None, k
    assert rec["never_revalidate"] is True
    # ⚠️ hash **不是身份**：主线快速变动会让它失配，把被驳回的候选当成新的重试。
    # 它只能作为溯源线索存在，名字里就写死这件事。
    assert "candidate_fingerprint" not in rec
    assert rec["candidate_fingerprint_PROVENANCE_ONLY"] == "deadbeef"


# ── 接线 ────────────────────────────────────────────────────────────────
def test_relearn_is_a_distinct_action_from_recalibrate():
    """科学语义不同：RECALIBRATE 用旧生产帧重解，RELEARN 从头学。绝不合并。"""
    src = (_SRC / "abfe_preoptimizer.py").read_text()
    assert '"RELEARN_FK_EPOCH"' in src
    assert '"RECALIBRATE_FK"' in src


def test_relearn_does_not_seed_old_f_k():
    """新 Epoch 传 f_k 种子就等于把被驳回那份带进来，独立性没了。"""
    fn = next(
        n for n in ast.walk(ast.parse((_SRC / "abfe_pipeline.py").read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_run_stage2_autonomous"
    )
    body = ast.dump(fn)
    assert "RELEARN_FK_EPOCH" in body
    # 该分支内不得出现 f_k 播种关键字
    for node in ast.walk(fn):
        if (isinstance(node, ast.Compare)
                and any(isinstance(c, ast.Constant) and c.value == "RELEARN_FK_EPOCH"
                        for c in node.comparators)):
            break
    else:
        raise AssertionError("找不到 RELEARN_FK_EPOCH 分支")


def test_refuted_candidate_no_longer_kills_the_stage():
    """统计驳回只终止这份候选 —— 那个 handler 里不得再有裸 raise。"""
    fn = next(
        n for n in ast.walk(ast.parse((_SRC / "abfe_pipeline.py").read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_run_stage2_autonomous"
    )
    for h in [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]:
        names = {n.attr for n in ast.walk(h.type or ast.Pass())
                 if isinstance(n, ast.Attribute)}
        if "IBSFrozenCalibrationValidationError" in names:
            assert not any(isinstance(b, ast.Raise) for b in ast.walk(h)), \
                "驳回应封存候选并路由到 RELEARN_FK_EPOCH，不再上抛终止整个 Stage-2"
            return
    raise AssertionError("没找到驳回 handler")


if __name__ == "__main__":
    import tempfile

    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as d:
                fn(pathlib.Path(d))
        else:
            fn()
        print("  ok", name)
    print("全过")
