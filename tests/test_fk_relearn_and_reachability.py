"""老板 2026-09-12 裁决的两条规则的自检。

1. `IBSFrozenCalibrationValidationError` 只终止**这份候选**，不终止 Stage-2；
   一个 (path_version, window) 只给**一次**替代候选。
2. 验证可达性预检**只改路由，不改 verdict**：UNMEASURED 绝不变 REJECTED。
"""
import ast
import os
import pathlib
import sys
import pytest

pytestmark = pytest.mark.cpu_only

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
    """统计驳回只终止这份候选 —— 那个 handler 里不得有**任何** raise。

    [2026-09-14] 原文写的是「裸 raise」，不准确：外层 `except Exception` 会接住
    并重新抛出这个 handler 里的**任何**异常 ⟹ 不管抛的是什么、为了什么，结果都是
    整个 run 死掉。所以判据就是「一个 raise 都不许有」，fail-closed 要停就走终态出口。
    """
    fn = next(
        n for n in ast.walk(ast.parse((_SRC / "abfe_pipeline.py").read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_run_stage2_autonomous"
    )
    for h in [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]:
        names = {n.attr for n in ast.walk(h.type or ast.Pass())
                 if isinstance(n, ast.Attribute)}
        if "IBSFrozenCalibrationValidationError" in names:
            # 🔑 [2026-09-14] **判据维持最严：这个 handler 里不得有任何 `raise`。**
            #
            # 我一度把它收窄成「只禁裸 raise / 重抛捕获变量 / 重抛同类型」，理由是
            # 同一天 handler 里加了一道**针对另一件事**的 fail-closed（封存记录的
            # 身份字段缺失时拒绝落一条注定匹配不上的空记录）。那个收窄是**错的**：
            #   · 这个 handler 里**任何** raise 都会被外层 `except Exception` 接住
            #     并重新抛出 ⟹ 不管抛的是什么、为了什么，结果都是整个 run 死掉。
            #     所以「不得有任何 raise」不是过严，它就是这个 handler 的契约。
            #   · 而那道 fail-closed 想表达的「台账写不成就别装作写成了」，用
            #     **终态出口**（`_finish("TERMINAL", "HALT_FK_REFUTED", …)` + `break`）
            #     一样能表达，而且更准确 —— 它本来就该停下并留诊断，不是抛异常。
            #     执行器那边已经按这个改法落地，所以本断言一个字都不用放宽。
            #
            # 教训值得留在这里：「为了让改动通过而放宽测试」和「改动本身就不该触发
            # 那条断言」是两件事。先确认是后者，再动测试。
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


# ── S2-B：完整性要求不许随预算浮动 ──────────────────────────────────────
def _assignments_of(module: str, target: str):
    """源码里对 `target` 的每一次赋值，返回右侧表达式里引用的名字集合。

    按**源码 AST** 断言而不是跑一遍：这个量活在 `run_ibs_bias_warmup` 内部，
    真要跑到它得起 OpenMM + 完整预热循环。同类源码断言的先例见
    `docs/archive/STAGE2_CONTROLLER_DESIGN_2026-09-12.md` §9.5。
    """
    out = []
    for node in ast.walk(ast.parse((_SRC / module).read_text())):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == target for t in node.targets):
            continue
        out.append({n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)})
    return out


def test_completeness_requirement_is_decoupled_from_budget():
    """`minimum_complete_validation_frames` 是**统计目标**，不是预算的函数（S2-B）。

    原式是 `max(统计目标, validation_attempt_budget_steps // check_chunk)`，而续验
    路径上 `validation_attempt_budget_steps = full_bias_step_budget` ⟹ **给的预算
    越多、要求的帧数越高**（实测 max(200, 515000//250) = 2061，是统计目标的 10 倍）。
    那不是一个"要求"，那是把预算改名叫要求。

    它只进报告、不当门，所以这条钉的不是判定，而是**这个数不许骗读它的人**：
    可达性预检的 T 一度就被错取成它，把 gcrit 算小 20 倍（见上面 win4 那条）；
    而且随预算浮动的数会让两次 run 的报告没法横向比。
    """
    rhs_names = _assignments_of("ibs_engine.py", "minimum_complete_validation_frames")
    assert rhs_names, "找不到 minimum_complete_validation_frames 的赋值"
    forbidden = {"validation_attempt_budget_steps", "full_bias_step_budget",
                 "frozen_validation_reserved_steps", "budget_remaining_steps"}
    for names in rhs_names:
        leaked = names & forbidden
        assert not leaked, (
            f"完整性要求又跟预算耦合上了：右侧引用了 {sorted(leaked)}。"
            "200 是统计目标，不该随预算浮动（S2-B）。"
        )


def test_reachability_T_is_the_decorrelated_floor_not_the_completeness_target():
    """两个量不许合并：T 是**去相关**帧数下限，完整性目标是**原始**帧数。

    混掉的实测后果就在本文件第一条测试里：gcrit 算小 20 倍。
    """
    src = (_SRC / "abfe_preoptimizer.py").read_text()
    # 控制器取 T 的那一处必须读 decorrelated_frames_required，且回退到引擎常量
    assert '"decorrelated_frames_required"' in src
    assert "_ie_min_frames()" in src
    # 完整性目标在控制器里只能是**报告字段**，名字自带 REPORT_ONLY 免得被误用
    assert "validation_completeness_frames_REPORT_ONLY" in src
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_ie_reach"):
            for kw in node.keywords:
                if kw.arg == "required_decorrelated_frames":
                    used = {n.id for n in ast.walk(kw.value) if isinstance(n, ast.Name)}
                    assert "T" in used or used, "T 的来源读不出来"
