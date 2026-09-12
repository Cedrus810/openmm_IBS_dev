"""Stage-2 λ 路径的**版本记录**：路径可以演化，已完成的工作不因编号后移而丢失。

背景：末段窗口的 f_k 调不动时需要在失败区间插一个 λ，后面的状态顺延。插点之前
的窗口 λ 值逐位未变、物理上完全可复用，但如果只存一份"当前路径"，重启后就分不清
"这是插过点的 v2" 还是 "预优化又算了一遍的 v1"，也无法判断哪些窗口该复用。

本模块只做记账，不做任何物理决策：
  · 每个版本一份**不可变**记录（v1.json, v2.json, ...），含父版本、插点事件与理由。
  · 一份**原子**指针 `path_current.json` 指向当前有效版本，**最后**写。
  · 插点事件带确定性 ID，重启后不会把同一个 NEW7 插第二次。
  · `changed_windows()` 给出"哪些窗口可复用、哪些必须重来"。

⚠️ 不在这里做的事（各有现成实现，别重复造）：
  · 窗口产物按 λ 集合改名复用 → `abfe_pipeline._invalidate_stage_window_files` 的 reuse_map。
  · 单窗口 checkpoint 能否续算 → `ibs_engine._resume_cached_window_gate_status`。
  · 插点位置怎么选 → `abfe_preoptimizer.insert_thermodynamic_midpoint_from_ibs_lse_failure`。
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 与 abfe_pipeline._lambda_signature 同口径：λ 一律按 8 位小数比较，避免
# 浮点尾数让"同一个状态"在两次运行里算成两个。
LAMBDA_DECIMALS = 8

PATH_VERSION_PROTOCOL_VERSION = 1

_VERSIONS_DIRNAME = "path_versions"
_POINTER_FILENAME = "path_current.json"
_VERSION_RE = re.compile(r"^v(\d+)\.json$")


def _q(value: float) -> float:
    """量化到比较口径。-0.0 归一成 0.0，否则同一个 λ 会有两个 ID。"""
    return round(float(value), LAMBDA_DECIMALS) + 0.0


def state_id(lambda_coul: float, lambda_vdw: float) -> str:
    """状态的稳定身份。

    **由 λ 值本身决定，不用单独的序号登记簿。** 这样 old7 后移到第 8 位仍然是同一个
    ID，无需维护计数器、也不会因为进程中断而发错号；两条不同路径上 λ 相同的状态
    本来就是同一个物理状态，共用 ID 是对的。
    """
    return "s_%s_%s" % (
        format(_q(lambda_coul), ".%df" % LAMBDA_DECIMALS),
        format(_q(lambda_vdw), ".%df" % LAMBDA_DECIMALS),
    )


def build_states(
    lambdas_coul: Sequence[float], lambdas_vdw: Sequence[float]
) -> List[Dict[str, Any]]:
    lc = [float(x) for x in lambdas_coul]
    lv = [float(x) for x in lambdas_vdw]
    if len(lc) != len(lv):
        raise ValueError(f"lambdas_coul({len(lc)}) 与 lambdas_vdw({len(lv)}) 长度不一致")
    if len(lc) < 2:
        raise ValueError("路径至少需要 2 个状态")
    states = [
        {"id": state_id(c, v), "lambda_coul": _q(c), "lambda_vdw": _q(v)}
        for c, v in zip(lc, lv)
    ]
    ids = [s["id"] for s in states]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"路径含重复状态（λ 在 {LAMBDA_DECIMALS} 位小数下相同）: {dup}")
    return states


def _canonical(states: List[Dict[str, Any]], window_ranges: Sequence[Sequence[int]]) -> str:
    return json.dumps(
        {
            "protocol": PATH_VERSION_PROTOCOL_VERSION,
            "states": states,
            "window_ranges": [[int(a), int(b)] for a, b in window_ranges],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def content_sha256(states, window_ranges) -> str:
    return hashlib.sha256(_canonical(states, window_ranges).encode("utf-8")).hexdigest()


def event_id(kind: str, detail: Dict[str, Any]) -> str:
    """插点事件的确定性 ID —— 幂等的依据。

    **只吃 (类型, 细节)，刻意不含父版本、时间戳或进程号。** 含父版本会让幂等失效：
    插点成功后当前版本已经前进，同一次重试算出的父版本变了、ID 就对不上，于是被
    当成新事件插第二次（本模块第一版真踩了这个，由
    `test_same_insert_event_does_not_insert_twice` 抓出）。事件身份应当由"在哪个
    区间插了什么 λ"决定，这在任何父版本下都是同一件事。
    """
    blob = json.dumps(
        {"kind": kind, "detail": detail},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# 落盘布局
# --------------------------------------------------------------------------

def versions_dir(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, _VERSIONS_DIRNAME)


def version_path(checkpoint_dir: str, version: int) -> str:
    return os.path.join(versions_dir(checkpoint_dir), "v%d.json" % int(version))


def pointer_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, _POINTER_FILENAME)


def _atomic_write(path: str, payload: Dict[str, Any]) -> None:
    # 惰性 import：本模块其余部分是纯 stdlib，不该为了写文件把 OpenMM 拖进来，
    # 也避免与 ibs_engine 形成 import 环。
    from ibs_engine import _atomic_write_json

    _atomic_write_json(path, payload)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _validate(record: Dict[str, Any]) -> Dict[str, Any]:
    """fail-closed：内容与自带 sha256 对不上就拒绝，不猜。"""
    for key in ("version", "protocol_version", "states", "window_ranges", "content_sha256"):
        if key not in record:
            raise ValueError(f"路径版本记录缺字段 {key!r}")
    if int(record["protocol_version"]) != PATH_VERSION_PROTOCOL_VERSION:
        raise ValueError(
            f"路径版本协议 {record['protocol_version']} != {PATH_VERSION_PROTOCOL_VERSION}"
        )
    actual = content_sha256(record["states"], record["window_ranges"])
    if actual != record["content_sha256"]:
        raise ValueError(
            f"路径版本 v{record['version']} 内容与 content_sha256 不符（{actual} != "
            f"{record['content_sha256']}），拒绝使用"
        )
    return record


def load_version(checkpoint_dir: str, version: int) -> Optional[Dict[str, Any]]:
    record = _read_json(version_path(checkpoint_dir, version))
    return None if record is None else _validate(record)


def existing_versions(checkpoint_dir: str) -> List[int]:
    out = []
    for path in glob.glob(os.path.join(versions_dir(checkpoint_dir), "v*.json")):
        match = _VERSION_RE.match(os.path.basename(path))
        if match:
            out.append(int(match.group(1)))
    return sorted(out)


def load_current(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    """读当前有效版本。指针缺失/损坏一律返回 None（当作没有路径记录）。"""
    pointer = _read_json(pointer_path(checkpoint_dir))
    if pointer is None or "version" not in pointer:
        return None
    return load_version(checkpoint_dir, int(pointer["version"]))


def _publish(checkpoint_dir: str, version: int) -> None:
    """原子推进指针。**只前进不后退**——防止半套恢复把路径倒回旧版本。"""
    pointer = _read_json(pointer_path(checkpoint_dir))
    if pointer is not None and int(pointer.get("version", -1)) > int(version):
        raise ValueError(
            f"拒绝把当前路径从 v{pointer['version']} 倒退到 v{version}"
        )
    _atomic_write(
        pointer_path(checkpoint_dir),
        {"version": int(version), "protocol_version": PATH_VERSION_PROTOCOL_VERSION},
    )


def _write_version(checkpoint_dir: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """先写完整版本文件并回读校验，**再**推指针。

    中途断电只会留下一个没人指向的 v{N}.json：`load_current` 仍然返回完整的
    v{N-1}，绝不会读到"新路径配旧窗口文件"的半套状态。下次 `append_version`
    会靠 event_id 认出这个孤儿并直接采纳它，而不是再插一次。
    """
    path = version_path(checkpoint_dir, record["version"])
    _atomic_write(path, record)
    readback = _read_json(path)
    if readback is None:
        raise RuntimeError(f"路径版本 v{record['version']} 写入后回读失败")
    _validate(readback)
    _publish(checkpoint_dir, int(record["version"]))
    return readback


def _make_record(version, parent_version, states, window_ranges, event) -> Dict[str, Any]:
    ranges = [[int(a), int(b)] for a, b in window_ranges]
    n_states = len(states)
    covered = sorted({i for a, b in ranges for i in range(a, b)})
    if covered != list(range(n_states)):
        raise ValueError(f"window_ranges 未完整覆盖 {n_states} 个状态：{ranges}")
    return {
        "protocol_version": PATH_VERSION_PROTOCOL_VERSION,
        "version": int(version),
        "parent_version": None if parent_version is None else int(parent_version),
        "states": states,
        "window_ranges": ranges,
        "windows": [
            {"index": i, "state_ids": [s["id"] for s in states[a:b]]}
            for i, (a, b) in enumerate(ranges)
        ],
        "event": event,
        "content_sha256": content_sha256(states, ranges),
    }


def init_version(
    checkpoint_dir: str,
    lambdas_coul: Sequence[float],
    lambdas_vdw: Sequence[float],
    window_ranges: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """建立 v1（幂等）。已有当前版本时原样返回，**不覆盖**。

    幂等而非覆盖，是为了满足"resume 不能重新预优化得到 v1"：已经演化到 v2 的目录
    再次启动时，这里必须把 v2 交回去，而不是拿初始配置把路径打回 v1。
    """
    current = load_current(checkpoint_dir)
    if current is not None:
        return current
    states = build_states(lambdas_coul, lambdas_vdw)
    record = _make_record(
        1, None, states, window_ranges,
        {"id": event_id("init", {}), "kind": "init", "reason": "initial_path"},
    )
    return _write_version(checkpoint_dir, record)


def append_version(
    checkpoint_dir: str,
    lambdas_coul: Sequence[float],
    lambdas_vdw: Sequence[float],
    window_ranges: Sequence[Sequence[int]],
    *,
    kind: str,
    reason: str,
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """在当前版本之上追加一个新版本（按 event_id 幂等）。

    幂等覆盖两种重复：已发布过同一事件（指针已指向它），以及**写完版本文件但还没
    来得及推指针就被杀**（孤儿版本）。两种情况都采纳既有版本，绝不重复插点。
    """
    current = load_current(checkpoint_dir)
    if current is None:
        raise ValueError("没有当前路径版本，先调用 init_version()")
    parent = int(current["version"])
    detail = dict(detail or {})
    eid = event_id(kind, detail)

    # 1) 这次插点已经在当前路径的**祖先链**里 → 路径里已经有它了，原样返回，不重插、
    #    也绝不把指针退回那个祖先。
    if any(rec.get("event", {}).get("id") == eid for rec in history(checkpoint_dir)):
        return current
    # 2) 存在写完但没来得及推指针的**孤儿**版本 → 采纳它（只前进）。
    for version in existing_versions(checkpoint_dir):
        if version <= parent:
            continue
        candidate = load_version(checkpoint_dir, version)
        if candidate is not None and candidate.get("event", {}).get("id") == eid:
            _publish(checkpoint_dir, version)
            return candidate

    states = build_states(lambdas_coul, lambdas_vdw)
    record = _make_record(
        max(existing_versions(checkpoint_dir) or [parent]) + 1,
        parent, states, window_ranges,
        {"id": eid, "kind": kind, "reason": reason, "detail": detail},
    )
    return _write_version(checkpoint_dir, record)


def changed_windows(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """哪些窗口可以复用、哪些必须重来。

    判据是**窗口的状态 ID 集合**，不是窗口编号：编号后移但组成不变的窗口算可复用
    （产物改名即可，见 reuse_map），组成变了的一律算 changed —— 采样势变了，旧
    checkpoint 不能原样续跑，只能当初始构象。
    """
    old_map = {frozenset(w["state_ids"]): w["index"] for w in old["windows"]}
    reusable, changed = [], []
    for window in new["windows"]:
        key = frozenset(window["state_ids"])
        if key in old_map:
            reusable.append({"new_index": window["index"], "old_index": old_map[key]})
        else:
            changed.append(window["index"])
    new_keys = {frozenset(w["state_ids"]) for w in new["windows"]}
    dropped = [w["index"] for w in old["windows"] if frozenset(w["state_ids"]) not in new_keys]
    return {"reusable": reusable, "changed": changed, "dropped_old_indices": dropped}


def history(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """从当前版本沿 parent_version 回溯到根，按时间正序返回。"""
    current = load_current(checkpoint_dir)
    chain: List[Dict[str, Any]] = []
    seen = set()
    while current is not None:
        version = int(current["version"])
        if version in seen:
            raise ValueError(f"路径版本链出现环：v{version}")
        seen.add(version)
        chain.append(current)
        parent = current.get("parent_version")
        current = None if parent is None else load_version(checkpoint_dir, int(parent))
    return list(reversed(chain))


def count_events(checkpoint_dir: str, kind: str) -> int:
    """当前路径链上某类事件发生过几次（跨 resume 累计）。

    用途：`max_path_insertions` 的预算。按"本次调用里的循环次数"计会在重启后归零，
    既不是新增态数上限、也不是总回退次数上限；从不可变的版本链里数才是跨进程的。
    """
    return sum(
        1 for record in history(checkpoint_dir)
        if (record.get("event") or {}).get("kind") == kind
    )


def resolve_path(
    checkpoint_dir: str,
    lambdas_coul: Sequence[float],
    lambdas_vdw: Sequence[float],
    window_ranges: Sequence[Sequence[int]],
) -> Tuple[Dict[str, Any], List[float], List[float], List[List[int]]]:
    """把"刚预优化出来的路径"换成"当前有效路径"，供流水线直接使用。

    首次运行：登记为 v1，原样返回传入的路径。
    已演化过（存在 v2+）：**返回记录里的路径，而不是传入的那份**——这是设计里
    "resume 不能重新预优化得到 v1" 那条：预优化是纯函数，重跑必然又给出 v1 的
    λ 表，若照单全收就把插过的点抹掉、并让已完成的末段窗口全部作废。

    Returns ``(record, lambdas_coul, lambdas_vdw, window_ranges)``。
    """
    record = init_version(checkpoint_dir, lambdas_coul, lambdas_vdw, window_ranges)
    lc = [s["lambda_coul"] for s in record["states"]]
    lv = [s["lambda_vdw"] for s in record["states"]]
    return record, lc, lv, [list(r) for r in record["window_ranges"]]
