def _ie_min_frames():
    """去相关帧数下限（可达性预检的 T）。惰性读，避免顶层拖 ibs_engine。"""
    try:
        from ibs_engine import IBS_LOCAL_MBAR_GATE_MIN_FRAMES
        return int(IBS_LOCAL_MBAR_GATE_MIN_FRAMES)
    except Exception:
        return 10



def _ie_reach(*args, **kwargs):
    """惰性转发到 `ibs_engine.validation_reachability_verdict`（别在顶层拖重依赖）。"""
    from ibs_engine import validation_reachability_verdict
    return validation_reachability_verdict(*args, **kwargs)


SEALED_CANDIDATES_FILENAME = "stage2_fk_sealed_candidates.json"


def _sealed_candidates_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, SEALED_CANDIDATES_FILENAME)


def read_sealed_candidates(checkpoint_dir):
    """读被统计驳回、**永不续验**的 f_k 候选台账。读不到返回空表。"""
    try:
        with open(_sealed_candidates_path(checkpoint_dir), encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return []
    return list((payload or {}).get("sealed") or [])


def seal_refuted_candidate(
    checkpoint_dir,
    *,
    path_version,
    window_idx,
    lambdas_vdw,
    f_k,
    reason,
    fingerprint=None,
):
    """封存一份被驳回的 f_k 候选：**永不续验**。

    身份 = `path_version + window + λ 身份 + f_k 向量本身`。

    ⚠️ **候选身份不用 hash。** 主线在快速反复变动，哈希里放什么一改，所有已封存
    记录就全部失配、被驳回的候选会被当成新的重新试一遍 —— 正是"反复试到偶然
    通过"。而且这是本仓库记录在案、**已经复发四次**的同一个坑：自产产物的
    sha256 进身份。规则是「只有用户输入才配做身份」，而 f_k 是我们自己算出来的。
    正解不是删掉身份，是改成**语义身份**：f_k 向量（mean-center 后比距离，
    见 `ibs_engine.sealed_candidate_matches`）。
    `fingerprint` 只作为可选的溯源线索留着，**不参与任何判定**。

    ⚠️ 这里**只记录**，不决定下一步动作。统计驳回不得被解释成"λ 太稀"，
    所以它永远不触发插 λ / 拆窗。
    """
    entries = read_sealed_candidates(checkpoint_dir)
    record = {
        "path_version": int(path_version),
        "window_idx": int(window_idx),
        "lambda_identity": [round(float(x), 10) for x in (lambdas_vdw or [])],
        # **身份**：f_k 向量本身。判"实质上是不是同一份"在 f_k 空间里比距离
        # （mean-center 后，见 ibs_engine.sealed_candidate_matches）。
        "f_k_kJ_mol": [float(x) for x in (f_k or [])],
        # 仅溯源，不参与判定。主线变动会让它失配，所以它不配做身份。
        "candidate_fingerprint_PROVENANCE_ONLY": fingerprint,
        "reason": reason,
        "never_revalidate": True,
    }
    entries.append(record)
    payload = {
        "note": (
            "被统计驳回的 f_k 候选。**永不续验**；每个 (path_version, window) "
            "只允许一次替代候选（RELEARN_FK_EPOCH），防止反复试到偶然通过。"
        ),
        "sealed": entries,
    }
    tmp = _sealed_candidates_path(checkpoint_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, _sealed_candidates_path(checkpoint_dir))
    return record


def relearn_epoch_used(checkpoint_dir, path_version, window_idx):
    """这个 (path_version, window) 是否已经用掉了那**唯一一次**替代候选。

    ⚠️ `REJECTED → RELEARN` 和 `UNREACHABLE → RELEARN` 共用同一个配额 ——
    否则一个窗口能走两扇门拿两次 fresh Epoch，一次性护栏就形同虚设。
    """
    return any(
        int(e.get("path_version", -1)) == int(path_version)
        and int(e.get("window_idx", -1)) == int(window_idx)
        and e.get("relearn_consumed")
        for e in read_sealed_candidates(checkpoint_dir)
    )


def mark_relearn_epoch_consumed(checkpoint_dir, path_version, window_idx, detail=None):
    """记下那唯一一次替代候选已被使用。找不到对应封存记录时补一条。"""
    entries = read_sealed_candidates(checkpoint_dir)
    hit = [
        e for e in entries
        if int(e.get("path_version", -1)) == int(path_version)
        and int(e.get("window_idx", -1)) == int(window_idx)
    ]
    if not hit:
        entries.append({
            "path_version": int(path_version),
            "window_idx": int(window_idx),
            "reason": "relearn_without_sealed_candidate",
        })
        hit = [entries[-1]]
    for e in hit:
        e["relearn_consumed"] = True
        if detail:
            e["relearn_detail"] = detail
    payload = {"note": "见 seal_refuted_candidate", "sealed": entries}
    tmp = _sealed_candidates_path(checkpoint_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, _sealed_candidates_path(checkpoint_dir))


