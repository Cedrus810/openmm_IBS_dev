"""体系元素能不能被指定的参考 ML 势表示——启动期判定，不加载模型跑力。

为什么需要：`local_residual` 的类型词表是从体系原子序数推的，而 ML 势各有各的
覆盖范围，两者从来不自动对齐。实测过的例子（EXP-033 §2.A）：MACE-OFF24 的
z-table 是 {1,6,7,8,9,15,16,17,35,53}，**连 Na 都没有**，而任何加过离子的水盒都有
Na。这道检查把"跑到一半才发现表示不了"提前到启动期，并且一次列全缺哪些元素。

**z-table 一律从模型文件里读，不抄常量。** MACE 的 `.model` 自带
`atomic_numbers`；抄一份迟早跟模型对不上——同一批 MACE-omol-0 里
`-extra-large-1024` 是 83 种元素、`-extra-large-4M` 是 82 种（少一个 Ne）。

这跟残差模型的训练/重训**没有关系**（R1 的训练信号是纯 MM 的）。它只回答一个
问题：这个体系的元素，这个模型表不表示得了。
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable, Sequence

#: 模型不在默认缓存目录时用它指路（本机模型在 /home/ruigengji/MLP/mace）。
#: 不把个人路径写进代码。
MODEL_SEARCH_ENV = "ABFE_MACE_MODEL_DIR"
MODEL_DEFAULT_DIRS = ("~/.cache/mace",)
#: 推荐的参考模型：83 种元素，是同族 `-4M`（82 种，少 Ne）的严格超集，含 Na。
RECOMMENDED_MODEL = "MACE-omol-0-extra-large-1024"


class ElementCoverageError(RuntimeError):
    """体系元素超出参考模型覆盖范围，或模型找不到/读不出 z-table。"""


def resolve_model(model: str | Path) -> Path:
    """名字或路径 → 具体的 .model 文件。换模型就是换这个参数。"""

    candidate = Path(model).expanduser()
    if candidate.is_file():
        return candidate
    searched: list[str] = []
    for root in [os.environ.get(MODEL_SEARCH_ENV), *MODEL_DEFAULT_DIRS]:
        if not root:
            continue
        directory = Path(root).expanduser()
        searched.append(str(directory))
        if not directory.is_dir():
            continue
        for name in (str(model), f"{model}.model"):
            hit = directory / name
            if hit.is_file():
                return hit
        # 名字大小写/分隔符不一致很常见（mace-omol-0-… ↔ MACE-omol-0-…）
        normalized = str(model).replace("-", "").replace("_", "").lower()
        for hit in sorted(directory.glob("*.model")):
            if hit.stem.replace("-", "").replace("_", "").lower() == normalized:
                return hit
    raise ElementCoverageError(
        f"找不到参考模型 {model!r}；已找过 {searched}。"
        f"给绝对路径，或把模型目录设进 ${MODEL_SEARCH_ENV}。"
    )


@lru_cache(maxsize=8)
def _z_table_cached(resolved: str) -> tuple[int, ...]:
    import torch  # 惰性：诊断路径不该在 import 期付 torch 的启动成本

    module = torch.load(resolved, map_location="cpu", weights_only=False)
    numbers = getattr(module, "atomic_numbers", None)
    if numbers is None:
        raise ElementCoverageError(f"{resolved} 不像 MACE 模型：没有 atomic_numbers")
    table = tuple(sorted(int(value) for value in numbers))
    if not table:
        raise ElementCoverageError(f"{resolved} 的 atomic_numbers 是空的")
    return table


def model_z_table(model: str | Path) -> tuple[int, ...]:
    """读模型自己的元素表。整份模型要 `torch.load`（extra-large 约 400 MB），
    按解析后的路径缓存，一次进程只读一遍。"""

    return _z_table_cached(str(resolve_model(model)))


def check_element_coverage(
    atomic_numbers: Iterable[int], model: str | Path
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """返回 (体系元素, 模型元素)；有任何元素不被覆盖就 fail-closed。"""

    system_elements = tuple(sorted({int(value) for value in atomic_numbers}))
    if not system_elements:
        raise ElementCoverageError("体系元素列表为空")
    table = model_z_table(model)
    missing = [z for z in system_elements if z not in set(table)]
    if missing:
        raise ElementCoverageError(
            f"参考模型 {Path(str(model)).name} 表示不了体系里的元素 {missing}"
            f"（它覆盖 {len(table)} 种）。换一个覆盖更广的模型——"
            f"推荐 {RECOMMENDED_MODEL}（83 种，含 Na）。"
        )
    return system_elements, table


def check_topology_element_coverage(topology, model: str | Path, *, system=None):
    """拓扑版入口。元素缺失时先由 `topology_atomic_numbers` 按
    「原子名 + System 质量双源一致」补，补不出来一样 fail-closed。"""

    from .openmm_plugin import topology_atomic_numbers

    return check_element_coverage(
        topology_atomic_numbers(topology, system=system), model
    )
