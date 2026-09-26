"""敏感片段选择与遮盖的核心算法。

本模块与 HTTP 框架解耦，便于单元测试。

约定：
- 文本位置按 Unicode 码点（Python ``str`` 的索引）计数，区间为左闭右开
  ``[start, end)``；``[0, 3)`` 与 ``[3, 5)`` 互不重叠，可同时入选。
- 片段 ID 允许为整数或字符串，一次请求内类型须一致。
- 编辑（``Edit``）把原文 ``[start, end)`` 替换为 ``replacement``；
  ``start == end`` 时为纯插入，``replacement == ""`` 时为纯删除。
  一批编辑须互不重叠，应用时按起点排序后一次性生效。
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence

# 遮盖字符：U+2588 FULL BLOCK，一个码点对应一个被遮盖码点。
MASK_CHAR = "█"

ID = int | str


@dataclass(frozen=True)
class Fragment:
    """一条候选敏感片段。"""

    id: ID
    weight: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Edit:
    """一处对原文的编辑：把 ``[start, end)`` 替换为 ``replacement``。"""

    start: int
    end: int
    replacement: str

    @property
    def is_insertion(self) -> bool:
        """纯插入（空区间）。"""
        return self.start == self.end

    @property
    def delta(self) -> int:
        """该编辑引起的码点位移（新长度 - 旧长度）。"""
        return len(self.replacement) - (self.end - self.start)


def apply_edits(text: str, edits: Sequence[Edit]) -> str:
    """把互不重叠的编辑一次性应用到原文，返回修订后文本。

    所有区间均按原文定位；编辑之间互不影响（同一位置的插入先于
    同起点的替换生效）。调用方须保证编辑已经过校验、互不重叠。
    """
    parts: list[str] = []
    cursor = 0
    for edit in sorted(edits, key=lambda e: (e.start, e.end)):
        parts.append(text[cursor : edit.start])
        parts.append(edit.replacement)
        cursor = edit.end
    parts.append(text[cursor:])
    return "".join(parts)


def migrate_fragment(fragment: Fragment, edits: Sequence[Edit]) -> Fragment | None:
    """把片段区间迁移到修订后文本；被编辑波及时返回 ``None``（作废）。

    作废规则：
    - 删除/替换区间与片段有交集；
    - 插入点严格落在片段内部（``start < p < end``）。

    迁移规则（片段长度不变，整体平移）：
    - 插入在片段起点：整段右移，插入内容不并入片段；
    - 插入在片段终点：不扩展片段，区间不变；
    - 其余情况按前方编辑的累计码点位移平移区间。
    """
    shift = 0
    for edit in edits:
        if edit.is_insertion:
            if fragment.start < edit.start < fragment.end:
                return None
            if edit.start <= fragment.start:
                shift += edit.delta
        else:
            if edit.start < fragment.end and fragment.start < edit.end:
                return None
            if edit.end <= fragment.start:
                shift += edit.delta
    return Fragment(
        id=fragment.id,
        weight=fragment.weight,
        start=fragment.start + shift,
        end=fragment.end + shift,
    )


def _insert_sorted(ids: tuple[ID, ...], value: ID) -> tuple[ID, ...]:
    """把 ``value`` 有序插入已升序排列的 ``ids``。"""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if ids[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return ids[:lo] + (value,) + ids[lo:]


def select_fragments(fragments: Sequence[Fragment]) -> tuple[ID, ...]:
    """从候选片段中选出互不重叠的最优组合。

    目标依次为：
    1. 权重总和最大；
    2. 权重总和并列时，遮盖字符数最多；
    3. 仍并列时，取入选 ID 升序列表字典序最小者。

    使用带权区间调度动态规划，按右端点排序，``p[i]`` 指向与第 ``i`` 个
    片段相容（``end <= start``）的最后一个片段。

    返回入选 ID 的升序元组。
    """
    if not fragments:
        return ()

    # 唯一确定的排序，便于结果可复现；最优解由比较目标决定，与顺序无关。
    ordered = sorted(fragments, key=lambda f: (f.end, f.start, f.id))
    ends = [f.end for f in ordered]

    # dp[i]：仅考虑前 i 个片段（ordered[0:i]）时的最优状态
    #   (权重总和, 遮盖字符数, 入选 ID 升序元组)
    dp: list[tuple[int, int, tuple[ID, ...]]] = [(0, 0, ())]

    for i, frag in enumerate(ordered, start=1):
        p = bisect_right(ends, frag.start, 0, i - 1)
        base_weight, base_len, base_ids = dp[p]
        take = (
            base_weight + frag.weight,
            base_len + frag.length,
            _insert_sorted(base_ids, frag.id),
        )
        skip = dp[i - 1]
        # 元组按 (权重, 字符数, ID 列表字典序) 比较；前两项求最大，
        # 第三项取字典序最小。
        if take[:2] > skip[:2] or (take[:2] == skip[:2] and take[2] < skip[2]):
            dp.append(take)
        else:
            dp.append(skip)

    return dp[-1][2]


def select_fragments_with_budget(
    fragments: Sequence[Fragment], max_masked_chars: int
) -> tuple[ID, ...]:
    """预算受限版的 :func:`select_fragments`。

    只在互不重叠（相邻可以）且入选片段长度之和——即遮盖码点总数——不超过
    ``max_masked_chars`` 的组合中择优，目标优先级与 :func:`select_fragments`
    完全相同：权重总和最大 → 遮盖码点数最多 → 入选 ID 升序列表字典序最小。

    先求无约束最优：若其遮盖码点数已在预算内则直接复用，保证预算宽松时与
    :func:`select_fragments` 结果一致；否则做带容量的带权区间调度 DP。
    容量较小时按容量稠密递推，容量较大而片段较少时按已用码点数稀疏枚举。
    """
    if not fragments or max_masked_chars <= 0:
        return ()

    by_id = {f.id: f for f in fragments}
    unconstrained = select_fragments(fragments)
    if sum(by_id[i].length for i in unconstrained) <= max_masked_chars:
        return unconstrained

    ordered = sorted(fragments, key=lambda f: (f.end, f.start, f.id))
    ends = [f.end for f in ordered]
    n = len(ordered)

    # 稠密递推每轮维护 budget+1 个容量格；稀疏枚举每轮至多 2^n 个可达和。
    # 取格数估计较小者；n 较大时 2^n 截断为无穷，自然走稠密递推。
    sparse_states = 1 << n if n < 32 else None
    if sparse_states is None or (max_masked_chars + 1) <= sparse_states:
        final_row = _budget_dp_dense(ordered, ends, max_masked_chars)
        candidates = (state for state in final_row if state is not None)
    else:
        final_row = _budget_dp_sparse(ordered, ends, max_masked_chars)
        candidates = final_row.values()

    best: BudgetState | None = None
    for state in candidates:
        if best is None or _prefer_budget_state(state, best):
            best = state
    return best[2] if best is not None else ()


# 预算 DP 状态：(权重总和, 遮盖码点数, 入选 ID 升序元组)。
BudgetState = tuple[int, int, tuple[ID, ...]]


def _prefer_budget_state(candidate: BudgetState, current: BudgetState) -> bool:
    """``candidate`` 是否按（权重最大、码点数最多、ID 列表字典序最小）更优。"""
    return candidate[:2] > current[:2] or (
        candidate[:2] == current[:2] and candidate[2] < current[2]
    )


def _budget_dp_dense(
    ordered: Sequence[Fragment], ends: Sequence[int], budget: int
) -> list[BudgetState | None]:
    """按容量稠密递推，返回末轮各“恰好使用码点数”上的最优状态。

    ``dp[k][c]`` 为只考虑前 ``k`` 个片段、恰好遮盖 ``c`` 个码点时的最优
    状态；第 ``i`` 个片段的相容前缀由 ``bisect_right`` 指向 ``dp[p]``。
    """
    dp: list[list[BudgetState | None]] = [
        [(0, 0, ())] + [None] * budget
    ]
    for i, frag in enumerate(ordered, start=1):
        p = bisect_right(ends, frag.start, 0, i - 1)
        row: list[BudgetState | None] = list(dp[i - 1])  # 不选：继承前轮
        base_row = dp[p]
        length = frag.length
        if length <= budget:
            for used in range(budget + 1 - length):
                base = base_row[used]
                if base is None:
                    continue
                take: BudgetState = (
                    base[0] + frag.weight,
                    used + length,
                    _insert_sorted(base[2], frag.id),
                )
                target = used + length
                current = row[target]
                if current is None or _prefer_budget_state(take, current):
                    row[target] = take
        dp.append(row)
    return dp[-1]


def _budget_dp_sparse(
    ordered: Sequence[Fragment], ends: Sequence[int], budget: int
) -> dict[int, BudgetState]:
    """:func:`_budget_dp_dense` 的稀疏版本：只记录可达的已用码点数。"""
    dp: list[dict[int, BudgetState]] = [{0: (0, 0, ())}]
    for i, frag in enumerate(ordered, start=1):
        p = bisect_right(ends, frag.start, 0, i - 1)
        row = dict(dp[i - 1])  # 不选：继承前轮
        for used, base in dp[p].items():
            target = used + frag.length
            if target > budget:
                continue
            take = (
                base[0] + frag.weight,
                target,
                _insert_sorted(base[2], frag.id),
            )
            current = row.get(target)
            if current is None or _prefer_budget_state(take, current):
                row[target] = take
        dp.append(row)
    return dp[-1]


def mask_text(text: str, fragments: Sequence[Fragment]) -> str:
    """用等长的 ``█`` 遮盖入选片段覆盖到的码点。

    ``fragments`` 须已保证互不重叠（相邻可以）。重叠区域只遮盖一次。
    """
    covered = bytearray(len(text))
    for frag in fragments:
        covered[frag.start : frag.end] = b"\x01" * frag.length
    chars = list(text)
    for i, flag in enumerate(covered):
        if flag:
            chars[i] = MASK_CHAR
    return "".join(chars)
