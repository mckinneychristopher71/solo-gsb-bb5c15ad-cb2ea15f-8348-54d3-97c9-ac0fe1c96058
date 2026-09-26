"""核心算法单元测试。"""

from __future__ import annotations

from itertools import combinations

from app.redaction import (
    Fragment,
    MASK_CHAR,
    mask_text,
    select_fragments,
    select_fragments_with_budget,
)


def f(fid: int, weight: int, start: int, end: int) -> Fragment:
    return Fragment(id=fid, weight=weight, start=start, end=end)


def test_empty_input_selects_nothing() -> None:
    assert select_fragments([]) == ()


def test_non_overlapping_all_selected_including_adjacent() -> None:
    frags = [f(2, 1, 3, 5), f(1, 2, 0, 3)]
    assert select_fragments(frags) == (1, 2)


def test_overlapping_prefers_higher_weight() -> None:
    # [0,5) 权重 5 vs [0,3)+[3,5) 权重 3+1=4
    frags = [f(1, 5, 0, 5), f(2, 3, 0, 3), f(3, 1, 3, 5)]
    assert select_fragments(frags) == (1,)


def test_weight_tie_prefers_more_chars() -> None:
    # 两组权重同为 5：单片段覆盖 5 码点，组合只覆盖 3 码点。
    frags = [f(1, 5, 0, 5), f(2, 3, 0, 2), f(3, 2, 2, 3)]
    assert select_fragments(frags) == (1,)


def test_full_tie_prefers_smallest_id_list_lexicographically() -> None:
    # {1} 与 {2} 权重、长度完全相同，取 ID 较小者。
    frags = [f(2, 1, 0, 1), f(1, 1, 0, 1)]
    assert select_fragments(frags) == (1,)

    # 比较完整升序列表：[2] < [2, 1] 不可能；此处 [1, 3] < [2]。
    frags = [f(1, 1, 0, 1), f(3, 1, 0, 1), f(2, 2, 1, 2)]
    assert select_fragments(frags) == (1, 2)


def test_string_ids_lexicographic() -> None:
    frags = [
        Fragment(id="b", weight=1, start=0, end=1),
        Fragment(id="a", weight=1, start=0, end=1),
    ]
    assert select_fragments(frags) == ("a",)


def test_mask_text_simple_and_adjacent() -> None:
    text = "你好，世界！"  # 6 个码点
    frags = [f(1, 1, 0, 2), f(2, 1, 3, 5)]
    masked = mask_text(text, frags)
    assert masked == MASK_CHAR * 2 + "，" + MASK_CHAR * 2 + "！"
    assert len(masked) == len(text)


def test_mask_text_handles_astral_code_points() -> None:
    text = "a😀b"  # 4 个码点：a, 😀(代理对), b -> 索引 0,1,2,3
    masked = mask_text(text, [f(1, 1, 1, 2)])
    assert masked == "a" + MASK_CHAR + "b"


# ---------------------------------------------------------------------------
# 随机化正确性：动态规划结果必须与 2^n 暴力枚举一致。
# ---------------------------------------------------------------------------


def _brute_force(frags: list[Fragment]) -> tuple[int, ...]:
    def compatible(subset: tuple[Fragment, ...]) -> bool:
        ordered = sorted(subset, key=lambda x: x.start)
        return all(a.end <= b.start for a, b in zip(ordered, ordered[1:]))

    best: tuple[int, int, tuple[int, ...]] | None = None
    for size in range(len(frags) + 1):
        for subset in combinations(frags, size):
            if not compatible(subset):
                continue
            key = (
                sum(x.weight for x in subset),
                sum(x.length for x in subset),
                tuple(sorted(x.id for x in subset)),
            )
            candidate = (key[0], key[1], key[2])
            if (
                best is None
                or candidate[:2] > best[:2]
                or (candidate[:2] == best[:2] and candidate[2] < best[2])
            ):
                best = candidate
    assert best is not None
    return best[2]


def test_dp_matches_brute_force_random() -> None:
    import random

    rng = random.Random(20260926)
    for trial in range(300):
        n = rng.randint(0, 7)
        horizon = rng.randint(1, 8)
        ids = rng.sample(range(100), n) if n else []
        frags: list[Fragment] = []
        for i in range(n):
            start = rng.randrange(horizon)
            end = rng.randrange(start + 1, horizon + 1)
            frags.append(Fragment(id=ids[i], weight=rng.randint(1, 4), start=start, end=end))
        expected = _brute_force(frags)
        assert select_fragments(frags) == expected, f"trial {trial}: {frags}"


# ---------------------------------------------------------------------------
# select_fragments_with_budget
# ---------------------------------------------------------------------------


def test_budget_zero_or_empty_selects_nothing() -> None:
    frags = [f(1, 1, 0, 1)]
    assert select_fragments_with_budget(frags, 0) == ()
    assert select_fragments_with_budget([], 0) == ()
    assert select_fragments_with_budget([], 5) == ()


def test_budget_large_enough_matches_unconstrained() -> None:
    frags = [f(1, 5, 0, 5), f(2, 3, 0, 3), f(3, 1, 3, 5)]
    assert select_fragments_with_budget(frags, 5) == (1,)
    assert select_fragments_with_budget(frags, 10**6) == select_fragments(frags)


def test_budget_excludes_too_expensive_optimum() -> None:
    # 无约束取 {1}（权重 5，长 3）；预算 2 时只有 {2} 可入选。
    frags = [f(1, 5, 0, 3), f(2, 3, 0, 2), f(3, 1, 2, 5)]
    assert select_fragments_with_budget(frags, 2) == (2,)
    # 预算 1 时所有片段都超长，无法入选。
    assert select_fragments_with_budget(frags, 1) == ()


def test_budget_adjacent_fragments_combined_when_fit() -> None:
    frags = [f(1, 3, 0, 2), f(2, 1, 2, 3), f(3, 5, 0, 3)]
    # 预算 3：{3} 权重 5 与 {1,2} 权重 4 均恰用 3 码点，取权重更高的 {3}。
    assert select_fragments_with_budget(frags, 3) == (3,)
    # 预算 2：{1} 权重 3 胜过 {3} 无法入选的情况，只剩 {1}。
    assert select_fragments_with_budget(frags, 2) == (1,)
    # 相邻片段 {1}、{2} 总长 3；预算 3 下另设场景使其胜出。
    frags2 = [f(1, 3, 0, 2), f(2, 2, 2, 3)]
    assert select_fragments_with_budget(frags2, 3) == (1, 2)


def test_budget_weight_tie_prefers_more_chars() -> None:
    # {1}（长 1，权重 2）与 {2}（长 2，权重 2）相邻；预算 2 下两者不能
    # 组合（合计 3 码点），权重并列时取遮盖码点更多的 {2}。
    frags = [f(1, 2, 0, 1), f(2, 2, 1, 3)]
    assert select_fragments_with_budget(frags, 2) == (2,)
    assert select_fragments_with_budget(frags, 3) == (1, 2)


def test_budget_full_tie_prefers_smallest_id_list() -> None:
    frags = [
        Fragment(id=2, weight=1, start=0, end=1),
        Fragment(id=1, weight=1, start=0, end=1),
    ]
    assert select_fragments_with_budget(frags, 1) == (1,)


def test_budget_string_ids_lexicographic() -> None:
    frags = [
        Fragment(id="b", weight=1, start=0, end=2),
        Fragment(id="a", weight=1, start=0, end=2),
        Fragment(id="c", weight=1, start=2, end=3),
    ]
    assert select_fragments_with_budget(frags, 3) == ("a", "c")
    assert select_fragments_with_budget(frags, 2) == ("a",)


def _brute_force_with_budget(
    frags: list[Fragment], budget: int
) -> tuple[int, ...]:
    def compatible(subset: tuple[Fragment, ...]) -> bool:
        ordered = sorted(subset, key=lambda x: x.start)
        return all(a.end <= b.start for a, b in zip(ordered, ordered[1:]))

    best: tuple[int, int, tuple[int, ...]] | None = None
    for size in range(len(frags) + 1):
        for subset in combinations(frags, size):
            if not compatible(subset):
                continue
            used = sum(x.length for x in subset)
            if used > budget:
                continue
            candidate = (
                sum(x.weight for x in subset),
                used,
                tuple(sorted(x.id for x in subset)),
            )
            if (
                best is None
                or candidate[:2] > best[:2]
                or (candidate[:2] == best[:2] and candidate[2] < best[2])
            ):
                best = candidate
    assert best is not None
    return best[2]


def test_budget_dp_matches_brute_force_random() -> None:
    import random

    rng = random.Random(20260927)
    for trial in range(400):
        n = rng.randint(0, 8)
        horizon = rng.randint(1, 9)
        ids = rng.sample(range(100), n) if n else []
        frags: list[Fragment] = []
        for i in range(n):
            start = rng.randrange(horizon)
            end = rng.randrange(start + 1, horizon + 1)
            frags.append(Fragment(id=ids[i], weight=rng.randint(1, 4), start=start, end=end))
        # 覆盖 0、紧预算、超过总长度等各种预算，稠密/稀疏两条实现路径都会走到。
        budget = rng.randint(0, horizon + 3)
        expected = _brute_force_with_budget(frags, budget)
        assert select_fragments_with_budget(frags, budget) == expected, (
            f"trial {trial}: budget={budget}, {frags}"
        )

