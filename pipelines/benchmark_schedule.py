from __future__ import annotations

import math


def build_sampling_groups(
    total_rounds: int,
    group_size: int,
    max_parallel_groups: int,
) -> list[dict[str, int | str]]:
    if total_rounds <= 0:
        raise ValueError("total_rounds must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if max_parallel_groups <= 0:
        raise ValueError("max_parallel_groups must be positive")

    group_count = math.ceil(total_rounds / group_size)
    groups: list[dict[str, int | str]] = []
    for group_index in range(1, group_count + 1):
        round_start = (group_index - 1) * group_size + 1
        round_end = min(total_rounds, round_start + group_size - 1)
        chain_index = ((group_index - 1) % max_parallel_groups) + 1
        groups.append(
            {
                "group_index": group_index,
                "group_name": f"group_{group_index:02d}",
                "round_start": round_start,
                "round_end": round_end,
                "round_count": round_end - round_start + 1,
                "chain_index": chain_index,
            }
        )
    return groups
