from __future__ import annotations

import math


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]
