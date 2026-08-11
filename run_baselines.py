"""
Run the baseline policies and print the score table.

    python run_baselines.py --seeds 32

Read the standard deviations before reading the means. Episode variance
is large relative to the differences between policies, so a comparison
on a handful of seeds is not a result. See the README.
"""

import argparse

import numpy as np

from env import (
    constant_policy,
    even_pace_policy,
    greedy_roas_policy,
    maturity_aware_policy,
    run,
)

# Factories, not instances. Several baselines carry state in a closure,
# so reusing one instance across seeds leaks the bid level from the last
# episode into the next. That is a 60x error on greedy_roas. Every episode
# gets a fresh policy.
BASELINES = {
    "constant_0.9": lambda: constant_policy(0.9),
    "constant_1.0": lambda: constant_policy(1.0),
    "constant_1.2": lambda: constant_policy(1.2),
    "constant_1.5": lambda: constant_policy(1.5),
    "even_pace": lambda: even_pace_policy(),
    "greedy_roas": lambda: greedy_roas_policy(),
    "maturity_aware": lambda: maturity_aware_policy(),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=32)
    args = ap.parse_args()

    hdr = (
        f"{'policy':<17}{'imps':>7}{'budget':>8}{'conv_obs':>10}"
        f"{'profit':>10}{'sd':>9}{'sem':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name, make_policy in BASELINES.items():
        rows = [run(make_policy(), seed=s) for s in range(args.seeds)]
        profit = np.array([r["profit_observed"] for r in rows])
        print(
            f"{name:<17}"
            f"{np.mean([r['impressions'] for r in rows]):>7.0f}"
            f"{np.mean([r['budget_used'] for r in rows]) * 100:>7.0f}%"
            f"{np.mean([r['conversions_observed'] for r in rows]):>10.1f}"
            f"{profit.mean():>10.0f}"
            f"{profit.std():>9.0f}"
            f"{profit.std() / np.sqrt(args.seeds):>8.0f}"
        )

    print(f"\n{args.seeds} seeds. profit = observed revenue - spend.")
    print("Differences smaller than ~2x sem are not distinguishable.")


if __name__ == "__main__":
    main()
