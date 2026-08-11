"""
Invariant tests. Run: python test_invariants.py

These prove dose-response and reproducibility under the perturbations
covered here. They do NOT prove the reward is unhackable. If you break
it, that is the most useful thing you can tell me.
"""

import numpy as np

from env import (
    CensoredMarketEnv,
    MarketConfig,
    constant_policy,
    even_pace_policy,
    run,
)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, PASS if cond else FAIL, detail))


# A. Bid monotonicity -- the load-bearing claim.
#    A policy bidding >= another at every step must win >= impressions.
def test_monotonicity():
    # Budget made non-binding. Under a binding budget this invariant is
    # FALSE by design -- a higher bid buys fewer impressions with the same
    # money. See the note on test B.
    big = MarketConfig(budget=1e9)
    for seed in range(8):
        lo = run(constant_policy(1.2), seed=seed, cfg=big)
        hi = run(constant_policy(2.4), seed=seed, cfg=big)
        ok = hi["impressions"] >= lo["impressions"]
        check(
            f"A/monotone_impressions seed={seed}",
            ok,
            f"{lo['impressions']} -> {hi['impressions']}",
        )


# B. Dose-response on WIN RATE, budget made non-binding.
#
#    NOTE: cumulative impressions are deliberately NOT asserted here.
#    A higher bid exhausts the budget sooner, so the impression curve
#    rises and then falls. That is correct economics, not a defect --
#    and it is exactly the kind of thing an agent has to reason about.
#    The invariant that must hold is: at a fixed step, with budget
#    non-binding, a higher bid cannot lower the win rate.
def test_dose_response():
    big = MarketConfig(budget=1e9, n_steps=3)
    for seed in range(4):
        rates = []
        for b in [1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]:
            env = CensoredMarketEnv(cfg=big, seed=seed)
            r = env.step(b)
            rates.append(r.obs.win_rate_last_step)
        ok = all(a <= b + 1e-12 for a, b in zip(rates, rates[1:]))
        # A grid that is mostly zeros would pass monotonicity vacuously.
        # Require a real curve: most points nonzero, and genuine spread.
        informative = sum(r > 0 for r in rates) >= 5 and (max(rates) - min(rates)) > 0.4
        check(f"B/dose_response_win_rate seed={seed}", ok and informative, str(rates))


# C. Determinism: same seed, same policy, identical result.
def test_determinism():
    for seed in range(4):
        a = run(even_pace_policy(), seed=seed)
        b = run(even_pace_policy(), seed=seed)
        check(f"C/determinism seed={seed}", a == b)


# D. Common random numbers: two different policies at the same seed
#    must face the same market. Check competitor bids match on step 0.
def test_common_random_numbers():
    e1 = CensoredMarketEnv(seed=3)
    e2 = CensoredMarketEnv(seed=3)
    b1 = np.stack([c.bid(50, 0.55, 0) for c in e1.competitors])
    b2 = np.stack([c.bid(50, 0.55, 0) for c in e2.competitors])
    check("D/crn_identical_market", np.allclose(b1, b2))


# E. Budget is a hard constraint.
def test_budget():
    for seed in range(4):
        s = run(constant_policy(5.0), seed=seed)
        check(f"E/budget_respected seed={seed}", s["spend"] <= 2000.0 + 1e-6,
              f"spend={s['spend']}")


# F. Censoring is real: the attribution window must hide value.
#    profit_true should exceed profit_observed when spend is nonzero.
def test_censoring_hides_value():
    hidden = 0
    for seed in range(8):
        s = run(constant_policy(2.0), seed=seed)
        if s["conversions_true"] > s["conversions_observed"]:
            hidden += 1
    check("F/window_censors_conversions", hidden > 0,
          f"{hidden}/8 seeds had unobserved conversions")


# G. The observation must not leak ground truth.
def test_observation_firewall():
    env = CensoredMarketEnv(seed=1)
    r = env.step(1.8)
    leaked = [k for k in r.obs.as_dict()
              if "true" in k or "clearing" in k or "floor" in k]
    check("G/no_ground_truth_in_obs", not leaked, str(leaked))


# H. Zero bid wins nothing and spends nothing.
def test_zero_bid():
    s = run(constant_policy(0.0), seed=0)
    check("H/zero_bid_no_spend", s["impressions"] == 0 and s["spend"] == 0.0)


if __name__ == "__main__":
    for t in [
        test_monotonicity,
        test_dose_response,
        test_determinism,
        test_common_random_numbers,
        test_budget,
        test_censoring_hides_value,
        test_observation_firewall,
        test_zero_bid,
    ]:
        t()

    width = max(len(n) for n, _, _ in results)
    for name, status, detail in results:
        print(f"{name:<{width}}  {status}  {detail}")
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    raise SystemExit(1 if n_fail else 0)
