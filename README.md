# Censored Market

A sealed-bid market where an agent bids under a budget against reactive
competitors, and **cannot observe whether its actions worked.**

Three things are hidden from it by construction:

1. **Censored prices** — on a lost auction it learns only that it lost.
   The clearing price is never revealed.
2. **Delayed outcomes** — a won impression may convert many steps later,
   or never. Reward arrives long after the action that caused it.
3. **No counterfactual** — it never learns what a different bid would
   have done in the same auction.

There is also an unannounced regime shift partway through each episode:
the conversion rate moves up or down, direction drawn per episode, and
the only evidence is the delayed arrivals themselves.

Written from scratch. numpy only.

---

## Status: this is not a benchmark yet

Read this section before running anything.

The environment works and the invariants hold. But it does **not**
currently separate policies well enough to score agents on, for two
measured reasons:

**Profit is dominated by budget mechanics.** Every impression has
positive expected value, so the best strategy is close to "spend the
whole budget as cheaply as possible." Bid 1.0 wins because it buys the
most impressions per euro, not because it reasons about anything. An
oracle told the hidden regime direction in advance gains only ~16% over
the best constant bid — the information the environment is supposed to
be testing is worth very little in it.

**Variance swamps the differences.** At 32 seeds the best baseline
returns 4134 with a standard deviation of 2944. Most pairwise
comparisons in the table below are noise. Distinguishing policies
reliably would need hundreds of seeds, or a redesign that reduces
episode variance.

Fixing the first needs a value model where indiscriminate buying loses
money, so the agent has to decide rather than just spend efficiently.
That is a rebuild, not a tune. It is not done.

What the repo is good for today: a working, tested, reproducible
simulation of censored and delayed feedback that you can attach a policy
or an agent to. Not a leaderboard.

---

## Baselines

```
python run_baselines.py --seeds 32
```

```
policy              imps  budget  conv_obs    profit       sd     sem
---------------------------------------------------------------------
constant_0.9        1050     47%      25.1      2272     1556     275
constant_1.0        1998    100%      47.8      4134     2944     521
constant_1.2        1666    100%      31.4      1996      571     101
constant_1.5        1333    100%      25.2      1176      676     120
even_pace           1250    100%      21.3       662      537      95
greedy_roas          669     41%      22.1      2013     3447     609
maturity_aware      1480    100%      27.9      1513      640     113
```

Note what this says. The best policy is a **fixed bid that ignores all
feedback.** Both adaptive policies — the naive one and the one that
corrects for maturity — lose to it. That is an honest null result, not a
demonstration of anything. If adaptation mattered here, `maturity_aware`
would win, and it does not.

`greedy_roas` reacts to observed ROAS, which is censored early because
conversions have not arrived yet. It underspends heavily (41% of budget)
and its variance is the largest in the table. That behaviour is real and
reproducible, but at 32 seeds it is not cleanly separated from the
constant baselines, and earlier much starker numbers for it turned out to
be a state-leakage bug in the runner rather than a property of the
policy. Do not quote a ratio from this table without rerunning it.

---

## Install and run

```bash
pip install numpy
python test_invariants.py           # 8 tests, 24 checks, <1s
python run_baselines.py --seeds 32  # ~2s
```

A policy is any callable `obs -> float`:

```python
from env import CensoredMarketEnv

env = CensoredMarketEnv(seed=0)
obs = env.reset()
done = False
while not done:
    result = env.step(my_policy(obs))
    obs, done = result.obs, result.done
print(env.score())
```

`Observation` exposes only what a real bidder would see: step, budget
remaining, last-step spend and impressions, cumulative impressions,
**conversions that have arrived so far**, and last-step win rate. Not the
floor, not the conversion rate, not pending conversions, not the regime.

---

## Invariants

`test_invariants.py` — 8 test functions, most parameterised over several
seeds, producing 24 checks in total. Runs in under a second.

- **A** bid monotonicity — a higher constant bid wins at least as many
  impressions, under a non-binding budget
- **B** dose-response on win rate, with a guard that fails the test if
  the bid grid degenerates to mostly zeros (this caught a vacuous
  version of itself)
- **C** determinism — same seed, same policy, identical result
- **D** common random numbers — two policies at the same seed face an
  identical market, so a score difference is a real difference
- **E** budget is a hard constraint
- **F** the attribution window actually censors conversions
- **G** no ground truth leaks into the observation
- **H** a zero bid wins nothing and spends nothing

**A and B only hold under a non-binding budget.** Under a binding budget
a higher bid buys *fewer* impressions with the same money, which is
correct economics and would make the invariant false. Both tests set the
budget high for this reason.

These prove dose-response and reproducibility under the perturbations
covered here. They do **not** prove the reward is unhackable. Simulator
bugs, observation artifacts, seed artifacts and unintended action
combinations are all live possibilities. If you break it, that is more
useful to me than a passing run.

---

## Calibration honesty

The market dynamics are structurally plausible but are **not** validated
against real auction data. Real exchanges do not report clearing prices
on lost auctions, which is the same censoring the environment models.

This is a synthetic environment. It is not a claim about any real
market's behaviour, and no result here should be read as one.

---

## Licence

MIT
