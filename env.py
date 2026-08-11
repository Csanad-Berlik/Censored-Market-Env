"""
Sealed-bid market with censored, delayed, unobservable feedback.

The agent bids repeatedly under a budget against reactive competitors.
Three things are deliberately hidden from it:

  1. CENSORED PRICES  - on a lost auction it learns only that it lost.
                        The clearing price is never revealed.
  2. DELAYED OUTCOMES - a won impression may convert many steps later,
                        or never. Reward arrives long after the action.
  3. NO COUNTERFACTUAL - it never learns what a different bid would
                        have done in the same auction.

Written from scratch. No dependencies beyond numpy.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------
# Market configuration
# ----------------------------------------------------------------------


@dataclass
class MarketConfig:
    n_steps: int = 60
    auctions_per_step: int = 200
    budget: float = 2_000.0

    # Value model (hidden from the agent)
    base_conv_rate: float = 0.020
    conv_value: float = 120.0

    # Delay model (hidden). Log-logistic-ish: median ~8 steps, fat tail.
    delay_median: float = 8.0
    delay_shape: float = 1.4

    # Attribution window. Conversions after this are lost -- censored
    # for the agent AND for the scorer. Both are blind to them.
    attribution_window: int = 25

    # Regime shift. At `shift_step` the conversion rate is multiplied by
    # shift_down or shift_up. The DIRECTION is drawn per episode and never
    # announced. The agent can only infer which world it is in from delayed,
    # censored conversion arrivals. Set shift_down = shift_up = 1.0 for a
    # stationary market.
    shift_step: int = 18
    shift_down: float = 0.25
    shift_up: float = 3.0

    # Publisher floor, adapts to fill rate
    floor_init: float = 0.55
    floor_adapt: float = 0.02

    max_bid: float = 6.0


# ----------------------------------------------------------------------
# Competitors -- reactive, each with its own persistent state
# ----------------------------------------------------------------------


class Competitor:
    """Base archetype. Bids are shaded and floor-constrained."""

    name = "base"

    def __init__(self, rng: np.random.Generator, cfg: MarketConfig):
        self.rng = rng
        self.cfg = cfg
        self.spend = 0.0
        self.wins = 0

    def bid(self, n: int, floor: float, step: int) -> np.ndarray:
        raise NotImplementedError

    def observe(self, won: np.ndarray, price: np.ndarray) -> None:
        self.wins += int(won.sum())
        self.spend += float(price[won].sum())


class FloorFollower(Competitor):
    """Sits just above the floor. Cheap volume."""

    name = "floor_follower"

    def bid(self, n, floor, step):
        return floor * (1.02 + 0.35 * self.rng.random(n))


class Aggressive(Competitor):
    """Bids high early to take share, tapers as budget burns."""

    name = "aggressive"

    def __init__(self, rng, cfg):
        super().__init__(rng, cfg)
        self.budget = cfg.budget * 0.8

    def bid(self, n, floor, step):
        remaining = max(0.0, 1.0 - self.spend / self.budget)
        level = 0.8 + 0.9 * remaining
        return np.maximum(floor, level * (0.6 + 0.9 * self.rng.random(n)))


class PIDPacer(Competitor):
    """Targets an even spend curve, corrects each step."""

    name = "pid_pacer"

    def __init__(self, rng, cfg):
        super().__init__(rng, cfg)
        self.budget = cfg.budget
        self.level = 1.0

    def bid(self, n, floor, step):
        target = self.budget * (step + 1) / self.cfg.n_steps
        err = (target - self.spend) / max(target, 1.0)
        self.level = float(np.clip(self.level * (1.0 + 0.25 * err), 0.4, 3.0))
        return np.maximum(floor, self.level * (0.65 + 0.7 * self.rng.random(n)))


class Conservative(Competitor):
    """Low, stable bids. Only takes clearly cheap inventory."""

    name = "conservative"

    def bid(self, n, floor, step):
        return np.maximum(floor * 0.98, 0.55 + 0.55 * self.rng.random(n))


ARCHETYPES = [FloorFollower, Aggressive, PIDPacer, Conservative]


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------


@dataclass
class Observation:
    """Exactly what the agent is allowed to see."""

    step: int
    steps_remaining: int
    budget_remaining: float
    spend_last_step: float
    impressions_last_step: int
    impressions_total: int
    # Conversions that have ARRIVED so far. Not the ones pending.
    conversions_observed: int
    revenue_observed: float
    # Win rate is observable. Clearing prices on losses are not.
    win_rate_last_step: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class StepResult:
    obs: Observation
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class CensoredMarketEnv:
    """
    Two RNG streams:
      - market_rng  drives competitors, values, delays
      - agent_rng   unused by the env; reserved for the policy

    The market stream is seeded independently of the policy, so two
    policies at the same seed face an IDENTICAL market. This is common
    random numbers: a score difference between two agents is a real
    difference, not seed noise.
    """

    def __init__(self, cfg: MarketConfig | None = None, seed: int = 0):
        self.cfg = cfg or MarketConfig()
        self.seed = seed
        self.reset()

    # ---------------- lifecycle ----------------

    def reset(self) -> Observation:
        cfg = self.cfg
        self.market_rng = np.random.default_rng(self.seed)
        comp_seed = np.random.default_rng(self.seed + 10_000)

        self.competitors = [
            A(np.random.default_rng(comp_seed.integers(1 << 31)), cfg)
            for A in ARCHETYPES
        ]

        self.step_idx = 0
        self.budget_left = cfg.budget
        self.floor = cfg.floor_init

        self.impressions = 0
        self.spend = 0.0
        self.conversions_observed = 0
        self.revenue_observed = 0.0

        # Ground truth the agent never sees
        self._pending: list[tuple[int, float]] = []  # (arrival_step, value)
        self._conversions_true = 0
        self._revenue_true = 0.0

        # Drawn once per episode from the market stream, before anything
        # else consumes it, so it is stable under common random numbers.
        self._regime_down = bool(self.market_rng.random() < 0.5)

        self._last = dict(spend=0.0, imps=0, n=0)
        return self._observe()

    def _observe(self) -> Observation:
        last = self._last
        wr = last["imps"] / last["n"] if last["n"] else 0.0
        return Observation(
            step=self.step_idx,
            steps_remaining=self.cfg.n_steps - self.step_idx,
            budget_remaining=round(self.budget_left, 2),
            spend_last_step=round(last["spend"], 2),
            impressions_last_step=last["imps"],
            impressions_total=self.impressions,
            conversions_observed=self.conversions_observed,
            revenue_observed=round(self.revenue_observed, 2),
            win_rate_last_step=round(wr, 4),
        )

    # ---------------- core ----------------

    def step(self, bid: float) -> StepResult:
        cfg = self.cfg
        rng = self.market_rng
        n = cfg.auctions_per_step

        bid = float(np.clip(bid, 0.0, cfg.max_bid))

        # Competitors bid. Highest competitor sets the price to beat.
        comp_bids = np.stack(
            [c.bid(n, self.floor, self.step_idx) for c in self.competitors]
        )
        best_comp = comp_bids.max(axis=0)
        clearing = np.maximum(best_comp, self.floor)

        # First-price: we win when we clear both the floor and the field.
        won = (bid >= clearing) & (bid >= self.floor)

        # Budget truncation, in auction order.
        cost = np.where(won, bid, 0.0)
        cum = np.cumsum(cost)
        affordable = cum <= self.budget_left
        won = won & affordable

        n_won = int(won.sum())
        step_spend = float(bid * n_won)

        self.budget_left -= step_spend
        self.spend += step_spend
        self.impressions += n_won

        # Competitors observe their own outcomes.
        for i, c in enumerate(self.competitors):
            c_won = (comp_bids[i] >= clearing) & ~won
            c.observe(c_won, comp_bids[i])

        # Publishers adapt the floor to fill rate.
        fill = (won.sum() + (best_comp >= self.floor).sum()) / n
        self.floor *= 1.0 + cfg.floor_adapt * (fill - 0.5)
        self.floor = float(np.clip(self.floor, 0.2, cfg.max_bid * 0.8))

        # Schedule delayed conversions for the impressions we won.
        self._schedule(n_won, rng)

        # Collect whatever has arrived by now.
        arrived, arrived_value = self._collect(self.step_idx)
        self.conversions_observed += arrived
        self.revenue_observed += arrived_value

        self._last = dict(spend=step_spend, imps=n_won, n=n)
        self.step_idx += 1
        done = self.step_idx >= cfg.n_steps or self.budget_left <= 0.01

        if done:
            # Drain only what lands inside the attribution window.
            cutoff = self.step_idx + cfg.attribution_window
            extra, extra_value = self._collect(cutoff)
            self.conversions_observed += extra
            self.revenue_observed += extra_value

        reward = arrived_value - step_spend
        return StepResult(
            obs=self._observe(),
            reward=reward,
            done=done,
            info={"clearing_mean": float(clearing.mean())},
        )

    def _schedule(self, n_won: int, rng) -> None:
        if n_won == 0:
            return
        cfg = self.cfg
        rate = cfg.base_conv_rate
        if self.step_idx >= cfg.shift_step:
            rate *= cfg.shift_down if self._regime_down else cfg.shift_up
        converts = rng.random(n_won) < rate
        k = int(converts.sum())
        if k == 0:
            return
        # Log-logistic delay: median at delay_median, fat right tail.
        u = rng.random(k)
        delay = cfg.delay_median * (u / (1.0 - u)) ** (1.0 / cfg.delay_shape)
        value = cfg.conv_value * rng.lognormal(0.0, 0.35, k)
        for d, v in zip(delay, value):
            self._conversions_true += 1
            self._revenue_true += float(v)
            heapq.heappush(self._pending, (self.step_idx + int(d), float(v)))

    def _collect(self, up_to: int) -> tuple[int, float]:
        n, total = 0, 0.0
        while self._pending and self._pending[0][0] <= up_to:
            _, v = heapq.heappop(self._pending)
            n += 1
            total += v
        return n, total

    # ---------------- scoring ----------------

    def score(self) -> dict:
        """
        Profit as the agent can measure it, plus the truth for diagnosis.
        A benchmark should report `profit_observed`. `profit_true` exists
        so you can see how much value the attribution window hid.
        """
        return {
            "spend": round(self.spend, 2),
            "impressions": self.impressions,
            "conversions_observed": self.conversions_observed,
            "revenue_observed": round(self.revenue_observed, 2),
            "profit_observed": round(self.revenue_observed - self.spend, 2),
            "conversions_true": self._conversions_true,
            "profit_true": round(self._revenue_true - self.spend, 2),
            "budget_used": round(1.0 - self.budget_left / self.cfg.budget, 3),
        }


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------


def constant_policy(level: float):
    def policy(obs: Observation) -> float:
        return level

    return policy


def even_pace_policy(target_bid: float = 1.6):
    """Spends evenly; nudges the bid to stay on the budget curve."""

    def policy(obs: Observation) -> float:
        if obs.steps_remaining <= 0:
            return 0.0
        per_step = obs.budget_remaining / obs.steps_remaining
        if obs.impressions_last_step > 0:
            cpi = obs.spend_last_step / obs.impressions_last_step
            wanted = per_step / max(obs.impressions_last_step, 1)
            return float(np.clip(target_bid * (wanted / max(cpi, 1e-6)), 0.3, 5.0))
        return target_bid

    return policy


def greedy_roas_policy(target_roas: float = 1.5):
    """
    The trap. Reacts to observed ROAS -- which is censored early,
    because conversions have not arrived yet. Cuts bids on a cohort
    that was fine, just not matured.
    """
    level = [1.8]

    def policy(obs: Observation) -> float:
        spent = 2000.0 - obs.budget_remaining
        if spent > 1.0:
            roas = obs.revenue_observed / spent
            if roas < target_roas:
                level[0] *= 0.88
            else:
                level[0] *= 1.05
            level[0] = float(np.clip(level[0], 0.2, 5.0))
        return level[0]

    return policy


def run(policy, seed: int = 0, cfg: MarketConfig | None = None) -> dict:
    env = CensoredMarketEnv(cfg=cfg, seed=seed)
    obs = env.reset()
    done = False
    while not done:
        r = env.step(policy(obs))
        obs, done = r.obs, r.done
    return env.score()


def maturity_aware_policy(target_roas: float = 2.0, assumed_median: float = 10.0,
                          assumed_shape: float = 1.4):
    """
    The same reactive logic as greedy_roas -- but it corrects the observed
    signal for maturity before reacting to it.

    At step t, an impression won at step s has had (t - s) steps to convert.
    Under an assumed log-logistic delay, the expected matured fraction is
    F(d) = d^k / (d^k + m^k). Dividing observed revenue by the spend-weighted
    matured fraction recovers an unbiased estimate of eventual ROAS.

    Note the assumed median is 10 while the environment's true median is 8.
    The correction does not require the right model, only a roughly right one.
    """
    state = {"level": 1.3, "imps": [], "spend": 0.0}

    def policy(obs: Observation) -> float:
        st = state
        st["imps"].append((obs.step, obs.impressions_last_step))
        st["spend"] = 2000.0 - obs.budget_remaining

        if st["spend"] < 50.0:
            return st["level"]

        total = sum(n for _, n in st["imps"])
        if total == 0:
            return st["level"]

        matured = 0.0
        for s, n in st["imps"]:
            d = max(obs.step - s, 0.0)
            f = 0.0 if d <= 0 else d ** assumed_shape / (
                d ** assumed_shape + assumed_median ** assumed_shape
            )
            matured += n * f
        frac = max(matured / total, 0.05)

        est_revenue = obs.revenue_observed / frac
        roas = est_revenue / st["spend"]

        # Maturity correction alone is not enough: early on, zero observed
        # conversions divided by any fraction is still zero. So weight the
        # reaction by how much evidence has actually arrived. With few
        # matured conversions the policy barely moves -- it does not read
        # "no conversions yet" as "this is failing".
        w = obs.conversions_observed / (obs.conversions_observed + 8.0)
        adj = (1.06 if roas > target_roas else 0.94) ** w
        st["level"] *= adj
        st["level"] = float(np.clip(st["level"], 0.3, 4.0))
        return st["level"]

    return policy
