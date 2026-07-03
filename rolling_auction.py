"""
Rolling Weekly Auction Model
============================
Extends the static LP from simulator.py into a 26-week time-series simulation.

Each week the auction runs three steps:
  1. Age funded invoices by one week; repay matured ones to the buyer.
  2. Each of the 45 suppliers independently submits a new invoice with
     probability ARRIVAL_PROB (default 35%).
  3. Run the LP on the current unfunded book; add winners to the funded
     book and reduce available capital.

Capital recycles: repayments (principal + discount) re-enter the pool each
week, letting the programme compound beyond the initial capital pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from simulator import DEFAULT_CAPITAL_POOL, TIERS, generate_suppliers, optimise_allocation

ARRIVAL_PROB: float = 0.35
N_WEEKS: int = 26


@dataclass
class WeeklySnapshot:
    week: int
    capital_available: float
    capital_deployed: float      # total currently locked in live funded invoices
    recycled_this_week: float    # principal + discount returned from maturities
    discount_this_week: float    # new discount earned from invoices funded this week
    cumulative_discount: float
    n_funded_this_week: int      # new invoices funded in this week's auction
    n_active: int                # total live funded invoices (not yet matured)


@dataclass
class RollingAuctionResult:
    weekly: List[WeeklySnapshot]
    scenario_label: str
    initial_capital: float

    @property
    def total_discount(self) -> float:
        return self.weekly[-1].cumulative_discount if self.weekly else 0.0

    @property
    def avg_weekly_deployed(self) -> float:
        return float(np.mean([s.capital_deployed for s in self.weekly]))

    @property
    def avg_funded_per_week(self) -> float:
        return float(np.mean([s.n_funded_this_week for s in self.weekly]))

    @property
    def peak_active_invoices(self) -> int:
        return max((s.n_active for s in self.weekly), default=0)

    @property
    def weeks_at_full_deployment(self) -> int:
        threshold = self.initial_capital * 0.99
        return sum(1 for s in self.weekly if s.capital_deployed >= threshold)

    def summary_dict(self) -> dict:
        return {
            "Scenario": self.scenario_label,
            "Total Discount ($k)": round(self.total_discount / 1e3, 1),
            "Avg Weekly Deployed ($M)": round(self.avg_weekly_deployed / 1e6, 2),
            "Avg Funded / Week": round(self.avg_funded_per_week, 1),
            "Peak Active Invoices": self.peak_active_invoices,
            "Weeks at Full Deployment": self.weeks_at_full_deployment,
        }

    def weekly_df(self) -> pd.DataFrame:
        return pd.DataFrame([vars(s) for s in self.weekly])


def run_rolling_auction(
    capital_pool: float = DEFAULT_CAPITAL_POOL,
    n_weeks: int = N_WEEKS,
    arrival_prob: float = ARRIVAL_PROB,
    stress_tier3: bool = False,
    rate_env_multiplier: float = 1.0,
    rng: np.random.Generator = None,
    scenario_label: str = "Baseline",
) -> RollingAuctionResult:
    """
    Simulate n_weeks of weekly auctions with capital recycling.

    Stress parameters mirror generate_suppliers() so the same economic
    scenarios (liquidity stress, rising rates, combined) can be applied.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Fixed supplier characteristics per scenario; invoice amounts/tenors
    # are re-drawn randomly each time a supplier submits a new invoice.
    supplier_base = generate_suppliers(
        stress_tier3=stress_tier3,
        rate_env_multiplier=rate_env_multiplier,
        rng=np.random.default_rng(42),
    )

    available_capital = float(capital_pool)
    funded_book: List[dict] = []   # live funded invoices not yet matured
    weekly: List[WeeklySnapshot] = []
    cumulative_discount = 0.0

    for week in range(1, n_weeks + 1):
        # Step 1: age funded invoices; recycle matured ones
        recycled = 0.0
        discount_this_week = 0.0  # discount realized from maturities this week
        surviving = []
        for inv in funded_book:
            inv = dict(inv)
            inv["weeks_left"] -= 1
            if inv["weeks_left"] <= 0:
                recycled += inv["allocation"] + inv["discount_earned"]
                discount_this_week += inv["discount_earned"]
            else:
                surviving.append(inv)
        funded_book = surviving
        available_capital += recycled
        cumulative_discount += discount_this_week

        # Step 2: new invoice arrivals (each supplier independent at arrival_prob)
        arrivals = []
        for _, sup in supplier_base.iterrows():
            if rng.random() < arrival_prob:
                tier_cfg = TIERS[int(sup["tier"])]
                inv_amt = round(float(rng.uniform(*tier_cfg["invoice_range"])), -3)
                days = int(rng.uniform(*tier_cfg["days_range"]))
                arrivals.append(
                    {
                        "supplier_id": sup["supplier_id"],
                        "tier": sup["tier"],
                        "tier_name": sup["tier_name"],
                        "invoice_amount": inv_amt,
                        "days_remaining": days,
                        "min_acceptable_rate": sup["min_acceptable_rate"],
                        "color": sup["color"],
                    }
                )

        # Step 3: LP auction — allocate available capital to unfunded arrivals
        n_funded_this_week = 0

        if arrivals and available_capital > 500:
            df_arr = pd.DataFrame(arrivals)
            res = optimise_allocation(
                df_arr,
                capital_pool=available_capital,
                scenario_label=scenario_label,
            )
            for _, row in res.suppliers[res.suppliers["funded"]].iterrows():
                funded_book.append(
                    {
                        "allocation": float(row["allocation"]),
                        "discount_earned": float(row["discount_earned"]),
                        # convert invoice days to weeks (round up; minimum 1)
                        "weeks_left": max(1, int(np.ceil(row["days_remaining"] / 7))),
                        "tier": int(row["tier"]),
                    }
                )
            available_capital -= res.capital_deployed
            n_funded_this_week = res.n_funded

        total_deployed = sum(inv["allocation"] for inv in funded_book)

        weekly.append(
            WeeklySnapshot(
                week=week,
                capital_available=available_capital,
                capital_deployed=total_deployed,
                recycled_this_week=recycled,
                discount_this_week=discount_this_week,
                cumulative_discount=cumulative_discount,
                n_funded_this_week=n_funded_this_week,
                n_active=len(funded_book),
            )
        )

    return RollingAuctionResult(
        weekly=weekly,
        scenario_label=scenario_label,
        initial_capital=capital_pool,
    )


def run_all_rolling_scenarios(
    capital_pool: float = DEFAULT_CAPITAL_POOL,
    n_weeks: int = N_WEEKS,
    arrival_prob: float = ARRIVAL_PROB,
) -> Dict[str, RollingAuctionResult]:
    """Run all four economic scenarios through the rolling auction model."""
    scenarios = {
        "Baseline":         dict(stress_tier3=False, rate_env_multiplier=1.0),
        "Liquidity Stress": dict(stress_tier3=True,  rate_env_multiplier=1.0),
        "Rising Rates":     dict(stress_tier3=False, rate_env_multiplier=1.5),
        "Combined Stress":  dict(stress_tier3=True,  rate_env_multiplier=1.5),
    }
    return {
        label: run_rolling_auction(
            capital_pool=capital_pool,
            n_weeks=n_weeks,
            arrival_prob=arrival_prob,
            **kwargs,
            rng=np.random.default_rng(42),
            scenario_label=label,
        )
        for label, kwargs in scenarios.items()
    }
