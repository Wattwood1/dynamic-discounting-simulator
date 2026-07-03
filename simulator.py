"""
Dynamic Discounting Simulator
==============================
Models a buyer with a pool of capital making early-payment offers
to a tiered supplier base. Solves a linear program to maximise
annualised yield on deployed capital, then re-runs under four
economic scenarios.

Companies like C2FO and Taulia run live marketplaces on this logic.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

warnings.filterwarnings("ignore")

# ── Tier configuration ────────────────────────────────────────────────────────

TIERS: Dict[int, dict] = {
    1: dict(
        name="Investment Grade",
        description="Large, creditworthy corporates with strong balance sheets.",
        credit_score_range=(720, 850),
        min_rate_range=(0.005, 0.012),   # annualised discount rate supplier will accept
        invoice_range=(500_000, 5_000_000),
        days_range=(30, 60),
        n_suppliers=8,
        color="#1565C0",
    ),
    2: dict(
        name="Mid-Market",
        description="Regional suppliers with moderate credit; occasional cash-flow lumps.",
        credit_score_range=(580, 720),
        min_rate_range=(0.015, 0.040),
        invoice_range=(100_000, 500_000),
        days_range=(30, 90),
        n_suppliers=15,
        color="#E65100",
    ),
    3: dict(
        name="SME / Constrained",
        description="Small businesses with thin margins, high cost of bank credit.",
        credit_score_range=(400, 580),
        min_rate_range=(0.040, 0.120),
        invoice_range=(10_000, 100_000),
        days_range=(15, 90),
        n_suppliers=22,
        color="#B71C1C",
    ),
}

DEFAULT_CAPITAL_POOL = 10_000_000   # $10 M


# ── Supplier generation ───────────────────────────────────────────────────────

def generate_suppliers(
    tiers: Dict[int, dict] = TIERS,
    stress_tier3: bool = False,
    rate_env_multiplier: float = 1.0,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Build a synthetic supplier portfolio.

    Parameters
    ----------
    tiers : dict
        Tier configuration (see module-level TIERS).
    stress_tier3 : bool
        If True, Tier-3 suppliers suffer a liquidity crunch and raise their
        minimum acceptable discount rate by 2–5 pp (they need cash urgently).
    rate_env_multiplier : float
        Scale all minimum rates proportionally.  >1 ⟹ rising-rate environment
        (bank credit becomes more expensive so all suppliers demand more).
    rng : numpy Generator
        For reproducibility.

    Returns
    -------
    pd.DataFrame with one row per supplier.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rows: List[dict] = []
    sid = 1

    for tier_id, cfg in tiers.items():
        lo_r, hi_r   = cfg["min_rate_range"]
        lo_inv, hi_inv = cfg["invoice_range"]
        lo_d, hi_d   = cfg["days_range"]
        lo_cs, hi_cs = cfg["credit_score_range"]

        for _ in range(cfg["n_suppliers"]):
            base_rate = rng.uniform(lo_r, hi_r) * rate_env_multiplier

            # Liquidity-stress bump: Tier 3 suppliers raise the bar
            stress_bump = 0.0
            if stress_tier3 and tier_id == 3:
                stress_bump = rng.uniform(0.02, 0.05)

            min_rate = float(np.clip(base_rate + stress_bump, lo_r, 0.30))

            rows.append(
                dict(
                    supplier_id=f"S{sid:03d}",
                    tier=tier_id,
                    tier_name=cfg["name"],
                    credit_score=int(rng.uniform(lo_cs, hi_cs)),
                    invoice_amount=round(float(rng.uniform(lo_inv, hi_inv)), -3),
                    days_remaining=int(rng.uniform(lo_d, hi_d)),
                    min_acceptable_rate=round(min_rate, 5),
                    color=cfg["color"],
                    liquidity_stressed=(stress_tier3 and tier_id == 3 and stress_bump > 0),
                )
            )
            sid += 1

    df = pd.DataFrame(rows)
    # Annualised yield score: how much does one dollar earn for one year?
    df["yield_score"] = df["min_acceptable_rate"] * df["days_remaining"] / 365
    return df


# ── Optimisation ──────────────────────────────────────────────────────────────

@dataclass
class AllocationResult:
    """Holds the full output of one optimisation run."""
    suppliers: pd.DataFrame           # includes 'allocation' and 'discount_earned' columns
    capital_pool: float
    capital_deployed: float
    capital_utilisation: float
    total_discount_earned: float
    n_funded: int
    n_total: int
    scenario_label: str = "Baseline"

    @property
    def annualised_yield(self) -> float:
        """Weighted-average annualised return on deployed capital."""
        funded = self.suppliers[self.suppliers["funded"]]
        if funded.empty or self.capital_deployed == 0:
            return 0.0
        avg_days = (funded["allocation"] * funded["days_remaining"]).sum() / self.capital_deployed
        return self.total_discount_earned / self.capital_deployed * (365 / avg_days) if avg_days else 0.0

    def summary_dict(self) -> dict:
        return {
            "Scenario": self.scenario_label,
            "Capital Pool ($M)": self.capital_pool / 1e6,
            "Deployed ($M)": round(self.capital_deployed / 1e6, 2),
            "Utilisation (%)": round(self.capital_utilisation * 100, 1),
            "Discount Earned ($k)": round(self.total_discount_earned / 1e3, 1),
            "Ann. Yield (%)": round(self.annualised_yield * 100, 2),
            "Suppliers Funded": self.n_funded,
        }


def optimise_allocation(
    df: pd.DataFrame,
    capital_pool: float = DEFAULT_CAPITAL_POOL,
    scenario_label: str = "Baseline",
) -> AllocationResult:
    """
    Solve the linear programme:

        maximise   Σᵢ xᵢ · rᵢ · (dᵢ / 365)
        subject to Σᵢ xᵢ ≤ capital_pool
                   0 ≤ xᵢ ≤ invoice_amount_i   ∀ i

    where
        xᵢ  = dollar amount advanced to supplier i (decision variable)
        rᵢ  = supplier i's minimum acceptable annualised discount rate
        dᵢ  = days remaining until invoice due date

    The objective is the buyer's total absolute discount earned (≈ gross profit
    on the early-payment programme).  Because the LP is continuous, the HiGHS
    solver finds the exact optimum efficiently for hundreds of suppliers.

    Note: this is trivially solvable by greedy ranking on rᵢ·dᵢ, but the LP
    formulation scales naturally to richer constraints (tier limits, min-ticket
    sizes, concentration caps) added in extensions.
    """
    n = len(df)

    # scipy minimises → negate objective coefficients
    c_obj = -(df["min_acceptable_rate"] * df["days_remaining"] / 365).values

    A_ub = [np.ones(n)]
    b_ub = [capital_pool]

    bounds = [(0.0, float(inv)) for inv in df["invoice_amount"].values]

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.status != 0:
        raise RuntimeError(f"LP solver failed: {result.message}")

    out = df.copy()
    out["allocation"]       = np.round(result.x, 2)
    out["funded"]           = out["allocation"] > 500          # noise threshold
    out["pct_funded"]       = out["allocation"] / out["invoice_amount"]
    out["discount_earned"]  = out["allocation"] * out["min_acceptable_rate"] * out["days_remaining"] / 365

    deployed   = float(out["allocation"].sum())
    disc_total = float(out["discount_earned"].sum())

    return AllocationResult(
        suppliers=out,
        capital_pool=capital_pool,
        capital_deployed=deployed,
        capital_utilisation=deployed / capital_pool,
        total_discount_earned=disc_total,
        n_funded=int(out["funded"].sum()),
        n_total=n,
        scenario_label=scenario_label,
    )


# ── Scenario runner ───────────────────────────────────────────────────────────

def run_all_scenarios(
    capital_pool: float = DEFAULT_CAPITAL_POOL,
) -> Dict[str, AllocationResult]:
    """
    Run four scenarios and return a dict of AllocationResult objects.

    Scenarios
    ---------
    baseline        : Normal market conditions.
    liquidity_stress: Tier-3 suppliers under cash pressure; they raise their
                      minimum acceptable rates (+2 – 5 pp).
    rising_rates    : Central-bank tightening cycle raises the cost of bank
                      credit uniformly (+50 % on all rates, simulating a
                      ~300 bp shift from very-low baseline).
    combined        : Both stresses simultaneously — the most severe case.
    """
    rng = np.random.default_rng(42)   # shared seed for comparability

    scenarios = {
        "Baseline":         dict(stress_tier3=False, rate_env_multiplier=1.0),
        "Liquidity Stress": dict(stress_tier3=True,  rate_env_multiplier=1.0),
        "Rising Rates":     dict(stress_tier3=False, rate_env_multiplier=1.5),
        "Combined Stress":  dict(stress_tier3=True,  rate_env_multiplier=1.5),
    }

    results: Dict[str, AllocationResult] = {}
    for label, kwargs in scenarios.items():
        df = generate_suppliers(**kwargs, rng=np.random.default_rng(42))
        results[label] = optimise_allocation(df, capital_pool=capital_pool, scenario_label=label)

    return results


# ── Capital sensitivity ───────────────────────────────────────────────────────

def capital_sensitivity(
    capital_range: Tuple[float, float] = (1_000_000, 25_000_000),
    n_points: int = 40,
    stress_tier3: bool = False,
    rate_env_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Sweep the capital pool from low to high and record yield & utilisation.
    Returns a DataFrame with one row per capital level.
    """
    df_sup = generate_suppliers(
        stress_tier3=stress_tier3,
        rate_env_multiplier=rate_env_multiplier,
        rng=np.random.default_rng(42),
    )

    records = []
    for cap in np.linspace(capital_range[0], capital_range[1], n_points):
        res = optimise_allocation(df_sup, capital_pool=cap)
        records.append(
            dict(
                capital_pool=cap,
                capital_deployed=res.capital_deployed,
                utilisation=res.capital_utilisation,
                discount_earned=res.total_discount_earned,
                annualised_yield=res.annualised_yield,
                n_funded=res.n_funded,
            )
        )

    return pd.DataFrame(records)


# ── Quick CLI demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_all_scenarios()
    rows = [r.summary_dict() for r in results.values()]
    print(pd.DataFrame(rows).to_string(index=False))
