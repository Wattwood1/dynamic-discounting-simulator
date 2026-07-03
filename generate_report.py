#!/usr/bin/env python3
"""
Generate the Dynamic Discounting Simulator HTML report.

Usage:
    python generate_report.py

Output:
    Dynamic_Discounting_Simulator.html  (overwrites existing file)
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from simulator import (
    DEFAULT_CAPITAL_POOL,
    TIERS,
    generate_suppliers,
    optimise_allocation,
    run_all_scenarios,
    capital_sensitivity,
    AllocationResult,
)
from rolling_auction import run_all_rolling_scenarios, RollingAuctionResult

# ── Palette ───────────────────────────────────────────────────────────────────

TIER_COLORS = {t: TIERS[t]["color"] for t in TIERS}
SCENARIO_COLORS = {
    "Baseline":         "#1565C0",
    "Liquidity Stress": "#E65100",
    "Rising Rates":     "#2E7D32",
    "Combined Stress":  "#B71C1C",
}

plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})

# ── Chart utilities ───────────────────────────────────────────────────────────

def _b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(fig: plt.Figure, width: str = "100%") -> str:
    return (
        f'<img src="data:image/png;base64,{_b64(fig)}" '
        f'style="width:{width};max-width:900px;" />'
    )


def _df_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=1, classes="dataframe tbl")


# ── §1 Portfolio overview ──────────────────────────────────────────────────────

def chart_portfolio(df: pd.DataFrame) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    names  = [TIERS[t]["name"] for t in sorted(TIERS)]
    colors = [TIERS[t]["color"] for t in sorted(TIERS)]

    counts = [df[df["tier"] == t].shape[0] for t in sorted(TIERS)]
    bars = ax1.bar(names, counts, color=colors, edgecolor="white", width=0.55)
    ax1.set_title("Suppliers per Tier", fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.set_ylim(0, max(counts) * 1.18)
    for b, c in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                 str(c), ha="center", va="bottom", fontsize=10)
    ax1.tick_params(axis="x", labelsize=8)

    totals = [df[df["tier"] == t]["invoice_amount"].sum() / 1e6 for t in sorted(TIERS)]
    bars2 = ax2.bar(names, totals, color=colors, edgecolor="white", width=0.55)
    ax2.set_title("Total Invoice Value per Tier", fontweight="bold")
    ax2.set_ylabel("$M")
    ax2.set_ylim(0, max(totals) * 1.18)
    for b, v in zip(bars2, totals):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                 f"${v:.1f}M", ha="center", va="bottom", fontsize=9)
    ax2.tick_params(axis="x", labelsize=8)

    fig.tight_layout(pad=2)
    return fig


def table_portfolio_summary(df: pd.DataFrame) -> str:
    rows = []
    for t in sorted(TIERS):
        sub = df[df["tier"] == t]
        rows.append(
            {
                "Tier": TIERS[t]["name"],
                "Count": len(sub),
                "Avg Invoice ($k)": f"${sub['invoice_amount'].mean()/1e3:.0f}k",
                "Avg Min Rate (%)": f"{sub['min_acceptable_rate'].mean()*100:.2f}%",
                "Avg Days": f"{sub['days_remaining'].mean():.0f}",
                "Total Invoices ($M)": f"${sub['invoice_amount'].sum()/1e6:.1f}M",
            }
        )
    return _df_html(pd.DataFrame(rows))


# ── §2 Baseline optimisation ───────────────────────────────────────────────────

def chart_baseline_allocation(res: AllocationResult) -> plt.Figure:
    df = res.suppliers.copy()
    df = df.sort_values(["tier", "min_acceptable_rate"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(12, 4.5))

    x = np.arange(len(df))
    colors = [TIER_COLORS[t] for t in df["tier"]]
    alphas_full  = [1.0 if f else 0.25 for f in df["funded"]]

    bars = ax.bar(x, df["invoice_amount"] / 1e3, color=colors,
                  alpha=0.3, edgecolor="none", label="_nolegend_")
    funded_bars = ax.bar(x, df["allocation"] / 1e3, color=colors,
                         alpha=0.9, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(df["supplier_id"], rotation=90, fontsize=6)
    ax.set_ylabel("Amount ($k)")
    ax.set_title("Baseline Allocation — funded (solid) vs invoice face value (faded)", fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TIERS[t]["color"], label=TIERS[t]["name"]) for t in sorted(TIERS)
    ]
    legend_elements.append(Patch(facecolor="#aaa", alpha=0.3, label="Unfunded portion"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}k"))
    fig.tight_layout()
    return fig


def table_baseline_by_tier(res: AllocationResult) -> str:
    rows = []
    for t in sorted(TIERS):
        sub = res.suppliers[res.suppliers["tier"] == t]
        funded = sub[sub["funded"]]
        rows.append(
            {
                "Tier": TIERS[t]["name"],
                "Suppliers": len(sub),
                "Funded": len(funded),
                "Capital Deployed ($M)": f"${funded['allocation'].sum()/1e6:.2f}M",
                "Discount Earned ($k)": f"${funded['discount_earned'].sum()/1e3:.1f}k",
                "Avg Ann. Rate (%)": (
                    f"{funded['min_acceptable_rate'].mean()*100:.2f}%"
                    if len(funded) else "—"
                ),
            }
        )
    return _df_html(pd.DataFrame(rows))


# ── §3 Scenario analysis ───────────────────────────────────────────────────────

def chart_scenario_comparison(results: dict[str, AllocationResult]) -> plt.Figure:
    labels  = list(results.keys())
    colors  = [SCENARIO_COLORS[l] for l in labels]
    metrics = {
        "Deployed ($M)":     [results[l].capital_deployed / 1e6 for l in labels],
        "Discount Earned ($k)": [results[l].total_discount_earned / 1e3 for l in labels],
        "Ann. Yield (%)":    [results[l].annualised_yield * 100 for l in labels],
        "Suppliers Funded":  [results[l].n_funded for l in labels],
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (title, vals) in zip(axes, metrics.items()):
        bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor="white", width=0.6)
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, max(vals) * 1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Four-Scenario Dashboard", fontweight="bold", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def chart_capital_sensitivity(df_sens: pd.DataFrame) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.plot(df_sens["capital_pool"] / 1e6, df_sens["annualised_yield"] * 100,
             color="#1565C0", lw=2, label="Ann. Yield (%)")
    ax2.plot(df_sens["capital_pool"] / 1e6, df_sens["utilisation"] * 100,
             color="#E65100", lw=2, linestyle="--", label="Utilisation (%)")

    ax1.set_xlabel("Capital Pool ($M)")
    ax1.set_ylabel("Annualised Yield (%)", color="#1565C0")
    ax2.set_ylabel("Capital Utilisation (%)", color="#E65100")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax2.tick_params(axis="y", labelcolor="#E65100")
    ax1.set_title("Capital Sensitivity — yield compression as pool grows", fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    fig.tight_layout()
    return fig


def chart_liquidity_stress(baseline: AllocationResult, stress: AllocationResult) -> plt.Figure:
    """Side-by-side bar: Baseline vs Liquidity Stress by tier."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, res, label in zip(axes, [baseline, stress], ["Baseline", "Liquidity Stress"]):
        names  = [TIERS[t]["name"] for t in sorted(TIERS)]
        colors = [TIERS[t]["color"] for t in sorted(TIERS)]
        disc   = [
            res.suppliers[res.suppliers["tier"] == t]["discount_earned"].sum() / 1e3
            for t in sorted(TIERS)
        ]
        bars = ax.bar(names, disc, color=colors, edgecolor="white", width=0.55)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Discount Earned ($k)")
        ax.set_ylim(0, max(disc) * 1.2 if disc else 1)
        for b, v in zip(bars, disc):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                    f"${v:.1f}k", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Discount Earned by Tier — Baseline vs Liquidity Stress",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def chart_rising_rates(baseline: AllocationResult, rising: AllocationResult) -> plt.Figure:
    """Side-by-side bar: Baseline vs Rising Rates — yield and deployment."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    metrics = ["Ann. Yield (%)", "Discount Earned ($k)"]
    vals_bl = [baseline.annualised_yield * 100, baseline.total_discount_earned / 1e3]
    vals_rr = [rising.annualised_yield * 100, rising.total_discount_earned / 1e3]

    for ax, metric, vbl, vrr in zip(axes, metrics, vals_bl, vals_rr):
        ax.bar(["Baseline", "Rising Rates"], [vbl, vrr],
               color=["#1565C0", "#2E7D32"], edgecolor="white", width=0.45)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, max(vbl, vrr) * 1.25)
        for x, v in enumerate([vbl, vrr]):
            ax.text(x, v + max(vbl, vrr) * 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Rising Rate Environment — Yield & Discount vs Baseline",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── §6 Rolling auction charts ──────────────────────────────────────────────────

def chart_rolling_capital_dynamics(res: RollingAuctionResult) -> plt.Figure:
    df = res.weekly_df()
    weeks = df["week"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.fill_between(weeks, df["capital_deployed"] / 1e6, alpha=0.3, color="#1565C0")
    ax1.plot(weeks, df["capital_deployed"] / 1e6, color="#1565C0", lw=2, label="Deployed")
    ax1.plot(weeks, df["capital_available"] / 1e6, color="#E65100", lw=1.5,
             linestyle="--", label="Available")
    ax1.axhline(DEFAULT_CAPITAL_POOL / 1e6, color="#999", lw=1, linestyle=":")
    ax1.set_ylabel("Capital ($M)")
    ax1.set_title("Baseline: Weekly Capital Dynamics", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_ylim(bottom=0)

    ax2.bar(weeks, df["recycled_this_week"] / 1e3, color="#2E7D32", alpha=0.7,
            label="Recycled from maturities")
    ax2.bar(weeks, df["discount_this_week"] / 1e3, bottom=0,
            color="#F57F17", alpha=0.8, label="New discount earned")
    ax2.set_xlabel("Week")
    ax2.set_ylabel("$k")
    ax2.set_title("Weekly Capital Recycled & New Discount Earned", fontweight="bold")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    return fig


def chart_rolling_scenario_comparison(results: dict[str, RollingAuctionResult]) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, res in results.items():
        df = res.weekly_df()
        color = SCENARIO_COLORS[label]
        ax1.plot(df["week"], df["cumulative_discount"] / 1e3,
                 color=color, lw=2, label=label)
        ax2.plot(df["week"], df["capital_deployed"] / 1e6,
                 color=color, lw=2, label=label)

    ax1.set_xlabel("Week")
    ax1.set_ylabel("Cumulative Discount ($k)")
    ax1.set_title("Cumulative Discount Earned", fontweight="bold")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Week")
    ax2.set_ylabel("Capital Deployed ($M)")
    ax2.set_title("Weekly Deployed Capital", fontweight="bold")
    ax2.axhline(DEFAULT_CAPITAL_POOL / 1e6, color="#bbb", lw=1, linestyle=":")
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)

    fig.suptitle("26-Week Rolling Auction — Scenario Comparison", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def chart_rolling_active_invoices(results: dict[str, RollingAuctionResult]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, res in results.items():
        df = res.weekly_df()
        ax.plot(df["week"], df["n_active"], color=SCENARIO_COLORS[label], lw=2, label=label)

    ax.set_xlabel("Week")
    ax.set_ylabel("Active Funded Invoices")
    ax.set_title("Live Invoice Count Over Time", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ── HTML assembly ──────────────────────────────────────────────────────────────

CSS = """
  :root { --blue:#1565C0; --orange:#E65100; --red:#B71C1C; }
  body  { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
           max-width:1100px; margin:0 auto; padding:24px 32px;
           background:#fafafa; color:#222; line-height:1.6; }
  nav   { position:sticky; top:0; background:#fff; border-bottom:2px solid var(--blue);
           padding:8px 0; margin-bottom:28px; z-index:100;
           display:flex; gap:18px; flex-wrap:wrap; }
  nav a { color:var(--blue); text-decoration:none; font-size:0.88em; font-weight:600; white-space:nowrap; }
  nav a:hover { text-decoration:underline; }
  h1    { font-size:2em; border-bottom:3px solid var(--blue); padding-bottom:8px; }
  h2    { font-size:1.4em; margin-top:2.5em; color:var(--blue);
           border-left:4px solid var(--blue); padding-left:10px; scroll-margin-top:48px; }
  h3    { font-size:1.1em; margin-top:1.5em; color:#333; }
  .subtitle { color:#555; font-size:1.05em; margin-top:-8px; margin-bottom:24px; }
  .tbl  { border-collapse:collapse; width:100%; margin:12px 0; font-size:0.9em; }
  .tbl th { background:var(--blue); color:white; padding:8px 12px; text-align:left; }
  .tbl td { border-bottom:1px solid #ddd; padding:7px 12px; }
  .tbl tr:nth-child(even) td { background:#f0f4ff; }
  img   { border-radius:6px; box-shadow:0 2px 8px rgba(0,0,0,.12); }
  ul,ol { padding-left:1.5em; }
  li    { margin-bottom:0.5em; }
  .pill { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.8em;
           background:#e8f0fe; color:var(--blue); margin:0 2px; }
  .kpi  { display:inline-block; background:#fff; border:1px solid #ddd; border-radius:8px;
           padding:12px 20px; margin:8px; text-align:center; min-width:140px; }
  .kpi .val { font-size:1.6em; font-weight:700; color:var(--blue); }
  .kpi .lbl { font-size:0.8em; color:#666; }
"""


def _kpi(label: str, value: str) -> str:
    return f'<div class="kpi"><div class="val">{value}</div><div class="lbl">{label}</div></div>'


def build_html(
    df_base: pd.DataFrame,
    base_res: AllocationResult,
    all_static: dict,
    df_sens: pd.DataFrame,
    rolling: dict[str, RollingAuctionResult],
) -> str:
    print("Generating charts...", flush=True)

    # §1
    img_portfolio   = _img(chart_portfolio(df_base))
    tbl_portfolio   = table_portfolio_summary(df_base)

    # §2
    img_baseline    = _img(chart_baseline_allocation(base_res))
    tbl_tier        = table_baseline_by_tier(base_res)

    # §3
    img_liq_stress  = _img(chart_liquidity_stress(
        all_static["Baseline"], all_static["Liquidity Stress"]))
    img_rising      = _img(chart_rising_rates(
        all_static["Baseline"], all_static["Rising Rates"]))
    img_sensitivity = _img(chart_capital_sensitivity(df_sens))

    # §4
    img_dashboard   = _img(chart_scenario_comparison(all_static))

    # §6
    img_dynamics    = _img(chart_rolling_capital_dynamics(rolling["Baseline"]))
    img_rolling_cmp = _img(chart_rolling_scenario_comparison(rolling))
    img_active      = _img(chart_rolling_active_invoices(rolling))

    # Rolling summary table (all four scenarios, including Combined Stress)
    rolling_rows = [r.summary_dict() for r in rolling.values()]
    tbl_rolling  = _df_html(pd.DataFrame(rolling_rows))

    # Static scenario summary table
    static_rows = [r.summary_dict() for r in all_static.values()]
    tbl_static  = _df_html(pd.DataFrame(static_rows))

    # KPIs from baseline
    b = base_res
    kpis = (
        _kpi("Capital Pool", f"${b.capital_pool/1e6:.0f}M")
        + _kpi("Deployed", f"${b.capital_deployed/1e6:.1f}M")
        + _kpi("Utilisation", f"{b.capital_utilisation*100:.1f}%")
        + _kpi("Discount Earned", f"${b.total_discount_earned/1e3:.1f}k")
        + _kpi("Ann. Yield", f"{b.annualised_yield*100:.2f}%")
        + _kpi("Suppliers Funded", str(b.n_funded))
    )

    print("Assembling HTML...", flush=True)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dynamic Discounting Simulator</title>
<style>
{CSS}
</style>
</head>
<body>
<h1>Dynamic Discounting Simulator</h1>
<p class="subtitle">Supply-Chain Finance · LP Optimisation · Rolling Auction · Scenario Analysis</p>
<p>A buyer firm holds a <b>${DEFAULT_CAPITAL_POOL/1e6:.0f}M capital pool</b> and can offer suppliers early payment
in exchange for a small discount — <em>dynamic discounting</em>. This report optimises allocation across
{base_res.n_total} suppliers, stress-tests four economic scenarios, and simulates a 26-week rolling
weekly auction. Real marketplaces:
<span class="pill">C2FO</span><span class="pill">Taulia</span><span class="pill">Greensill</span></p>
<nav>
  <a href="#s1">1 · Portfolio</a>
  <a href="#s2">2 · Baseline</a>
  <a href="#s3">3 · Scenarios</a>
  <a href="#s4">4 · Dashboard</a>
  <a href="#s5">5 · Insights</a>
  <a href="#s6">6 · Rolling Auction</a>
  <a href="#s7">7 · Extensions</a>
</nav>

<h2 id="s1">1 · Supplier Portfolio</h2>
<p>{base_res.n_total} synthetic suppliers across three credit tiers. Each supplier reports a
<em>minimum acceptable discount rate</em> (annualised) on its outstanding invoice.</p>
{tbl_portfolio}
{img_portfolio}

<h2 id="s2">2 · Baseline Optimisation</h2>
<p>The LP maximises total discount earned subject to the ${DEFAULT_CAPITAL_POOL/1e6:.0f}M capital
constraint. Each bar shows the amount advanced (solid) against the full invoice face value (faded).
Suppliers are sorted within each tier by descending minimum rate.</p>
{img_baseline}

<h3>Funding by Tier</h3>
{tbl_tier}

<h3>Baseline Key Metrics</h3>
<div>{kpis}</div>

<h2 id="s3">3 · Scenario Analysis</h2>

<h3>Scenario A — Supplier Liquidity Stress</h3>
<p>Tier-3 (SME) suppliers face a cash crunch and raise their minimum acceptable discount rate by
2–5 percentage points. The buyer earns more per dollar deployed as distressed suppliers compete
harder for early payment.</p>
{img_liq_stress}

<h3>Scenario B — Rising Rate Environment</h3>
<p>A central-bank tightening cycle makes bank credit 50% more expensive across the board
(approximately +300 bp from a low base). All suppliers raise their minimum rates proportionally,
lifting the buyer's annualised yield on the programme.</p>
{img_rising}

<h3>Scenario C — Capital Sensitivity</h3>
<p>Sweeping the capital pool from $1M to $25M reveals a yield cliff: below ~$8M the buyer is
capital-constrained and yield is flat; above ~$15M the high-rate SME book is exhausted and
yield compresses as cheaper Tier-1 invoices must fill the gap.</p>
{img_sensitivity}

<h3>All-Scenario Summary</h3>
{tbl_static}

<h2 id="s4">4 · Four-Scenario Dashboard</h2>
<p>Side-by-side comparison of deployed capital, discount earned, annualised yield, and number of
suppliers funded across all four economic scenarios.</p>
{img_dashboard}

<h2 id="s5">5 · Key Insights</h2>
<ul>
  <li><b>Tier-3 suppliers are the yield engine.</b> SMEs accept the highest discount rates but have
  the smallest invoices. The LP funds them first until capital is exhausted, then fills remaining
  capacity with Tier-2 and finally Tier-1 invoices.</li>
  <li><b>Liquidity stress is good for buyers.</b> When SME suppliers need cash urgently, they accept
  higher discount rates, increasing the buyer's yield with no additional capital commitment.</li>
  <li><b>Rising rates lift all tiers.</b> A uniform rate increase (simulating a tightening cycle)
  improves the buyer's return proportionally across the whole portfolio.</li>
  <li><b>Capital sensitivity shows a natural ceiling.</b> Once the buyer's pool exceeds the total
  SME book, additional capital earns diminishing returns — it must fund lower-rate Tier-1 invoices.</li>
  <li><b>The combined stress scenario is the most lucrative for the buyer</b> — higher rates and
  more urgent suppliers simultaneously maximise discount earned.</li>
</ul>

<h2 id="s6">6 · Rolling Weekly Auction Model</h2>

<h3>What changes from the static model</h3>
<p>The static LP clears the market once. Real platforms like C2FO run <em>continuous</em> or
<em>weekly</em> auctions where:</p>
<ul>
  <li><b>Invoices arrive stochastically</b> — each supplier independently submits a new invoice
  each week with probability 35%.</li>
  <li><b>Funded invoices mature</b> — when an invoice's payment term expires, the buyer is repaid
  in full (principal + discount), recycling capital back into the pool.</li>
  <li><b>Capital compounds</b> — reinvested repayments let the programme grow beyond the initial
  ${DEFAULT_CAPITAL_POOL/1e6:.0f}M over the simulation horizon.</li>
</ul>
<p>We run 26 weeks (≈ one half-year). Each week the LP clears on the live invoice book.</p>

<h3>Baseline: capital dynamics and weekly flow</h3>
{img_dynamics}

<h3>Scenario comparison</h3>
{img_rolling_cmp}

<h3>Active invoice count over time</h3>
{img_active}

<h3>26-Week Summary</h3>
{tbl_rolling}

<h3>Key dynamics</h3>
<ul>
  <li><b>Capital recycling accelerates returns.</b> As short-tenor Tier-3 invoices mature quickly
  (2–4 weeks), their principal flows back and funds the next cohort — the programme compounds
  without adding external capital.</li>
  <li><b>Steady state reached in ~8 weeks.</b> Active invoice count stabilises as arrivals balance
  maturities, giving the buyer a predictable monthly income stream.</li>
  <li><b>Liquidity stress lifts cumulative discount</b> over baseline, as higher SME rates persist
  across all 26 auctions.</li>
  <li><b>Rising rates compress invoice tenure.</b> Higher cost of waiting means more suppliers
  accept very short tenors, accelerating capital recycling and increasing weekly throughput.</li>
  <li><b>Combined Stress maximises total return</b> — both stress mechanisms active simultaneously
  produce the highest 26-week discount earned.</li>
</ul>

<h2 id="s7">7 · Model Limitations &amp; Extensions</h2>
<table class="tbl"><thead><tr><th>Limitation</th><th>Extension</th></tr></thead><tbody>
<tr><td>Partial invoice payments unrealistic for some platforms</td>
    <td>Add minimum-ticket constraint; binary allocation per supplier (MIP)</td></tr>
<tr><td>Rolling model ignores default risk</td>
    <td>Add credit-loss term; tier-dependent default probability</td></tr>
<tr><td>Deterministic arrival rate</td>
    <td>Monte Carlo over arrival probability and invoice size</td></tr>
<tr><td>Single buyer, no competition</td>
    <td>Model platform marketplace with multiple competing buyers</td></tr>
<tr><td>Static rate preferences</td>
    <td>Adaptive suppliers who lower rates as liquidity improves</td></tr>
<tr><td>Weekly granularity</td>
    <td>Daily clearing loop; intra-week capital deployment</td></tr>
</tbody></table>

<h3>Real-World Connections</h3>
<p><b>C2FO</b> — continuous rate auction; our weekly LP approximates each clearing cycle.<br>
<b>Taulia</b> — ERP-integrated; suppliers see real-time early-payment offers based on their rate.<br>
<b>Greensill</b> — used supply-chain finance at scale; collapsed 2021 partly due to concentration risk
and undisclosed receivables — a real-world reminder of the credit-risk limitations flagged above.</p>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    out_path = Path(__file__).parent / "Dynamic_Discounting_Simulator.html"

    print("Running static scenarios...", flush=True)
    df_base  = generate_suppliers(rng=np.random.default_rng(42))
    base_res = optimise_allocation(df_base, scenario_label="Baseline")
    all_static = run_all_scenarios()
    df_sens    = capital_sensitivity()

    print("Running rolling auction scenarios (26 weeks × 4)...", flush=True)
    rolling = run_all_rolling_scenarios()

    html = build_html(df_base, base_res, all_static, df_sens, rolling)

    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
