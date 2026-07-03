# Dynamic Discounting Simulator

A supply-chain finance model that optimises a buyer's early-payment programme across a synthetic supplier portfolio, stress-tests four economic scenarios, and simulates 26 weeks of rolling weekly auctions with capital recycling.

---

## What is dynamic discounting?

When a company (the buyer) owes money to its suppliers, it normally pays on a fixed schedule — say, 60 days after receiving an invoice. Dynamic discounting lets suppliers request early payment in exchange for a small discount. A supplier that needs cash now might accept payment today at a 3% annualised discount rather than wait two months for the full amount.

The buyer earns a return on capital it would otherwise leave idle. The supplier gets liquidity without taking on debt. Platforms like **C2FO** and **Taulia** run live marketplaces on exactly this logic.

---

## What this simulator models

### Static LP allocation (`simulator.py`)

The buyer holds a **$10M capital pool** and faces 45 synthetic suppliers across three credit tiers:

| Tier | Type | Suppliers | Invoice size | Min. discount rate |
|---|---|---|---|---|
| 1 | Investment Grade | 8 | $500k – $5M | 0.5 – 1.2% |
| 2 | Mid-Market | 15 | $100k – $500k | 1.5 – 4.0% |
| 3 | SME / Constrained | 22 | $10k – $100k | 4.0 – 12.0% |

A **linear programme** (via `scipy`'s HiGHS solver) allocates capital to maximise total discount earned, subject to the pool limit and per-invoice caps. SME suppliers offer the highest rates but the smallest invoices; the LP funds them first, then fills remaining capacity down the credit stack.

### Rolling weekly auction (`rolling_auction.py`)

The static model clears the market once. The rolling model adds time:

- **Each week**, every supplier independently has a 35% chance of submitting a new invoice.
- **Funded invoices mature** on schedule; the buyer is repaid principal + discount, recycling capital back into the pool.
- **The LP re-runs** each week on the current unfunded book.

Simulating 26 weeks (≈ one half-year) shows how capital compounds through recycling, when the programme reaches steady state, and how total returns differ across economic conditions.

### Four scenarios

| Scenario | What changes |
|---|---|
| **Baseline** | Normal market conditions |
| **Liquidity Stress** | Tier-3 suppliers face a cash crunch; raise minimum rates by 2–5 pp |
| **Rising Rates** | Central-bank tightening makes bank credit 50% more expensive; all supplier rates shift up proportionally |
| **Combined Stress** | Both stresses simultaneously — the most severe (and most lucrative for the buyer) case |

---

## How to run it

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Regenerate the HTML report**

```bash
python3 generate_report.py
```

This runs all four scenarios through both the static LP and the 26-week rolling auction, generates embedded charts, and writes `Dynamic_Discounting_Simulator.html`. Open that file in any browser.

---

## Project structure

```
simulator.py          # Core model: supplier generation, LP optimisation, scenario runner
rolling_auction.py    # 26-week rolling auction with capital recycling
generate_report.py    # Produces the self-contained HTML report
requirements.txt      # numpy, pandas, scipy, matplotlib
Dynamic_Discounting_Simulator.html  # Pre-built report (regenerate with generate_report.py)
Explainer.docx        # Extended write-up
```

---

## Notes

The rolling auction module (`rolling_auction.py`) was **reconstructed** from the analysis visible in the original HTML report, which showed results but no corresponding source code. The reconstructed model reproduces the key dynamics from the report — capital staying near fully deployed throughout, steady state reached around week 8, and the ordering Baseline < Liquidity Stress < Rising Rates < Combined Stress — and adds the **Combined Stress** scenario that was missing from the original report's 26-week summary table.

Discount totals are counted only when invoices mature within the simulation window (realised, not accrued), matching the accounting convention of the original report.
