# Credit Risk AI Workbench

> A professional-grade commercial credit risk analytics platform built with Python, Streamlit, and scikit-learn — designed to mirror the quantitative risk frameworks used by large commercial banks.

---

## Live Demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## What This Does (Plain English)

Commercial banks lend billions of dollars to corporations every year. Before approving a loan, credit analysts ask: *"How likely is this borrower to stop making payments, and how much money would we lose if they did?"* This tool automates and visualises that entire analytical workflow.

### Key Financial Concepts

| Term | What It Means |
|---|---|
| **PD — Probability of Default** | The chance that a borrower fails to repay within the next 12 months. A AAA-rated firm might have PD < 0.1%; a CCC-rated firm might exceed 20%. |
| **LGD — Loss Given Default** | If a borrower does default, how much of the loan do we actually lose after recovering collateral? Senior secured loans typically have LGD ~30-40%; unsecured exposure can be 60-70%. |
| **EL — Expected Loss** | EL = PD × LGD × Exposure. This is the actuarial loss the bank budgets for via loan loss provisions. |
| **Credit Grade** | A letter rating (AAA → CCC) summarising the borrower's creditworthiness. Investment grade (AAA–BBB) is considered low risk; sub-investment grade (BB and below) is "high yield" or "junk". |
| **VaR — Value at Risk** | The maximum loss the portfolio is expected to suffer at a given confidence level (e.g., VaR 99% means: we are 99% confident losses won't exceed this amount). Banks use VaR to set aside economic capital. |
| **Stress Testing** | Simulating how the portfolio performs under adverse economic conditions (recession, rising rates, sector shocks). Regulators like the Fed mandate annual stress tests (DFAST) for large banks. |

---

## Tab-by-Tab Breakdown

### Tab 1 — Portfolio Overview
- 50 simulated commercial & industrial (C&I) loans across five sectors
- KPI cards: total exposure, weighted-average PD, total expected loss, composite credit rating
- Industry exposure pie chart and credit grade distribution bar chart
- Risk map scatter plot (PD vs LGD) sized by exposure
- Full loan table with colour-coded credit grades

### Tab 2 — ML Default Prediction
- Random Forest classifier trained on 2,000+ synthetic borrower records
- SHAP (SHapley Additive exPlanations) values quantify each feature's contribution to the default probability — the same explainability framework now required by regulators under SR 11-7 model risk guidance
- Interactive sliders let you adjust borrower characteristics and see the live predicted default probability update in real-time
- Confusion matrix and ROC/AUC curve show model discrimination power

### Tab 3 — Stress Testing
- Three scenarios: Base Case, Recession, Severe Recession
- PD multipliers escalate by credit grade (lower-rated borrowers suffer disproportionately in downturns)
- Monte Carlo loss distribution (10,000 simulations) with correlated defaults — because in a recession, firms fail together, not independently
- VaR lines at 95th and 99th percentiles; industry-level heatmap shows which sectors are most exposed

### Tab 4 — Individual Loan Analysis
- Drill into any single loan via dropdown
- SHAP waterfall chart: shows exactly which borrower characteristics are pushing default probability up or down relative to the base rate
- Peer comparison: scatter plot of the selected loan vs. all loans in the same industry and credit grade
- Radar chart: borrower profile normalised against the industry median

---

## Tech Stack

| Layer | Library | Why |
|---|---|---|
| UI | Streamlit 1.35 | Rapid prototyping of data dashboards |
| Visualisation | Plotly 5.22 | Interactive, publication-quality charts |
| ML | scikit-learn 1.4 | Random Forest classifier with calibrated probabilities |
| Explainability | SHAP 0.45 | Model-agnostic Shapley values for regulatory interpretability |
| Numerics | NumPy / Pandas | Vectorised portfolio calculations |

---

## Financial Methodology Notes

- **Default correlation**: The Monte Carlo stress engine uses a single-factor Gaussian copula (similar to the Basel II IRB Advanced model) with an asset correlation of 25% across loans in the same scenario.
- **Capital requirement**: Estimated as 108% of stressed expected loss, approximating a simplified Basel III buffer.
- **SHAP for SR 11-7**: U.S. banking regulators require banks to be able to explain model outputs to auditors. SHAP values provide an additive, feature-level breakdown that satisfies this requirement.

---

## Project Structure

```
credit-risk-ai-workbench/
├── app.py            # Main Streamlit application (all tabs)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

*Built as a portfolio project demonstrating quantitative credit risk analysis, machine learning model development, and financial data visualisation skills relevant to Commercial & Specialized Industries banking roles.*
