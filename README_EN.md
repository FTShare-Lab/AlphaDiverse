<div align="center">

<h1>AlphaDiverse</h1>
<p><strong>Post-Training Local Quantitative Research Agents<br>for Diverse Exploration in Alpha Factor Mining</strong></p>
<p>Quantitative research agents that keep exploring complementary economic mechanisms</p>

<p>
  <a href="https://github.com/FTShare-Lab/AlphaDiverse"><img src="https://img.shields.io/badge/GitHub-FTShare--Lab%2FAlphaDiverse-181717.svg?logo=github" alt="GitHub: FTShare-Lab/AlphaDiverse"></a>
  <a href="https://arxiv.org/abs/2609.29014"><img src="https://img.shields.io/badge/arXiv-2609.29014-b31b1b.svg" alt="arXiv:2609.29014"></a>
  <a href="https://arxiv.org/pdf/2609.29014"><img src="https://img.shields.io/badge/Paper-PDF-315b7d.svg" alt="Paper PDF"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg" alt="Python 3.10+"></a>
</p>

<p><a href="README.md">简体中文</a> · <strong>English</strong></p>
<p><a href="#overview">Overview</a> · <a href="#quick-start">Quick Start</a> · <a href="#data">Data</a> · <a href="#evaluation">Evaluation</a> · <a href="#citation">Citation</a></p>

</div>

**AlphaDiverse** is a multi-agent framework for alpha factor mining. It connects economic-mechanism planning, feature and operator retrieval, structured factor realization, and evaluation feedback in an iterative loop, using accumulated evidence to guide the next research direction.

> **This repository implements the agent loop in Section 2.1, Research Agent System, of the paper.** It runs on a personal server with a compatible model endpoint. The paper's post-trained models will be made available through an **API**; access details will be added here.

<a id="overview"></a>

## Overview

<p align="center">
  <a href="assets/overview.svg"><img src="assets/overview.svg" width="100%" alt="AlphaDiverse Fig. 2: research agent system, training data construction, and local post-training"></a>
</p>

### From research plans to executable factors

| Stage | Role |
| :--- | :--- |
| **Planning** | Propose four complementary plans from feasible economic mechanisms, covering at least three themes. |
| **Retrieval** | Deterministically select the features and operators available to each plan. |
| **Realization** | Produce two structured factor specifications per plan, yielding eight candidates per round. |
| **Execution** | Validate JSON contracts and compute factors through the controlled operator runtime. |
| **Validation** | Screen individual factors, then measure their contribution to the predictive model. |
| **Analysis** | Summarize evidence, update research memory, and guide the next round. |

- **Mechanism-driven exploration**: explicit plans and complementarity constraints organize the search across rounds.
- **Executable, traceable research**: factor specifications, screening decisions, and feedback connect research intent to numerical results.
- **Separate final evaluation**: the loop uses inner data; factors and model choices are frozen before a single outer evaluation.

<a id="quick-start"></a>

## Quick Start

Use **Python 3.10+**, a local Parquet feature panel, and a `/chat/completions` endpoint that supports structured JSON output. Factor research and model evaluation run on CPUs; model inference is provided by the connected service.

### 1. Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Configure a model

Connect a hosted API or a compatible endpoint you deploy yourself.

```bash
export ALPHADIVERSE_BASE_URL="https://your-provider.example/v1"
export ALPHADIVERSE_MODEL="your-model-id"
read -r -s -p "API key: " ALPHADIVERSE_API_KEY
export ALPHADIVERSE_API_KEY
```

Planner, Realizer, and Analysis share this model by default. For a locally hosted inference service, set `ALPHADIVERSE_BACKEND=local`; local endpoints without authentication may omit the key.

<details>
<summary><strong>Per-role configuration and generation settings</strong></summary>

Override settings with the prefixes `ALPHADIVERSE_PLANNER_`, `ALPHADIVERSE_REALIZER_`, or `ALPHADIVERSE_ANALYSIS_`, for example `ALPHADIVERSE_PLANNER_MODEL` and `ALPHADIVERSE_REALIZER_BASE_URL`.

| Environment variable suffix | Default | Description |
| :--- | :--- | :--- |
| `BACKEND` | `api` | `api` or `local` |
| `CONTEXT_TOKENS` | `131072` | Set to the model's actual context limit |
| `MAX_TOKENS` | `8192` | Initial output budget; bounded retries can expand it within the context limit |
| `TIMEOUT_SECONDS` | `300` | Per-request read timeout in seconds |
| `TEMPERATURE` | `0.1` | Sampling temperature |
| `THINKING` | Unset | `0` / `1`, when supported by the endpoint |
| `REASONING_EFFORT` | Unset | A reasoning-effort value supported by the endpoint |

</details>

### 3. Run a research loop

Prepare a feature panel following the [data interface](docs/data_EN.md), then run:

```bash
python -m alphadiverse \
  --data data/base_panel \
  --membership-data data/daily_weights \
  --market hs300 \
  --base level1 \
  --folds 1 \
  --rounds 20 \
  --output results/hs300_run
```

**Defaults: HS300 · Level1 · 1 fold · 20 rounds · feature mask 70%.** Omit `--membership-data` if the panel already contains complete point-in-time daily membership flags. `bash run.sh` is equivalent to `python -m alphadiverse`; use a new output directory for each run.

<details>
<summary><strong>Markets, date ranges, and research settings</strong></summary>

| Argument | Default | Options |
| :--- | :--- | :--- |
| `--market` | `hs300` | `ashare`, `hs300`, `csi500`, `csi1000` |
| `--base` | `level1` | `level1`, `alpha20` |
| `--folds` | `1` | Set to `n` for n-fold research |
| `--rounds` | `20` | Number of research rounds |
| `--feature-mask` | `70` | Percentage of optional research features to mask; core inputs remain available |
| `--threads` | `4` | Local compute threads |
| `--inner-start` / `--inner-end` | `2022-05-01` / `2025-12-31` | Inner research interval |
| `--outer-start` / `--outer-end` | `2026-01-01` / `2026-08-31` | Final held-out evaluation interval |

Supply enough trading days for the chronological train, validation, and research splits. `--base-features` also accepts an explicit base-factor subset; see the [data interface](docs/data_EN.md).

</details>

<a id="data"></a>

## Data

The system reads local, precomputed stock-date feature panels from a Parquet file or non-overlapping Parquet partitions. Each `date`–`symbol` pair must be unique.

| Input | Contents |
| :--- | :--- |
| Identifiers and market data | `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `turnover` |
| Research features | Base factors and available daily / minute-derived price-volume features |
| Return target | `label_<market>_excess_ret_1d` |
| Historical universe | `in_<market>`, or daily historical constituents supplied through `--membership-data` |

Features for signal date **t** use information available by that day's close. The target is the market-excess return from **open(t+1) to open(t+2)**. Use point-in-time daily membership and apply tradability filters during data preparation.

See the **[data interface documentation](docs/data_EN.md)** for the full schema, Level1 / Alpha20 base factors, rolling-feature warm-up, and membership-table format.

<a id="evaluation"></a>

## Evaluation and Outputs

The loop screens factors and selects models on inner chronological folds, evaluating Ridge and LightGBM under the same splits. New factors enter the retained set when they pass screening and improve the model's composite score. After research, the factor set and model choice are frozen, the final predictor is fitted using inner data, and outer evaluation is performed.

<details>
<summary><strong>Composite score and temporal splits</strong></summary>

Inner data are divided into the requested number of chronological folds. Each fold uses **70% training / 10% validation / 20% research evaluation**, with target-overlap purging at boundaries. Retention requires an improvement in the trading-day-weighted score and in a strict majority of corresponding folds.

The composite score averages these five normalized metrics:

| Metric | RankIC | RankICIR | IC | ICIR | Long-short annual return |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Normalization scale | 0.05 | 0.35 | 0.03 | 0.25 | 0.30 |

ICIR / RankICIR are the mean daily correlation divided by its daily standard deviation; Sharpe is annualized. Returns are reported before costs.

</details>

```text
results/hs300_run/
├── run.json             # Run settings and state
├── rounds/              # Plans, factors, decisions, and feedback by round
├── factors.json         # Factor specifications and accepted / retained flags
├── summary.json         # Completion summary and retained factors
├── metrics.csv          # Baseline / final outer metrics
├── daily_ic.csv         # Daily IC and RankIC
├── returns.csv          # Daily portfolio returns used for scoring
├── final_groups.csv     # Final-model decile returns
├── final_returns.csv    # Final-model top10%−bottom10% returns
└── equity.png           # Four-panel final-model figure
```

`equity.png` shows **ten group equity curves, top10%−bottom10% equity, drawdown with maximum drawdown marked, and 20-day rolling RankIC**, all for the final model. If a run stops early, `status.json` records the failing stage and completed rounds remain available.

## Code Map

| Module | Contents |
| :--- | :--- |
| [`alphadiverse/agent/`](alphadiverse/agent/) | Planning, retrieval, realization, validation, and feedback |
| [`alphadiverse/operators/`](alphadiverse/operators/) | Operator registration, specification validation, and execution |
| [`alphadiverse/factors/`](alphadiverse/factors/) | Price-volume features and base-factor definitions |
| [`alphadiverse/model_eval.py`](alphadiverse/model_eval.py) | Chronological model selection and final held-out evaluation |
| [`alphadiverse/config.py`](alphadiverse/config.py) | Default research and evaluation settings |

<a id="citation"></a>

## Citation

If this project supports your research, please cite the [paper](https://arxiv.org/abs/2609.29014):

```bibtex
@article{wang2026alphadiverse,
  title   = {AlphaDiverse: Post-Training Local Quantitative Research Agents for Diverse Exploration in Alpha Factor Mining},
  author  = {Wang, Qingzhuo and Wei, Zikun and Wei, Zhihua and Shen, Wen},
  journal = {arXiv preprint arXiv:2609.29014},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.29014}
}
```
