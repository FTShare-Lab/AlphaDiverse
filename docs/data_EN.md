# Data interface

[Project home](../README_EN.md) · [中文](data.md)

`--data` accepts a Parquet file, a directory of Parquet partitions, or a cache directory containing `base_panel/`. Dates are inclusive at the CLI. The reader loads the configured inner-to-outer range. Precomputed rolling features should already include sufficient historical warm-up; newly generated rolling factors may be missing during their initial windows. Input partitions must not overlap: each `date`–`symbol` pair must be unique. If an upstream cache retains historical refresh snapshots, select the active partitions according to its manifest and place them in a separate directory. The reader rejects duplicate rows rather than deduplicating them automatically.

Dataset columns must use the following conventions:

| Column | Meaning |
|---|---|
| `date` | Signal formation date, YYYY-MM-DD |
| `symbol` | Shanghai/Shenzhen stock identifier, e.g. `600000.XSHG` or `000001.XSHE`; `.SH`/`.SZ` suffixes are also accepted |
| `open`, `high`, `low`, `close` | Daily prices using a consistent adjustment convention |
| `volume` | Traded share quantity |
| `turnover` | Traded monetary amount, not turnover rate |
| Base and available features | Precomputed causal daily/minute price-volume features |
| `label_<market>_excess_ret_1d` | Signed next-open to following-open return minus the corresponding market benchmark return |
| `in_<market>` | Boolean historical membership on each signal date |
| `universe_eligible` | Optional precomputed tradability flag; in its absence the panel must already be filtered |

For signal date t, features may use information available by close t and the one-day target covers open(t+1) to open(t+2). Labels and tradability filters are taken from the supplied dataset; the program does not rebuild or silently change them. Provide mature labels through the outer end date. Apply the intended ST, suspended-stock, price-limit and 60-day new-listing filters during dataset preparation. Beijing-listed stocks are excluded.

The default Level1 base contains 21 basic price-volume factors: `ret_1`, `open_close_ret`, `high_low_range`, `close_position`, `vwap`, `vwap_deviation`, `log_turnover`, `log_volume`, nine candlestick-shape factors (`KMID`, `KLEN`, `KMID2`, `KUP`, `KUP2`, `KLOW`, `KLOW2`, `KSFT`, `KSFT2`), and `OPEN0`, `HIGH0`, `LOW0`, `VWAP0`. `--base alpha20` selects the supported Alpha20 feature set. Missing base factors cause an explicit failure. Available research inputs are restricted to the supported price-volume feature catalogue; labels, membership, weights and unrelated numeric columns are excluded from factor inputs.

For a predefined ablation, `--base-features ret_1 high_low_range close_position log_volume ROC60 STD5 VSTD5 CORR5` selects an explicit subset from the union of Level1 and Alpha20. It overrides `--base`; the same list forms the baseline and final model base features and is saved in `run.json`. Duplicate names and noncandidate inputs such as labels or membership flags are rejected.

For index markets, use point-in-time daily membership. `--membership-data` can supply a separate Parquet table with `date`, `market` and `component_symbol`; additional weight columns are ignored. It overrides cached membership and must cover every loaded observation date. No current-constituent or static fallback is applied. If this option is omitted, `in_<market>` in the panel must already contain correct daily membership. For `ashare`, the universe is all supplied eligible Shanghai/Shenzhen A-shares.
