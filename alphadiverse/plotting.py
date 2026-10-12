"""Final-model outer diagnostics, separate from the research score."""

from pathlib import Path

import numpy as np
import pandas as pd


def final_model_figure(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter

    output = Path(output)
    frames = []
    for filename in ("final_groups.csv", "final_returns.csv", "daily_ic.csv"):
        frame = pd.read_csv(output / filename)
        frame = frame.loc[frame.model.eq("final")].copy()
        if frame.empty or frame.market.nunique() != 1:
            raise ValueError("Final figure requires one market with final-model data")
        frame["date"] = pd.to_datetime(frame.date)
        frames.append(frame.sort_values("date"))
    groups, returns, ic = frames
    if len({f.market.iloc[0] for f in frames}) != 1:
        raise ValueError("Final figure market mismatch")
    buckets = groups.pivot(index="date", columns="group", values="return")
    if set(buckets.columns) != set(range(1, 11)) or buckets.isna().any().any():
        raise ValueError("Final figure requires all ten groups on each plotted day")
    if returns.date.duplicated().any() or ic.date.duplicated().any():
        raise ValueError("Duplicate dates in final figure data")
    if not buckets.index.equals(pd.DatetimeIndex(returns.date)):
        raise ValueError("Group and long-short dates differ")
    # Include initial capital in the running peak, including a first-day loss.
    equity = (1.0 + returns.long_short).cumprod()
    drawdown = equity / np.maximum.accumulate(np.r_[1.0, equity])[1:] - 1.0
    worst = int(np.argmin(drawdown.to_numpy()))
    rolling = ic.rank_ic.rolling(20, min_periods=20).mean()
    styles = {
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
        "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "legend.fontsize": 7, "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    }
    with plt.rc_context(styles):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.8))
        fig.subplots_adjust(left=0.085, right=0.975, bottom=0.085, top=0.82,
                            wspace=0.29, hspace=0.48)
        a, b, c, d = axes.flat
        colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, 10))
        group_equity = (1.0 + buckets).cumprod()
        for group, color in zip(range(1, 11), colors):
            a.plot(buckets.index, group_equity[group], color=color, lw=1.1,
                   label=f"G{group}")
        a.set(title="Decile equity", ylabel="Excess-return NAV")
        fig.legend(*a.get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(0.53, 0.925), ncol=10, handlelength=1.4,
                   columnspacing=0.9)
        b.plot(returns.date, equity, color="#286F8E", lw=1.5)
        b.set(title="Top 10% − bottom 10%", ylabel="Long-short NAV")
        c.fill_between(returns.date, drawdown, 0, color="#B8CED9")
        c.plot(returns.date, drawdown, color="#286F8E", lw=1)
        c.scatter([returns.date.iloc[worst]], [drawdown.iloc[worst]],
                  color="#AE6543", s=16, zorder=3)
        c.set(title=f"Drawdown · maximum {drawdown.iloc[worst]:.2%}", ylabel="Drawdown")
        c.yaxis.set_major_formatter(PercentFormatter(1))
        d.plot(ic.date, rolling, color="#286F8E", lw=1.4)
        d.axhline(0, color="0.6", lw=0.7, linestyle="--")
        d.set(title="RankIC · 20-day rolling mean", ylabel="RankIC")
        for letter, ax in zip("abcd", axes.flat):
            ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-30, 10),
                        textcoords="offset points", weight="bold", fontsize=10)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(labelsize=7)
            ax.grid(axis="y", color="0.9", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_xlim(returns.date.min(), returns.date.max())
        market = str(returns.market.iloc[0]).upper()
        fig.suptitle(f"{market} · Final model · Outer evaluation", y=0.98, fontsize=12)
        return fig


def plot_final_results(output):
    import matplotlib.pyplot as plt

    fig = final_model_figure(output)
    try:
        fig.savefig(Path(output) / "equity.png", dpi=300)
    finally:
        plt.close(fig)
