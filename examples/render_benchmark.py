"""Render the committed benchmark summary. Requires the optional plots extra."""

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports/market_benchmark.json").read_text())
    windows = report["windows"]
    names = ["2020", "2022", "2025–Oct 2026"]
    improvements = [
        100 * (1 - w["metrics_model"]["crps_mean"] / w["metrics_baseline"]["crps_mean"])
        for w in windows
    ]
    breach = {
        method: [100 * w["portfolio"][method]["0.05"]["violation_rate"] for w in windows]
        for method in ["model", "baseline"]
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "synthetic-market-benchmark-v1",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1, 1.5]})
    fig.patch.set_facecolor("#fafaf8")
    for ax in axes:
        ax.set_facecolor("#fafaf8")
        ax.grid(axis="y", alpha=0.2, zorder=0)
    x = np.arange(3)
    bars = axes[0].bar(x, improvements, color="#235b78", width=0.55, zorder=3)
    axes[0].bar_label(bars, labels=[f"{v:.1f}%" for v in improvements], padding=5)
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(min(0, min(improvements) * 1.32), max(1, max(improvements) * 1.32))
    axes[0].axhline(0, color="#555555", linewidth=0.7)
    axes[0].set_ylabel("Reduction in average marginal CRPS (%)")
    axes[0].set_title("Distributional score comparison", loc="left", fontweight="bold")
    for shift, method, color, label in [
        (-0.18, "model", "#235b78", "Filtered model"),
        (0.18, "baseline", "#a5aaa9", "Unfiltered baseline"),
    ]:
        bars = axes[1].bar(
            x + shift, breach[method], width=0.34, color=color, label=label, zorder=3
        )
        axes[1].bar_label(bars, labels=[f"{v:.1f}%" for v in breach[method]], padding=4, fontsize=9)
    axes[1].axhline(5, color="#a8443c", linestyle="--", linewidth=1.3, label="Nominal 5%")
    axes[1].set_xticks(x, names)
    axes[1].set_ylim(0, max(25, max(max(v) for v in breach.values()) * 1.25))
    axes[1].set_ylabel("Observed portfolio loss-tail breaches (%)")
    axes[1].set_title("Observed portfolio tail frequency", loc="left", fontweight="bold")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(
        "SPY / TLT / GLD · equal weights · chronological one-step evaluation",
        x=0.08,
        y=1.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    tail2022 = windows[1]["portfolio"]["model"]["0.05"]
    fig.text(
        0.08,
        -0.01,
        f"Lower average scores do not certify calibration. 2022: {tail2022['violations']} breaches / {tail2022['observations']} sessions at the nominal 5% tail.",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    destination = root / "docs/assets"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "benchmark.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(
        Path(tempfile.gettempdir()) / "synthetic_benchmark_preview.png",
        dpi=150,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
