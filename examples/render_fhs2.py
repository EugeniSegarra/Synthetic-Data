"""Render separate evidence figures from complete research bundles.

Matplotlib is optional for rendering. No plotting changes research calculations.
Path scenarios are conditional model draws, not confidence bands for calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from synthetic_market.data import read_csv
from synthetic_market.research2.generators import fit_factorial

LABELS = {
    "fhs_global_zero": "FHS de referencia",
    "fhs_global_shrink": "FHS + media contraída",
    "harch_state_shrink": "HARCH + estado + media",
    "delayed_selector": "Selector con resultados maduros",
}


def render(root, csv, out):
    root, out = Path(root), Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    frames = {
        asset: pd.read_csv(root / asset / "forecasts.csv", float_precision="round_trip")
        for asset in ("SPY", "EWJ", "EEM", "VGK")
    }
    summaries = {a: json.loads((root / a / "summary.json").read_text()) for a in frames}
    files = []

    def save(fig, name, note):
        fig.text(0.09, 0.025, note, fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        for extension in ("png", "svg"):
            path = out / f"{name}.{extension}"
            fig.savefig(path, dpi=170, bbox_inches="tight")
            files.append(str(path))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.3))
    positions = np.arange(4)
    for k, method in enumerate(("fhs_global_shrink", "harch_state_shrink", "delayed_selector")):
        values, counts = [], []
        for asset in frames:
            record = next(
                x
                for x in summaries[asset]["simultaneous_comparisons"]
                if x["horizon"] == 126 and x["block_length"] == 12 and x["method"] == method
            )
            values.append(record["relative_change"])
            counts.append(record["common_origins"])
        bars = ax.bar(positions + (k - 1) * 0.25, values, 0.25, label=LABELS[method])
        ax.bar_label(bars, labels=[f"{v:+.1%}" for v in values], padding=3, fontsize=9)
    ax.set_xticks(positions, [f"{a}\nn = {n}" for a, n in zip(frames, counts, strict=True)])
    ax.axhline(0, linewidth=1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(
        "Caída máxima a seis meses · cambio en el error probabilístico", fontsize=14, pad=14
    )
    ax.set_ylabel("Cambio relativo del CRPS frente a FHS; negativo es mejor")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.18)
    save(
        fig,
        "comparacion_externa",
        "Mismas fechas por mercado · Desde 2015 · Ninguna variante supera el criterio de sustitución predefinido.",
    )

    spy = frames["SPY"]
    six = spy[(spy.origin >= "2015-01-01") & (spy.horizon == 126)]
    fig, ax = plt.subplots(figsize=(11, 6.3))
    groups = ("low", "medium", "high")
    counts = []
    for k, method in enumerate(("fhs_global_zero", "fhs_global_shrink")):
        part = six[six.method == method]
        values = [
            part.loc[part.initial_vol_group == g, "drawdown_hit10_prob"].mean() for g in groups
        ]
        ax.bar(positions[:3] + (k - 1) * 0.25, values, 0.25, label=LABELS[method])
    base = six[six.method == "fhs_global_zero"]
    values = [base.loc[base.initial_vol_group == g, "drawdown_hit10_actual"].mean() for g in groups]
    counts = [int((base.initial_vol_group == g).sum()) for g in groups]
    bars = ax.bar(positions[:3] + 0.25, values, 0.25, label="Frecuencia realizada")
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3)
    ax.set_xticks(
        positions[:3],
        [f"<15%\nn = {counts[0]}", f"15–25%\nn = {counts[1]}", f"≥25%\nn = {counts[2]}"],
    )
    ax.set_xlabel("Volatilidad RMS anualizada de las 252 sesiones anteriores a la previsión")
    ax.set_ylabel("Probabilidad / frecuencia de drawdown ≥10% en 126 sesiones")
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("SPY · el desajuste depende del estado inicial", fontsize=14, pad=14)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.18)
    save(
        fig,
        "calibracion_condicional",
        "136 ventanas solapadas; solo 12 en el grupo de volatilidad alta · Diagnóstico descriptivo, no prueba causal.",
    )

    dates = pd.to_datetime(base.origin)
    fig, ax = plt.subplots(figsize=(12, 6.3))
    ax.fill_between(
        dates,
        base.max_drawdown_q05,
        base.max_drawdown_q95,
        alpha=0.18,
        label="Escenarios FHS: percentiles 5–95",
    )
    ax.plot(dates, base.max_drawdown_actual, linewidth=1.8, label="Drawdown realizado")
    altered = six[six.method == "fhs_global_shrink"]
    ax.plot(
        pd.to_datetime(altered.origin),
        altered.max_drawdown_q50,
        linestyle="--",
        linewidth=1.4,
        label="Mediana FHS + media contraída",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Caída máxima desde un máximo, incluido el precio inicial")
    ax.set_xlabel("Fecha de emisión de la previsión; no fecha del mínimo")
    ax.set_title("SPY · drawdown de las 126 sesiones siguientes", fontsize=14, pad=14)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.18)
    save(
        fig,
        "drawdown_realizado",
        "Las bandas son distribuciones de escenarios, no intervalos de confianza del estudio. Cada camino usa cierres diarios completos.",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    hs = (21, 63, 126, 252)
    for method in ("fhs_global_zero", "fhs_global_shrink", "harch_state_shrink"):
        part = spy[(spy.origin >= "2015-01-01") & (spy.method == method)]
        ax.plot(
            hs,
            [part.loc[part.horizon == h, "recovery_prob"].mean() for h in hs],
            marker="o",
            label=LABELS[method],
        )
    part = spy[(spy.origin >= "2015-01-01") & (spy.method == "fhs_global_zero")]
    ax.plot(
        hs,
        [part.loc[part.horizon == h, "recovery_actual"].mean() for h in hs],
        marker="x",
        linestyle="--",
        label="Frecuencia realizada (muestra FHS)",
    )
    ax.set_xticks(hs)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Horizonte en sesiones")
    ax.set_ylabel("Probabilidad media / frecuencia de recuperación")
    ax.set_title("SPY · recuperación del máximo ya conocido", fontsize=14, pad=14)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.18)
    save(
        fig,
        "recuperacion",
        "Solo orígenes bajo un máximo previo; denominadores variables por madurez y disponibilidad · No recuperación al horizonte ≠ recuperación al final.",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in ("fhs_global_zero", "harch_global_zero"):
        part = spy[(spy.origin >= "2015-01-01") & (spy.method == method)]
        label = "FHS" if method.startswith("fhs") else "HARCH diario"
        ax.plot(
            hs,
            [part.loc[part.horizon == h, "realized_volatility_q50"].mean() for h in hs],
            marker="o",
            label=f"{label}: mediana prevista media",
        )
    part = spy[(spy.origin >= "2015-01-01") & (spy.method == "fhs_global_zero")]
    ax.plot(
        hs,
        [part.loc[part.horizon == h, "realized_volatility_actual"].mean() for h in hs],
        marker="x",
        linestyle="--",
        label="RMS realizada media (muestra FHS)",
    )
    ax.set_xticks(hs)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("RMS anualizada de rentabilidades diarias")
    ax.set_xlabel("Horizonte en sesiones")
    ax.set_title("SPY · escala de variación a distintos horizontes", fontsize=14, pad=14)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.18)
    save(
        fig,
        "volatilidad_horizontes",
        "Comparación descriptiva de medianas previstas y RMS realizadas, no igualdad esperada de ambas medias · Denominadores en resultados.",
    )

    data = read_csv(csv, price_basis="adjusted")
    fitted = fit_factorial(data)
    paths, errors = fitted.simulate(n_paths=2000, horizon=252, seed=20261030)
    if "fhs_global_zero" not in paths or "fhs_global_shrink" not in paths:
        raise ValueError(errors)
    prices = 100 * np.exp(np.c_[np.zeros(2000), paths["fhs_global_zero"].cumsum(axis=1)])
    x = np.arange(253)
    fig, ax = plt.subplots(figsize=(12, 6.3))
    ax.plot(x, prices[:100].T, linewidth=0.65, alpha=0.12)
    q05, q50, q95 = np.quantile(prices, [0.05, 0.5, 0.95], axis=0)
    ax.fill_between(x, q05, q95, alpha=0.16, label="Percentiles 5–95 de 2.000 caminos")
    ax.plot(x, q50, linewidth=2.3, label="Mediana FHS")
    ax.set_xlabel("Sesiones simuladas; 0 es la condición inicial")
    ax.set_ylabel("Precio ajustado normalizado: inicio = 100")
    ax.set_title(f"SPY · escenarios FHS condicionados al {fitted.origin}", fontsize=14, pad=14)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.18)
    save(
        fig,
        "trayectorias_condicionales",
        "Se muestran 100 de 2.000 caminos, sin reducir la frecuencia diaria · Ilustración del modelo, no validación ni predicción de alpha.",
    )

    fig, ax = plt.subplots(figsize=(11, 6.3))
    terminal = {
        m: np.expm1(paths[m][:, :126].sum(axis=1)) for m in ("fhs_global_zero", "fhs_global_shrink")
    }
    edges = np.histogram_bin_edges(np.concatenate(list(terminal.values())), bins=70)
    for method, values in terminal.items():
        ax.hist(
            values, bins=edges, density=True, histtype="step", linewidth=1.8, label=LABELS[method]
        )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Rentabilidad ajustada acumulada a 126 sesiones")
    ax.set_ylabel("Densidad empírica simulada")
    ax.set_title(
        "SPY · cambiar la media no corrige toda la distribución del camino", fontsize=14, pad=14
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.18)
    save(
        fig,
        "distribucion_terminal",
        f"Condición inicial {fitted.origin} · 2.000 simulaciones pareadas con los mismos shocks · Sin recortar extremos.",
    )
    (out / "figures.json").write_text(
        json.dumps({"files": files, "input_asof": fitted.origin}, indent=2)
    )
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    render(args.results_root, args.csv, args.out)


if __name__ == "__main__":
    main()
