import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm


# CONFIGURATION
CSV_PATH      = "ffnn/results/C_default_lr0.001_wd1e-05_bs32.csv"       
HEXBIN_OUT    = "figures/ffnn_hexbin.png"   
ERROR_OUT     = "figures/ffnn_error_plot.png"    
BUCKET_WIDTH  = 10 # relative error bucket width in %
MAX_PCT       = 100 # buckets beyond ±MAX_PCT% are collapsed into tails
ERROR_YMAX    = 900 # set to None to auto-scale

HEXBIN_TITLE = "Predicted vs Actual - FFNN"
SIGNED_BARS_TITLE  = "Signed Relative Error Distribution - FFNN"

# colour palette
BLUE      = "#3A86FF"
RED       = "#FF595E"
DARK_BG   = "#F7F9FC"
GRID_COL  = "#DDE3ED"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df = df.dropna(subset=["y_true", "y_pred"])
    return df

def signed_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    sre = (y_pred - y_true) / np.abs(y_true) * 100.0
    mask = y_true == 0
    if mask.any():
        raise ValueError(f"  {mask.sum()} rows have y_true=0")
    return sre

def make_buckets(bucket_width: int, max_pct: int):
    edges = list(range(-max_pct, max_pct + 1, bucket_width))
    buckets = []
    # left tail: open on both ends
    buckets.append((f"(-∞, {edges[0]})", -np.inf, edges[0]))
    # interior bands
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == 0:
            label = f"({lo:d}, {hi:+d})"
        elif hi == 0:
            label = f"[{lo:+d}, {hi:d})"
        else:
            label = f"[{lo:+d}, {hi:+d})"
        buckets.append((label, lo, hi))
    # right tail: closed on left, open on right
    buckets.append((f"[{edges[-1]}, +∞)", edges[-1], np.inf))
    return buckets

def bin_errors(sre: np.ndarray, bucket_width: int, max_pct: int):
    buckets = make_buckets(bucket_width, max_pct)
    sre_clean = sre[~np.isnan(sre) & (sre != 0)]   
    counts, labels = [], []
    for label, lo, hi in buckets:
        n = np.sum((sre_clean >= lo) & (sre_clean < hi))
        counts.append(n)
        labels.append(label)
    return labels, np.array(counts)

def plot_hexbin(ax, y_true: np.ndarray, y_pred: np.ndarray):
    """Hexbin predicted vs actual with y=x line."""

    cmap = LinearSegmentedColormap.from_list(
        "hex_cmap", ["#C8E0F8", "#BDD7F5", "#3A86FF", "#003F8A"]
    )

    x_pad = (y_true.max()  - y_true.min())  * 0.05
    y_pad = (y_pred.max()  - y_pred.min())  * 0.05
    x_min, x_max = y_true.min() - x_pad, y_true.max() + x_pad
    y_min, y_max = y_pred.min() - y_pad, y_pred.max() + y_pad

    hb = ax.hexbin(
        y_true, y_pred,
        gridsize=40,
        cmap=cmap,
        mincnt=1,
        linewidths=0.2,
        edgecolors="none",
        extent=[x_min, x_max, y_min, y_max],
        norm=LogNorm(),
    )

    ref_min = min(x_min, y_min)
    ref_max = max(x_max, y_max)
    ax.plot(
        [ref_min, ref_max], [ref_min, ref_max],
        color="#FF595E", linewidth=1.5, linestyle="--"
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    cb = plt.colorbar(hb, ax=ax, pad=0.02)
    cb.ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    ax.set_xlabel("Actual", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.set_title(HEXBIN_TITLE, fontsize=13, fontweight="bold", pad=10)
    ax.set_facecolor(DARK_BG)
    ax.grid(True, color=GRID_COL, linewidth=0.6, zorder=0)

def plot_signed_bars(ax, labels, counts):
    """Standard bar chart of signed relative error buckets."""
    x = np.arange(len(labels))
    colours = [RED if lbl.startswith("(-∞") or lbl.startswith("[-") else BLUE
               for lbl in labels]
    ax.bar(x, counts, color=colours, edgecolor="white", linewidth=0.5, zorder=3)
    if ERROR_YMAX is not None:
        ax.set_ylim(0, ERROR_YMAX)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("number of samples", fontsize=10)
    ax.set_xlabel("signed relative error (%)", fontsize=10)
    ax.set_title(SIGNED_BARS_TITLE, fontsize=12, fontweight="bold")
    ax.set_facecolor(DARK_BG)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.6, zorder=0)
    mid = len(labels) // 2
    ax.axvline(mid - 0.5, color="#718096", linewidth=1, linestyle=":", zorder=4)

def save_hexbin_figure(y_true: np.ndarray, y_pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    plot_hexbin(ax, y_true, y_pred)
    plt.savefig(HEXBIN_OUT, dpi=150, bbox_inches="tight")
    print(f"Saved -> {HEXBIN_OUT}")

def save_signed_bars(labels, counts):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    plot_signed_bars(ax, labels, counts)
    plt.tight_layout()
    plt.savefig(ERROR_OUT, dpi=150, bbox_inches="tight")
    print(f"Saved -> {ERROR_OUT}")

if __name__ == "__main__":
    print(f"Loading {CSV_PATH} …")
    df     = load_data(CSV_PATH)
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    sre    = signed_relative_error(y_true, y_pred)
    labels, counts = bin_errors(sre, BUCKET_WIDTH, MAX_PCT)
    save_hexbin_figure(y_true, y_pred)
    save_signed_bars(labels, counts)