# %%
"""Collapse-design grid with the soft (linear) and quadratic collapse side by side.

Same 3x5 layout as collapse_grid.py (rows = models, columns = benchmarks), but each
cell has 13 rows: the 6 collapse variants with the soft collapse (pale colours),
the same 6 with the quadratic collapse (saturated colours), and "no collapse".
Cells missing any of the 13 runs are left blank.

Run names are the ones collapse_grid.py uses: collapse_{exp}_{model}_{suffix} for
soft, collapse2_{exp}_{model}_{suffix}_quadratic for quadratic; the "no collapse"
row is the soft forget_none run (collapse_on=none skips the collapse entirely).
Shares collapse_cache.pkl with collapse_grid.py.

NOTE n_pcs mismatch on bio and aa: the soft runs (2026-04-24) used n_pcs=500, the
quadratic ones (2026-07-26) 512. Left as is: July's soft bio reruns at 512
(collapse2_bio_*) match the 500 runs to within 0.03 points, and Fig. 8 shows a flat
plateau over 128-1024. cyber, rwku and sycophancy use 512 on both sides.
"""
import pickle
import time
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import yaml

import wandb

try:  # wandb credentials for non-interactive runs; a no-op if the key is in the env
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).parent
CACHE_FILE = SCRIPT_DIR / "collapse_cache.pkl"
REL_PROJECT = "filyp/rel-selective-unlearning"
REL_STEPS = 11  # epoch 0 + 10 relearning epochs
PALE = 0.45  # soft-collapse bars: this fraction of the colour, the rest white

# Baselines from dedicated reference runs in wandb. Shape: {dataset: {model: value}}.
_BENCHMARKS_DIR = SCRIPT_DIR.parent.parent / "benchmarks"
baselines: dict[str, dict[str, float]] = {}
for _path in [
    _BENCHMARKS_DIR / "wmdp_low_mi" / "baselines.yaml",
    _BENCHMARKS_DIR / "beavertails" / "baselines.yaml",
    _BENCHMARKS_DIR / "rwku" / "baselines.yaml",
    _BENCHMARKS_DIR / "sycophancy" / "baselines.yaml",
]:
    with open(_path) as _f:
        baselines.update(yaml.safe_load(_f))

plt.style.use("default")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 10

MODELS = [
    ("Llama-3.1-8B", "Llama-3.1-8B"),
    ("Qwen3.5-9B", "Qwen3.5-9B"),
    ("DeepSeek-V2-Lite", "DeepSeek-V2-Lite"),
]

# (exp_name_in_task, display, reference_benchmark_tag, metric)
BENCHMARKS = [
    ("bio", "WMDP-Bio", "bio", "train/recall_prob"),
    ("cyber", "WMDP-Cyber", "cyber", "train/recall_prob"),
    ("rwku", "RWKU", "rwku", "train/recall_cloze_prob"),
    ("sycophancy", "Sycophancy", "sycophancy", "train/recall_prob"),
    ("AA", "Animal Abuse", "animal_abuse", "train/holdout_harmful_prob"),
]

VARIANTS = ["forget_act", "forget_grad", "forget_both", "retain_act", "retain_grad", "retain_both"]

# rows of one cell, top to bottom: (label, task-name suffix, collapse kind)
ROWS = (
    [(f"linear: {s.replace('_', ' / ')}", s, "soft") for s in VARIANTS]  # "soft" in the code, "linear" in the plot
    + [(f"quad: {s.replace('_', ' / ')}", s, "quadratic") for s in VARIANTS]
    + [("no collapse", "forget_none", "soft")]
)


def task_name(exp_name, model, suffix, kind):
    if kind == "soft" or suffix.endswith("_none"):
        return f"collapse_{exp_name}_{model}_{suffix}"
    return f"collapse2_{exp_name}_{model}_{suffix}_{kind}"


# %%
# === CELL 1: fetch relearning trajectories (cached, shared with collapse_grid.py) ===

if CACHE_FILE.exists():
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded {len(cache)} cached runs from {CACHE_FILE}")
else:
    cache = {}

expected = []
for _, model_field in MODELS:
    for exp_name, _, _, _ in BENCHMARKS:
        for _, suffix, kind in ROWS:
            expected.append((exp_name, task_name(exp_name, model_field, suffix, kind)))


def _needs_fetch(t):
    hist = cache.get(t)
    return hist is None or len(hist) == 0


missing = [(e, t) for e, t in expected if _needs_fetch(t)]
if missing:
    api = wandb.Api(timeout=3600)
    for exp_name, t in missing:
        metric = next(m for en, _, _, m in BENCHMARKS if en == exp_name)
        print(f"Fetching {t} (metric={metric})...")
        runs = list(api.runs(REL_PROJECT, filters={"display_name": t}))
        if len(runs) == 0:
            print("  no run found; marking None")
            cache[t] = None
            continue
        if len(runs) > 1:
            print(f"  warning: {len(runs)} runs, taking first")
        for i in range(10):
            try:
                hist = runs[0].history(keys=[metric])
                break
            except Exception as e:
                print(f"  attempt {i} failed: {e}")
                time.sleep(2**i)
        else:
            raise RuntimeError(f"failed to fetch history for {t}")
        cache[t] = hist
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved {len(cache)} runs to {CACHE_FILE}")


def max_over_attack(t, metric):
    """Max of the metric over the relearn trajectory, or None if the run is missing."""
    hist = cache.get(t)
    if hist is None or len(hist) == 0 or metric not in hist.columns:
        return None
    head = hist.head(REL_STEPS)[metric].dropna()
    return head.max() * 100 if len(head) else None


# %%
# === CELL 2: plot ===

nrows, ncols = len(MODELS), len(BENCHMARKS)
# same width as collapse_grid.py; height scaled to the 13 rows per cell
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5, 0.8 * (1.0 + 2.4 * nrows)))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
collapse_color = {"act": colors[0], "grad": colors[1], "both": colors[2], "none": colors[3]}


def pale(color):
    r, g, b = mcolors.to_rgb(color)
    return (1 - PALE * (1 - r), 1 - PALE * (1 - g), 1 - PALE * (1 - b))


def style_for(suffix, kind):
    dist, coll = suffix.rsplit("_", 1)
    color = collapse_color[coll]
    if kind == "soft" and coll != "none":
        color = pale(color)
    hatch = "///" if dist == "retain" else ""
    return color, hatch


n_blank = 0
for row_idx, (model_display, model_field) in enumerate(MODELS):
    for col_idx, (exp_name, bench_display, bench_tag, metric) in enumerate(BENCHMARKS):
        ax = axes[row_idx][col_idx]
        assert baselines[bench_tag][model_field] is not None
        baseline = baselines[bench_tag][model_field] * 100

        values = [
            max_over_attack(task_name(exp_name, model_field, suffix, kind), metric)
            for _, suffix, kind in ROWS
        ]

        if row_idx == 0:
            ax.set_title(bench_display)
        if col_idx == 0:
            ax.set_ylabel(model_display)

        # blank panel unless every one of the 13 runs is there
        if any(v is None for v in values):
            n_blank += 1
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        y_positions = list(range(len(ROWS) - 1, -1, -1))  # first row on top
        bars = ax.barh(
            y_positions,
            [v - baseline for v in values],
            left=baseline,
            height=0.9,
            color=[style_for(s, k)[0] for _, s, k in ROWS],
        )
        for patch, (_, s, k) in zip(bars, ROWS):
            hatch = style_for(s, k)[1]
            if hatch:
                patch.set_hatch(hatch)
                patch.set_edgecolor("white")
        # thin gap between the soft block, the quadratic block and "no collapse"
        for y in (len(ROWS) - len(VARIANTS) - 0.5, 0.5):  # soft|quad and quad|none
            ax.axhline(y, color="0.6", lw=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        min_val = min(values)
        min_val -= (baseline - min_val) * 0.05 if baseline > min_val else 0.5
        ax.set_xlim(min_val, baseline)
        ax.set_ylim(min(y_positions) - 0.5, max(y_positions) + 0.5)

        if col_idx == ncols - 1:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([label for label, _, _ in ROWS])
            ax.yaxis.tick_right()
        else:
            ax.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
fig.text(0.5, -0.01, "Post-Attack Answer Probability (%) ↓", ha="center", va="bottom")

save_path = SCRIPT_DIR / "collapse_grid_combined.pdf"
fig.savefig(save_path, bbox_inches="tight", dpi=150)
print(f"Saved plot to {save_path} ({n_blank} of {nrows * ncols} cells blank)")
plt.show()

# %%
