# %%
# Collapse along per-token retain covariances (RepSelectCov, collapse_cov.sh) vs the
# retain / both cell of the collapse-design grid (collapse_grid.py, paper Fig. 7).
# Same protocol in every row: step size bisected onto the KL budget, no LoRA.
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

import wandb

_BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"
baselines: dict[str, dict[str, float]] = {}
for _dir in ["wmdp_low_mi", "beavertails", "rwku", "sycophancy"]:
    with open(_BENCHMARKS_DIR / _dir / "baselines.yaml") as _f:
        baselines.update(yaml.safe_load(_f))

plt.style.use("default")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 10

SCRIPT_DIR = Path(__file__).parent
CACHE_FILE = SCRIPT_DIR / "collapse_cov_cache.pkl"

REL_PROJECT = "filyp/rel-selective-unlearning"
REL_STEPS = 11

MODELS = ["Llama-3.1-8B", "Qwen3.5-9B"]

# (exp name in the collapse*_ runs, in the cov_ runs, display, baselines tag, metric)
BENCHMARKS = [
    ("bio", "bio", "WMDP-Bio", "bio", "train/recall_prob"),
    ("cyber", "cyber", "WMDP-Cyber", "cyber", "train/recall_prob"),
    ("rwku", "rwku", "RWKU", "rwku", "train/recall_cloze_prob"),
    ("sycophancy", "sycophancy", "Sycophancy", "sycophancy", "train/recall_prob"),
    ("AA", "aa", "Animal Abuse", "animal_abuse", "train/holdout_harmful_prob"),
]

# (label, run name); top to bottom
ROWS = [
    ("no collapse", "collapse_{exp}_{model}_forget_none"),
    ("RepSelect", "collapse_{exp}_{model}_retain_both"),
    ("RepSelect quadratic", "collapse2_{exp}_{model}_retain_both_quadratic"),
    ("cov unweighted", "cov_{cov_exp}_{model}_retain_both_unweighted"),
    ("cov unw. inverse", "cov_{cov_exp}_{model}_retain_both_unweighted_inverse"),
    ("cov weighted", "cov_{cov_exp}_{model}_retain_both_weighted"),
    ("cov w. inverse", "cov_{cov_exp}_{model}_retain_both_weighted_inverse"),
    # as "cov w. inverse", with each batch's cov computed in fp64 instead of fp32
    ("cov w. inv. fp64", "cov_{cov_exp}_{model}_retain_both_weighted_inverse_fp64"),
    # ridge = the cov's 512th eigenvalue instead of 1e-3 x its mean diagonal
    ("cov w. inv. eps512", "cov_{cov_exp}_{model}_retain_both_weighted_inverse_eps512"),
]

# runs that crash (Cholesky: cov + ridge not positive definite); marked in the plot
FAILING = {"cov_aa_Llama-3.1-8B_retain_both_unweighted_inverse"}


def task_name(template, model, exp, cov_exp):
    return template.format(exp=exp, cov_exp=cov_exp, model=model)


# %%
# === CELL 1: fetch relearning trajectories (cached) ===

cache = pickle.load(open(CACHE_FILE, "rb")) if CACHE_FILE.exists() else {}

expected = [
    (task_name(template, model, exp, cov_exp), metric)
    for model in MODELS
    for exp, cov_exp, _, _, metric in BENCHMARKS
    for _, template in ROWS
]
missing = [(t, m) for t, m in expected if t not in cache and t not in FAILING]
if missing:
    api = wandb.Api(timeout=3600)
    for t, metric in missing:
        runs = list(api.runs(REL_PROJECT, filters={"display_name": t}))
        if len(runs) != 1 or runs[0].state != "finished":
            # not cached, so it is retried next time
            print(f"SKIPPED {t}: {[(r.id, r.state) for r in runs]}")
            continue
        for i in range(10):
            try:
                cache[t] = runs[0].history(keys=[metric])
                break
            except Exception as e:
                print(f"  attempt {i} failed: {e}")
                time.sleep(2**i)
        else:
            raise RuntimeError(f"failed to fetch history for {t}")
        if cache[t][metric].notna().sum() < REL_STEPS:  # epochs 0..10
            print(f"SKIPPED {t}: only {len(cache.pop(t))} relearning points")
            continue
        print(f"fetched {t}: {len(cache[t])} points")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

# %%
# === CELL 2: plot ===

nrows, ncols = len(MODELS), len(BENCHMARKS)
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5, 1.0 + 1.7 * nrows))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# "no collapse" keeps its color from the collapse grid
# the inverse variants share their cov's color, hatched
bar_colors = [colors[3], colors[2], colors[0], colors[1], colors[1]] + [colors[4]] * 4
bar_hatches = ["", "", "", "", "///", "", "///", "...", "xxx"]
y_positions = list(range(len(ROWS) - 1, -1, -1))  # first row on top

for row_idx, model in enumerate(MODELS):
    for col_idx, (exp, cov_exp, bench_display, bench_tag, metric) in enumerate(
        BENCHMARKS
    ):
        ax = axes[row_idx][col_idx]
        baseline = baselines[bench_tag][model] * 100

        values = []
        for label, template in ROWS:
            hist = cache.get(task_name(template, model, exp, cov_exp))
            if hist is None:
                values.append(baseline)  # missing run: no bar
                continue
            values.append(hist.head(REL_STEPS)[metric].dropna().max() * 100)
            print(f"{model:13s} {bench_display:13s} {label:20s} {values[-1]:6.2f}")

        bars = ax.barh(
            y_positions,
            [v - baseline for v in values],
            height=0.9,
            color=bar_colors,
            left=baseline,
        )
        for patch, hatch in zip(bars, bar_hatches):
            if hatch:
                patch.set_hatch(hatch)
                patch.set_edgecolor("white")
        for y, (_, template) in zip(y_positions, ROWS):
            if task_name(template, model, exp, cov_exp) in FAILING:
                ax.text(
                    0.97, y, "FAILING", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=6, color="red",
                )  # fmt: skip
        if row_idx == 0:
            ax.set_title(bench_display)
        if col_idx == 0:
            ax.set_ylabel(model)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)

        min_val = min(values)
        min_val -= (baseline - min_val) * 0.05 if baseline > min_val else 0.5
        ax.set_xlim(min_val, baseline)
        ax.set_ylim(min(y_positions) - 0.5, max(y_positions) + 0.5)
        if col_idx == ncols - 1:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([label for label, _ in ROWS])
            ax.yaxis.tick_right()
        else:
            ax.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
fig.text(0.5, -0.02, "Post-Attack Answer Probability (%) ↓", ha="center", va="bottom")

save_path = SCRIPT_DIR / "collapse_cov_grid.pdf"
fig.savefig(save_path, bbox_inches="tight", dpi=150)
print(f"Saved plot to {save_path}")
plt.show()
