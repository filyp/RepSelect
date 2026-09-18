# %%
"""Robustness vs. disruption across WikiText-KL budgets (Llama-3.1-8B / WMDP-Bio).

One point per (method, budget): x = disruption of the unlearned checkpoint that gets
attacked (WikiText KL at the last step within budget; or MMLU accuracy), y = worst-case
recall probability over the relearning attack.

All budgets: runs launched by community/benchmarks/wmdp_low_mi/run_kl0.1.sh with budget=
edited (top-1 hyperparameters of the paper's 0.01 search, 100 epochs max), fetched from wandb.
The 0.01 budget is rerun rather than taken from the search, so that MMLU is logged.

Fetched data is cached in kl_budget_sweep.json; delete it to refetch.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.style.use("default")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 10

SCRIPT_DIR = Path(__file__).parent
CACHE_FILE = SCRIPT_DIR / "kl_budget_sweep.json"
BASELINES_FILE = SCRIPT_DIR.parent.parent / "benchmarks" / "wmdp_low_mi" / "baselines.yaml"
OUT_FILE = SCRIPT_DIR / "kl_budget_sweep.pdf"

UNL_PROJECT = "filyp/selective-unlearning"
REL_PROJECT = "filyp/rel-selective-unlearning"
MODEL = "Llama-3.1-8B"
DATASET = "bio"
KL_METRIC = "train/wikitext_kl"
ROB_METRIC = "train/recall_prob"
MMLU_METRIC = "train/mmlu_total/acc"

# same order as main_grid.py so colours match
titles_dict = {
    "RepSelectSimple_forget": "RepSelect",
    "RepSelect2_forget": None,
    "RepSelectSimple_forget_no_lora": None,
    "NPO": "NPO",
    "RMU": "RMU",
    "UNDIAL": "UNDIAL",
    "SimNPO": "SimNPO",
    "GradDiff": "GradDiff",
}
_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
method_to_color = {m: _colors[i % len(_colors)] for i, m in enumerate(titles_dict)}
METHODS = [m for m, t in titles_dict.items() if t]

# budget -> method -> wandb run name; all from run_kl0.1.sh with budget= edited
# (top-1 hyperparameters of the paper's 0.01 search; 0.01 rerun so that MMLU is logged)
NEW_BUDGETS = [0.01, 0.03, 0.1, 0.3]
RUNS = {b: {m: f"kl{b}_Llama-3.1-8B_bio_{m}" for m in METHODS} for b in NEW_BUDGETS}
REFERENCE_RUN = "kl0.1_Llama-3.1-8B_bio_reference"


def last_valid_kl(kl, budget):
    kl = np.asarray(kl, dtype=float)
    valid = np.where(kl <= budget)[0]
    return float(kl[valid[-1]]) if len(valid) else None


def fetch():
    """Returns {budget: {method: {kl, robustness, mmlu, source}}} plus a 'reference' entry."""
    from dotenv import load_dotenv
    import wandb

    load_dotenv(SCRIPT_DIR.parent.parent.parent / ".env")
    api = wandb.Api()

    def one_run(project, name):
        runs = list(api.runs(project, filters={"display_name": name}))
        assert len(runs) <= 1, f"{name}: {len(runs)} runs"
        return runs[0] if runs else None

    data = {}

    def flat_lr(run):
        cfg = {k: v for k, v in run.config.items()}
        return cfg.get("trainer", {}).get("args", {}).get("learning_rate") or cfg.get(
            "trainer.args.learning_rate"
        )

    # new budgets from wandb
    for budget in NEW_BUDGETS:
      data[str(budget)] = {}
      for method, name in RUNS[budget].items():
        unl_run = one_run(UNL_PROJECT, name)
        rel_run = one_run(REL_PROJECT, name)
        if unl_run is None or rel_run is None or rel_run.state != "finished":
            print(f"  {budget} {method}: not finished yet, skipping")
            continue
        kl_hist = unl_run.history(keys=[KL_METRIC])[KL_METRIC].dropna().values
        rob_hist = rel_run.history(keys=[ROB_METRIC])[ROB_METRIC].dropna().values
        data[str(budget)][method] = {
            "kl": last_valid_kl(kl_hist, budget),
            "robustness": float(np.max(rob_hist)),
            "mmlu": rel_run.summary.get(MMLU_METRIC),
            "run": name,
            "unlearning_lr": flat_lr(unl_run),
        }

    # RepSelect's single step at the top-1 LR lands at KL ~0.010 +- run-to-run noise: the
    # kl0.01 rerun overshot (0.0106 > 0.01) so its "last valid" checkpoint is the base
    # model. Use the paper's trial for KL / robustness (it landed at 0.0095), and MMLU
    # from the kl0.03 run, whose attacked checkpoint is that same single step (KL 0.0106).
    rs = "RepSelectSimple_forget"
    if data["0.01"].get(rs, {}).get("kl") == 0.0 and rs in data["0.03"]:
        paper = "v5.3_Llama-3.1-8B_bio_RepSelectSimple_forget_26"
        unl_run = one_run(UNL_PROJECT, paper)
        rel_run = one_run(REL_PROJECT, paper)
        data["0.01"][rs] = {
            "kl": last_valid_kl(unl_run.history(keys=[KL_METRIC])[KL_METRIC].dropna().values, 0.01),
            "robustness": float(rel_run.history(keys=[ROB_METRIC])[ROB_METRIC].dropna().max()),
            "mmlu": data["0.03"][rs]["mmlu"],
            "run": paper,
            "note": f"kl0.01 rerun overshot the budget at step 1; mmlu taken from {data['0.03'][rs]['run']}",
        }

    ref = one_run(REL_PROJECT, REFERENCE_RUN)
    rob_hist = ref.history(keys=[ROB_METRIC])[ROB_METRIC].dropna().values
    data["reference"] = {
        "kl": 0.0,
        "robustness": float(np.max(rob_hist)),
        "mmlu": ref.summary.get(MMLU_METRIC),
        "run": REFERENCE_RUN,
    }
    return data


if CACHE_FILE.exists():
    data = json.loads(CACHE_FILE.read_text())
    print(f"Loaded {CACHE_FILE}")
else:
    data = fetch()
    CACHE_FILE.write_text(json.dumps(data, indent=2))
    print(f"Saved {CACHE_FILE}")

# %%
BUDGETS = ["0.01", "0.03", "0.1", "0.3"]
budget_marker = {"0.01": "o", "0.03": "^", "0.1": "s", "0.3": "D"}

fig, (ax_kl, ax_mmlu) = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=True)

ref = data["reference"]
for ax, key in [(ax_kl, "kl"), (ax_mmlu, "mmlu")]:
    ax.scatter(ref[key], ref["robustness"] * 100, marker="*", s=70, color="black", zorder=3)
    for method in METHODS:
        pts = [
            (data[b][method][key], data[b][method]["robustness"] * 100, b)
            for b in BUDGETS
            if method in data[b] and data[b][method][key] is not None
        ]
        if not pts:
            continue
        color = method_to_color[method]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=0.8, alpha=0.6)
        for x, y, b in pts:
            ax.scatter(x, y, marker=budget_marker[b], s=22, color=color, zorder=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax_kl.set_xlabel("WikiText KL of attacked checkpoint")
ax_mmlu.set_xlabel("MMLU accuracy of attacked checkpoint")
ax_mmlu.invert_xaxis()  # more disruption to the right, like the KL panel
ax_kl.set_ylabel("Post-attack answer prob. (%) ↓")

# legend: methods by colour, budgets by marker, reference
from matplotlib.lines import Line2D

handles = [Line2D([], [], color=method_to_color[m], lw=2, label=titles_dict[m]) for m in METHODS]
handles += [
    Line2D([], [], color="gray", marker=budget_marker[b], ls="", label=f"KL budget {b}")
    for b in BUDGETS
]
handles += [Line2D([], [], color="black", marker="*", ms=8, ls="", label="No unlearning")]
plt.tight_layout()
fig.legend(
    handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False,
    fontsize=8, ncol=6, handlelength=1.5, columnspacing=1.0,
)
fig.savefig(OUT_FILE, bbox_inches="tight", dpi=150)
print(f"Saved {OUT_FILE}")
plt.show()
