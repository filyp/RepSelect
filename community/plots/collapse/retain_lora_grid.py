# %%
"""LoRA elicitation on vs off for the recommended variant (retain / both / quadratic).

Rows = models, columns = Sycophancy and Animal Abuse. Each cell: post-attack answer
probability (max over the 10-epoch relearning attack) without LoRA
(collapse2_{ds}_{model}_retain_both_quadratic, from the collapse grid) and with LoRA
(retainlora_{ds}_{model}_retain_both_quadratic_lora, from retain_lora.sh), drawn as
bars from the no-unlearning reference like collapse_grid.py.
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

import wandb

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).parent
REL_PROJECT = "filyp/rel-selective-unlearning"
REL_STEPS = 11  # epoch 0 + 10 relearning epochs

_BENCHMARKS_DIR = SCRIPT_DIR.parent.parent / "benchmarks"
baselines = {}
for _p in ["beavertails", "sycophancy"]:
    with open(_BENCHMARKS_DIR / _p / "baselines.yaml") as _f:
        baselines.update(yaml.safe_load(_f))

plt.style.use("default")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 10

MODELS = ["Llama-3.1-8B", "Qwen3.5-9B", "DeepSeek-V2-Lite"]
# (exp name in run names, display, baseline tag, metric)
BENCHMARKS = [
    ("sycophancy", "Sycophancy", "sycophancy", "train/recall_prob"),
    ("aa", "Animal Abuse", "animal_abuse", "train/holdout_harmful_prob"),
]
ROWS = [
    ("no LoRA", "collapse2_{ds}_{model}_retain_both_quadratic"),
    ("LoRA", "retainlora_{ds}_{model}_retain_both_quadratic_lora"),
]

api = wandb.Api(timeout=600)


def max_over_attack(name, metric):
    runs = [r for r in api.runs(REL_PROJECT, filters={"display_name": name}) if r.state == "finished"]
    assert len(runs) == 1, f"{name}: {len(runs)} finished runs"
    for i in range(6):
        try:
            hist = runs[0].history(keys=[metric])
            break
        except Exception as e:
            print(f"  retry {i}: {e}")
            time.sleep(2**i)
    return hist.head(REL_STEPS)[metric].dropna().max() * 100


# %%
results = {}  # (model, ds) -> [no-LoRA, LoRA] in %
for model in MODELS:
    for ds, _, _, metric in BENCHMARKS:
        results[(model, ds)] = [max_over_attack(t.format(ds=ds, model=model), metric) for _, t in ROWS]
        print(f"{model:18s} {ds:11s} " + "  ".join(f"{l}={v:5.2f}%" for (l, _), v in zip(ROWS, results[(model, ds)])))

# %%
fig, axes = plt.subplots(len(MODELS), len(BENCHMARKS), figsize=(3.3, 0.6 + 0.85 * len(MODELS)))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
bar_colors = [colors[2], colors[4]]  # green = retain/both like the collapse grid; purple = +LoRA

for r, model in enumerate(MODELS):
    for c, (ds, display, tag, _) in enumerate(BENCHMARKS):
        ax = axes[r][c]
        baseline = baselines[tag][model] * 100
        vals = results[(model, ds)]
        y = list(range(len(ROWS) - 1, -1, -1))
        ax.barh(y, [v - baseline for v in vals], left=baseline, height=0.8, color=bar_colors, hatch="///", edgecolor="white")
        lo = min(vals)
        ax.set_xlim(lo - (baseline - lo) * 0.08, baseline)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if r == 0:
            ax.set_title(display)
        if c == 0:
            ax.set_ylabel(model, fontsize=8)
        if c == len(BENCHMARKS) - 1:
            ax.set_yticks(y)
            ax.set_yticklabels([l for l, _ in ROWS])
            ax.yaxis.tick_right()
        else:
            ax.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
fig.text(0.5, -0.02, "Post-Attack Answer Probability (%) ↓", ha="center", va="bottom")
save_path = SCRIPT_DIR / "retain_lora_grid.pdf"
fig.savefig(save_path, bbox_inches="tight", dpi=150)
print(f"Saved plot to {save_path}")
plt.show()
