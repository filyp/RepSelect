#!/bin/bash
# Rerun of the collapse-design grid (paper Fig. 7 / fig:collapse_grid, plotted by
# collapse_grid.py) with the *quadratic* collapse instead of the soft one.
#
# Everything else is identical to collapse.sh, which produced the Fig. 7 runs
# (`collapse_{exp}_{model}_{dist}_{collapse}` on wandb):
#   RepSelectAdaptive (bisects the step size onto the KL budget), no LoRA,
#   defaults of RepSelectSimple.yaml otherwise, 10 relearning epochs.
# One model per invocation (picker below); the 3 datasets missing quadratic runs; 6 variants each. The
# "no collapse" row is unaffected by hard_soft, so collapse_grid.py reuses the
# soft runs for it and it is not rerun here.
#
# Comparability notes (from the wandb configs of the Fig. 7 runs):
#   - n_pcs MISMATCH on bio and aa: their soft Fig. 7 runs (2026-04-24, commit
#     2e5eec9) used n_pcs=500, while their existing quadratic runs (2026-07-26)
#     used 512. Deliberately left as is: July's soft bio reruns at 512
#     (collapse2_bio_*) reproduce the April 500 runs to within 0.03 points, and
#     Fig. 8 shows a flat plateau over 128-1024. cyber (07-28), rwku and
#     sycophancy (07-30) used 512 for soft, and 512 here, so they match.
#   - The collapse code is unchanged since July; pipeline changes since are
#     evaluator additions and refactors that do not touch training.
#   - MMLU and few-shot evals are dropped here: they only run at relearn epoch 0,
#     do not affect training or the robustness metric, and cost most of the time.
#
# Run names keep the collapse2_ prefix of the July quadratic runs, so the plot
# script (hard_soft = "quadratic") picks up old and new cells alike.

source .venv/bin/activate
run() {
  bash verda_runner.sh $*
}

# model picker: `bash collapse_quadratic_rest.sh Qwen3.5-9B` (default Llama-3.1-8B).
# verda_runner.sh picks the GPU from the model name in the command.
model=${1:-Llama-3.1-8B}
hard_soft=quadratic

# same overrides as collapse.sh, plus the eval trims
common="python src/unlearn_relearn.py --config-name=unlearn.yaml trainer=RepSelectSimple trainer.handler=RepSelectAdaptive trainer.method_args.use_lora=false model=${model} ~eval.general_caps"

###############################################################
# Regression test: repeat one July quadratic run on today's code. Compare max
# train/recall_prob over the relearn history with the original
# collapse2_bio_Llama-3.1-8B_forget_act_quadratic (4.50%).
# (Llama only; done, reproduced 4.50%.)
if [[ ${model} == Llama-3.1-8B ]]; then
  run ${common} experiment=unlearn/wmdp_low_mi/default ~eval.fewshot_attack_5 \
    trainer.method_args.hard_soft=quadratic \
    trainer.method_args.n_pcs=512 \
    trainer.method_args.distribution=forget \
    trainer.method_args.collapse_on=act \
    task_name=collapse2_bio_${model}_forget_act_quadratic_regression
fi

###############################################################
# The quadratic grid for the datasets that did not exist in July. Bio and aa
# already have quadratic runs from 2026-07-26 (collapse2_{bio,aa}_*_quadratic,
# commit fd6dfb0): the quadratic collapse code is unchanged since, and July's
# soft reruns of bio (collapse2_bio_*, n_pcs=512) reproduce the April Fig. 7
# runs (n_pcs=500) to within 0.03 points, so neither a rerun nor a separate
# regression run is needed. n_pcs=512 matches the Fig. 7 runs of these datasets.
# (experiment, exp_name used in the run name, n_pcs, extra overrides)
configs=(
  "unlearn/wmdp_low_mi/default cyber 512 wmdp_domain=cyber ~eval.fewshot_attack_5"
  "unlearn/rwku/default rwku 512"
  "unlearn/sycophancy/default sycophancy 512 ~eval.fewshot_attack_5"
)

for cfg in "${configs[@]}"; do
  set -- ${cfg}
  experiment=$1; exp_name=$2; n_pcs=$3; shift 3; extra="$*"
  for dist in forget retain; do
    for collapse in act grad both; do
      run ${common} experiment=${experiment} ${extra} \
        trainer.method_args.hard_soft=${hard_soft} \
        trainer.method_args.n_pcs=${n_pcs} \
        trainer.method_args.distribution=${dist} \
        trainer.method_args.collapse_on=${collapse} \
        task_name=collapse2_${exp_name}_${model}_${dist}_${collapse}_${hard_soft}
    done
  done
done
