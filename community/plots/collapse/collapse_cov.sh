#!/bin/bash
# Collapse along per-token retain covariances (RepSelectCov) instead of the SVD of
# the summed retain gradient: weighted and unweighted cov, on the retain / both /
# quadratic cell of the collapse-design grid.
#
# Same protocol as that grid (collapse_quadratic_rest.sh): step size bisected onto
# the KL budget (RepSelectCovAdaptive), no LoRA, n_pcs=512, 10 relearning epochs,
# MMLU and few-shot evals dropped (they do not affect training or the metric).
# Reference runs: collapse2_{exp}_{model}_retain_both_quadratic.
#
# Dense models only (no DeepSeek). verda_runner.sh picks the GPU from the model
# name: Llama-3.1-8B -> RTX PRO 6000 96GB, Qwen3.5-9B -> H200 141GB.
#
# Usage: bash community/plots/collapse/collapse_cov.sh   (from the repo root)

source .venv/bin/activate
run() {
  bash verda_runner.sh $*
}

common="python src/unlearn_relearn.py --config-name=unlearn.yaml trainer=RepSelectCov trainer.handler=RepSelectCovAdaptive trainer.method_args.n_pcs=512 ~eval.general_caps"

# (experiment, exp_name used in the run name, extra overrides)
configs=(
  "unlearn/wmdp_low_mi/default bio ~eval.fewshot_attack_5"
  "unlearn/wmdp_low_mi/default cyber wmdp_domain=cyber ~eval.fewshot_attack_5"
  "unlearn/rwku/default rwku"
  "unlearn/sycophancy/default sycophancy ~eval.fewshot_attack_5"
  "unlearn/beavertails/curated_contrast aa ~eval.fewshot_attack_5"
)

for model in Llama-3.1-8B Qwen3.5-9B; do
  for cfg in "${configs[@]}"; do
    set -- ${cfg}
    experiment=$1; exp_name=$2; shift 2; extra="$*"
    for weighted in true false; do
      if [[ ${weighted} == true ]]; then variant=weighted; else variant=unweighted; fi
      run ${common} experiment=${experiment} ${extra} \
        model=${model} \
        trainer.method_args.weighted=${weighted} \
        task_name=cov_${exp_name}_${model}_retain_both_${variant}
    done
  done
done
