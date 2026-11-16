#!/bin/bash
# RepSelectCov with the full-rank ridge inverses of the per-token retain covariances
# (cov_grad^-1 G cov_act^-1, as the init of als.solve) instead of the collapse along
# their top 512 eigenvectors: weighted and unweighted cov.
#
# Same protocol as collapse_cov.sh, which ran the top-512 variant: step size bisected
# onto the KL budget (RepSelectCovAdaptive), no LoRA, 10 relearning epochs, MMLU and
# few-shot evals dropped (they do not affect training or the metric).
# Reference runs: cov_{exp}_{model}_retain_both_{variant} (top 512) and
# collapse2_{exp}_{model}_retain_both_quadratic.
#
# Dense models only (no DeepSeek). Everything goes to the H200 141GB: the inverse
# needs fp32 cov accumulators (~46 GB for Llama-3.1-8B), too tight for the RTX PRO
# 6000 96GB that verda_runner.sh would pick for Llama. So this script posts to the
# H200 endpoint itself instead of calling verda_runner.sh.
#
# Usage: bash community/plots/collapse/collapse_cov_inverse.sh   (from the repo root)

source .venv/bin/activate
run() {
  [[ "$*" == *'"'* ]] && echo "Error: command must not contain double quotes" && exit 1
  curl -X POST "https://tasks.datacrunch.io/open-unlearning3/run" \
    -H "Authorization: Bearer $(cat verda_token.txt)" \
    -H "Content-Type: application/json" \
    -d "{\"command\": \"$*\"}"
}

common="python src/unlearn_relearn.py --config-name=unlearn.yaml trainer=RepSelectCov trainer.handler=RepSelectCovAdaptive trainer.method_args.inverse=true ~eval.general_caps"

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
        task_name=cov_${exp_name}_${model}_retain_both_${variant}_inverse
    done
  done
done
