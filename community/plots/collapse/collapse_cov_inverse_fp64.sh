#!/bin/bash
# RepSelectCov inverse (collapse_cov_inverse.sh), weighted cov only, with each batch's
# cov computed in fp64 (acts and grads cast to fp64) instead of fp32 (cov_fp64=true).
# The accumulators stay fp32, the ridge stays 1e-3 x mean diagonal.
#
# Same protocol otherwise. Reference runs: cov_{exp}_{model}_retain_both_weighted_inverse
# (fp32 products).
#
# Everything goes to the H200 141GB, posted directly.
#
# Usage: bash community/plots/collapse/collapse_cov_inverse_fp64.sh   (from the repo root)

source .venv/bin/activate
run() {
  [[ "$*" == *'"'* ]] && echo "Error: command must not contain double quotes" && exit 1
  curl -X POST "https://tasks.datacrunch.io/open-unlearning3/run" \
    -H "Authorization: Bearer $(cat verda_token.txt)" \
    -H "Content-Type: application/json" \
    -d "{\"command\": \"$*\"}"
}

common="python src/unlearn_relearn.py --config-name=unlearn.yaml trainer=RepSelectCov trainer.handler=RepSelectCovAdaptive trainer.method_args.inverse=true trainer.method_args.weighted=true trainer.method_args.cov_fp64=true ~eval.general_caps"

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
    run ${common} experiment=${experiment} ${extra} \
      model=${model} \
      task_name=cov_${exp_name}_${model}_retain_both_weighted_inverse_fp64
  done
done
