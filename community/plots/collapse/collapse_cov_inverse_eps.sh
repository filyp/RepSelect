#!/bin/bash
# RepSelectCov inverse (collapse_cov_inverse.sh), weighted cov only, with the ridge of
# each cov set to its 512th eigenvalue (eps_pc=512) instead of 1e-3 x its mean
# diagonal. This is a smooth version of the top-512 collapse: a component with
# eigenvalue e is scaled by ridge / (e + ridge), against min(1, ridge / e) there.
#
# Same protocol otherwise. Reference runs: cov_{exp}_{model}_retain_both_weighted
# (top 512) and cov_{exp}_{model}_retain_both_weighted_inverse (ridge 1e-3).
#
# Everything goes to the H200 141GB (fp32 cov accumulators), posted directly.
#
# Usage: bash community/plots/collapse/collapse_cov_inverse_eps.sh   (from the repo root)

source .venv/bin/activate
run() {
  [[ "$*" == *'"'* ]] && echo "Error: command must not contain double quotes" && exit 1
  curl -X POST "https://tasks.datacrunch.io/open-unlearning3/run" \
    -H "Authorization: Bearer $(cat verda_token.txt)" \
    -H "Content-Type: application/json" \
    -d "{\"command\": \"$*\"}"
}

common="python src/unlearn_relearn.py --config-name=unlearn.yaml trainer=RepSelectCov trainer.handler=RepSelectCovAdaptive trainer.method_args.inverse=true trainer.method_args.weighted=true trainer.method_args.eps_pc=512 ~eval.general_caps"

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
      task_name=cov_${exp_name}_${model}_retain_both_weighted_inverse_eps512
  done
done
