#!/bin/bash
# Follow-up for reviewers (ARR Aug 2026): a published relearning-robust baseline,
# NPO + sharpness-aware minimization (Fan et al., ICML 2025), under our protocol.
#
# Standard paper setup: Llama-3.1-8B / WMDP-Bio, WikiText KL <= 0.01, 10 epochs.
# SAM is a modifier on NPO, so NPO's top-1 hyperparameters from the main Optuna
# search (results_bio/Llama-3.1-8B.json) are reused unchanged and only rho is added:
# 0.01 is the paper's default, 0.1 the top of its sweep. No new search.
# The comparison point is the kl0.01_*_NPO run (same hyperparameters). ROUGE is off (yaml).

model=Llama-3.1-8B
wmdp_domain=bio
version=kl0.01  # same prefix as the kl0.01_* baselines so the sweep plot picks them up

common="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=${model} wmdp_domain=${wmdp_domain} eval.wikitext_kl.disr_budget=0.01 trainer.args.num_train_epochs=10"
prefix="${version}_${model}_${wmdp_domain}"

run() {
  bash verda_runner.sh $*
}

###############################################################

run ${common} trainer=NPOSAM \
  trainer.args.learning_rate=3.847572780666156e-06 \
  trainer.method_args.alpha=1.5133906918539597 \
  trainer.method_args.beta=0.47790320784689266 \
  trainer.method_args.sam_rho=0.01 \
  task_name=${prefix}_NPOSAM_rho0.01

run ${common} trainer=NPOSAM \
  trainer.args.learning_rate=3.847572780666156e-06 \
  trainer.method_args.alpha=1.5133906918539597 \
  trainer.method_args.beta=0.47790320784689266 \
  trainer.method_args.sam_rho=0.1 \
  task_name=${prefix}_NPOSAM_rho0.1
