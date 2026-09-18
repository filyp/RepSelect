#!/bin/bash
# Follow-up for reviewers (ARR Aug 2026): rerun the Llama-3.1-8B / Bio comparison
# under a 10x looser disruption budget (wikitext KL <= 0.1 instead of 0.01).
#
# - hyperparameters: the top-1 trial of the KL=0.01 Optuna search, taken from
#   results_bio/Llama-3.1-8B.json (no new search)
# - num_train_epochs raised 10 -> 100 so methods have room to reach the budget
# - attack, MMLU (general_caps) and few-shot evals unchanged, plus new IFEval:
#   all run at relearn epoch 0 on the unlearned model, then the same fine-tuning attack

model=Llama-3.1-8B
wmdp_domain=bio
budget=0.1
version=kl${budget}

# ifeval deps (lm-eval[ifeval] extra) are not in the verda image; installed at job start
deps="pip install -q langdetect immutabledict nltk &&"

common="${deps} python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=${model} wmdp_domain=${wmdp_domain} eval.wikitext_kl.disr_budget=${budget} trainer.args.num_train_epochs=100"
prefix="${version}_${model}_${wmdp_domain}"
# no unlearning: baseline for MMLU / IFEval / few-shot / attack on the base model
reference="${deps} python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=${model} wmdp_domain=${wmdp_domain} trainer.args.num_train_epochs=0"

run() {
  bash verda_runner.sh $*
}

###############################################################

run ${reference} trainer=GradDiff task_name=${prefix}_reference

# NPO (trial 24 of v5_Llama-3.1-8B_bio_NPO)
run ${common} trainer=NPO \
  trainer.args.learning_rate=3.847572780666156e-06 \
  trainer.method_args.alpha=1.5133906918539597 \
  trainer.method_args.beta=0.47790320784689266 \
  task_name=${prefix}_NPO

# RepSelect (trial 26 of v5.3_Llama-3.1-8B_bio_RepSelectSimple_forget)
run ${common} trainer=RepSelectSimple \
  trainer.args.learning_rate=0.11564327765540657 \
  trainer.method_args.lora_lr=0.05012932753237797 \
  task_name=${prefix}_RepSelectSimple_forget
# (the json also lists peft_config.default.target_modules=[gate_proj,up_proj,down_proj];
#  that key no longer exists, the modules are hardcoded in RepSelectSimple to the same value)

# # remaining baselines (top-1 trials), to add later:
# run ${common} trainer=GradDiff \
#   trainer.args.learning_rate=4.538706382911242e-06 \
#   trainer.method_args.alpha=6.541067245834811 \
#   task_name=${prefix}_GradDiff
# run ${common} trainer=RMU \
#   trainer.args.learning_rate=3.388114022223246e-06 \
#   trainer.method_args.module_regex=model\\.layers\\.11 \
#   trainer.method_args.steering_coeff=1.1494196400307433 \
#   task_name=${prefix}_RMU
# run ${common} trainer=SimNPO \
#   trainer.args.learning_rate=1.0963072160039623e-06 \
#   trainer.method_args.beta=4.1169339968747565 \
#   trainer.method_args.delta=0.9437480785146242 \
#   trainer.method_args.gamma=0.21022753738793545 \
#   task_name=${prefix}_SimNPO
# run ${common} trainer=UNDIAL \
#   trainer.args.learning_rate=4.423136522639717e-06 \
#   trainer.method_args.alpha=2.543875814472088 \
#   trainer.method_args.beta=5.7685847199969285 \
#   task_name=${prefix}_UNDIAL
