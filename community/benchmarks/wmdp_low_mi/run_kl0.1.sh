#!/bin/bash
# Follow-up for reviewers (ARR Aug 2026): rerun the Llama-3.1-8B / Bio comparison
# under a 10x looser disruption budget (wikitext KL <= 0.1 instead of 0.01).
#
# - hyperparameters: the top-1 trial of the KL=0.01 Optuna search, taken from
#   results_bio/Llama-3.1-8B.json (no new search)
# - num_train_epochs raised 10 -> 100 so methods have room to reach the budget
# - attack, MMLU (general_caps) and few-shot evals unchanged: they run at
#   relearn epoch 0 on the unlearned model, then the same fine-tuning attack

model=Llama-3.1-8B
wmdp_domain=bio
# budget=0.03
# budget=0.1
budget=0.3
version=kl${budget}

# ifeval is disabled in the experiment yaml (uninformative on base models); if re-enabled,
# its deps (lm-eval[ifeval] extra) are not in the verda image:
# deps="pip install -q langdetect immutabledict nltk &&"
deps=""

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
  trainer.method_args.distribution=forget trainer.method_args.hard_soft=soft \
  trainer.args.learning_rate=0.11564327765540657 \
  trainer.method_args.lora_lr=0.05012932753237797 \
  task_name=${prefix}_RepSelectSimple_forget
# (the json also lists peft_config.default.target_modules=[gate_proj,up_proj,down_proj];
#  that key no longer exists, the modules are hardcoded in RepSelectSimple to the same value)

# remaining baselines (top-1 trials), to add later:
run ${common} trainer=GradDiff \
  trainer.args.learning_rate=4.538706382911242e-06 \
  trainer.method_args.alpha=6.541067245834811 \
  task_name=${prefix}_GradDiff

# module_regex has unescaped dots on purpose: backslashes break the JSON body sent
# to verda (HTTP 422); RMU uses re.fullmatch, so model.layers.11 matches exactly one module
run ${common} trainer=RMU \
  trainer.args.learning_rate=3.388114022223246e-06 \
  trainer.method_args.module_regex=model.layers.11 \
  trainer.method_args.steering_coeff=1.1494196400307433 \
  task_name=${prefix}_RMU

# Note, SimNPO's LR tuned on the KL=0.01 search was much too low and couldn't complete in 100 epochs. That's why we retried it with 10x larger LR.
run ${common} trainer=SimNPO \
  trainer.args.learning_rate=1.0963072160039623e-05 \
  trainer.method_args.beta=4.1169339968747565 \
  trainer.method_args.delta=0.9437480785146242 \
  trainer.method_args.gamma=0.21022753738793545 \
  task_name=${prefix}_SimNPO

run ${common} trainer=UNDIAL \
  trainer.args.learning_rate=4.423136522639717e-06 \
  trainer.method_args.alpha=2.543875814472088 \
  trainer.method_args.beta=5.7685847199969285 \
  task_name=${prefix}_UNDIAL

###############################################################
# Same top-1 trials at the paper's budget (0.01, 10 epochs), rerun only to log
# MMLU at relearn epoch 0 (the original search runs predate the MMLU eval).
# Needed for the MMLU panel of community/plots/kl_budget_sweep/kl_budget_sweep.py.
common01="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=${model} wmdp_domain=${wmdp_domain} eval.wikitext_kl.disr_budget=0.01 trainer.args.num_train_epochs=10"
prefix01="kl0.01_${model}_${wmdp_domain}"
# run ${common01} trainer=NPO \
#   trainer.args.learning_rate=3.847572780666156e-06 \
#   trainer.method_args.alpha=1.5133906918539597 \
#   trainer.method_args.beta=0.47790320784689266 \
#   task_name=${prefix01}_NPO
# # # note: the LR is slightly smaller, than the 0.11 value from Optuna search, because the original one can randomly overshoot the kl0.01 target in one epoch, which makes relearning be run on the base model
# run ${common01} trainer=RepSelectSimple \
#   trainer.method_args.distribution=forget trainer.method_args.hard_soft=soft \
#   trainer.args.learning_rate=0.1 \
#   trainer.method_args.lora_lr=0.05012932753237797 \
#   task_name=${prefix01}_RepSelectSimple_forget
# run ${common01} trainer=GradDiff \
#   trainer.args.learning_rate=4.538706382911242e-06 \
#   trainer.method_args.alpha=6.541067245834811 \
#   task_name=${prefix01}_GradDiff
# run ${common01} trainer=RMU \
#   trainer.args.learning_rate=3.388114022223246e-06 \
#   trainer.method_args.module_regex=model.layers.11 \
#   trainer.method_args.steering_coeff=1.1494196400307433 \
#   task_name=${prefix01}_RMU
# run ${common01} trainer=SimNPO \
#   trainer.args.learning_rate=1.0963072160039623e-06 \
#   trainer.method_args.beta=4.1169339968747565 \
#   trainer.method_args.delta=0.9437480785146242 \
#   trainer.method_args.gamma=0.21022753738793545 \
#   task_name=${prefix01}_SimNPO
# run ${common01} trainer=UNDIAL \
#   trainer.args.learning_rate=4.423136522639717e-06 \
#   trainer.method_args.alpha=2.543875814472088 \
#   trainer.method_args.beta=5.7685847199969285 \
#   task_name=${prefix01}_UNDIAL

###############################################################
# Budget 0.003, RepSelect only. The update is a cached gradient applied once per epoch,
# so a smaller LR just gives a finer step ladder (n steps at LR/n == 1 step at LR) and
# lands closer to the budget instead of overshooting it. One step at the top-1 LR
# (0.1156) gives KL ~0.010; KL grows ~quadratically, so 0.02 gives ~0.0003 per step
# and should stop after ~3 steps at KL ~0.003.
common003="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=${model} wmdp_domain=${wmdp_domain} eval.wikitext_kl.disr_budget=0.003 trainer.args.num_train_epochs=100"
run ${common003} trainer=RepSelectSimple \
  trainer.method_args.distribution=forget trainer.method_args.hard_soft=soft \
  trainer.args.learning_rate=0.02 \
  trainer.method_args.lora_lr=0.05012932753237797 \
  task_name=kl0.003_${model}_${wmdp_domain}_RepSelectSimple_forget
