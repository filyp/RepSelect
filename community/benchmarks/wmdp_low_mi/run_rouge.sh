#!/bin/bash
# Follow-up for reviewers (ARR Aug 2026): free-generation ROUGE-1 recall alongside the
# teacher-forced answer probability, to show that the unlearned model cannot *produce*
# the answer either, not merely that it assigns it low probability.
#
# Standard paper setup rederived: Llama-3.1-8B / WMDP-Bio, WikiText KL <= 0.01, 10 epochs,
# top-1 hyperparameters of the main Optuna search (results_bio/Llama-3.1-8B.json).
# The `recall_rouge` evaluator (configs/experiment/unlearn/wmdp_low_mi/default.yaml) runs
# at every epoch of both stages, so each run gives the full pre- and post-attack curve.
# Unlearned checkpoints are not saved, hence the rerun rather than an eval-only pass.

model=Llama-3.1-8B
wmdp_domain=bio
version=rouge1

common="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=${model} wmdp_domain=${wmdp_domain} eval.wikitext_kl.disr_budget=0.01 trainer.args.num_train_epochs=10"
prefix="${version}_${model}_${wmdp_domain}"
# no unlearning: ROUGE of the untouched model, under the same attack
reference="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=${model} wmdp_domain=${wmdp_domain} trainer.args.num_train_epochs=0"

run() {
  bash verda_runner.sh $*
}

###############################################################

# run ${reference} trainer=GradDiff task_name=${prefix}_reference

# # note, the learning rate is lowered, because the original 0.11 is right at the edge of 0.01 KL in one epoch, so sometimes it makes this run unusable (the base model is the latest valid checkpoint)
# run ${common} trainer=RepSelectSimple \
#   trainer.args.learning_rate=0.10 \
#   trainer.method_args.lora_lr=0.05012932753237797 \
#   task_name=${prefix}_RepSelectSimple_forget

# run ${common} trainer=NPO \
#   trainer.args.learning_rate=3.847572780666156e-06 \
#   trainer.method_args.alpha=1.5133906918539597 \
#   trainer.method_args.beta=0.47790320784689266 \
#   task_name=${prefix}_NPO

# run ${common} trainer=GradDiff \
#   trainer.args.learning_rate=4.538706382911242e-06 \
#   trainer.method_args.alpha=6.541067245834811 \
#   task_name=${prefix}_GradDiff

# # module_regex has unescaped dots on purpose: backslashes break the JSON body sent
# # to verda (HTTP 422); RMU uses re.fullmatch, so model.layers.11 matches exactly one module
# run ${common} trainer=RMU \
#   trainer.args.learning_rate=3.388114022223246e-06 \
#   trainer.method_args.module_regex=model.layers.11 \
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


# (finished: RepSelect's ROUGE did not rise further between epochs 10 and 30)
# # note, the learning rate is lowered, because the original 0.11 is right at the edge of 0.01 KL in one epoch, so sometimes it makes this run unusable (the base model is the latest valid checkpoint)
# common="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=${model} wmdp_domain=${wmdp_domain} eval.wikitext_kl.disr_budget=0.01 trainer.args.num_train_epochs=10 relearning_trainer.args.num_train_epochs=30"
# run ${common} trainer=RepSelectSimple \
#   trainer.args.learning_rate=0.10 \
#   trainer.method_args.lora_lr=0.05012932753237797 \
#   task_name=${prefix}_RepSelectSimple_forget_30epoch

###############################################################
# Retain-side ROUGE (reviewer request): same runs as rouge1_* above, same hyperparameters
# (incl. RepSelect's lowered LR), with `retain_rouge` on and `recall_rouge` still off in
# the experiment yaml, so the forget-side generations are not repeated.
version=rouge_retain
prefix="${version}_${model}_${wmdp_domain}"

run ${reference} trainer=GradDiff task_name=${prefix}_reference

run ${common} trainer=RepSelectSimple \
  trainer.args.learning_rate=0.10 \
  trainer.method_args.lora_lr=0.05012932753237797 \
  task_name=${prefix}_RepSelectSimple_forget

run ${common} trainer=NPO \
  trainer.args.learning_rate=3.847572780666156e-06 \
  trainer.method_args.alpha=1.5133906918539597 \
  trainer.method_args.beta=0.47790320784689266 \
  task_name=${prefix}_NPO

run ${common} trainer=GradDiff \
  trainer.args.learning_rate=4.538706382911242e-06 \
  trainer.method_args.alpha=6.541067245834811 \
  task_name=${prefix}_GradDiff

run ${common} trainer=RMU \
  trainer.args.learning_rate=3.388114022223246e-06 \
  trainer.method_args.module_regex=model.layers.11 \
  trainer.method_args.steering_coeff=1.1494196400307433 \
  task_name=${prefix}_RMU

run ${common} trainer=SimNPO \
  trainer.args.learning_rate=1.0963072160039623e-06 \
  trainer.method_args.beta=4.1169339968747565 \
  trainer.method_args.delta=0.9437480785146242 \
  trainer.method_args.gamma=0.21022753738793545 \
  task_name=${prefix}_SimNPO

run ${common} trainer=UNDIAL \
  trainer.args.learning_rate=4.423136522639717e-06 \
  trainer.method_args.alpha=2.543875814472088 \
  trainer.method_args.beta=5.7685847199969285 \
  task_name=${prefix}_UNDIAL
