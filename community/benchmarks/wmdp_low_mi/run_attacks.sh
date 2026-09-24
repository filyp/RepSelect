#!/bin/bash
# Follow-up for reviewer ydez (ARR Aug 2026): attacks beyond adversarial fine-tuning.
# Llama-3.1-8B / WMDP-Bio, WikiText KL <= 0.01, same unlearning setup and hyperparameters
# as the kl0.01_* runs (RepSelect at LR 0.02 walking up to the budget, NPO top-1).
#
#   nf4, int8:   quantization attack (src/model/quantize.py) on the unlearned model,
#                evaluated without any fine-tuning (relearning epochs = 0)
#   benign:      relearning on held-out FineFineWeb biology text (benign, same domain;
#                docs 2000-2282, disjoint from retain / retain_eval), same attack
#                settings as the adversarial fine-tuning (10 epochs, LR 7e-6)
#   unrelated:   the same on WikiText (docs 1000-1282, disjoint from the KL eval set)
# The comparison is the adversarial fine-tuning of kl0.01_* (same unlearning runs).
# Reference (no unlearning) for every attack; for quantization it gives the answer
# probability of the quantized base model, i.e. what full recovery would look like.

model=Llama-3.1-8B
wmdp_domain=bio
prefix="attack_${model}_${wmdp_domain}"

run() {
  bash verda_runner.sh $*
}

common="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=${model} wmdp_domain=${wmdp_domain} eval.wikitext_kl.disr_budget=0.01"
repselect="${common} trainer=RepSelectSimple trainer.method_args.distribution=forget trainer.method_args.hard_soft=soft trainer.args.num_train_epochs=100 trainer.args.learning_rate=0.02 trainer.method_args.lora_lr=0.05012932753237797"
npo="${common} trainer=NPO trainer.args.num_train_epochs=10 trainer.args.learning_rate=3.847572780666156e-06 trainer.method_args.alpha=1.5133906918539597 trainer.method_args.beta=0.47790320784689266"
reference="${common} trainer=GradDiff trainer.args.num_train_epochs=0"

quant() { echo "relearn_quantize=$1 relearning_trainer.args.num_train_epochs=0"; }
relearn_on() { echo "data.relearn_split=$1"; }

###############################################################
for attack in nf4 int8; do
  run ${repselect} $(quant ${attack}) task_name=${prefix}_${attack}_RepSelectSimple_forget
  run ${npo} $(quant ${attack}) task_name=${prefix}_${attack}_NPO
  run ${reference} $(quant ${attack}) task_name=${prefix}_${attack}_reference
done

for pair in "benign benign_relearn" "unrelated unrelated_relearn"; do
  set -- ${pair}
  run ${repselect} $(relearn_on $2) task_name=${prefix}_$1_RepSelectSimple_forget
  run ${npo} $(relearn_on $2) task_name=${prefix}_$1_NPO
  run ${reference} $(relearn_on $2) task_name=${prefix}_$1_reference
done

# The standard adversarial fine-tuning on this same unlearned RepSelect model (LR 0.02
# walk-up), so the table's adversarial row starts from the same checkpoint as the others;
# kl0.01_*_RepSelectSimple_forget used LR 0.10 (one step, KL 0.0081, 1.85%).
run ${repselect} task_name=${prefix}_adversarial_RepSelectSimple_forget
