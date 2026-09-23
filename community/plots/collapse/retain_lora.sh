#!/bin/bash
# Does LoRA elicitation help the recommended variant (retain / both / quadratic)?
# Sycophancy and Animal Abuse x Llama, Qwen, DeepSeek; LoRA on vs off.
# Step size via RepSelectAdaptive (bisected onto WikiText KL = 0.01), n_pcs=512.
# LoRA LR: top-1 of each model's Optuna search, RepSelectSimple_forget in
# {sycophancy,beavertails}/results/<model>.json (tuned for forget / soft, not retrained here).
#
# The no-LoRA half already exists and is not rerun (block at the bottom, commented):
# collapse2_{sycophancy,aa}_<model>_retain_both_quadratic, same settings with use_lora=false.
# Without LoRA the retain SVD is unaffected by where it sits relative to the elicitation,
# so those runs are on the same code path as today.

source .venv/bin/activate
run() {
  bash verda_runner.sh $*
}

common="python src/unlearn_relearn.py --config-name=unlearn.yaml trainer=RepSelectSimple trainer.handler=RepSelectAdaptive ~eval.general_caps ~eval.fewshot_attack_5 trainer.method_args.distribution=retain trainer.method_args.collapse_on=both trainer.method_args.hard_soft=quadratic trainer.method_args.n_pcs=512"
syco="experiment=unlearn/sycophancy/default"
aa="experiment=unlearn/beavertails/curated_contrast category=animal_abuse"

# (model, dataset tag, experiment overrides, lora_lr)
configs=(
  "Llama-3.1-8B sycophancy syco 0.04141775214615277"
  "Qwen3.5-9B sycophancy syco 0.044949535848587574"
  "DeepSeek-V2-Lite sycophancy syco 1.609140546466229"
  "Llama-3.1-8B aa aa 0.03295172397738558"
  "Qwen3.5-9B aa aa 0.20319517493171765"
  "DeepSeek-V2-Lite aa aa 7.319141553558477"
)

for cfg in "${configs[@]}"; do
  set -- ${cfg}
  model=$1; ds=$2; exp=${!3}; lora_lr=$4
  run ${common} ${exp} model=${model} \
    trainer.method_args.use_lora=true \
    trainer.method_args.lora_lr=${lora_lr} \
    task_name=retainlora_${ds}_${model}_retain_both_quadratic_lora
done

# # no-LoRA counterparts, already on wandb as collapse2_{ds}_{model}_retain_both_quadratic:
# for cfg in "${configs[@]}"; do
#   set -- ${cfg}
#   model=$1; ds=$2; exp=${!3}
#   run ${common} ${exp} model=${model} \
#     trainer.method_args.use_lora=false \
#     task_name=retainlora_${ds}_${model}_retain_both_quadratic_nolora
# done
